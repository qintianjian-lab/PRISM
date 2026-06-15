import os

import h5py
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Resize

MAG_PREFIX = "magnitudes"
MAG_DIFF_PREFIX = "magnitudes_diff"


class BuildDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root_dir: str,
        label_dir_name: str,
        photo_min_max: tuple[tuple[float]],
        target_photo_size: int,
        hdf5_file_name: str | None = None,
        load_mode: str = "train",
        eps: float | int = 1e-8,
    ):
        """
        Build a dataloader for the dataset.
        :param dataset_root_dir: dataset root directory
        :param label_dir_name : label directory name
        :param photo_min_max: min and max value for photometric data normalization, tuple of tuples with length equal to number of photometric channels
        :param target_photo_size: target size for photometric data, if not equal to the original size, will be resized
        :param hdf5_file_name: optional default hdf5 filename (e.g. dataset.h5) if csv has no h5_file column
        :param load_mode: mode to load the dataset, can be "train", "val", or "test"
        :param eps: epsilon value for numerical stability, default is 1e-8
        """
        super().__init__()
        assert load_mode in [
            "train",
            "val",
            "test",
        ], ValueError("load_mode must be in ['train', 'val', 'test']")

        self.dataset_root_dir = dataset_root_dir
        self.hdf5_file_name = hdf5_file_name
        self.label_df = pd.read_csv(
            os.path.join(
                dataset_root_dir,
                label_dir_name,
                f"{load_mode}.csv",
            ),
            header=0,
            converters={
                "TARGETID": str,
                "TARGET_RA": str,
                "TARGET_DEC": str,
                "Z": float,
                "h5_index": int,
                "south_north_flag": str,
            },
        )
        self.eps = eps
        self.train = load_mode == "train"
        self.photo_min_max = np.asarray(photo_min_max, dtype=np.float32)
        self.target_photo_size = target_photo_size
        self.resize_transform = Resize(
            size=target_photo_size,
            antialias=False,
        )
        self._h5_handles: dict[str, h5py.File] = {}

        # Pre-extract all fields
        self.target_ids = self.label_df["TARGETID"].values
        self.target_ras = self.label_df["TARGET_RA"].values
        self.target_decs = self.label_df["TARGET_DEC"].values
        self.zs = self.label_df["Z"].values.astype(np.float32)
        self.h5_indices = self.label_df["h5_index"].values.astype(np.int64)
        self.south_north_flags = self.label_df["south_north_flag"].values

        # Pre-extract mag data
        self.mag_cols = [
            c
            for c in self.label_df.columns
            if c.startswith(MAG_PREFIX) and not c.startswith(MAG_DIFF_PREFIX)
        ]
        self.mag_diff_cols = [
            c for c in self.label_df.columns if c.startswith(MAG_DIFF_PREFIX)
        ]
        self.mags_array = self.label_df[self.mag_cols].values.astype(np.float32)
        self.mags_diff_array = self.label_df[self.mag_diff_cols].values.astype(
            np.float32
        )

        # Pre-resolve h5 paths for all rows
        if "h5_file" in self.label_df.columns:
            h5_files = (
                self.label_df["h5_file"]
                .fillna(self.hdf5_file_name if self.hdf5_file_name else "")
                .values
            )
        else:
            if not self.hdf5_file_name:
                raise RuntimeError(
                    "Cannot resolve hdf5 path: neither csv h5_file nor hdf5_file_name is provided."
                )
            h5_files = np.array([self.hdf5_file_name] * len(self.label_df))

        self.resolved_h5_paths = []
        for p in h5_files:
            h5_path = str(p)
            if not h5_path.strip():
                raise RuntimeError(
                    "Cannot resolve hdf5 path: neither csv h5_file nor hdf5_file_name is provided."
                )
            if not os.path.isabs(h5_path):
                h5_path = os.path.join(self.dataset_root_dir, h5_path)
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
            self.resolved_h5_paths.append(h5_path)

    def __len__(self):
        return len(self.label_df)

    def _get_h5_handle(self, h5_path: str) -> h5py.File:
        handle = self._h5_handles.get(h5_path)
        if handle is None:
            handle = h5py.File(h5_path, "r")
            self._h5_handles[h5_path] = handle
        return handle

    def _load_photo_from_hdf5(
        self,
        h5_path: str,
        h5_index: int,
        nan_to_num: bool = False,
        min_max_normalize: bool = True,
    ) -> torch.Tensor:
        h5f = self._get_h5_handle(h5_path)
        photo_flat = h5f["photo_flat"][h5_index]
        photo_shape = h5f["photo_shape"][h5_index]
        _data = np.asarray(photo_flat, dtype=np.float32).reshape(
            int(photo_shape[0]),
            int(photo_shape[1]),
            int(photo_shape[2]),
        )
        if nan_to_num:
            _data = np.nan_to_num(_data, nan=0, posinf=0, neginf=0)
        if min_max_normalize:
            assert self.photo_min_max.shape[-1] == 2, RuntimeError(
                f"photo_min_max last dimension must be 2 (e.g. min, max), got {self.photo_min_max.shape[-1]}"
            )
            assert self.photo_min_max.ndim == 2, RuntimeError(
                f"photo_min_max must be 2D array, got {self.photo_min_max.ndim}D array"
            )
            assert _data.shape[0] == self.photo_min_max.shape[0], RuntimeError(
                f"photometric channels: {_data.shape[0]} must be equal to the photo_min_max channels: {self.photo_min_max.shape[0]}"
            )
            min_vals = self.photo_min_max[:, 0].reshape(-1, 1, 1)
            max_vals = self.photo_min_max[:, 1].reshape(-1, 1, 1)
            _data = (_data - min_vals) / (max_vals - min_vals + self.eps)
        return torch.tensor(_data, dtype=torch.float32)

    def _load_spec_from_hdf5(
        self,
        h5_path: str,
        h5_index: int,
        nan_to_num: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h5f = self._get_h5_handle(h5_path)
        spec_flat = h5f["spec_flat"][h5_index]
        spec_shape = h5f["spec_shape"][h5_index]
        _data = np.asarray(spec_flat, dtype=np.float32).reshape(
            int(spec_shape[0]),
            int(spec_shape[1]),
        )
        if nan_to_num:
            _data = np.nan_to_num(_data, nan=0, posinf=0, neginf=0)

        # _data shape is (3, length), where:
        # [0] is wavelength
        # [1] is flux
        # [2] is ivar
        wavelength = torch.tensor(_data[0], dtype=torch.float32).unsqueeze(0)
        flux = torch.tensor(_data[1], dtype=torch.float32).unsqueeze(0)
        # Per-spectrum z-score normalisation: removes flux-scale differences
        # between sources so the cosine alignment loss is scale-invariant and
        # the SSCNN z-regression head is not dominated by bright objects.
        _mean = flux.mean()
        _std = flux.std()
        flux = (flux - _mean) / (_std + self.eps)
        return wavelength, flux

    def __getitem__(self, idx):
        id = self.target_ids[idx]
        ra = self.target_ras[idx]
        dec = self.target_decs[idx]
        z = self.zs[idx]
        h5_index = self.h5_indices[idx]
        h5_path = self.resolved_h5_paths[idx]
        south_north_flag = self.south_north_flags[idx]

        # Load photometric data from HDF5
        photometric = self._load_photo_from_hdf5(
            h5_path=h5_path,
            h5_index=h5_index,
            nan_to_num=True,
            min_max_normalize=True,
        )
        # resize if needed
        if (
            photometric.shape[1] != self.target_photo_size
            or photometric.shape[2] != self.target_photo_size
        ):
            photometric = self.resize_transform(photometric)

        # load mag data
        mags = torch.cat(
            [
                torch.tensor(
                    self.mags_array[idx],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    self.mags_diff_array[idx],
                    dtype=torch.float32,
                ),
            ],
            dim=0,
        ).unsqueeze(0)
        label = torch.tensor(z, dtype=torch.float32).unsqueeze(0)

        wavelength, flux = self._load_spec_from_hdf5(
            h5_path=h5_path,
            h5_index=h5_index,
            nan_to_num=True,
        )

        return (
            id,
            ra,
            dec,
            photometric,
            mags,
            label,
            flux,
            south_north_flag,
        )

    def __del__(self):
        for _, handle in self._h5_handles.items():
            try:
                handle.close()
            except Exception:
                pass


def build_dataloader(
    config: dict, mode: str, cross_val_name: str = ""
) -> torch.utils.data.DataLoader:
    """
    Build a dataloader for the dataset.
    :param config: configuration dictionary
    :param mode: mode to load the dataset, can be "train", "val", or "test"
    :param cross_val_name: name of the cross-validation, if any
    :return: DataLoader object
    """
    assert mode in ["train", "val", "test"], ValueError(
        "mode must be in ['train', 'val', 'test']"
    )
    # check keys in config
    keys = [
        "eps",
        # dataset settings
        "dataset_root_dir",
        "label_dir_name",
        "hdf5_file_name",
        "photo_min_max",
        "photo_in_size",
        # dataloader settings
        "batch_size",
        "num_workers",
    ]
    for key in keys:
        assert key in config, ValueError(f"{key} not found in config")
    dataset_root_dir = config["dataset_root_dir"]
    label_dir_name = config["label_dir_name"]
    if cross_val_name is not None and cross_val_name != "":
        label_dir_name = os.path.join(dataset_root_dir, label_dir_name, cross_val_name)
    _dataset = BuildDataset(
        dataset_root_dir=dataset_root_dir,
        label_dir_name=label_dir_name,
        hdf5_file_name=config["hdf5_file_name"],
        load_mode=mode,
        photo_min_max=config["photo_min_max"],
        target_photo_size=config["photo_in_size"],
        eps=config["eps"],
    )
    _dataloader = torch.utils.data.DataLoader(
        dataset=_dataset,
        batch_size=config["batch_size"],
        shuffle=(mode == "train"),
        num_workers=config["num_workers"],
        pin_memory=False,
        drop_last=(mode == "train"),
    )
    return _dataloader
