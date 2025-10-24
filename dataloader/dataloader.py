import os

import numpy as np
import pandas as pd
import torch

MAG_PREFIX = "magnitudes"
MAG_DIFF_PREFIX = "magnitudes_diff"


class BuildDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root_dir: str,
        photometric_dir_name: str,
        label_dir_name: str,
        photo_min_max: tuple[tuple[float]],
        load_mode: str = "train",
        spectrum_dir_name: str = None,
        spectrum_size: int = None,
        eps: float | int = 1e-8,
    ):
        """
        Build a dataloader for the dataset.
        :param dataset_root_dir: dataset root directory
        :param photometric_dir_name: photometric data directory name
        :param label_dir_name : label directory name
        :param photo_min_max: min and max value for photometric data normalization, tuple of tuples with length equal to number of photometric channels
        :param load_mode: mode to load the dataset, can be "train", "val", or "test"
        :param spectrum_dir_name: spectrum data directory name, default is None, meaning no spectrum data will be loaded
        :param spectrum_size: size of the spectrum data, like 3522, only works when spectrum_dir_name is not None
        :param eps: epsilon value for numerical stability, default is 1e-8
        """
        super().__init__()
        assert load_mode in [
            "train",
            "val",
            "test",
        ], ValueError("load_mode must be in ['train', 'val', 'test']")

        self.photometric_dir = os.path.join(dataset_root_dir, photometric_dir_name)
        if spectrum_dir_name is not None:
            self.spectrum_dir = os.path.join(dataset_root_dir, spectrum_dir_name)
            self.spectrum_size = spectrum_size
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
                "lp_zPDF": float,
            },
        )
        self.eps = eps
        self.train = load_mode == "train"
        self.photo_min_max = np.asarray(photo_min_max, dtype=np.float32)

    def __len__(self):
        return len(self.label_df)

    def _load_data(
        self,
        file_path: str,
        allow_pickle: bool = False,
        nan_to_num: bool = False,
        min_max_normalize: bool = True,
    ) -> torch.Tensor:
        _data = np.load(
            os.path.join(file_path),
            allow_pickle=allow_pickle,
        ).astype(np.float32)
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

    def __getitem__(self, idx):
        data_row = self.label_df.iloc[idx]
        id = data_row["TARGETID"]
        ra = data_row["TARGET_RA"]
        dec = data_row["TARGET_DEC"]
        z = data_row["lp_zPDF"]
        photometric_file_name = data_row["photo_name"]
        # Load photometric data
        photometric = self._load_data(
            os.path.join(self.photometric_dir, photometric_file_name),
            nan_to_num=True,
            min_max_normalize=True,
        )
        # load mag data
        mags = []
        for col in data_row.index:
            if col.startswith(MAG_PREFIX) or col.startswith(MAG_DIFF_PREFIX):
                mags.append(data_row[col].astype(float))
        mags = torch.tensor(
            mags,
            dtype=torch.float32,
        ).unsqueeze(0)
        # min max norm
        mags = (mags - mags.min()) / (mags.max() - mags.min() + self.eps)
        label = torch.tensor(z, dtype=torch.float32).unsqueeze(0)

        wavelength = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        flux = torch.zeros(1, dtype=torch.float32).unsqueeze(0)
        if hasattr(self, "spectrum_dir"):
            spectrum_file_name = data_row["spec_name"]
            # Load spectrum data
            spectrum_data = np.load(
                os.path.join(self.spectrum_dir, spectrum_file_name)
            ).astype(np.float32)
            assert spectrum_data.shape[-1] >= self.spectrum_size, RuntimeError(
                f"spectrum_size must be shorter than {spectrum_data.shape[-1]}"
            )
            half_index = (spectrum_data.shape[-1] - self.spectrum_size) // 2
            spectrum_data = spectrum_data[
                :, half_index : half_index + self.spectrum_size
            ].astype(np.float32)
            wavelength = spectrum_data[0]
            wavelength = torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0)
            flux = spectrum_data[1]
            flux = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)

        return (
            id,
            ra,
            dec,
            photometric,
            mags,
            label,
            wavelength,
            flux,
        )


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
        "photometric_dir_name",
        "spectrum_extractor_settings",
        "label_dir_name",
        "photo_min_max",
        # dataloader settings
        "batch_size",
        "num_workers",
    ]
    spectrum_extractor_settings_keys = [
        "spectrum_size",
        "spectrum_dir_name",
        "enable_spectrum_auxiliary",
    ]
    for key in keys:
        assert key in config, ValueError(f"{key} not found in config")
    for key in spectrum_extractor_settings_keys:
        assert key in config["spectrum_extractor_settings"], ValueError(
            f"{key} not found in spectrum_extractor_settings"
        )
    enable_spectrum_auxiliary = config["spectrum_extractor_settings"][
        "enable_spectrum_auxiliary"
    ]
    dataset_root_dir = config["dataset_root_dir"]
    label_dir_name = config["label_dir_name"]
    if cross_val_name is not None and cross_val_name != "":
        print("[INFO] Using cross validation with dataset fold: ", cross_val_name)
        label_dir_name = os.path.join(dataset_root_dir, label_dir_name, cross_val_name)
    _dataset = BuildDataset(
        dataset_root_dir=dataset_root_dir,
        photometric_dir_name=config["photometric_dir_name"],
        spectrum_dir_name=(
            config["spectrum_extractor_settings"]["spectrum_dir_name"]
            if enable_spectrum_auxiliary
            else None
        ),
        spectrum_size=config["spectrum_extractor_settings"]["spectrum_size"],
        label_dir_name=label_dir_name,
        load_mode=mode,
        photo_min_max=config["photo_min_max"],
        eps=config["eps"],
    )
    _dataloader = torch.utils.data.DataLoader(
        dataset=_dataset,
        batch_size=config["batch_size"],
        shuffle=(mode == "train"),
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    return _dataloader
