import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.rich import tqdm

from config.config import config
from dataloader.dataloader import build_dataloader
from model.lightning_model import BuildLightningModel
from model.loss import delta_z, nmad_z, sigma_n, z_estimate, outline_fraction

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODE = "DESI_WISE"  # DESI_WISE or DESI
PRED_CONFIG_DESI_WISE = {
    "list": [
        {
            "model_name": "PRISM",
            "ckpt_name": {
                "fold_0": "",
                "fold_1": "",
                "fold_2": "",
                "fold_3": "",
                "fold_4": "",
            },
        },
    ],
    "save_path": "./results/DESI_WISE",
}
PRED_CONFIG_DESI = {
    "list": [
        {
            "model_name": "PRISM",
            "ckpt_name": {
                "fold_0": "",
                "fold_1": "",
                "fold_2": "",
                "fold_3": "",
                "fold_4": "",
            },
        },
    ],
    "save_path": "./results/DESI",
}
MODEL_PATH_LIST = (
    PRED_CONFIG_DESI_WISE["list"] if MODE == "DESI_WISE" else PRED_CONFIG_DESI["list"]
)
RES_SAVE_DIR = (
    PRED_CONFIG_DESI_WISE["save_path"]
    if MODE == "DESI_WISE"
    else PRED_CONFIG_DESI["save_path"]
)
EPS = config["eps"]
MODEL_STAGE = 4
PLOT_SPEC_RECON = True
PRINT_SR_PHOTO = True
SAVE_DEEP_FEATURE = True
PDF_MODE = "mean"
FEATURE_LINE_BASE = {
    "Ly_alpha": {
        "name": r"$Ly\alpha$",
        "wavelength": 1216,
    },
    "C_IV": {
        "name": r"$C~IV$",
        "wavelength": 1548,
    },
    "C_III": {
        "name": r"$C~III$",
        "wavelength": 1909,
    },
    "O_II": {
        "name": r"$[OII]$",
        "wavelength": [3726, 3729],
    },
    "Ca_II_H": {
        "name": r"$Ca~II~H$",
        "wavelength": 3934,
    },
    "Ca_II_K": {
        "name": r"$Ca~II~K$",
        "wavelength": 3969,
    },
    "H_beta": {
        "name": r"$H\beta$",
        "wavelength": 4861,
    },
    "O_III": {
        "name": r"$[OIII]$",
        "wavelength": [4959, 5007],
    },
    "H_alpha": {
        "name": r"$H\alpha$",
        "wavelength": 6563,
    },
    "N_II": {
        "name": r"$[NII]$",
        "wavelength": [6548, 6583],
    },
    "S_II": {
        "name": r"$[SII]$",
        "wavelength": [6716, 6731],
    },
}


def feature_lines_at_z(z: float, wavelength_min: float, wavelength_max: float) -> dict:
    lines = {}
    for key, value in FEATURE_LINE_BASE.items():
        if isinstance(value["wavelength"], list):
            lines[key] = {
                "name": value["name"],
                "wavelength": [wl * (1 + z) for wl in value["wavelength"]],
            }
        else:
            lines[key] = {
                "name": value["name"],
                "wavelength": value["wavelength"] * (1 + z),
            }
    # filter lines out of range
    for key in list(lines.keys()):
        if isinstance(lines[key]["wavelength"], list):
            if all(
                wl < wavelength_min or wl > wavelength_max
                for wl in lines[key]["wavelength"]
            ):
                lines.pop(key)
        else:
            if (
                lines[key]["wavelength"] < wavelength_min
                or lines[key]["wavelength"] > wavelength_max
            ):
                lines.pop(key)
    return lines


def save_deep_features(
    spec_partial_features: list[torch.Tensor],
    photo_partial_features: list[torch.Tensor],
    save_path: str,
    target_id: str,
) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    # for i in range(MODEL_STAGE):
    #     print(spec_partial_features[i].shape, photo_partial_features[i].shape)
    np.savez_compressed(
        file=os.path.join(save_path, f"{target_id}.npz"),
        **{
            "spec_features_l{}".format(i): spec_partial_features[i].cpu().numpy()
            for i in range(MODEL_STAGE)
        },
        **{
            "photo_features_l{}".format(i): photo_partial_features[i].cpu().numpy()
            for i in range(MODEL_STAGE)
        },
    )


def plot_reconstruction_spec(
    wavelength: torch.Tensor,
    truth_flux: torch.Tensor,
    recon_flux: torch.Tensor,
    truth_z: torch.Tensor,
    z_mae: float,
    target_id: str,
    save_dir: str,
) -> None:
    # to cpu and numpy
    wavelength = wavelength.cpu().numpy()
    truth_flux = truth_flux.cpu().numpy()
    recon_flux = recon_flux.cpu().numpy()
    # min-max normalize to [0, 1]
    truth_flux = (truth_flux - truth_flux.min()) / (
        truth_flux.max() - truth_flux.min() + EPS
    )
    recon_flux = (recon_flux - recon_flux.min()) / (
        recon_flux.max() - recon_flux.min() + EPS
    )
    # z to float
    truth_z = truth_z.cpu().item()
    # plot on the same figure
    plt.figure(figsize=(20, 5), dpi=256)
    plt.plot(
        wavelength,
        truth_flux,
        label="Truth Spectrum",
        color="#fb8500",
    )
    plt.plot(
        wavelength,
        recon_flux,
        label="Reconstructed Spectrum",
        color="#219ebc",
        alpha=0.7,
    )
    # plot feature lines, keep the label in the lines
    feature_lines = feature_lines_at_z(truth_z, wavelength.min(), wavelength.max())
    label_y_counter = 0
    label_y_step = max(truth_flux) * 0.2
    for key, value in feature_lines.items():
        if isinstance(value["wavelength"], list):
            for wl in value["wavelength"]:
                plt.axvline(
                    x=wl,
                    color="#495057",
                    linestyle="--",
                    alpha=0.7,
                )
                plt.text(
                    x=wl + 12,
                    y=max(truth_flux) * 0.9 - label_y_counter * label_y_step,
                    s=value["name"],
                    rotation=90,
                    color="#000814",
                )
                label_y_counter += 1
        else:
            plt.axvline(
                x=value["wavelength"],
                color="#495057",
                linestyle="--",
                alpha=0.7,
            )
            plt.text(
                x=value["wavelength"] + 12,
                y=max(truth_flux) * 0.9 - label_y_counter * label_y_step,
                s=value["name"],
                rotation=90,
                color="#000814",
            )
            label_y_counter += 1
        if label_y_counter >= 4:
            label_y_counter = 0
    plt.xlabel("Wavelength (AA)")
    plt.ylabel("Relative Flux")
    plt.title(
        r"$ID:~{},~z_{{Truth}}={:.4f},~z_{{MAE}}={:.4f}$".format(
            target_id, truth_z, z_mae
        )
    )
    plt.legend()
    plt.grid(False)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            save_dir, "{}_Z_{:.4f}_MAE_{:.4f}.png".format(target_id, truth_z, z_mae)
        )
    )
    plt.close()


def plot_sr_photo(
    photometric: torch.Tensor,
    sr_features: torch.Tensor,
    truth_z: torch.Tensor,
    z_mae: float,
    target_id: str,
    save_dir: str,
) -> None:
    photometric = photometric.cpu().numpy()
    sr_features = sr_features.cpu().numpy()
    truth_z = truth_z.cpu().item()
    # min-max normalize to [0, 1] for each channel
    for i in range(photometric.shape[0]):
        photometric[i] = (photometric[i] - photometric[i].min()) / (
            photometric[i].max() - photometric[i].min() + EPS
        )
    for i in range(sr_features.shape[0]):
        sr_features[i] = (sr_features[i] - sr_features[i].min()) / (
            sr_features[i].max() - sr_features[i].min() + EPS
        )
    # diff
    diff = np.abs(sr_features - photometric)
    for i in range(diff.shape[0]):
        diff[i] = (diff[i] - diff[i].min()) / (diff[i].max() - diff[i].min() + EPS)
    # plot on the same figure
    # there are n-rows for band_names, and each row has 3 subplots for photometric, spectrum_features, and diff
    # each subplot is a CHW image
    band_names = ["g", "r", "i", "z", "W1", "W2"]
    plt.figure(figsize=(15, 3 * len(band_names)), dpi=256)
    for i in range(len(band_names)):
        # photometric
        plt.subplot(6, 3, i * 3 + 1)
        plt.imshow(photometric[i], cmap="viridis", aspect="equal")
        plt.colorbar()
        plt.title(r"$Band:~{},~Photometric$".format(band_names[i]))
        plt.axis("off")
        # spectrum features
        plt.subplot(6, 3, i * 3 + 2)
        plt.imshow(sr_features[i], cmap="viridis", aspect="equal")
        plt.colorbar()
        plt.title(r"$Band:~{},~Reconstruction$".format(band_names[i]))
        plt.axis("off")
        # diff
        plt.subplot(6, 3, i * 3 + 3)
        plt.imshow(diff[i], cmap="viridis", aspect="equal")
        plt.colorbar()
        plt.title(r"$Band:~{},~Difference$".format(band_names[i]))
        plt.axis("off")
    plt.suptitle(
        r"$ID:~{},~z_{{Truth}}={:.4f},~z_{{MAE}}={:.4f}$".format(
            target_id, truth_z, z_mae
        ),
        fontsize=16,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            save_dir, "{}_Z_{:.4f}_MAE_{:.4f}.png".format(target_id, truth_z, z_mae)
        )
    )
    plt.close()


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    try:
        model = BuildLightningModel.load_from_checkpoint(
            os.path.join(model_path),
            map_location=device,
        ).model
        # remove model. prefix
        state_dict = {k.replace("model.", ""): v for k, v in model.state_dict().items()}
        model.load_state_dict(state_dict)
        print("[Info] Load model from {}".format(model_path))
    except Exception as e:
        print("[Error] Load model from {} failed".format(model_path))
        raise e
    return model


def inference(
    settings: dict,
    ckpt_path: str,
    dataset_fold: str,
    res_save_dir: str,
    clear_res_dir: bool = False,
    device: torch.device = torch.device("cpu"),
    cuda_infer_timer: bool = True,
    plot_spec_recon: bool = False,
    plot_self_recon_photo: bool = False,
    save_deep_feature: bool = False,
) -> None:
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir, exist_ok=True)
    else:
        if clear_res_dir:
            print(
                "[WARNING] Find existing result directory: {}, clear it.".format(
                    res_save_dir
                )
            )
            shutil.rmtree(res_save_dir)
            os.makedirs(res_save_dir, exist_ok=True)
        else:
            raise FileExistsError(res_save_dir)
    test_dataloader = build_dataloader(
        settings,
        mode="test",
        cross_val_name=dataset_fold,
    )
    test_model = load_model(ckpt_path, device)
    data_info = []
    pred = []
    z_pred = []
    z_label = []
    infer_time_ms = 0.0
    total_samples = 0

    test_model.eval()
    with torch.no_grad():
        with tqdm(
            total=len(test_dataloader),
            ncols=150,
        ) as pbar:
            for idx, batch in enumerate(test_dataloader):
                pbar.set_description("Predicting BS-{}".format(idx + 1))
                (
                    id,
                    ra,
                    dec,
                    photometric,
                    mags,
                    label,
                    wavelength,
                    flux,
                ) = batch
                photometric = photometric.to(device)
                mags = mags.to(device)
                label = label.to(device)
                infer_start = torch.cuda.Event(enable_timing=cuda_infer_timer)
                infer_end = torch.cuda.Event(enable_timing=cuda_infer_timer)
                for i in range(photometric.shape[0]):
                    data_info.append(
                        {
                            "TARGETID": id[i],
                            "TARGET_RA": ra[i],
                            "TARGET_DEC": dec[i],
                        }
                    )
                # ======================================================
                # start inference time record
                if cuda_infer_timer:
                    infer_start.record(torch.cuda.current_stream(device))
                (
                    _pred,
                    _spectrum_features,
                    _sr_features,
                    _deep_features,
                    _photometric_features,
                ) = test_model(
                    photometric=photometric,
                    magnitudes=mags,
                    output_spectrum=plot_spec_recon,
                )
                if cuda_infer_timer:
                    infer_end.record(torch.cuda.current_stream(device))
                    torch.cuda.synchronize(device)
                    infer_time_ms += infer_start.elapsed_time(infer_end)
                # end inference time record
                # ======================================================
                pred.append(_pred.detach().cpu())
                _pred_z = z_estimate(
                    pred=_pred,
                    mode=PDF_MODE,
                )
                z_pred.append(_pred_z.detach().cpu())
                d_z_pdf = delta_z(
                    pred=_pred,
                    label=label,
                    mode=PDF_MODE,
                )
                pbar.set_postfix(
                    {
                        "MAE": (_pred_z - label.squeeze(-1)).abs().mean().item(),
                        "NMAD-1.48": nmad_z(d_z_pdf, factor=1.4826).item(),
                        "Sigma_0.15": sigma_n(d_z_pdf, n=0.15).item(),
                    },
                    refresh=False,
                )
                z_label.append(label.detach().cpu())
                total_samples += photometric.shape[0]

                if save_deep_feature:
                    for i in range(photometric.shape[0]):
                        pbar.set_description(
                            "Saving Deep Feature-{} in BS-{}".format(i + 1, idx + 1)
                        )
                        _spec_f = []
                        _photo_f = []
                        for stage_idx in range(MODEL_STAGE):
                            _spec_f.append(_deep_features[stage_idx][i])
                            _photo_f.append(_photometric_features[stage_idx][i])
                        save_deep_features(
                            spec_partial_features=_spec_f,
                            photo_partial_features=_photo_f,
                            save_path=os.path.join(res_save_dir, "deep_features"),
                            target_id=id[i],
                        )
                if plot_spec_recon:
                    for i in range(photometric.shape[0]):
                        for stage_idx in range(len(_spectrum_features)):
                            pbar.set_description(
                                "Plotting Spec-{} in BS-{}".format(i + 1, idx + 1)
                            )
                            plot_reconstruction_spec(
                                wavelength=wavelength[i].squeeze(),
                                truth_flux=flux[i].squeeze(),
                                recon_flux=_spectrum_features[stage_idx][i].squeeze(),
                                truth_z=label[i].squeeze(),
                                z_mae=(_pred_z - label.squeeze(-1)).abs()[i].item(),
                                target_id=id[i],
                                save_dir=os.path.join(
                                    res_save_dir,
                                    "reconstruction_spec",
                                    "stage_{}".format(stage_idx),
                                ),
                            )
                if plot_self_recon_photo:
                    for i in range(photometric.shape[0]):
                        pbar.set_description(
                            "Plotting SR-Photo-{} in BS-{}".format(i + 1, idx + 1)
                        )
                        plot_sr_photo(
                            photometric=photometric[i].squeeze(),
                            sr_features=_sr_features[i].squeeze(),
                            truth_z=label[i].squeeze(),
                            z_mae=(_pred_z - label.squeeze(-1)).abs()[i].item(),
                            target_id=id[i],
                            save_dir=os.path.join(
                                res_save_dir,
                                "sf_photo",
                            ),
                        )
                pbar.update(1)

    print("=" * 50)
    fps = total_samples / (infer_time_ms / 1000.0)
    print(
        "[Info] FPS: {:.2f}, Total Samples: {}, Total Time: {:.2f} ms".format(
            fps,
            total_samples,
            infer_time_ms,
        )
    )
    print("=" * 50)
    data_df_header = [
        "TARGETID",
        "TARGET_RA",
        "TARGET_DEC",
        "Z_TRUTH",
        "Z_ESTIMATION",
    ]
    res_json = {
        "cross_val_name": dataset_fold,
        "ckpt_path": ckpt_path,
        "total_samples": int(total_samples),
        "infer_time_ms": infer_time_ms,
        "fps": fps,
        "mae": -1,
        "nmad_1.48": -1,
        "sigma_0.15": -1,
        "outline_fraction_0.15": -1,
    }
    z_label = torch.cat(z_label, dim=0)
    z_pdf = torch.cat(z_pred, dim=0)
    pred_pdf = torch.cat(pred, dim=0)
    print("[Info] PDF Estimation Results:")
    mae_pdf = (z_pdf - z_label.squeeze(-1)).abs().mean().item()
    d_z_epoch = delta_z(
        pred=pred_pdf,
        label=z_label,
        mode=PDF_MODE,
    )
    nmad_pdf = nmad_z(
        d_z=d_z_epoch,
        factor=1.4826,
    ).item()
    sigma_015_pdf = sigma_n(
        d_z=d_z_epoch,
        n=0.15,
    ).item()
    outline_frac = outline_fraction(
        d_z=d_z_epoch,
        threshold=0.1,
    ).item()
    print(
        "MAE: {:.4f}, NMAD-1.48: {:.4f}, Sigma_0.15: {:.4f}, Outline Fraction-0.15: {:.4f}".format(
            mae_pdf,
            nmad_pdf,
            sigma_015_pdf,
            outline_frac,
        )
    )
    res_json["mae"] = mae_pdf
    res_json["nmad_1.48"] = nmad_pdf
    res_json["sigma_0.15"] = sigma_015_pdf
    res_json["outline_fraction_0.15"] = outline_frac
    # save json
    with open(os.path.join(res_save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res_json, f, indent=4)
    # save data info
    data_df = []
    for i in range(len(data_info)):
        data_df.append(
            {
                "TARGETID": str(data_info[i]["TARGETID"]),
                "TARGET_RA": str(data_info[i]["TARGET_RA"]),
                "TARGET_DEC": str(data_info[i]["TARGET_DEC"]),
                "Z_TRUTH": str(z_label[i].item()),
                "Z_ESTIMATION": (str(z_pdf[i].item())),
            }
        )
    data_df = pd.DataFrame(data_df, columns=data_df_header)
    data_df.to_csv(
        os.path.join(res_save_dir, "result.csv"),
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--clear",
        "-c",
        action="store_true",
        default=False,
        help="Clear the result directory before inference",
    )
    opts = args.parse_args()

    print("[INFO] Using device: {}".format(DEVICE))
    with tqdm(total=len(MODEL_PATH_LIST), ncols=150) as pbar:
        for model_obj in MODEL_PATH_LIST:
            pbar.set_description("Predicting: {}".format(model_obj["model_name"]))
            for fold_name, ckpt_path in model_obj["ckpt_name"].items():
                pbar.set_description(
                    "[INFO] Start inference for model: {}, fold: {}".format(
                        model_obj["model_name"], fold_name
                    )
                )
                inference(
                    settings=config,
                    ckpt_path=os.path.join(ckpt_path),
                    dataset_fold=fold_name,
                    res_save_dir=os.path.join(
                        RES_SAVE_DIR,
                        model_obj["model_name"],
                        fold_name,
                    ),
                    clear_res_dir=opts.clear,
                    device=DEVICE,
                    cuda_infer_timer=True,
                    plot_spec_recon=PLOT_SPEC_RECON,
                    plot_self_recon_photo=PRINT_SR_PHOTO,
                    save_deep_feature=SAVE_DEEP_FEATURE,
                )
            pbar.update(1)
