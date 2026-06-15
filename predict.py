import argparse
import json
import math
import os
import shutil
from contextlib import nullcontext

import pandas as pd
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from config.config import config
from dataloader.dataloader import build_dataloader
from model.lightning_model import BuildLightningModel
from model.loss import nmad_z, sigma_n, z_bias, z_estimate, outlier_fraction

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_CONFIG = {
    "list": [
        {
            "model_name": "PRISM",
            "ckpt_name": {
                "fold_0": "weights/2026-05-19_19-14-42/checkpoints/best-epoch=50-val_sweep_score_epoch=0.07066.ckpt",
                "fold_1": "weights/2026-05-19_18-46-10/checkpoints/best-epoch=49-val_sweep_score_epoch=0.07030.ckpt",
                "fold_2": "weights/2026-05-19_12-40-00/checkpoints/best-epoch=50-val_sweep_score_epoch=0.07096.ckpt",
                "fold_3": "weights/2026-05-20_16-09-18/checkpoints/best-epoch=58-val_sweep_score_epoch=0.07104.ckpt",
                "fold_4": "weights/2026-05-20_15-17-28/checkpoints/best-epoch=64-val_sweep_score_epoch=0.07097.ckpt",
            },
        },
    ],
    "save_path": "./results/DATASET_1.9M",
}
MODEL_PATH_LIST = PRED_CONFIG["list"]
RES_SAVE_DIR = PRED_CONFIG["save_path"]
PDF_MODE = "mean"
EUCLID_Z_MIN = 0.2
EUCLID_Z_MAX = 2.6
EUCLID_BIN_WIDTH = 0.2
EUCLID_MODE_GRID_SIZE = 401
EUCLID_MODE_MAX_SAMPLES = 100000
CHUNK_SIZE = 8192
MAX_GMM_COMPONENTS = 5
CONSOLE = Console()

BASE_RESULT_CSV_COLUMNS = [
    "TARGETID",
    "TARGET_RA",
    "TARGET_DEC",
    "SOURCE",
    "Ground Truth",
    "Prediction Value",
    "delta_z",
    "|delta_z|",
    "NLL",
    "CRPS",
]
GMM_RESULT_CSV_COLUMNS = [
    "GMM_{}_{}".format(component_idx, field)
    for component_idx in range(1, MAX_GMM_COMPONENTS + 1)
    for field in ("mean", "std", "weight")
]
RESULT_CSV_COLUMNS = BASE_RESULT_CSV_COLUMNS + GMM_RESULT_CSV_COLUMNS


def make_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[metrics]}"),
        TimeElapsedColumn(),
        console=CONSOLE,
    )


def ensure_result_dir(res_save_dir: str, clear_res_dir: bool) -> None:
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
        return
    if clear_res_dir:
        CONSOLE.print(
            "[WARNING] Find existing result directory: {}, clear it.".format(
                res_save_dir
            )
        )
        shutil.rmtree(res_save_dir)
        os.makedirs(res_save_dir)
        return
    raise FileExistsError(res_save_dir)


def to_python_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def make_euclid_bin_edges() -> list[float]:
    steps = int(round((EUCLID_Z_MAX - EUCLID_Z_MIN) / EUCLID_BIN_WIDTH))
    return [round(EUCLID_Z_MIN + i * EUCLID_BIN_WIDTH, 10) for i in range(steps + 1)]


def finite_prediction_mask(
    pred_pdf: torch.Tensor,
    z_label: torch.Tensor,
    z_pdf: torch.Tensor,
) -> torch.Tensor:
    label = z_label.squeeze(-1) if z_label.dim() > 1 else z_label
    delta_z = (label - z_pdf) / (1.0 + label)
    mask = (
        torch.isfinite(pred_pdf).reshape(pred_pdf.shape[0], -1).all(dim=1)
        & torch.isfinite(z_label).reshape(z_label.shape[0], -1).all(dim=1)
        & torch.isfinite(z_pdf).reshape(z_pdf.shape[0], -1).all(dim=1)
        & torch.isfinite(delta_z).reshape(delta_z.shape[0], -1).all(dim=1)
    )
    if pred_pdf.dim() == 3 and pred_pdf.shape[-1] >= 3:
        mean, std, _, weight = gmm_parameters(pred_pdf, eps=config["eps"])
        mask = (
            mask
            & torch.isfinite(mean).all(dim=1)
            & torch.isfinite(std).all(dim=1)
            & torch.isfinite(weight).all(dim=1)
        )
    return mask


def rebuild_source_indices(data_info: list[dict]) -> dict:
    source_indices = {}
    for idx, item in enumerate(data_info):
        source_value = to_python_scalar(item["SOURCE"])
        source_indices.setdefault(source_value, []).append(idx)
    return source_indices


def _ks_pvalue_asymptotic(ks_stat: float, n: int) -> float:
    if n <= 0:
        return 0.0
    if ks_stat <= 0:
        return 1.0
    sqn = math.sqrt(n)
    x = (sqn + 0.12 + 0.11 / sqn) * ks_stat
    series_sum = 0.0
    for k in range(1, 101):
        term = (-1) ** (k - 1) * math.exp(-2.0 * (k**2) * (x**2))
        series_sum += term
        if abs(term) < 1e-12:
            break
    p_value = 2.0 * series_sum
    return max(0.0, min(1.0, p_value))


def evaluate_pit_population(
    pred_pdf: torch.Tensor,
    z_label: torch.Tensor,
    bins: int = 50,
    eps: float = 1e-12,
) -> dict:
    label = z_label.squeeze(-1) if z_label.dim() > 1 else z_label
    mean = pred_pdf[:, :, 0]
    std = torch.nn.functional.softplus(pred_pdf[:, :, 1]) + eps
    weight = torch.softmax(pred_pdf[:, :, 2], dim=1)
    standard_normal = torch.distributions.Normal(0.0, 1.0)
    z = (label.unsqueeze(-1) - mean) / std
    cdf = standard_normal.cdf(z)
    pit_values = torch.sum(weight * cdf, dim=-1).clamp(0.0, 1.0)
    pit_values = pit_values[
        torch.isfinite(pit_values) & (pit_values >= 0.0) & (pit_values <= 1.0)
    ]

    n = pit_values.numel()
    if n == 0:
        return {
            "KS_stat": 0.0,
            "KS_pval": 1.0,
            "CvM_stat": 0.0,
            "KLD": 0.0,
        }

    sorted_pit = torch.sort(pit_values).values
    ecdf = torch.arange(1, n + 1, dtype=sorted_pit.dtype) / float(n)
    ecdf_prev = torch.arange(0, n, dtype=sorted_pit.dtype) / float(n)
    ks_stat = torch.max(torch.max(ecdf - sorted_pit, sorted_pit - ecdf_prev)).item()
    ks_pval = _ks_pvalue_asymptotic(ks_stat, n)

    i = torch.arange(1, n + 1, dtype=sorted_pit.dtype)
    cvm_stat = (
        1.0 / (12.0 * n) + torch.sum((sorted_pit - (2 * i - 1) / (2.0 * n)) ** 2).item()
    )

    hist_bins = max(1, int(bins))
    hist = torch.histc(pit_values, bins=hist_bins, min=0.0, max=1.0)
    prob = hist / hist.sum().clamp_min(eps)
    uniform = torch.full_like(prob, 1.0 / hist_bins)
    kld = torch.sum(prob * torch.log((prob + eps) / (uniform + eps))).item()

    result = {
        "KS_stat": float(ks_stat),
        "KS_pval": float(ks_pval),
        "CvM_stat": float(cvm_stat),
        "KLD": float(kld),
    }
    return result


def save_pit_metrics(
    pred_pdf: torch.Tensor,
    z_label: torch.Tensor,
    bins: int = 50,
) -> dict:
    pit_result = evaluate_pit_population(
        pred_pdf=pred_pdf,
        z_label=z_label,
        bins=bins,
        eps=config["eps"],
    )
    return pit_result


def gmm_parameters(
    pred_pdf: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Mirrors model.loss.NLLLoss/z_estimate transforms for exported GMM parameters.
    mean = pred_pdf[:, :, 0]
    std = torch.nn.functional.softplus(pred_pdf[:, :, 1]) + eps
    log_weight = torch.log_softmax(pred_pdf[:, :, 2], dim=1)
    weight = torch.exp(log_weight)
    return mean, std, log_weight, weight


def empty_gmm_csv_values() -> dict:
    return {column: "" for column in GMM_RESULT_CSV_COLUMNS}


def build_gmm_csv_value_rows(pred_pdf: torch.Tensor) -> list[dict]:
    if pred_pdf.dim() != 3 or pred_pdf.shape[-1] < 3:
        return [empty_gmm_csv_values() for _ in range(pred_pdf.shape[0])]
    mean, std, _, weight = gmm_parameters(pred_pdf=pred_pdf, eps=config["eps"])
    component_count = min(MAX_GMM_COMPONENTS, mean.shape[1])
    rows = []
    for row_idx in range(pred_pdf.shape[0]):
        values = empty_gmm_csv_values()
        for component_idx in range(component_count):
            csv_idx = component_idx + 1
            values["GMM_{}_mean".format(csv_idx)] = float(
                mean[row_idx, component_idx].item()
            )
            values["GMM_{}_std".format(csv_idx)] = float(
                std[row_idx, component_idx].item()
            )
            values["GMM_{}_weight".format(csv_idx)] = float(
                weight[row_idx, component_idx].item()
            )
        rows.append(values)
    return rows


def build_result_dataframe(
    data_info: list[dict],
    z_label: torch.Tensor,
    z_pdf: torch.Tensor,
    delta_z_values: torch.Tensor,
    nll_values: torch.Tensor,
    crps_values: torch.Tensor,
    pred_pdf: torch.Tensor,
) -> pd.DataFrame:
    rows = []
    gmm_rows = build_gmm_csv_value_rows(pred_pdf)
    for i in range(len(data_info)):
        row = {
            "TARGETID": str(to_python_scalar(data_info[i]["TARGETID"])),
            "TARGET_RA": to_python_scalar(data_info[i]["TARGET_RA"]),
            "TARGET_DEC": to_python_scalar(data_info[i]["TARGET_DEC"]),
            "SOURCE": str(to_python_scalar(data_info[i]["SOURCE"])),
            "Ground Truth": float(z_label[i].item()),
            "Prediction Value": float(z_pdf[i].item()),
            "delta_z": float(delta_z_values[i].item()),
            "|delta_z|": float(delta_z_values[i].abs().item()),
            "NLL": float(nll_values[i].item()),
            "CRPS": float(crps_values[i].item()),
        }
        row.update(gmm_rows[i])
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_CSV_COLUMNS)


def gaussian_a_term(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    normal = torch.distributions.Normal(0.0, 1.0)
    z = x / scale.clamp_min(config["eps"])
    return 2.0 * scale * torch.exp(normal.log_prob(z)) + x * (2.0 * normal.cdf(z) - 1.0)


def calc_nll_values(
    pred_pdf: torch.Tensor,
    z_label: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    label = z_label.squeeze(-1) if z_label.dim() > 1 else z_label
    mean, std, log_weight, _ = gmm_parameters(pred_pdf, eps=eps)
    target_expanded = label.unsqueeze(-1).expand_as(mean)
    log_gaussian_pdf = (
        -torch.log(std)
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * ((target_expanded - mean) / std) ** 2
    )
    return -torch.logsumexp(log_weight + log_gaussian_pdf, dim=1)


def calc_crps_values(
    pred_pdf: torch.Tensor,
    z_label: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    label = z_label.squeeze(-1) if z_label.dim() > 1 else z_label
    mean, std, _, weight = gmm_parameters(pred_pdf, eps=eps)
    y_minus_mu = label.unsqueeze(-1) - mean
    first = torch.sum(weight * gaussian_a_term(y_minus_mu, std), dim=1)

    mu_diff = mean.unsqueeze(2) - mean.unsqueeze(1)
    pair_scale = torch.sqrt(std.unsqueeze(2) ** 2 + std.unsqueeze(1) ** 2)
    pair_weight = weight.unsqueeze(2) * weight.unsqueeze(1)
    second = 0.5 * torch.sum(
        pair_weight * gaussian_a_term(mu_diff, pair_scale), dim=(1, 2)
    )
    return first - second


def calc_point_delta_z(z_pdf: torch.Tensor, z_label: torch.Tensor) -> torch.Tensor:
    label = z_label.squeeze(-1) if z_label.dim() > 1 else z_label
    return (label - z_pdf) / (1.0 + label)


def subset_by_indices(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return tensor[torch.as_tensor(indices, dtype=torch.long)]


def calc_metrics(
    pred_pdf: torch.Tensor,
    z_label: torch.Tensor,
    z_pdf: torch.Tensor,
    mode: str,
) -> dict:
    label = z_label.squeeze(-1) if z_label.dim() > 1 else z_label
    abs_error = (z_pdf - label).abs()
    residual = calc_point_delta_z(z_pdf=z_pdf, z_label=z_label)
    crps_values = calc_crps_values(
        pred_pdf=pred_pdf, z_label=z_label, eps=config["eps"]
    )
    nll_values = calc_nll_values(pred_pdf=pred_pdf, z_label=z_label, eps=config["eps"])
    return {
        "mae": abs_error.mean().item(),
        "median_abs_error": torch.median(abs_error).item(),
        "rmse": torch.sqrt(torch.mean((z_pdf - label) ** 2)).item(),
        "nmad_1.48": nmad_z(d_z=residual, factor=1.4826).item(),
        "sigma_0.05": sigma_n(d_z=residual, n=0.05).item(),
        "sigma_0.15": sigma_n(d_z=residual, n=0.15).item(),
        "z_bias": z_bias(d_z=residual).item(),
        "median_delta_z": torch.median(residual).item(),
        "outlier_fraction_0.10": outlier_fraction(
            d_z=residual,
            threshold=0.1,
        ).item(),
        "outlier_fraction": outlier_fraction(
            d_z=residual,
            threshold=0.15,
        ).item(),
        "outlier_fraction_0.15": outlier_fraction(
            d_z=residual,
            threshold=0.15,
        ).item(),
        "nll": nll_values.mean().item(),
        "crps_mean": crps_values.mean().item(),
        "crps_median": torch.median(crps_values).item(),
    }


def format_metrics(metrics: dict) -> str:
    return (
        "MAE: {:.4f}, RMSE: {:.4f}, NMAD-1.48: {:.4f}, Sigma_0.05: {:.4f}, "
        "Sigma_0.15: {:.4f}, Z Bias: {:.4f}, Outlier-0.15: {:.4f}, "
        "NLL: {:.4f}, CRPS-med: {:.4f}"
    ).format(
        metrics["mae"],
        metrics["rmse"],
        metrics["nmad_1.48"],
        metrics["sigma_0.05"],
        metrics["sigma_0.15"],
        metrics["z_bias"],
        metrics["outlier_fraction"],
        metrics["nll"],
        metrics["crps_median"],
    )


def format_batch_metrics(metrics: dict) -> str:
    return ", ".join("{}: {:.4f}".format(key, value) for key, value in metrics.items())


def batch_metrics_postfix(
    pred: torch.Tensor,
    label: torch.Tensor,
    mode: str,
) -> dict:
    pred_z = z_estimate(
        pred=pred,
        mode=mode,
    )
    d_z = calc_point_delta_z(z_pdf=pred_z, z_label=label)
    return {
        "MAE": (pred_z - label.squeeze(-1)).abs().mean().item(),
        "NMAD-1.48": nmad_z(d_z, factor=1.4826).item(),
        "Sigma_0.05": sigma_n(d_z, n=0.05).item(),
        "Sigma_0.15": sigma_n(d_z, n=0.15).item(),
        "Z Bias": z_bias(d_z).item(),
    }


def _iter_chunks(n: int, chunk_size: int = CHUNK_SIZE):
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def estimate_stacked_pdz_mode(
    shifted_mean: torch.Tensor,
    std: torch.Tensor,
    weight: torch.Tensor,
    grid_size: int = EUCLID_MODE_GRID_SIZE,
    max_samples: int = EUCLID_MODE_MAX_SAMPLES,
) -> float:
    n = shifted_mean.shape[0]
    if n == 0:
        return 0.0
    if n > max_samples:
        sample_idx = torch.linspace(0, n - 1, max_samples).long()
        shifted_mean = shifted_mean[sample_idx]
        std = std[sample_idx]
        weight = weight[sample_idx]

    center = torch.median(shifted_mean.flatten()).item()
    spread = torch.quantile(shifted_mean.flatten().abs(), 0.995).item()
    std_pad = torch.quantile(std.flatten(), 0.995).item() * 5.0
    half_width = max(0.5, min(3.0, spread + std_pad))
    grid = torch.linspace(center - half_width, center + half_width, grid_size)
    density = torch.zeros_like(grid)
    normal_const = math.sqrt(2.0 * math.pi)

    for start, end in _iter_chunks(shifted_mean.shape[0], chunk_size=2048):
        mu = shifted_mean[start:end].unsqueeze(-1)
        sigma = std[start:end].unsqueeze(-1).clamp_min(config["eps"])
        w = weight[start:end].unsqueeze(-1)
        z = (grid.view(1, 1, -1) - mu) / sigma
        pdf = torch.exp(-0.5 * z**2) / (sigma * normal_const)
        density += torch.sum(w * pdf, dim=(0, 1))
    return grid[torch.argmax(density)].item()


def calc_euclid_pdz_fractions(
    pred_pdf: torch.Tensor,
    z_label: torch.Tensor,
    z_pdf: torch.Tensor,
) -> dict:
    label = z_label.squeeze(-1) if z_label.dim() > 1 else z_label
    mean, std, _, weight = gmm_parameters(pred_pdf, eps=config["eps"])
    edges = make_euclid_bin_edges()
    normal = torch.distributions.Normal(0.0, 1.0)
    bins = []
    total_count = 0
    weighted_f005 = 0.0
    weighted_f015 = 0.0

    for left, right in zip(edges[:-1], edges[1:]):
        if right >= EUCLID_Z_MAX:
            mask = (z_pdf >= left) & (z_pdf <= right)
        else:
            mask = (z_pdf >= left) & (z_pdf < right)
        count = int(mask.sum().item())
        center = 0.5 * (left + right)
        if count == 0:
            bins.append(
                {
                    "z_min": float(left),
                    "z_max": float(right),
                    "z_center": float(center),
                    "count": 0,
                    "mode": None,
                    "F005": None,
                    "F015": None,
                }
            )
            continue

        shifted_mean = mean[mask] - label[mask].unsqueeze(-1)
        std_bin = std[mask]
        weight_bin = weight[mask]
        mode = estimate_stacked_pdz_mode(
            shifted_mean=shifted_mean,
            std=std_bin,
            weight=weight_bin,
        )

        def interval_fraction(width: float) -> float:
            lower = mode - width
            upper = mode + width
            prob_sum = 0.0
            for start, end in _iter_chunks(count):
                mu = shifted_mean[start:end]
                sigma = std_bin[start:end].clamp_min(config["eps"])
                w = weight_bin[start:end]
                upper_cdf = normal.cdf((upper - mu) / sigma)
                lower_cdf = normal.cdf((lower - mu) / sigma)
                prob_sum += torch.sum(w * (upper_cdf - lower_cdf)).item()
            return prob_sum / count

        f005 = interval_fraction(0.05 * (1.0 + center))
        f015 = interval_fraction(0.15 * (1.0 + center))
        total_count += count
        weighted_f005 += f005 * count
        weighted_f015 += f015 * count
        bins.append(
            {
                "z_min": float(left),
                "z_max": float(right),
                "z_center": float(center),
                "count": count,
                "mode": float(mode),
                "F005": float(f005),
                "F015": float(f015),
            }
        )

    return {
        "z_bin_variable": "Prediction Value",
        "z_min": EUCLID_Z_MIN,
        "z_max": EUCLID_Z_MAX,
        "bin_width": EUCLID_BIN_WIDTH,
        "F005": float(weighted_f005 / total_count) if total_count else None,
        "F015": float(weighted_f015 / total_count) if total_count else None,
        "total_binned_samples": int(total_count),
        "bins": bins,
    }


def calc_redshift_binned_metrics(
    pred_pdf: torch.Tensor,
    z_label: torch.Tensor,
    z_pdf: torch.Tensor,
    mode: str,
) -> dict:
    label = z_label.squeeze(-1) if z_label.dim() > 1 else z_label
    edges = make_euclid_bin_edges()
    bins = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= EUCLID_Z_MAX:
            mask = (label >= left) & (label <= right)
        else:
            mask = (label >= left) & (label < right)
        count = int(mask.sum().item())
        if count == 0:
            bins.append(
                {
                    "z_min": float(left),
                    "z_max": float(right),
                    "z_center": float(0.5 * (left + right)),
                    "count": 0,
                    "metrics": None,
                }
            )
            continue
        bins.append(
            {
                "z_min": float(left),
                "z_max": float(right),
                "z_center": float(0.5 * (left + right)),
                "count": count,
                "metrics": calc_metrics(
                    pred_pdf=pred_pdf[mask],
                    z_label=z_label[mask],
                    z_pdf=z_pdf[mask],
                    mode=mode,
                ),
            }
        )
    return {
        "z_bin_variable": "Ground Truth",
        "z_min": EUCLID_Z_MIN,
        "z_max": EUCLID_Z_MAX,
        "bin_width": EUCLID_BIN_WIDTH,
        "bins": bins,
    }


def percent_diff(value: float, base: float):
    if base == 0:
        return 0.0 if value == 0 else None
    return (value - base) / abs(base) * 100.0


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    try:
        lightning_model = BuildLightningModel.load_from_checkpoint(
            os.path.join(model_path),
            map_location=device,
        )
        model = lightning_model.model
        model.load_state_dict(model.state_dict())
        CONSOLE.print("[Info] Load model from {}".format(model_path))
    except Exception as e:
        CONSOLE.print("[Error] Load model from {} failed".format(model_path))
        raise e
    return model


def get_infer_autocast(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def inference(
    settings: dict,
    ckpt_path: str,
    dataset_fold: str,
    res_save_dir: str,
    clear_res_dir: bool = False,
    device: torch.device = torch.device("cpu"),
    cuda_infer_timer: bool = True,
    model_name: str = "model",
    progress: Progress | None = None,
    batch_task_id: int | None = None,
    infer_precision: str = "fp32",
) -> dict:
    ensure_result_dir(res_save_dir, clear_res_dir)
    test_dataloader = build_dataloader(
        settings,
        mode="test",
        cross_val_name=dataset_fold,
    )
    test_model = load_model(ckpt_path, device)
    data_info = []
    source_indices = {}
    pred = []
    z_pred = []
    z_label = []
    infer_time_ms = 0.0
    total_samples = 0

    test_model.eval()
    owns_progress = progress is None
    progress_context = make_progress() if owns_progress else nullcontext(progress)
    with progress_context as active_progress:
        if batch_task_id is None:
            batch_task = active_progress.add_task(
                "{} / {}".format(model_name, dataset_fold),
                total=len(test_dataloader),
                metrics="",
            )
        else:
            batch_task = batch_task_id
            active_progress.update(
                batch_task,
                description="{} / {} batches".format(model_name, dataset_fold),
                total=len(test_dataloader),
                completed=0,
                metrics="",
            )
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                active_progress.update(
                    batch_task,
                    description="{} / {} batch {}".format(
                        model_name,
                        dataset_fold,
                        idx + 1,
                    ),
                )
                (
                    id,
                    ra,
                    dec,
                    photometric,
                    mags,
                    label,
                    flux,
                    south_north_flag,
                ) = batch
                photometric = photometric.to(device)
                mags = mags.to(device)
                label = label.to(device)
                infer_start = None
                infer_end = None
                if cuda_infer_timer and device.type == "cuda":
                    infer_start = torch.cuda.Event(enable_timing=True)
                    infer_end = torch.cuda.Event(enable_timing=True)
                sample_offset = total_samples
                for i in range(photometric.shape[0]):
                    source_value = to_python_scalar(south_north_flag[i])
                    source_indices.setdefault(source_value, []).append(
                        sample_offset + i
                    )
                    data_info.append(
                        {
                            "TARGETID": id[i],
                            "TARGET_RA": ra[i],
                            "TARGET_DEC": dec[i],
                            "SOURCE": south_north_flag[i],
                        }
                    )
                # ======================================================
                # start inference time record
                if infer_start is not None:
                    infer_start.record(torch.cuda.current_stream(device))
                with get_infer_autocast(device=device, precision=infer_precision):
                    _pred, _, _ = test_model(
                        photometric=photometric,
                        magnitudes=mags,
                        output_spectrum=False,
                    )
                if infer_end is not None:
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
                active_progress.update(
                    batch_task,
                    metrics=format_batch_metrics(
                        batch_metrics_postfix(
                            _pred,
                            label,
                            PDF_MODE,
                        )
                    ),
                )
                z_label.append(label.detach().cpu())
                total_samples += photometric.shape[0]
                active_progress.advance(batch_task)
        if batch_task_id is None:
            active_progress.remove_task(batch_task)

    CONSOLE.print("=" * 50)
    fps = total_samples / (infer_time_ms / 1000.0) if infer_time_ms > 0 else -1
    CONSOLE.print(
        "[Info] FPS: {:.2f}, Total Samples: {}, Total Time: {:.2f} ms".format(
            fps,
            total_samples,
            infer_time_ms,
        )
    )
    CONSOLE.print("=" * 50)
    z_label = torch.cat(z_label, dim=0).float()
    z_pdf = torch.cat(z_pred, dim=0).float()
    pred_pdf = torch.cat(pred, dim=0).float()
    finite_mask = finite_prediction_mask(
        pred_pdf=pred_pdf,
        z_label=z_label,
        z_pdf=z_pdf,
    )
    if not bool(finite_mask.all().item()):
        kept = int(finite_mask.sum().item())
        dropped = int(finite_mask.numel() - kept)
        CONSOLE.print(
            "[Warning] Dropping {} non-finite samples before metrics/export.".format(
                dropped
            )
        )
        keep_indices = finite_mask.nonzero(as_tuple=False).flatten()
        pred_pdf = pred_pdf[keep_indices]
        z_label = z_label[keep_indices]
        z_pdf = z_pdf[keep_indices]
        data_info = [data_info[i] for i in keep_indices.tolist()]
        source_indices = rebuild_source_indices(data_info)
        total_samples = kept
    delta_z_values = calc_point_delta_z(z_pdf=z_pdf, z_label=z_label)
    nll_values = calc_nll_values(pred_pdf=pred_pdf, z_label=z_label, eps=config["eps"])
    crps_values = calc_crps_values(
        pred_pdf=pred_pdf, z_label=z_label, eps=config["eps"]
    )
    CONSOLE.print("[Info] PDF Estimation Results:")
    overall_metrics = calc_metrics(pred_pdf, z_label, z_pdf, PDF_MODE)
    pdz_metrics = calc_euclid_pdz_fractions(
        pred_pdf=pred_pdf,
        z_label=z_label,
        z_pdf=z_pdf,
    )
    binned_metrics = calc_redshift_binned_metrics(
        pred_pdf=pred_pdf,
        z_label=z_label,
        z_pdf=z_pdf,
        mode=PDF_MODE,
    )
    CONSOLE.print(format_metrics(overall_metrics))
    overall_pit = save_pit_metrics(
        pred_pdf=pred_pdf,
        z_label=z_label,
        bins=50,
    )
    res_json = {
        "cross_val_name": dataset_fold,
        "ckpt_path": ckpt_path,
        "total_samples": int(total_samples),
        "infer_time_ms": float(infer_time_ms),
        "fps": float(fps),
        "mae": float(overall_metrics["mae"]),
        "median_abs_error": float(overall_metrics["median_abs_error"]),
        "rmse": float(overall_metrics["rmse"]),
        "nmad_1.48": float(overall_metrics["nmad_1.48"]),
        "sigma_0.05": float(overall_metrics["sigma_0.05"]),
        "sigma_0.15": float(overall_metrics["sigma_0.15"]),
        "z_bias": float(overall_metrics["z_bias"]),
        "median_delta_z": float(overall_metrics["median_delta_z"]),
        "outlier_fraction_0.10": float(overall_metrics["outlier_fraction_0.10"]),
        "outlier_fraction": float(overall_metrics["outlier_fraction"]),
        "outlier_fraction_0.15": float(overall_metrics["outlier_fraction_0.15"]),
        "nll": float(overall_metrics["nll"]),
        "crps_mean": float(overall_metrics["crps_mean"]),
        "crps_median": float(overall_metrics["crps_median"]),
        "pdz_metrics": pdz_metrics,
        "redshift_binned_metrics": binned_metrics,
        "pit_metrics": overall_pit,
        "by_source": [],
    }
    by_source_metrics = []
    for source_value, indices in source_indices.items():
        if not indices:
            continue
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        z_label_source = z_label[idx_tensor]
        pred_pdf_source = pred_pdf[idx_tensor]
        z_pdf_source = z_pdf[idx_tensor]
        source_metrics = calc_metrics(
            pred_pdf_source,
            z_label_source,
            z_pdf_source,
            PDF_MODE,
        )
        source_pit = save_pit_metrics(
            pred_pdf=pred_pdf_source,
            z_label=z_label_source,
            bins=50,
        )
        CONSOLE.print(
            "[Info] SOURCE {} -> {}".format(
                source_value,
                format_metrics(source_metrics),
            )
        )
        source_entry = {
            "total_samples": int(len(indices)),
            "metrics": source_metrics,
            "pit_metrics": source_pit,
        }
        by_source_metrics.append(
            {
                "source": source_value,
                **source_entry,
            }
        )
    by_source_metrics.sort(key=lambda item: item["metrics"]["outlier_fraction"])
    if len(by_source_metrics) > 0:
        base_metrics = by_source_metrics[0]
        res_json["by_source"] = []
        for idx, item in enumerate(by_source_metrics):
            diff_percent = {}
            if idx > 0:
                for key in [
                    "mae",
                    "median_abs_error",
                    "rmse",
                    "nmad_1.48",
                    "sigma_0.05",
                    "sigma_0.15",
                    "z_bias",
                    "median_delta_z",
                    "outlier_fraction_0.10",
                    "outlier_fraction",
                    "outlier_fraction_0.15",
                    "nll",
                    "crps_mean",
                    "crps_median",
                ]:
                    diff_percent[key] = percent_diff(
                        item["metrics"][key], base_metrics["metrics"][key]
                    )
            res_json["by_source"].append(
                {
                    "source": str(item["source"]),
                    "total_samples": item["total_samples"],
                    "metrics": {
                        "mae": item["metrics"]["mae"],
                        "median_abs_error": item["metrics"]["median_abs_error"],
                        "rmse": item["metrics"]["rmse"],
                        "nmad_1.48": item["metrics"]["nmad_1.48"],
                        "sigma_0.05": item["metrics"]["sigma_0.05"],
                        "sigma_0.15": item["metrics"]["sigma_0.15"],
                        "z_bias": item["metrics"]["z_bias"],
                        "median_delta_z": item["metrics"]["median_delta_z"],
                        "outlier_fraction_0.10": item["metrics"][
                            "outlier_fraction_0.10"
                        ],
                        "outlier_fraction": item["metrics"]["outlier_fraction"],
                        "outlier_fraction_0.15": item["metrics"][
                            "outlier_fraction_0.15"
                        ],
                        "nll": item["metrics"]["nll"],
                        "crps_mean": item["metrics"]["crps_mean"],
                        "crps_median": item["metrics"]["crps_median"],
                    },
                    "pit_metrics": item["pit_metrics"],
                    "diff_percent_vs_best": diff_percent or None,
                }
            )
        CONSOLE.print("[Info] Sorted by outlier_fraction (asc):")
        for idx, item in enumerate(res_json["by_source"]):
            if idx == 0:
                CONSOLE.print(
                    "[Info] BASE SOURCE {} -> {}".format(
                        item["source"],
                        format_metrics(item["metrics"]),
                    )
                )
                continue
            CONSOLE.print(
                "[Info] SOURCE {} -> {}".format(
                    item["source"],
                    format_metrics(item["metrics"]),
                )
            )
            diff = item["diff_percent_vs_best"]
            if diff is None:
                CONSOLE.print("[Info]    Diff% vs base: N/A")
            else:
                CONSOLE.print(
                    "[Info]    Diff% vs base (MAE/RMSE/NMAD/Sigma0.05/Sigma0.15/ZBias/Outlier0.15/NLL/CRPSmed): "
                    "{:.2f}% / {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%".format(
                        diff["mae"],
                        diff["rmse"],
                        diff["nmad_1.48"],
                        diff["sigma_0.05"],
                        diff["sigma_0.15"],
                        diff["z_bias"],
                        diff["outlier_fraction"],
                        diff["nll"],
                        diff["crps_median"],
                    )
                )
    # save json
    with open(os.path.join(res_save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res_json, f, indent=4)
    # save data info
    data_df = build_result_dataframe(
        data_info=data_info,
        z_label=z_label,
        z_pdf=z_pdf,
        delta_z_values=delta_z_values,
        nll_values=nll_values,
        crps_values=crps_values,
        pred_pdf=pred_pdf,
    )
    data_df.to_csv(
        os.path.join(res_save_dir, "result.csv"),
        index=False,
        encoding="utf-8",
    )
    return {
        "dataset_fold": dataset_fold,
        "pred_pdf": pred_pdf,
        "z_label": z_label,
        "z_pdf": z_pdf,
        "source_values": [to_python_scalar(item["SOURCE"]) for item in data_info],
        "result_csv": data_df,
    }


def save_all_folds_summary(
    model_save_dir: str,
    fold_outputs: list[dict],
) -> None:
    all_dir = os.path.join(model_save_dir, "all_folds")
    os.makedirs(all_dir, exist_ok=True)

    pred_pdf = torch.cat([item["pred_pdf"] for item in fold_outputs], dim=0).float()
    z_label = torch.cat([item["z_label"] for item in fold_outputs], dim=0).float()
    z_pdf = torch.cat([item["z_pdf"] for item in fold_outputs], dim=0).float()
    source_values = []
    for item in fold_outputs:
        source_values.extend(item["source_values"])
    result_csv = pd.concat(
        [item["result_csv"] for item in fold_outputs],
        ignore_index=True,
    )
    finite_mask = finite_prediction_mask(
        pred_pdf=pred_pdf,
        z_label=z_label,
        z_pdf=z_pdf,
    )
    if not bool(finite_mask.all().item()):
        kept = int(finite_mask.sum().item())
        dropped = int(finite_mask.numel() - kept)
        CONSOLE.print(
            "[Warning] Dropping {} non-finite all-fold samples before metrics/export.".format(
                dropped
            )
        )
        keep_indices = finite_mask.nonzero(as_tuple=False).flatten()
        pred_pdf = pred_pdf[keep_indices]
        z_label = z_label[keep_indices]
        z_pdf = z_pdf[keep_indices]
        source_values = [source_values[i] for i in keep_indices.tolist()]
        result_csv = result_csv.iloc[keep_indices.tolist()].reset_index(drop=True)

    overall_metrics = calc_metrics(pred_pdf, z_label, z_pdf, PDF_MODE)
    pdz_metrics = calc_euclid_pdz_fractions(
        pred_pdf=pred_pdf,
        z_label=z_label,
        z_pdf=z_pdf,
    )
    binned_metrics = calc_redshift_binned_metrics(
        pred_pdf=pred_pdf,
        z_label=z_label,
        z_pdf=z_pdf,
        mode=PDF_MODE,
    )
    overall_pit = save_pit_metrics(
        pred_pdf=pred_pdf,
        z_label=z_label,
        bins=50,
    )
    res_json = {
        "cross_val_name": "all_folds",
        "ckpt_path": "multiple_folds",
        "total_samples": int(z_label.shape[0]),
        "infer_time_ms": -1,
        "fps": -1,
        "mae": float(overall_metrics["mae"]),
        "median_abs_error": float(overall_metrics["median_abs_error"]),
        "rmse": float(overall_metrics["rmse"]),
        "nmad_1.48": float(overall_metrics["nmad_1.48"]),
        "sigma_0.05": float(overall_metrics["sigma_0.05"]),
        "sigma_0.15": float(overall_metrics["sigma_0.15"]),
        "z_bias": float(overall_metrics["z_bias"]),
        "median_delta_z": float(overall_metrics["median_delta_z"]),
        "outlier_fraction_0.10": float(overall_metrics["outlier_fraction_0.10"]),
        "outlier_fraction": float(overall_metrics["outlier_fraction"]),
        "outlier_fraction_0.15": float(overall_metrics["outlier_fraction_0.15"]),
        "nll": float(overall_metrics["nll"]),
        "crps_mean": float(overall_metrics["crps_mean"]),
        "crps_median": float(overall_metrics["crps_median"]),
        "pdz_metrics": pdz_metrics,
        "redshift_binned_metrics": binned_metrics,
        "pit_metrics": overall_pit,
        "by_source": [],
    }

    source_indices = {}
    for idx, source_value in enumerate(source_values):
        source_indices.setdefault(source_value, []).append(idx)

    by_source_metrics = []
    for source_value, indices in source_indices.items():
        if not indices:
            continue
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        z_label_source = z_label[idx_tensor]
        pred_pdf_source = pred_pdf[idx_tensor]
        z_pdf_source = z_pdf[idx_tensor]
        source_metrics = calc_metrics(
            pred_pdf_source,
            z_label_source,
            z_pdf_source,
            PDF_MODE,
        )
        source_pit = save_pit_metrics(
            pred_pdf=pred_pdf_source,
            z_label=z_label_source,
            bins=50,
        )
        by_source_metrics.append(
            {
                "source": source_value,
                "total_samples": int(len(indices)),
                "metrics": source_metrics,
                "pit_metrics": source_pit,
            }
        )

    by_source_metrics.sort(key=lambda item: item["metrics"]["outlier_fraction"])
    if len(by_source_metrics) > 0:
        base_metrics = by_source_metrics[0]["metrics"]
        for idx, item in enumerate(by_source_metrics):
            diff_percent = {}
            if idx > 0:
                for key in [
                    "mae",
                    "median_abs_error",
                    "rmse",
                    "nmad_1.48",
                    "sigma_0.05",
                    "sigma_0.15",
                    "z_bias",
                    "median_delta_z",
                    "outlier_fraction_0.10",
                    "outlier_fraction",
                    "outlier_fraction_0.15",
                    "nll",
                    "crps_mean",
                    "crps_median",
                ]:
                    diff_percent[key] = percent_diff(
                        item["metrics"][key], base_metrics[key]
                    )
            res_json["by_source"].append(
                {
                    "source": str(item["source"]),
                    "total_samples": item["total_samples"],
                    "metrics": {
                        "mae": item["metrics"]["mae"],
                        "median_abs_error": item["metrics"]["median_abs_error"],
                        "rmse": item["metrics"]["rmse"],
                        "nmad_1.48": item["metrics"]["nmad_1.48"],
                        "sigma_0.05": item["metrics"]["sigma_0.05"],
                        "sigma_0.15": item["metrics"]["sigma_0.15"],
                        "z_bias": item["metrics"]["z_bias"],
                        "median_delta_z": item["metrics"]["median_delta_z"],
                        "outlier_fraction_0.10": item["metrics"][
                            "outlier_fraction_0.10"
                        ],
                        "outlier_fraction": item["metrics"]["outlier_fraction"],
                        "outlier_fraction_0.15": item["metrics"][
                            "outlier_fraction_0.15"
                        ],
                        "nll": item["metrics"]["nll"],
                        "crps_mean": item["metrics"]["crps_mean"],
                        "crps_median": item["metrics"]["crps_median"],
                    },
                    "pit_metrics": item["pit_metrics"],
                    "diff_percent_vs_best": diff_percent or None,
                }
            )

    with open(os.path.join(all_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res_json, f, indent=4)
    result_csv = result_csv.reindex(columns=RESULT_CSV_COLUMNS)
    result_csv.to_csv(
        os.path.join(all_dir, "result.csv"),
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
    args.add_argument(
        "--infer-precision",
        "--infer_precision",
        choices=("fp32", "bf16"),
        default="bf16",
        help="Inference precision: fp32 disables autocast; bf16 enables CUDA bfloat16 autocast.",
    )
    opts = args.parse_args()

    CONSOLE.print("[INFO] Using device: {}".format(DEVICE))
    CONSOLE.print("[INFO] Inference precision: {}".format(opts.infer_precision))
    with make_progress() as progress:
        model_task = progress.add_task(
            "Models",
            total=len(MODEL_PATH_LIST),
            metrics="",
        )
        fold_task = progress.add_task("Folds", total=1, metrics="")
        batch_task = progress.add_task("Prediction batches", total=1, metrics="")
        for model_info in MODEL_PATH_LIST:
            model_name = model_info["model_name"]
            ckpt_name_dict = model_info["ckpt_name"]
            CONSOLE.print("[INFO] Predicting model: {}".format(model_name))
            fold_outputs = []
            progress.update(
                fold_task,
                description="{} folds".format(model_name),
                total=len(ckpt_name_dict),
                completed=0,
                metrics="",
            )
            progress.update(
                batch_task,
                description="{} batches".format(model_name),
                total=1,
                completed=0,
                metrics="",
            )
            for fold, ckpt_name in ckpt_name_dict.items():
                CONSOLE.print("[INFO] Predicting fold: {}".format(fold))
                res_save_path = os.path.join(RES_SAVE_DIR, model_name, fold)
                output = inference(
                    settings=config,
                    ckpt_path=str(os.path.join(ckpt_name)),
                    dataset_fold=fold,
                    res_save_dir=res_save_path,
                    clear_res_dir=opts.clear,
                    device=DEVICE,
                    cuda_infer_timer=True,
                    model_name=model_name,
                    progress=progress,
                    batch_task_id=batch_task,
                    infer_precision=opts.infer_precision,
                )
                fold_outputs.append(output)
                progress.advance(fold_task)
            if len(fold_outputs) > 0:
                save_all_folds_summary(
                    model_save_dir=os.path.join(RES_SAVE_DIR, model_name),
                    fold_outputs=fold_outputs,
                )
            progress.advance(model_task)
