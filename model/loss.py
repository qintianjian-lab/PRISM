import math

import torch
import torch.nn as nn
import torch.nn.functional as f


def z_estimate(
    pred: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Calculate the delta_z from the prediction.
    :param pred: predictions, shape like (batch_size, N, 3)
    :param mode: Gaussian mixture mode, can be 'mean' or 'max', 'mean' will return the weighted mean of the
        Gaussian components, 'max' will return the component with the highest weight
    :return: z estimate, shape like (batch_size, 1)
    """
    mean = pred[:, :, 0]  # (batch_size, N)
    weight = f.log_softmax(pred[:, :, 2], dim=1)  # (batch_size, N)
    if mode == "mean":
        x = torch.sum(mean * torch.exp(weight), dim=-1)  # (batch_size,)
        return x
    elif mode == "max":
        max_index = torch.argmax(weight, dim=-1)  # (batch_size,)
        x = mean[torch.arange(mean.size(0)), max_index]  # (batch_size,)
        return x
    else:
        raise ValueError("mode must be 'mean' or 'max', got {}".format(mode))


def delta_z(
    pred: torch.Tensor,
    label: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Calculate the delta_z from the prediction.
    :param pred: predictions, shape like (batch_size, N, 3)
    :param label: labels, actual redshift, shape like (batch_size, 1)
    :param mode: Gaussian mixture mode, can be 'mean' or 'max', 'mean' will return the weighted mean of the
        Gaussian components, 'max' will return the component with the highest weight
    :return: delta_z, shape like (batch_size, 1)
    """
    pred_z = z_estimate(
        pred=pred,
        mode=mode,
    )
    pred_z = pred_z.unsqueeze(-1) if pred_z.dim() == 1 else pred_z
    label = label.unsqueeze(-1) if label.dim() == 1 else label
    return (pred_z - label) / (1 + label)


def z_bias(
    d_z: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the bias of the redshift estimation.
    :param d_z: predictions, shape like (all,1)
    :return: Mean(delta_z), shape like (1)
    """
    return d_z.mean()


def nmad_z(
    d_z: torch.Tensor,
    factor: float = 1.4826,
) -> torch.Tensor:
    """
    Calculate the Normalized Median Absolute Deviation (NMAD) of the redshift estimation.
    :param d_z: predictions, shape like (all,1)
    :param factor: factor to multiply with the median absolute deviation, default is 1.48
    :return: NMAD, shape like (1)
    """
    median_d_z = torch.median(d_z)
    return factor * torch.median(torch.abs(d_z - median_d_z))


def sigma_n(
    d_z: torch.Tensor,
    n: float = 0.05,
) -> torch.Tensor:
    """
    Sigma_n metric
    :param d_z: predictions, shape like (all,1)
    :param n: threshold for the sigma_n metric, default is 0.05
    :return: Sigma_n, shape like (1), bigger is better
    """
    d_z = d_z.abs()
    mask = n > d_z
    return torch.sum(mask) / len(d_z)


def outlier_fraction(
    d_z: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Outlier fraction metric
    :param d_z: predictions, shape like (all,1)
    :param threshold: threshold for the outlier fraction metric, default is 0.1
    :return: Outlier fraction, shape like (1), smaller is better
    """
    d_z = d_z.abs()
    mask = d_z >= threshold
    return torch.sum(mask) / len(d_z)


class NLLLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert torch.isnan(pred).sum() == 0, "Prediction contains NaN"

        mean = pred[:, :, 0]
        std = f.softplus(pred[:, :, 1]) + self.eps
        log_weight = f.log_softmax(pred[:, :, 2] / self.temperature, dim=1)
        target_expanded = label.expand_as(mean)
        # -log(std) - 0.5*log(2*pi) - 0.5*((y-mu)/std)^2
        log_gaussian_pdf = (
            -torch.log(std)
            - 0.5 * math.log(2 * math.pi)
            - 0.5 * ((target_expanded - mean) / std) ** 2
        )
        # log P(y|x) = logsumexp_k( log(weight_k) + log(N_k) )
        log_mixture_likelihood = torch.logsumexp(log_weight + log_gaussian_pdf, dim=1)
        return -torch.mean(log_mixture_likelihood), torch.exp(log_weight)


class PITModule(nn.Module):

    def __init__(
        self,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.standard_normal = torch.distributions.Normal(0.0, 1.0)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mean = pred[:, :, 0]  # (B, N)
        std = f.softplus(pred[:, :, 1]) + self.eps  # (B, N)
        weights = f.softmax(pred, dim=-1)  # (B, N)
        if label.dim() == 1:
            label = label.unsqueeze(-1)  # (B, 1)
        z = (label - mean) / std  # (B, N)
        cdf = self.standard_normal.cdf(z)  # (B, N)
        pit = torch.sum(weights * cdf, dim=-1)  # (B,)
        return pit


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity alignment loss.
    """

    def __init__(
        self,
        eps: float = 1e-8,
    ):
        """
        :param eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        photo_emb: torch.Tensor,
        spec_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param photo_emb: (B, D) photometric projection embedding.
        :param spec_emb:  (B, D) spectral encoder embedding (frozen).
        :return: scalar cosine similarity loss.
        """
        assert (
            photo_emb.shape == spec_emb.shape
        ), "photo_emb and spec_emb must have the same shape"

        similarity = f.cosine_similarity(photo_emb, spec_emb, dim=1, eps=self.eps)
        return (1.0 - similarity).mean()
