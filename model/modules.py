import torch
import torch.nn as nn
import torch.nn.functional as f
from timm.layers import DropPath

from model.layers import (
    PoolMixer,
    RMSNorm1d,
    MHSA1D,
)


class CenterAwareStem(nn.Module):

    def __init__(
        self,
        in_channel: int,
        in_dim: int,
        out_channel: int,
        kernel_size_list: list[int] = None,
    ):
        super().__init__()
        if kernel_size_list is None:
            kernel_size_list = [7, 5, 3, 1]
        scale_count = len(kernel_size_list)
        self.origin_dim = in_dim
        self.in_dims = [in_dim // (2**i) for i in range(scale_count)]
        self.cutout_paddings = [
            (in_dim - self.in_dims[i]) // 2 for i in range(scale_count)
        ]
        self.stem = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size_list[i],
                        stride=1,
                        padding="same",
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.Conv2d(
                        in_channels=out_channel,
                        out_channels=out_channel * 4,
                        kernel_size=1,
                        stride=1,
                        padding="same",
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        in_channels=out_channel * 4,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride=1,
                        padding="same",
                    ),
                )
                for i in range(scale_count)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scales = []
        for i, stem in enumerate(self.stem):
            if self.cutout_paddings[i] > 0:
                x_cut = x[
                    :,
                    :,
                    self.cutout_paddings[i] : -self.cutout_paddings[i],
                    self.cutout_paddings[i] : -self.cutout_paddings[i],
                ]
            else:
                x_cut = x
            x_cut = stem(x_cut)
            padding_left = (self.origin_dim - self.in_dims[i]) // 2
            padding_top = (self.origin_dim - self.in_dims[i]) // 2
            padding_right = self.origin_dim - self.in_dims[i] - padding_left
            padding_bottom = self.origin_dim - self.in_dims[i] - padding_top
            x_cut = f.pad(
                x_cut,
                pad=(padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            x_scales.append(x_cut)
        x = torch.stack(x_scales, dim=0).sum(dim=0)
        x = x.view(x.shape[0], x.shape[1], -1)
        return x


class MultiStageRegressiveHead(nn.Module):

    def __init__(
        self,
        in_channels: list[int],
        out_channel: int,
        out_dim: int,
        dropout: float = None,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        total_channel = sum(in_channels) * 2
        hidden = max(total_channel // 2, 64)
        self.head = nn.Sequential(
            nn.BatchNorm1d(total_channel),
            nn.Linear(total_channel, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(hidden // 2, out_channel * out_dim),
        )
        self.out_channel = out_channel
        self.out_dim = out_dim

    def forward(self, deep_features: list[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for x in deep_features:
            avg_x = self.avg_pool(x)
            max_x = self.max_pool(x)
            pooled.append(torch.cat([avg_x, max_x], dim=1).squeeze(-1))
        x = torch.cat(pooled, dim=1)
        x = self.head(x)
        return x.reshape(-1, self.out_channel, self.out_dim)


class MagFeatureExtractor(nn.Module):

    def __init__(
        self,
        in_channel: int,
        in_dim: int,
        hidden_dim: int,
        hidden_channels: list[int],
        dropout: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channels = hidden_channels
        self.kernel_size = [16, 11, 7, 3]
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 4),
            nn.BatchNorm1d(in_channel),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(in_channel),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.extractor = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=(in_channel if i == 0 else self.channels[i - 1]),
                        out_channels=self.channels[i],
                        kernel_size=self.kernel_size[i],
                        stride=1,
                        padding="same",
                    ),
                    nn.BatchNorm1d(self.channels[i]),
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels=self.channels[i],
                        out_channels=self.channels[i],
                        kernel_size=self.kernel_size[i],
                        stride=1,
                        padding="same",
                    ),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
                )
                for i in range(len(self.channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        mag_features = []
        for layer in self.extractor:
            x = layer(x)
            mag_features.append(x)
        return mag_features


class ImgPhotometricMixFusion(nn.Module):

    def __init__(
        self,
        photometric_channel: int,
        mag_channel: int,
        hidden_feature: int,
        num_heads: int,
        dropout: float = None,
        eps: float | int = 1e-8,
    ):
        super().__init__()
        self.mags_attn = MHSA1D(
            q_channel=mag_channel,
            kv_channel=photometric_channel,
            embedding_dim=hidden_feature,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.token_mixer_1 = PoolMixer(
            in_channel=mag_channel,
            embedding_dim=hidden_feature,
            dropout=dropout,
            dim="1d",
        )
        self.photo_attn = MHSA1D(
            q_channel=photometric_channel,
            kv_channel=mag_channel,
            embedding_dim=hidden_feature,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.token_mixer_2 = PoolMixer(
            in_channel=photometric_channel,
            embedding_dim=hidden_feature,
            dropout=dropout,
            dim="1d",
        )
        self.inter_attn = MHSA1D(
            q_channel=photometric_channel,
            kv_channel=mag_channel,
            embedding_dim=hidden_feature * 2,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            RMSNorm1d(photometric_channel, eps=eps),
            nn.Linear(
                in_features=hidden_feature * 2,
                out_features=hidden_feature * 4,
            ),
            nn.GELU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(
                in_features=hidden_feature * 4,
                out_features=hidden_feature,
            ),
        )
        self.dropout = DropPath(dropout) if dropout is not None else nn.Identity()
        self.gamma_photo = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.gamma_mag = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.gamma_fuse = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(
        self,
        photometric: torch.Tensor,
        mags: torch.Tensor,
    ) -> torch.Tensor:
        mag_x = self.mags_attn(mags, photometric, photometric)
        mag_x = self.dropout(mag_x) + mags
        mag_x = self.token_mixer_1(mag_x) + mag_x

        photometric_x = self.photo_attn(photometric, mags, mags)
        photometric_x = self.dropout(photometric_x) + photometric
        photometric_x = self.token_mixer_2(photometric_x) + photometric_x

        x = torch.cat([photometric_x, mag_x], dim=-1)
        x = self.inter_attn(x, x, x)
        x = self.dropout(x) + x
        return (
            self.gamma_photo * photometric
            + self.gamma_mag * mags
            + self.gamma_fuse * self.dropout(self.ffn(x))
        )


class PhotometricImgExtractor(nn.Module):

    def __init__(
        self,
        in_channel: int,
        output_channel: int,
        embedding_dim: int,
        blks: int,
        dropout: float = None,
        down_sample: bool = True,
    ):
        super().__init__()

        self.blk = nn.ModuleList(
            [
                PoolMixer(
                    in_channel=in_channel,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    dim="1d",
                )
                for _ in range(blks)
            ]
        )
        self.down_sample = (
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=output_channel,
                kernel_size=2,
                stride=2,
                padding=0,
            )
            if down_sample
            else nn.Conv1d(
                in_channels=in_channel,
                out_channels=output_channel,
                kernel_size=3,
                padding="same",
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blk:
            x = blk(x)
        x = self.down_sample(x)
        return x


class BuildModel(nn.Module):

    def __init__(
        self,
        photo_in_channel: int,
        photo_in_dim: int,
        mag_in_channel: int,
        mag_in_dim: int,
        out_channel: int,
        blk_count: list[int],
        hidden_channels: list[int],
        num_heads: list[int],
        spectrum_extractor_config: dict,
        dropout: float = None,
        enable_spectrum: bool = True,
        eps: float | int = 1e-8,
    ):
        super().__init__()
        assert len(hidden_channels) == len(blk_count), ValueError(
            "hidden_channels length must be equal to blk_count length"
        )
        assert len(num_heads) == len(hidden_channels), ValueError(
            "num_heads length must be equal to hidden_channels length"
        )
        for i, (ch, nh) in enumerate(zip(hidden_channels, num_heads)):
            assert ch % nh == 0, ValueError(
                f"hidden_channels[{i}]={ch} must be divisible by num_heads[{i}]={nh}"
            )
        self.hidden_channels = hidden_channels
        self.hidden_features = [
            photo_in_dim**2 // (2**i) for i in range(len(self.hidden_channels))
        ]
        self.photometric_stem = CenterAwareStem(
            in_channel=photo_in_channel,
            in_dim=photo_in_dim,
            out_channel=self.hidden_channels[0],
            kernel_size_list=[7, 5, 3, 1],
        )
        self.photometric_extractor = nn.ModuleList(
            [
                PhotometricImgExtractor(
                    in_channel=(
                        self.hidden_channels[0]
                        if i == 0
                        else self.hidden_channels[i - 1]
                    ),
                    output_channel=self.hidden_channels[i],
                    embedding_dim=self.hidden_features[i],
                    blks=blk_count[i],
                    dropout=dropout,
                    down_sample=True,
                )
                for i in range(len(self.hidden_channels))
            ]
        )
        self.mag_extractor = MagFeatureExtractor(
            in_channel=mag_in_channel,
            in_dim=mag_in_dim,
            hidden_dim=self.hidden_features[0],
            hidden_channels=self.hidden_channels,
            dropout=dropout,
        )
        self.feature_mixers = nn.ModuleList(
            [
                ImgPhotometricMixFusion(
                    photometric_channel=self.hidden_channels[i],
                    mag_channel=self.hidden_channels[i],
                    hidden_feature=self.hidden_features[i] // 2,
                    num_heads=num_heads[i],
                    dropout=dropout,
                    eps=eps,
                )
                for i in range(len(self.hidden_channels))
            ]
        )

        self.reg_head = MultiStageRegressiveHead(
            in_channels=self.hidden_channels,
            out_dim=3,
            out_channel=out_channel,
            dropout=dropout,
        )
        self.enable_spectrum = enable_spectrum
        if enable_spectrum:
            embedding_dim = spectrum_extractor_config["spectrum_embedding_dim"]
            proj_hidden_dim = max(embedding_dim // 2, 32)
            supervised = spectrum_extractor_config["spectral_supervised_layers"]
            align_split_ratio = float(spectrum_extractor_config["align_split_ratio"])
            self.align_channels = {
                i: max(1, int(self.hidden_channels[i] * align_split_ratio))
                for i in supervised
            }
            self.projection_heads = nn.ModuleDict(
                {
                    str(i): nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Flatten(),
                        nn.Linear(self.align_channels[i], proj_hidden_dim),
                        nn.GELU(),
                        nn.Linear(proj_hidden_dim, embedding_dim),
                    )
                    for i in supervised
                }
            )
        self.eps = eps

    def forward(
        self,
        photometric: torch.Tensor,
        magnitudes: torch.Tensor,
        output_spectrum: bool = True,
    ) -> tuple:
        x = self.photometric_stem(photometric)
        mag_features = self.mag_extractor(magnitudes)
        deep_features = []
        for i in range(len(self.photometric_extractor)):
            x = self.photometric_extractor[i](x)
            x = self.feature_mixers[i](x, mag_features[i])
            deep_features.append(x)

        # Keep final GMM parameters in fp32 even when the backbone uses autocast.
        if photometric.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                pred = self.reg_head([feature.float() for feature in deep_features])
        else:
            pred = self.reg_head(deep_features)

        if self.enable_spectrum and output_spectrum:
            spectrum_features = {
                int(i): f.normalize(
                    proj(deep_features[int(i)][:, : self.align_channels[int(i)], :]),
                    p=2,
                    dim=-1,
                )
                for i, proj in self.projection_heads.items()
            }
        else:
            spectrum_features = None
        return (
            pred,
            spectrum_features,
            deep_features,
        )
