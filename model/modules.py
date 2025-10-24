import torch
import torch.nn as nn
from timm.layers import DropPath

from model.layers import (
    MultiHeadSelfAttention1D,
    LayerNormChannel,
    FFN,
    PoolMixer,
    StarFusion,
    UpSampleBlk,
)


class TokenMixBlk(nn.Module):

    def __init__(
        self,
        in_channel: int,
        embedding_dim: int,
        dropout: float | None = None,
        token_mixer: str = "star",
        dim: str = "2d",
        norm_type: str = "dyt",
        eps: float | int = 1e-8,
    ):
        super().__init__()
        assert dim in ["1d", "2d"], ValueError("dim must be '1d' or '2d'")
        if token_mixer == "poolformer":
            self.toke_mixer = PoolMixer(
                in_channel=in_channel,
                kernel_size=7,
                dim=dim,
            )
        elif token_mixer == "star":
            self.toke_mixer = StarFusion(
                in_channel=in_channel,
                kernel_size=7,
                dropout=dropout,
                dim=dim,
                eps=eps,
            )
        else:
            raise ValueError(
                f"token_mixer must be one of ['poolformer', 'star'], got {token_mixer}"
            )
        self.ffn = FFN(
            in_channel=in_channel,
            in_dim=embedding_dim,
            norm=norm_type,
            dim=dim,
            eps=eps,
        )
        self.toke_mixer_name = token_mixer
        self.dropout = DropPath(dropout) if dropout is not None else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
        if x2 is None:
            x = x1 + self.toke_mixer(x1)
        else:
            assert (
                x1.shape == x2.shape
            ), "[Error] x1 and x2 must have the same shape for token mixing"
            x = x1 + self.toke_mixer(x1, x2)
        x_ffn = self.ffn(x)
        return x + self.dropout(x_ffn)


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
        assert in_dim / (2**scale_count) > 1, ValueError(
            "in_dim is too small for the given scale_count"
        )
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
                    nn.Flatten(start_dim=2),  # (B, C, H, W) -> (B, C, H*W)
                    (
                        nn.Linear(self.in_dims[i] * self.in_dims[i], in_dim * in_dim)
                        if self.in_dims[i] != in_dim
                        else nn.Identity()
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
            x_scales.append(stem(x_cut))
        # sum multi-scale features
        x = torch.stack(x_scales, dim=0).mean(dim=0)
        return x


class SelfReconstructionHead(nn.Module):

    def __init__(
        self,
        in_channel: int,
        in_dim: int,
        out_channel: int,
        out_dim: int,
        up_sample_times: int,
        sr_norm: bool = True,
    ):
        super().__init__()
        self.up_sample = nn.ModuleList(
            [
                UpSampleBlk(
                    in_channel=in_channel // (2**i),
                    dim="2d",
                )
                for i in range(up_sample_times)
            ]
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel // (2**up_sample_times),
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid() if sr_norm else nn.Identity(),
        )
        self.in_channel = in_channel
        self.in_dim = in_dim
        self.out_channel = out_channel
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.in_channel, self.in_dim, self.in_dim)
        for up in self.up_sample:
            x = up(x)
        x = self.out_head(x)
        return x


class GaussianMixtureDensityModule(nn.Module):

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        out_dim: int,
        dropout: float = None,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_channel * 2),
            nn.Linear(in_channel * 2, in_channel),
            nn.GELU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(in_channel, in_channel // 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(in_channel // 2, out_channel * out_dim),
        )
        self.out_channel = out_channel
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        x = torch.cat([avg_x, max_x], dim=1).squeeze(-1)
        x = self.head(x)
        return x.reshape(-1, self.out_channel, self.out_dim)


class PhotometryDataExtractor(nn.Module):

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
        self.linear_projection_stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 4),
            nn.BatchNorm1d(in_channel),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(in_channel),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.photometry_data_representation_modules = nn.ModuleList(
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
        x = self.linear_projection_stem(x)
        mag_features = []
        for layer in self.photometry_data_representation_modules:
            x = layer(x)
            mag_features.append(x)
        return mag_features


class PhotometricDiffCrossAttn(nn.Module):

    def __init__(
        self,
        photometric_channel: int,
        mag_channel: int,
        hidden_feature: int,
        num_heads: int,
        dropout: float = None,
        norm_type: str = "dyt",
        eps: float | int = 1e-8,
    ):
        super().__init__()
        self.mags_attn = MultiHeadSelfAttention1D(
            q_channel=mag_channel,
            kv_channel=photometric_channel,
            embedding_dim=hidden_feature,
            num_heads=1,
            norm=norm_type,
            dropout=dropout,
            eps=eps,
        )
        self.token_mixer_1 = TokenMixBlk(
            in_channel=mag_channel,
            embedding_dim=hidden_feature,
            dropout=dropout,
            token_mixer="poolformer",
            dim="1d",
            norm_type=norm_type,
            eps=eps,
        )
        self.photo_attn = MultiHeadSelfAttention1D(
            q_channel=photometric_channel,
            kv_channel=mag_channel,
            embedding_dim=hidden_feature,
            num_heads=num_heads,
            norm=norm_type,
            dropout=dropout,
            eps=eps,
        )
        self.token_mixer_2 = TokenMixBlk(
            in_channel=photometric_channel,
            embedding_dim=hidden_feature,
            dropout=dropout,
            token_mixer="poolformer",
            dim="1d",
            norm_type=norm_type,
            eps=eps,
        )
        self.inter_attn = MultiHeadSelfAttention1D(
            q_channel=photometric_channel,
            kv_channel=mag_channel,
            embedding_dim=hidden_feature * 2,
            num_heads=num_heads,
            norm=norm_type,
            dropout=dropout,
            eps=eps,
        )
        self.ffn = nn.Sequential(
            LayerNormChannel(photometric_channel, dim="1d", eps=eps),
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

    def forward(
        self,
        photometric: torch.Tensor,
        mags: torch.Tensor,
    ) -> torch.Tensor:
        mag_x = self.dropout(self.mags_attn(mags, photometric, photometric)) + mags
        mag_x = self.token_mixer_1(mag_x) + mag_x
        photometric_x = (
            self.dropout(self.photo_attn(photometric, mags, mags)) + photometric
        )
        photometric_x = self.token_mixer_2(photometric_x) + photometric_x
        x = torch.cat([photometric_x, mag_x], dim=-1)
        x = self.dropout(self.inter_attn(x, x, x)) + x
        return self.dropout(self.ffn(x)) + photometric + mags


class ImagePhotometryMixFusion(nn.Module):

    def __init__(
        self,
        in_channel: int,
        output_channel: int,
        embedding_dim: int,
        blks: int,
        token_mixer: str = "poolformer",
        dropout: float = None,
        down_sample: bool = True,
        eps: float | int = 1e-8,
    ):
        super().__init__()

        self.blk = nn.ModuleList(
            [
                TokenMixBlk(
                    in_channel=in_channel,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    token_mixer=token_mixer,
                    dim="1d",
                    norm_type="dyt",
                    eps=eps,
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
        spectrum_extractor_config: dict,
        dropout: float = None,
        enable_spectrum: bool = True,
        eps: float | int = 1e-8,
    ):
        super().__init__()
        self.hidden_channels = [16, 32, 64, 128]
        self.contrastive_feature_ratio = [0.2, 0.4, 0.6, 0.8]
        self.hidden_features = [
            photo_in_dim**2 // (2**i) for i in range(len(self.hidden_channels))
        ]
        self.contrastive_feature_dim = [
            int(self.hidden_features[i] // 2 * self.contrastive_feature_ratio[i])
            for i in range(len(self.hidden_features))
        ]
        self.photometric_stem = CenterAwareStem(
            in_channel=photo_in_channel,
            in_dim=photo_in_dim,
            out_channel=self.hidden_channels[0],
            kernel_size_list=[7, 3, 3, 1],
        )
        self.photometric_extractor = nn.ModuleList(
            [
                ImagePhotometryMixFusion(
                    in_channel=(
                        self.hidden_channels[0]
                        if i == 0
                        else self.hidden_channels[i - 1]
                    ),
                    output_channel=self.hidden_channels[i],
                    embedding_dim=self.hidden_features[i],
                    blks=2,
                    token_mixer="poolformer",
                    dropout=dropout,
                    down_sample=True,
                    eps=eps,
                )
                for i in range(len(self.hidden_channels))
            ]
        )
        self.mag_extractor = PhotometryDataExtractor(
            in_channel=mag_in_channel,
            in_dim=mag_in_dim,
            hidden_dim=self.hidden_features[0],
            hidden_channels=self.hidden_channels,
            dropout=dropout,
        )
        self.feature_mixers = nn.ModuleList(
            [
                PhotometricDiffCrossAttn(
                    photometric_channel=self.hidden_channels[i],
                    mag_channel=self.hidden_channels[i],
                    hidden_feature=self.hidden_features[i] // 2,
                    num_heads=8,
                    dropout=dropout,
                    norm_type="dyt",
                    eps=eps,
                )
                for i in range(len(self.hidden_channels))
            ]
        )
        # use the 2nd scale features for reconstruction
        self.sr_head = SelfReconstructionHead(
            in_channel=self.hidden_channels[1],
            in_dim=int(self.hidden_features[2] ** 0.5),
            out_channel=photo_in_channel,
            out_dim=photo_in_dim,
            up_sample_times=1,
            sr_norm=True,
        )
        self.sr_diff_representation_channel = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hidden_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=self.hidden_channels[0],
                out_channels=self.hidden_channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )
        self.sr_diff_representation_max = nn.MaxPool1d(kernel_size=4, stride=4)
        self.sr_diff_representation_mean = nn.AvgPool1d(kernel_size=4, stride=4)
        self.sr_diff_representation_spatial = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=self.hidden_channels[1],
                kernel_size=7,
                padding="same",
            ),
            nn.Sigmoid(),
        )

        self.reg_head = GaussianMixtureDensityModule(
            in_channel=self.hidden_channels[-1],
            out_dim=3,
            out_channel=out_channel,
            dropout=dropout,
        )
        self.enable_spectrum = enable_spectrum
        if enable_spectrum:
            self.feature_reconstructor = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.hidden_channels[i],
                            out_channels=spectrum_extractor_config["spectrum_channel"],
                            kernel_size=7,
                            padding="same",
                        ),
                        nn.LayerNorm(self.contrastive_feature_dim[i]),
                        nn.GELU(),
                        nn.Linear(
                            in_features=self.contrastive_feature_dim[i],
                            out_features=spectrum_extractor_config["spectrum_size"],
                        ),
                    )
                    for i in range(len(self.hidden_features))
                ]
            )
        self.eps = eps

    def forward(
        self,
        photometric: torch.Tensor,
        magnitudes: torch.Tensor,
        output_spectrum: bool = False,
    ) -> tuple:
        x = self.photometric_stem(photometric)
        b, c, _ = x.shape
        mag_features = self.mag_extractor(magnitudes)
        sr_features = None
        sr_channel_weight = None
        sr_spatial_weight = None
        deep_features = []
        photometric_features = []
        for i in range(len(self.photometric_extractor)):
            x = self.photometric_extractor[i](x)
            if i == 1:
                sr_features = self.sr_head(x)
                sr_diff = torch.mean(
                    (sr_features - photometric).abs(), dim=1, keepdim=True
                )
                sr_diff = sr_diff.reshape(photometric.shape[0], 1, -1)
                sr_channel_weight = self.sr_diff_representation_channel(sr_diff)
                sr_spatial_mean = self.sr_diff_representation_mean(sr_diff)
                sr_spatial_max = self.sr_diff_representation_max(sr_diff)
                sr_spatial_weight = self.sr_diff_representation_spatial(
                    torch.cat([sr_spatial_mean, sr_spatial_max], dim=1)
                )
            x = self.feature_mixers[i](x, mag_features[i])
            _, _, d = x.shape
            deep_features.append(x[:, :, 0 : self.contrastive_feature_dim[i]])
            photometric_features.append(x[:, :, self.contrastive_feature_dim[i] :])
            if i == 1:
                x = x * sr_channel_weight * sr_spatial_weight

        pred = self.reg_head(x)

        if self.enable_spectrum and output_spectrum:
            spectrum_features = []
            for i, reconstructor in enumerate(self.feature_reconstructor):
                spec_feature = self.feature_reconstructor[i](deep_features[i])
                spectrum_features.append(spec_feature)
        else:
            spectrum_features = None
        return (
            pred,
            spectrum_features,
            sr_features,
            deep_features,
            photometric_features,
        )
