import torch
import torch.nn as nn
from timm.layers import DropPath


class DyT(nn.Module):
    """
    Dynamic Tanh activation function. From https://jiachenzhu.github.io/DyT/
    """

    def __init__(self, in_shape: list[int], alpha=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, *in_shape))
        self.bias = nn.Parameter(torch.zeros(1, *in_shape))
        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(self.alpha * x)
        return self.weight * x + self.bias


class LayerNormChannel(nn.Module):

    def __init__(
        self,
        num_channels: int,
        dim: str = "2d",
        eps: float | int = 1e-8,
    ):
        super().__init__()
        assert dim in ["1d", "2d"], ValueError("dim must be '1d' or '2d'")
        self.weight = nn.Parameter(torch.ones(1, num_channels))
        self.bias = nn.Parameter(torch.zeros(1, num_channels))
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.dim == "1d":
            x = self.weight.unsqueeze(-1) * x + self.bias.unsqueeze(-1)
        else:
            x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(
                -1
            ).unsqueeze(-1)
        return x


class FFN(nn.Module):

    def __init__(
        self,
        in_channel: int,
        in_dim: int,
        expansion_ratio: int = 4,
        norm: str = "dyt",
        dim: str = "2d",
        eps: float | int = 1e-8,
    ):
        super().__init__()
        norm = norm.lower()
        assert dim in ["1d", "2d"], ValueError("dim must be '1d' or '2d'")
        preset_conv = nn.Conv2d if dim == "2d" else nn.Conv1d
        self.ln = (
            DyT(
                [int(in_channel), int(in_dim), int(in_dim)]
                if dim == "2d"
                else [int(in_channel), int(in_dim)]
            )
            if norm == "dyt"
            else LayerNormChannel(in_channel, dim=dim, eps=eps)
        )
        self.ffn = nn.Sequential(
            preset_conv(
                in_channel,
                in_channel * expansion_ratio,
                1,
                stride=1,
                padding="same",
            ),
            nn.GELU(),
            preset_conv(
                in_channel * expansion_ratio,
                in_channel,
                1,
                stride=1,
                padding="same",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ln(x))


class StarFusion(nn.Module):

    def __init__(
        self,
        in_channel: int,
        kernel_size: int,
        dropout: float | None = None,
        mlp_ratio: int = 4,
        act_in_x1: bool = False,
        dim: str = "2d",
        eps: float | int = 1e-8,
    ):
        super().__init__()
        assert dim in ["1d", "2d"], ValueError("dim must be '1d' or '2d'")
        preset_conv = nn.Conv2d if dim == "2d" else nn.Conv1d
        self.branch_1 = nn.Sequential(
            preset_conv(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                groups=in_channel,
            ),
            preset_conv(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            LayerNormChannel(
                num_channels=in_channel,
                dim=dim,
                eps=eps,
            ),
            preset_conv(
                in_channels=in_channel,
                out_channels=in_channel * mlp_ratio,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
        )
        self.branch_2 = nn.Sequential(
            preset_conv(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                groups=in_channel,
            ),
            preset_conv(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            LayerNormChannel(
                num_channels=in_channel,
                dim=dim,
                eps=eps,
            ),
            preset_conv(
                in_channels=in_channel,
                out_channels=in_channel * mlp_ratio,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
        )
        self.act = nn.GELU()
        self.act_in_x1 = act_in_x1
        self.fusion = nn.Sequential(
            preset_conv(
                in_channels=in_channel * mlp_ratio,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            LayerNormChannel(
                num_channels=in_channel,
                dim=dim,
                eps=eps,
            ),
            preset_conv(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                groups=in_channel,
            ),
            preset_conv(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
        )
        self.dropout = DropPath(dropout) if dropout is not None else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
        x1_input = x1
        x2_input = x2
        x1 = self.branch_1(x1)
        x2 = self.branch_2(x2) if x2 is not None else self.branch_2(x1_input)
        if self.act_in_x1:
            x = self.act(x1) * x2
        else:
            x = x1 * self.act(x2)
        x = self.fusion(x)
        x = x + self.dropout(x1_input)
        if x2_input is not None:
            x = x + self.dropout(x2_input)
        return x


class PoolMixer(nn.Module):

    def __init__(
        self,
        in_channel: int,
        kernel_size: int = 3,
        enable_fusion: bool = False,
        dim: str = "2d",
    ):
        super().__init__()
        assert dim in ["1d", "2d"], ValueError("dim must be '1d' or '2d'")
        preset_conv = nn.Conv2d if dim == "2d" else nn.Conv1d
        preset_avg_pool = nn.AvgPool2d if dim == "2d" else nn.AvgPool1d
        self.pool = preset_avg_pool(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        self.conv = (
            preset_conv(
                in_channels=in_channel * 2,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding="same",
            )
            if enable_fusion
            else nn.Identity()
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
        if x2 is not None:
            x = torch.cat([x1, x2], dim=1)
            x = self.conv(self.pool(x))
        else:
            x = self.pool(x1)
        return x


class MultiHeadSelfAttention1D(nn.Module):

    def __init__(
        self,
        q_channel: int,
        kv_channel: int,
        embedding_dim: int,
        num_heads: int,
        norm: str = "dyt",
        dropout: float | None = None,
        eps: float | int = 1e-8,
    ):
        super().__init__()
        norm = norm.lower()
        assert norm in ["dyt", "ln"], ValueError("norm must be 'dyt' or 'ln'")
        assert q_channel % num_heads == 0, ValueError(
            "channel must be divisible by num_heads"
        )
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.ln_q = (
            DyT([q_channel, embedding_dim])
            if norm == "dyt"
            else LayerNormChannel(
                q_channel,
                dim="1d",
                eps=eps,
            )
        )
        self.ln_k = (
            DyT([kv_channel, embedding_dim])
            if norm == "dyt"
            else LayerNormChannel(
                kv_channel,
                dim="1d",
                eps=eps,
            )
        )
        self.ln_v = (
            DyT([kv_channel, embedding_dim])
            if norm == "dyt"
            else LayerNormChannel(
                kv_channel,
                dim="1d",
                eps=eps,
            )
        )
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.fc_q = nn.Conv1d(q_channel, q_channel, kernel_size=1, stride=1, padding=0)
        self.fc_k = nn.Conv1d(
            kv_channel,
            q_channel if q_channel != kv_channel else kv_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.fc_v = nn.Conv1d(
            kv_channel,
            q_channel if q_channel != kv_channel else kv_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.out_projection = nn.Conv1d(
            q_channel, q_channel, kernel_size=1, stride=1, padding=0
        )
        self.head_dim = q_channel // num_heads
        self.embedding_dim = embedding_dim

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        b, q_c, _ = q.shape
        q, k, v = self.ln_q(q), self.ln_k(k), self.ln_v(v)
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        q = q.reshape(b, self.num_heads, self.head_dim, self.embedding_dim).permute(
            0, 1, 3, 2
        )
        k = k.reshape(b, self.num_heads, self.head_dim, self.embedding_dim).permute(
            0, 1, 3, 2
        )
        v = v.reshape(b, self.num_heads, self.head_dim, self.embedding_dim).permute(
            0, 1, 3, 2
        )
        attn = (q @ k.transpose(-1, -2)) / (self.head_dim**0.5)
        attn = (
            (self.dropout(self.softmax(attn)) @ v)
            .permute(0, 1, 3, 2)
            .reshape(b, q_c, self.embedding_dim)
        )
        return self.out_projection(attn)


class UpSampleBlk(nn.Module):

    def __init__(
        self,
        in_channel: int,
        dim: str = "2d",
    ):
        super().__init__()
        trans_conv = nn.ConvTranspose2d if dim == "2d" else nn.ConvTranspose1d
        conv = nn.Conv2d if dim == "2d" else nn.Conv1d
        norm = nn.BatchNorm2d if dim == "2d" else nn.BatchNorm1d
        self.blk = nn.Sequential(
            trans_conv(
                in_channels=in_channel,
                out_channels=in_channel // 2,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            norm(in_channel // 2),
            conv(
                in_channels=in_channel // 2,
                out_channels=in_channel,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.GELU(),
            conv(
                in_channels=in_channel,
                out_channels=in_channel // 2,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blk(x)
