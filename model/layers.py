import torch
import torch.nn as nn
import torch.nn.functional as f
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


class RMSNorm1d(nn.Module):

    def __init__(self, channel, eps=1e-8):
        super().__init__()
        self.scale = channel**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, channel, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g


class FFN(nn.Module):

    def __init__(
        self,
        in_channel: int,
        in_dim: int,
        expansion_ratio: int = 4,
        dim: str = "2d",
    ):
        super().__init__()
        assert dim in ["1d", "2d"], ValueError("dim must be '1d' or '2d'")
        preset_conv = nn.Conv2d if dim == "2d" else nn.Conv1d
        self.ln = DyT(
            [int(in_channel), int(in_dim), int(in_dim)]
            if dim == "2d"
            else [int(in_channel), int(in_dim)]
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


class PoolMixer(nn.Module):

    def __init__(
        self,
        in_channel: int,
        embedding_dim: int,
        kernel_size: int = 7,
        dropout: float | None = None,
        dim: str = "2d",
    ):
        super().__init__()
        assert dim in ["1d", "2d"], ValueError("dim must be '1d' or '2d'")
        preset_avg_pool = nn.AvgPool2d if dim == "2d" else nn.AvgPool1d
        self.pool = preset_avg_pool(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        self.ffn = FFN(
            in_channel=in_channel,
            in_dim=embedding_dim,
            dim=dim,
        )
        self.dropout = DropPath(dropout) if dropout is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.pool(x)) + x
        x = self.dropout(self.ffn(x)) + x
        return x


class MHSA1D(nn.Module):

    def __init__(
        self,
        q_channel: int,
        kv_channel: int,
        embedding_dim: int,
        num_heads: int,
        dropout: float | None = None,
    ):
        super().__init__()
        assert q_channel % num_heads == 0, ValueError(
            "channel must be divisible by num_heads"
        )
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.ln_q = DyT([q_channel, embedding_dim])
        self.ln_k = DyT([kv_channel, embedding_dim])
        self.ln_v = DyT([kv_channel, embedding_dim])
        self.fc_q = nn.Conv1d(
            q_channel,
            q_channel if q_channel != kv_channel else kv_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
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
        self.dropout_p = dropout if dropout is not None else 0.0

        self.head_dim = q_channel // num_heads
        self.embedding_dim = embedding_dim
        self.q_channel = q_channel
        self.kv_channel = kv_channel
        self.gate_fc = nn.Sequential(
            nn.Conv1d(
                q_channel, q_channel, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        b, q_c, _ = q.shape
        q, k, v = self.ln_q(q), self.ln_k(k), self.ln_v(v)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q = (
            q.reshape(b, self.num_heads, self.head_dim, self.embedding_dim)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        k = (
            k.reshape(b, self.num_heads, self.head_dim, self.embedding_dim)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        v = (
            v.reshape(b, self.num_heads, self.head_dim, self.embedding_dim)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        v = f.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        v = v.permute(0, 1, 3, 2).reshape(b, q_c, self.embedding_dim)
        v = self.gate_fc(v) * v
        return v
