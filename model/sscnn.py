"""
title: Classification of large-scale stellar spectra based on deep convolutional neural network
doi: 10.1093/mnras/sty3020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSCNNEncoder(nn.Module):
    """
    Spectral encoder (SSCNN conv backbone + linear projection).
    flat_dim = 32 * (spectrum_size // 16), e.g. 32 * 476 = 15232 for 7626-point spectra.
    """

    def __init__(
        self,
        in_channel: int,
        spectrum_size: int,
        embedding_dim: int,
        pretrained_ckpt_path: str = "",
    ):
        super().__init__()
        adaptive_seq_len = 8
        self.conv_structure = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=16, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=16, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(adaptive_seq_len),
        )
        self.flat_dim = 32 * adaptive_seq_len
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.flat_dim, embedding_dim),
        )
        self.z_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if pretrained_ckpt_path:
            print(
                f"[INFO] SSCNNEncoder: Loading pre-trained weights from {pretrained_ckpt_path}"
            )
            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            # Filter 'model.' prefix since pre-training lightning module saves it as self.model
            sscnn_state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
            if not sscnn_state_dict:
                sscnn_state_dict = state_dict  # Fallback
            self.load_state_dict(sscnn_state_dict)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (spec_emb, z_pred):
          spec_emb -- L2-normalised embedding for cosine alignment, shape (B, D)
          z_pred   -- redshift prediction from z_head, shape (B, 1)
        """
        x = self.conv_structure(x)
        x = x.view(x.size(0), -1)
        pre_norm = self.fc(x)
        spec_emb = F.normalize(pre_norm, p=2, dim=-1)
        z_pred = self.z_head(pre_norm)
        return spec_emb, z_pred
