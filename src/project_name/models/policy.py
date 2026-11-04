"""Example multimodal policy: image + pose + instruction -> action chunk.

Small on purpose; replace with your architecture. Pattern worth keeping: one encoder
per modality -> fuse -> head, with sizes coming from the `model:` config section.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class ImageEncoder(nn.Module):
    def __init__(self, image_size: int, channels: tuple[int, ...], dim: int):
        super().__init__()
        layers, c = [], 3
        for c_out in channels:
            layers.append(conv_block(c, c_out))
            c = c_out
        self.convs = nn.Sequential(*layers)
        spatial = image_size // 2 ** len(channels)
        # flatten (not avg-pool): the goal's *position* in the image matters
        self.proj = nn.Linear(c * spatial * spatial, dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.convs(image).flatten(1))


class TextEncoder(nn.Module):
    """Mean-pooled byte embeddings; swap for a pretrained language model if needed."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).unsqueeze(-1).float()
        return (self.embed(ids) * mask).sum(1) / mask.sum(1).clamp_min(1.0)


class MultimodalPolicy(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        channels: tuple[int, ...] = (32, 64, 64),
        pose_dim: int = 7,
        action_dim: int = 3,
        horizon: int = 8,
        dim: int = 256,
        vocab_size: int = 257,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.image_enc = ImageEncoder(image_size, tuple(channels), dim)
        self.pose_enc = nn.Sequential(nn.Linear(pose_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.text_enc = TextEncoder(vocab_size, dim)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, horizon * action_dim),
        )

    def forward(
        self, image: torch.Tensor, pose: torch.Tensor, text_ids: torch.Tensor
    ) -> torch.Tensor:
        z = torch.cat([self.image_enc(image), self.pose_enc(pose), self.text_enc(text_ids)], dim=-1)
        return self.head(z).view(-1, self.horizon, self.action_dim)
