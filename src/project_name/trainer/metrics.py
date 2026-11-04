"""Accumulate scalar metrics across steps and ranks without per-step GPU syncs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from project_name.utils.dist import all_reduce


class MetricAccumulator:
    """Weighted running means. Values stay on device until `compute()`."""

    def __init__(self, device: torch.device):
        self.device = device
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}

    def update(self, metrics: Mapping[str, torch.Tensor | float], weight: float = 1.0) -> None:
        for k, v in metrics.items():
            v = v.detach().float() if isinstance(v, torch.Tensor) else torch.tensor(float(v))
            v = v.to(self.device)
            if v.numel() != 1:
                raise ValueError(f"Metric '{k}' must be a scalar, got shape {tuple(v.shape)}")
            if k not in self.sums:
                self.sums[k] = torch.zeros((), device=self.device)
                self.counts[k] = torch.zeros((), device=self.device)
            self.sums[k] += v.reshape(()) * weight
            self.counts[k] += weight

    def compute(self, sync: bool = True) -> dict[str, float]:
        """Mean of each metric, reduced over all ranks (every rank must call this)."""
        if not self.sums:
            return {}
        keys = sorted(self.sums)
        packed = torch.stack([self.sums[k] for k in keys] + [self.counts[k] for k in keys])
        if sync:
            all_reduce(packed, "sum")
        n = len(keys)
        means = (packed[:n] / packed[n:].clamp_min(1e-12)).tolist()  # one device->host sync
        return dict(zip(keys, means, strict=True))

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()
