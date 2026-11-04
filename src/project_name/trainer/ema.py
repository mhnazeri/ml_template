"""Exponential moving average of model parameters (works with DDP and FSDP2 DTensors)."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.num_updates = 0
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        params = dict(model.named_parameters())
        self.shadow = {n: params[n].detach().clone() for n in self.names}

    def _params(self, model: nn.Module) -> list[torch.Tensor]:
        params = dict(model.named_parameters())
        return [params[n] for n in self.names]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # warm-up: early averages shouldn't be dominated by the random init
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for s, p in zip(self.shadow.values(), self._params(model), strict=True):
            s.lerp_(p.detach(), 1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.values(), self._params(model), strict=True):
            p.copy_(s)

    @contextmanager
    def swapped_in(self, model: nn.Module):
        """Temporarily load EMA weights into the model (e.g. for validation)."""
        params = self._params(model)
        backup = [p.detach().clone() for p in params]
        self.copy_to(model)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(params, backup, strict=True):
                    p.copy_(b)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow
