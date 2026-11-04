"""nn.Module helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP / torch.compile wrappers (FSDP2 modifies modules in place, nothing to strip)."""
    while True:
        if hasattr(model, "module") and isinstance(
            model, nn.parallel.DistributedDataParallel | nn.DataParallel
        ):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params/total": total, "params/trainable": trainable}


def freeze(model: nn.Module, exclude: Iterable[str] = ()) -> nn.Module:
    """Freeze every parameter whose name doesn't contain one of `exclude`."""
    exclude = tuple(exclude)
    for name, p in model.named_parameters():
        p.requires_grad_(any(e in name for e in exclude))
    return model


_WEIGHT_LAYERS = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_NORM_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)


def init_weights(
    method: str = "kaiming_normal",
    nonlinearity: str = "relu",
    mode: str = "fan_in",
    std: float = 0.02,
    gain: float = 1.0,
) -> Callable[[nn.Module], None]:
    """Return a function for `model.apply(...)` that initialises Linear/Conv/Norm layers.

    method: kaiming_normal | kaiming_uniform | xavier_normal | xavier_uniform | normal | orthogonal
    """
    weight_inits = {
        "kaiming_normal": lambda w: nn.init.kaiming_normal_(
            w, mode=mode, nonlinearity=nonlinearity
        ),
        "kaiming_uniform": lambda w: nn.init.kaiming_uniform_(
            w, mode=mode, nonlinearity=nonlinearity
        ),
        "xavier_normal": lambda w: nn.init.xavier_normal_(w, gain=gain),
        "xavier_uniform": lambda w: nn.init.xavier_uniform_(w, gain=gain),
        "normal": lambda w: nn.init.normal_(w, std=std),
        "orthogonal": lambda w: nn.init.orthogonal_(w, gain=gain),
    }
    if method not in weight_inits:
        raise ValueError(f"Unknown init method '{method}', choose from {list(weight_inits)}")

    @torch.no_grad()
    def _init(m: nn.Module) -> None:
        if isinstance(m, _WEIGHT_LAYERS):
            weight_inits[method](m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, _NORM_LAYERS) and m.weight is not None:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    return _init
