"""Small generic helpers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def locate(path: str) -> Any:
    """Import `package.module.Name` and return `Name`."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"'{path}' is not a fully qualified name like 'pkg.module.Class'")
    return getattr(importlib.import_module(module_name), attr)


def instantiate(cfg: Mapping | DictConfig, *args: Any, **kwargs: Any) -> Any:
    """Build an object from a config holding a `_target_` key (Hydra-style, non-recursive).

    Remaining keys are passed as keyword arguments; explicit `kwargs` take precedence.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = dict(cfg)
    if "_target_" not in cfg:
        raise KeyError(f"Config needs a '_target_' key to be instantiated, got keys {list(cfg)}")
    target = locate(cfg.pop("_target_"))
    cfg.update(kwargs)
    return target(*args, **cfg)


def move_to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """Recursively move tensors in nested dicts/lists/tuples to `device`; leave the rest."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return type(batch)({k: move_to_device(v, device, non_blocking) for k, v in batch.items()})
    if isinstance(batch, Sequence) and not isinstance(batch, str):
        moved = [move_to_device(v, device, non_blocking) for v in batch]
        return type(batch)(moved) if isinstance(batch, tuple) else moved
    return batch


def infer_batch_size(batch: Any) -> int:
    """Leading dimension of the first tensor found in the batch."""
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]
    if isinstance(batch, Mapping):
        values = batch.values()
    elif isinstance(batch, Sequence) and not isinstance(batch, str):
        values = batch
    else:
        values = []
    for v in values:
        try:
            return infer_batch_size(v)
        except ValueError:
            continue
    raise ValueError("Could not infer batch size: no tensor in batch")
