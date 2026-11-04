"""Optimizer and learning-rate schedule construction."""

from __future__ import annotations

import math

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from project_name.utils.misc import locate

_NO_DECAY_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.Embedding,
)


def default_param_groups(model: nn.Module, weight_decay: float, split: bool = True) -> list[dict]:
    """Two groups: decayed weights, and non-decayed biases / norm / embedding params."""
    if not split:
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]
    no_decay_ids = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if name.endswith("bias") or isinstance(module, _NO_DECAY_TYPES) or p.ndim < 2:
                no_decay_ids.add(id(p))
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if id(p) in no_decay_ids else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(param_groups: list[dict], cfg: DictConfig, device: torch.device):
    extra = OmegaConf.to_container(cfg.kwargs, resolve=True) if cfg.kwargs else {}
    name = cfg.name.lower()
    if name in {"adamw", "adam"}:
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        kwargs = dict(lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
        if device.type == "cuda":
            kwargs["fused"] = True
    elif name == "sgd":
        cls = torch.optim.SGD
        kwargs = dict(lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        cls = locate(cfg.name)
        kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    kwargs.update(extra)
    return cls(param_groups, **kwargs)


def lr_multiplier(step: int, total: int, cfg: DictConfig) -> float:
    """LR multiplier in [min_lr_ratio, 1] for optimizer step `step` (0-based)."""
    warmup, floor = cfg.warmup_steps, cfg.min_lr_ratio
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if cfg.name == "constant":
        return 1.0
    if cfg.name == "wsd":
        decay_steps = max(1, int(total * cfg.decay_ratio))
        decay_start = total - decay_steps
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / decay_steps)
        return floor + (1 - floor) * (1 - progress)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    if cfg.name == "linear":
        return floor + (1 - floor) * (1 - progress)
    # cosine
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, cfg: DictConfig
) -> LambdaLR:
    return LambdaLR(optimizer, lambda step: lr_multiplier(step, total_steps, cfg))
