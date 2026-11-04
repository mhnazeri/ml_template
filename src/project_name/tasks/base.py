"""The `Task` is the one class you write per project.

It owns everything problem-specific (model, datasets, loss, metrics, visualizations);
the `Trainer` owns everything engineering-specific (devices, precision, distributed,
checkpointing, logging, schedules). Swap tasks via the `task._target_` config key.

Minimal subclass:

    class MyTask(Task):
        def training_step(self, model, batch):
            pred = model(batch["image"])
            loss = F.mse_loss(pred, batch["target"])
            return {"loss": loss}

`build_model` / `build_datasets` instantiate `cfg.model` / `cfg.data.train|val` via their
`_target_` keys, so for many projects they don't need overriding at all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from project_name.utils.misc import infer_batch_size, instantiate


class Task:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ construction
    def build_model(self) -> nn.Module:
        return instantiate(self.cfg.model)

    def build_datasets(self) -> tuple[Dataset, Dataset | None]:
        """Return (train, val). Called on every rank; guard downloads with `main_process_first`."""
        data = self.cfg.data
        train = instantiate(data.train)
        val = instantiate(data.val) if data.get("val") else None
        return train, val

    def collate_fn(self) -> Callable | None:
        """Custom collate (e.g. padding variable-length sequences). None = default_collate."""
        return None

    def param_groups(self, model: nn.Module) -> list[dict] | None:
        """Custom optimizer param groups (e.g. lower LR for a pretrained encoder).

        None = default groups (weight decay on weights, not on biases/norms/embeddings).
        Group dicts may set "lr_scale" to scale the group's LR relative to optim.lr.
        """
        return None

    def fsdp_modules(self, model: nn.Module) -> list[nn.Module]:
        """Sub-modules to shard as separate FSDP units (e.g. transformer blocks).

        Sharding per block lets FSDP free each block's full weights right after use.
        Empty list = only the root module is sharded (fine for small models).
        """
        return []

    # ------------------------------------------------------------------ steps
    def training_step(self, model: nn.Module, batch: Any) -> dict[str, torch.Tensor]:
        """Return a dict with a scalar "loss" (backpropagated) plus any other scalar metrics."""
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(self, model: nn.Module, batch: Any) -> dict[str, torch.Tensor]:
        """Return scalar metrics averaged over the batch. Defaults to the training metrics.

        Every rank must return the same keys.
        """
        return self.training_step(model, batch)

    def batch_size(self, batch: Any) -> int:
        """Used to weight per-batch metric means into exact dataset means."""
        return infer_batch_size(batch)

    @torch.no_grad()
    def visualize(self, model: nn.Module, batch: Any) -> dict[str, Any]:
        """Qualitative outputs on the fixed visualization batch.

        Return {name: image | str}. Images can be CHW/HWC tensors/arrays in [0, 1] or
        uint8; strings are logged as text. Runs on all ranks (FSDP needs every rank to
        join the forward pass); only rank 0's result is logged.
        """
        return {}
