"""DataLoader construction for training and (fixed-sample) validation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset

from project_name.data.samplers import (
    ResumableDistributedSampler,
    fixed_subset_indices,
    shard_evenly,
)
from project_name.utils.dist import get_rank, get_world_size
from project_name.utils.log import get_logger
from project_name.utils.seed import worker_init_fn

log = get_logger(__name__)


def _loader_kwargs(cfg: DictConfig, num_workers: int) -> dict:
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=cfg.dataloader.pin_memory and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        # A private generator: creating an iterator draws the workers' base seed from it.
        # With the default (global RNG) every new iterator would shift the global stream,
        # so a resumed run would see different dropout/augmentation than an uninterrupted one.
        generator=torch.Generator(),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = cfg.dataloader.persistent_workers
        kwargs["prefetch_factor"] = cfg.dataloader.prefetch_factor
    return kwargs


def build_train_loader(
    dataset: Dataset, cfg: DictConfig, collate_fn: Callable | None = None
) -> tuple[DataLoader, ResumableDistributedSampler]:
    sampler = ResumableDistributedSampler(
        len(dataset),  # type: ignore[arg-type]
        shuffle=cfg.dataloader.shuffle,
        seed=cfg.run.seed,
        rank=get_rank(),
        world_size=get_world_size(),
        drop_last=cfg.dataloader.drop_last,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        sampler=sampler,
        drop_last=cfg.dataloader.drop_last,
        collate_fn=collate_fn,
        **_loader_kwargs(cfg, cfg.dataloader.num_workers),
    )
    return loader, sampler


def seed_loader_for_epoch(loader: DataLoader, seed: int, epoch: int, rank: int) -> None:
    """Make worker seeds a pure function of (seed, epoch, rank)."""
    loader.generator.manual_seed(seed * 1_000_003 + epoch * 1_009 + rank)


class ValidationSet:
    """The fixed validation subset, plus its quick-validation prefix and visualization samples.

    The subset depends only on `val.max_samples` and `val.subset_seed`, never on the
    training seed, so every run (and every resume) is evaluated on identical samples.
    """

    def __init__(self, dataset: Dataset, cfg: DictConfig, collate_fn: Callable | None = None):
        rank, world = get_rank(), get_world_size()
        self.indices = fixed_subset_indices(len(dataset), cfg.val.max_samples, cfg.val.subset_seed)  # type: ignore[arg-type]
        if len(self.indices) % world:
            log.warning(
                f"Validation subset of {len(self.indices)} is not divisible by world size {world}; "
                f"dropping {len(self.indices) % world} samples so all ranks stay in lock-step."
            )
        bs = cfg.dataloader.val_batch_size or cfg.dataloader.batch_size
        # validation workers are not persistent: they'd otherwise idle in memory between evals
        kwargs = dict(batch_size=bs, shuffle=False, collate_fn=collate_fn, drop_last=False)
        kwargs.update(_loader_kwargs(cfg, cfg.dataloader.num_workers))
        if "persistent_workers" in kwargs:
            kwargs["persistent_workers"] = False

        self.full = DataLoader(Subset(dataset, shard_evenly(self.indices, rank, world)), **kwargs)
        quick = self.indices[: cfg.val.quick_samples]
        self.quick = DataLoader(Subset(dataset, shard_evenly(quick, rank, world)), **kwargs)
        for loader in (self.full, self.quick):
            seed_loader_for_epoch(loader, cfg.val.subset_seed, 0, rank)

        n_vis = min(cfg.val.num_vis_samples, len(self.indices))
        self.vis_batch = None
        if n_vis > 0:
            items = [dataset[i] for i in self.indices[:n_vis]]
            self.vis_batch = (collate_fn or torch.utils.data.default_collate)(items)

    def __len__(self) -> int:
        return len(self.indices)
