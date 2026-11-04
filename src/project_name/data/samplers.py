"""Samplers that make training resumable mid-epoch and validation reproducible."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ResumableDistributedSampler(Sampler[int]):
    """Shuffles with (seed, epoch), shards across ranks, and can skip already-seen samples.

    * The permutation depends only on `seed` and `epoch`, so a resumed run sees exactly
      the same order as an uninterrupted one.
    * `start_index` skips the first N indices of this rank's shard for the current epoch;
      the trainer sets it on resume to `consumed_batches * batch_size`.
    * Every rank gets the same number of samples (truncated if `drop_last`, else padded
      by wrapping around), which DDP/FSDP need to stay in lock-step.
    """

    def __init__(
        self,
        length: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ):
        if length < world_size:
            raise ValueError(f"Dataset of {length} samples is smaller than world size {world_size}")
        self.length = length
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0
        self.start_index = 0

    def _shard(self) -> list[int]:
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.length, generator=g).tolist()
        else:
            indices = list(range(self.length))
        if self.drop_last:
            indices = indices[: self.length - self.length % self.world_size]
        else:
            pad = (-self.length) % self.world_size
            indices += indices[:pad]
        return indices[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[int]:
        return iter(self._shard()[self.start_index :])

    def __len__(self) -> int:
        per_rank = (
            self.length // self.world_size if self.drop_last else -(-self.length // self.world_size)
        )
        return max(0, per_rank - self.start_index)

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        self.epoch = epoch
        self.start_index = start_index


def fixed_subset_indices(length: int, max_samples: int | None, seed: int) -> list[int]:
    """A deterministic, run-independent subset of a dataset (sorted for stable order)."""
    if max_samples is None or max_samples >= length:
        return list(range(length))
    g = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(length, generator=g)[:max_samples].tolist())


def shard_evenly(indices: list[int], rank: int, world_size: int) -> list[int]:
    """Strided shard with equal length on every rank (drops at most world_size-1 samples)."""
    usable = len(indices) - len(indices) % world_size
    return indices[:usable][rank::world_size]
