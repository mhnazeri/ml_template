"""Thin helpers around torch.distributed that also work in a single process.

Launch multi-GPU runs with `torchrun`; it sets RANK / WORLD_SIZE / LOCAL_RANK which
`init_distributed` picks up. Without those env vars everything degrades to a
single process and all collectives become no-ops.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    return get_rank() == 0


def init_distributed(timeout_minutes: int = 30) -> torch.device:
    """Initialise the process group (if launched by torchrun) and return this rank's device."""
    launched_by_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", get_local_rank())
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # also with a single process under torchrun, so strategy=ddp/fsdp can be debugged on 1 GPU
    if launched_by_torchrun and not is_dist():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
            device_id=device if use_cuda else None,
        )
    return device


def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """In-place all-reduce; returns the tensor for convenience."""
    if is_dist():
        ops = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}
        dist.all_reduce(tensor, op=ops[op])
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_dist():
        return obj
    container = [obj]
    dist.broadcast_object_list(container, src=src)
    return container[0]


@contextmanager
def main_process_first():
    """Let rank 0 run a block first (e.g. dataset download / caching), then the others."""
    if not is_main():
        barrier()
    yield
    if is_main():
        barrier()
