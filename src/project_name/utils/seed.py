"""Seeding and RNG state capture for exact resumption."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def configure_determinism(deterministic: bool, tf32: bool = True) -> None:
    """`deterministic=True` trades speed for bitwise reproducibility on the same hardware."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker distinct numpy/python seeds.

    PyTorch seeds torch per worker, but numpy/random would otherwise be identical
    copies in every worker (a classic duplicated-augmentation bug).
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
