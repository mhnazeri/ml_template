"""Checkpointing built on torch.distributed.checkpoint (DCP).

One code path for single-GPU, DDP and FSDP2: model/optimizer/EMA state is saved with
DCP (sharded when using FSDP, so no rank ever materialises the full model), and the
small trainer state (step, scheduler, sampler position, RNG, ...) with torch.save.

Layout of <run_dir>/checkpoints:
    step_00001000/        a complete checkpoint (DCP shards + trainer_state.pt + rng_rank*.pt)
    step_00002000/
    latest                text file naming the newest complete checkpoint
    best.json             which checkpoint is best on `val.monitor` (never pruned)

Writes are atomic: everything goes to `<name>.tmp/` first and is renamed only once all
ranks are done, so a crash mid-save never leaves a corrupt "latest" checkpoint.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)

from project_name.utils.dist import barrier, get_rank, get_world_size, is_main
from project_name.utils.log import get_logger
from project_name.utils.nn import unwrap_model
from project_name.utils.seed import get_rng_state, set_rng_state

log = get_logger(__name__)
# DCP warns on every call when used without a process group; that's intended here.
warnings.filterwarnings("ignore", message="torch.distributed is (disabled|unavailable)")

_STEP_DIR = re.compile(r"^step_(\d+)$")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, ckpt_dir: str | Path, keep_last: int = 3):
        self.dir = Path(ckpt_dir)
        self.keep_last = keep_last

    # ---------------------------------------------------------------- discovery
    def list_checkpoints(self) -> list[Path]:
        if not self.dir.exists():
            return []
        dirs = [p for p in self.dir.iterdir() if p.is_dir() and _STEP_DIR.match(p.name)]
        return sorted(dirs, key=lambda p: int(_STEP_DIR.match(p.name).group(1)))  # type: ignore[union-attr]

    def latest(self) -> Path | None:
        pointer = self.dir / "latest"
        if pointer.exists():
            path = self.dir / pointer.read_text().strip()
            if (path / "trainer_state.pt").exists():
                return path
        ckpts = [p for p in self.list_checkpoints() if (p / "trainer_state.pt").exists()]
        return ckpts[-1] if ckpts else None

    def best(self) -> dict | None:
        path = self.dir / "best.json"
        return json.loads(path.read_text()) if path.exists() else None

    # ---------------------------------------------------------------- save
    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        trainer_state: dict[str, Any],
        ema_state: dict[str, torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        """Collective: every rank must call this."""
        name = f"step_{step:08d}"
        final, tmp = self.dir / name, self.dir / f"{name}.tmp"
        if is_main():
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
        barrier()

        t0 = time.perf_counter()
        model_sd, optim_sd = get_state_dict(model, optimizer)
        state: dict[str, Any] = {"model": model_sd, "optim": optim_sd}
        if ema_state is not None:
            state["ema"] = ema_state
        dcp.save(state, checkpoint_id=tmp)
        torch.save(get_rng_state(), tmp / f"rng_rank{get_rank()}.pt")
        if is_main():
            torch.save(trainer_state, tmp / "trainer_state.pt")
            meta = {
                "step": step,
                "world_size": get_world_size(),
                "time": time.time(),
                **(meta or {}),
            }
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        barrier()
        if is_main():
            shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
            _atomic_write_text(self.dir / "latest", name)
            self._prune()
        barrier()
        log.info(f"Saved checkpoint {final} ({time.perf_counter() - t0:.1f}s)")
        return final

    def mark_best(self, path: Path, metric: str, value: float, step: int) -> None:
        if is_main():
            info = {"name": path.name, "metric": metric, "value": value, "step": step}
            _atomic_write_text(self.dir / "best.json", json.dumps(info, indent=2))

    def _prune(self) -> None:
        if self.keep_last <= 0:
            return
        best = self.best()
        protected = {best["name"]} if best else set()
        ckpts = self.list_checkpoints()
        for path in ckpts[: max(0, len(ckpts) - self.keep_last)]:
            if path.name not in protected:
                shutil.rmtree(path, ignore_errors=True)

    # ---------------------------------------------------------------- load
    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        ema_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Restore model (+optimizer, EMA, this rank's RNG) in place; return the trainer state."""
        path = Path(path)
        if optimizer is not None:
            model_sd, optim_sd = get_state_dict(model, optimizer)
            state: dict[str, Any] = {"model": model_sd, "optim": optim_sd}
        else:
            state = {"model": get_model_state_dict(model)}
        if ema_state is not None:
            state["ema"] = ema_state
        dcp.load(state, checkpoint_id=path)
        if optimizer is not None:
            set_state_dict(
                model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"]
            )
        else:
            set_model_state_dict(model, state["model"])
        if ema_state is not None:
            ema_state.update(state["ema"])  # DCP loads in place; this keeps references valid

        rng_file = path / f"rng_rank{get_rank()}.pt"
        if rng_file.exists():
            set_rng_state(torch.load(rng_file, weights_only=False))
        else:
            log.warning(f"No RNG state for rank {get_rank()} in {path} (world size changed?)")
        trainer_state_file = path / "trainer_state.pt"
        return (
            torch.load(trainer_state_file, weights_only=False)
            if trainer_state_file.exists()
            else {}
        )


def load_model_weights(
    path: str | Path, model: nn.Module, strict: bool = True, use_ema: bool = False
) -> None:
    """Load weights only (for fine-tuning / evaluation) from a checkpoint dir or .safetensors.

    Works for plain, DDP and FSDP2 models. Collective under distributed training.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        full_sd = load_file(path) if is_main() or get_world_size() == 1 else {}
        opts = StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=get_world_size() > 1, strict=strict
        )
        set_model_state_dict(model, full_sd, options=opts)
        return

    sd = get_model_state_dict(model)
    planner = DefaultLoadPlanner(allow_partial_load=not strict)
    dcp.load({"model": sd}, checkpoint_id=path, planner=planner)
    if use_ema:
        # overlay EMA parameters; buffers (e.g. BatchNorm stats) come from the model state
        param_names = {n for n, _ in unwrap_model(model).named_parameters()}
        ema_sd = {k: v for k, v in sd.items() if k in param_names}
        dcp.load(
            {"ema": ema_sd}, checkpoint_id=path, planner=DefaultLoadPlanner(allow_partial_load=True)
        )
        sd.update(ema_sd)
    set_model_state_dict(model, sd, options=StateDictOptions(strict=strict))


def export_safetensors(model: nn.Module, path: str | Path) -> Path | None:
    """Gather the full (unsharded) weights and write them on rank 0. Collective."""
    full_sd = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    path = Path(path)
    if is_main():
        path.parent.mkdir(parents=True, exist_ok=True)
        # clone breaks shared storage (tied weights), which safetensors refuses to save
        tensors = {k: v.detach().clone().contiguous() for k, v in full_sd.items()}
        tmp = path.with_suffix(".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, path)
        log.info(f"Exported weights to {path}")
    barrier()
    return path if is_main() else None
