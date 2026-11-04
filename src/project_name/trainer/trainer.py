"""The training engine: step-based loop, mixed precision, gradient accumulation,
DDP/FSDP2, EMA, fixed-sample validation, checkpoint/resume, preemption handling.

Problem-specific code lives in a `Task` (see tasks/base.py); this file shouldn't need
editing for a new project.
"""

from __future__ import annotations

import json
import math
import signal
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from project_name.config import save_config, to_dict
from project_name.data.loaders import ValidationSet, build_train_loader, seed_loader_for_epoch
from project_name.tasks.base import Task
from project_name.tracking import NullTracker, Tracker, build_tracker
from project_name.trainer.checkpoint import (
    CheckpointManager,
    export_safetensors,
    load_model_weights,
)
from project_name.trainer.ema import EMA
from project_name.trainer.metrics import MetricAccumulator
from project_name.trainer.optim import build_optimizer, build_scheduler, default_param_groups
from project_name.utils.dist import (
    all_reduce,
    broadcast_object,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main,
    main_process_first,
)
from project_name.utils.env import save_reproducibility_info
from project_name.utils.log import get_logger, setup_logging
from project_name.utils.misc import locate, move_to_device
from project_name.utils.nn import count_parameters, unwrap_model
from project_name.utils.seed import configure_determinism, seed_everything

log = get_logger(__name__)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class Trainer:
    def __init__(self, cfg: DictConfig, run_dir: str | Path | None = None, read_only: bool = False):
        """Common setup for training and evaluation.

        `run_dir` defaults to <run.root>/<run.name>. `read_only=True` (evaluation) never
        creates or modifies run artifacts except the eval log/results.
        """
        self.cfg = cfg
        self.device = init_distributed()
        self.rank, self.world_size = get_rank(), get_world_size()
        self.run_dir = Path(run_dir) if run_dir else self._resolve_run_dir()
        setup_logging(self.run_dir / ("eval.log" if read_only else "train.log"))
        configure_determinism(cfg.run.deterministic, cfg.run.tf32)
        torch.autograd.set_detect_anomaly(cfg.trainer.detect_anomaly)

        self.precision = cfg.trainer.precision
        if self.device.type == "cpu" and self.precision == "fp16":
            log.warning("fp16 is not supported on CPU; using fp32")
            self.precision = "fp32"
        self.strategy = self._resolve_strategy()
        log.info(
            f"Run dir: {self.run_dir} | device: {self.device} | world size: {self.world_size} | "
            f"strategy: {self.strategy} | precision: {self.precision}"
        )

        # Same seed on every rank while building the model -> identical init everywhere.
        seed_everything(cfg.run.seed)
        self.task: Task = locate(cfg.task._target_)(cfg)
        with main_process_first():
            self.train_dataset, self.val_dataset = self.task.build_datasets()
        model = self.task.build_model().to(self.device)
        self.param_counts = count_parameters(model)
        log.info(f"Model: {type(model).__name__} | {self.param_counts}")
        self.model = self._wrap_model(model)
        # From here on each rank gets its own stream (dropout, augmentations).
        seed_everything(cfg.run.seed + self.rank)

        collate = self.task.collate_fn()
        self.collate_fn = collate
        self.val_set = (
            ValidationSet(self.val_dataset, cfg, collate) if self.val_dataset is not None else None
        )
        self.ckpt = CheckpointManager(self.run_dir / "checkpoints", cfg.checkpoint.keep_last)
        self.tracker: Tracker = NullTracker()
        self.ema: EMA | None = None

        # training state (saved in checkpoints)
        self.step = 0  # optimizer steps completed
        self.epoch = 0
        self.batches_in_epoch = 0  # micro-batches consumed in the current epoch (this rank)
        self.best_value: float | None = None
        self.evals_without_improvement = 0
        self.bad_steps = 0
        self._stop_requested = False

    # ======================================================================== setup helpers
    def _resolve_run_dir(self) -> Path:
        name = self.cfg.run.name
        if not name:
            name = broadcast_object(f"{self.cfg.run.prefix}-{datetime.now():%Y%m%d-%H%M%S}")
        return Path(self.cfg.run.root) / name

    def _resolve_strategy(self) -> str:
        strategy = self.cfg.trainer.strategy
        if strategy == "auto":
            return "ddp" if self.world_size > 1 else "single"
        if strategy == "single" and self.world_size > 1:
            raise ValueError("trainer.strategy=single but launched with multiple processes")
        return strategy

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        if self.strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=self.cfg.trainer.ddp_find_unused_parameters,
            )
        elif self.strategy == "fsdp":
            from torch.distributed.fsdp import fully_shard

            reshard = self.cfg.trainer.fsdp_reshard_after_forward
            for sub in self.task.fsdp_modules(model):
                fully_shard(sub, reshard_after_forward=reshard)
            fully_shard(model, reshard_after_forward=reshard)
        if self.cfg.trainer.compile:
            model = torch.compile(model)
        return model

    @property
    def unwrapped_model(self) -> nn.Module:
        return unwrap_model(self.model)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        ocfg = self.cfg.optim
        groups = self.task.param_groups(self.unwrapped_model) or default_param_groups(
            self.unwrapped_model, ocfg.weight_decay, ocfg.no_decay_on_bias_and_norm
        )
        for g in groups:
            if "lr_scale" in g:
                g["lr"] = ocfg.lr * g.pop("lr_scale")
        return build_optimizer(groups, ocfg, self.device)

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(self.device.type, dtype=_DTYPES[self.precision])

    @contextmanager
    def _grad_sync(self, sync: bool):
        """Skip gradient all-reduce on all but the last micro-batch of an accumulation."""
        if sync or self.strategy == "single":
            yield
        elif self.strategy == "ddp":
            with self.unwrapped_ddp().no_sync():
                yield
        else:  # fsdp
            self.unwrapped_model.set_requires_gradient_sync(False)
            try:
                yield
            finally:
                self.unwrapped_model.set_requires_gradient_sync(True)

    def unwrapped_ddp(self) -> nn.Module:
        m = self.model
        return m._orig_mod if hasattr(m, "_orig_mod") else m

    # ======================================================================== checkpoint I/O
    def _trainer_state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "batches_in_epoch": self.batches_in_epoch,
            "best_value": self.best_value,
            "evals_without_improvement": self.evals_without_improvement,
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "ema_num_updates": self.ema.num_updates if self.ema else None,
            "experiment_key": self.tracker.experiment_key,
            "world_size": self.world_size,
            "batch_size": self.cfg.dataloader.batch_size,
        }

    def save_checkpoint(self) -> Path:
        return self.ckpt.save(
            self.step,
            self.model,
            self.optimizer,
            self._trainer_state(),
            ema_state=self.ema.state_dict() if self.ema else None,
            meta={
                "epoch": self.epoch,
                "best_value": self.best_value,
                "has_ema": self.ema is not None,
            },
        )

    def _find_resume_checkpoint(self) -> Path | None:
        resume = self.cfg.checkpoint.resume
        if resume == "never":
            if self.ckpt.latest() is not None:
                raise FileExistsError(
                    f"{self.run_dir} already has checkpoints but checkpoint.resume=never. "
                    "Pick a new run.name or use checkpoint.resume=auto."
                )
            return None
        if resume == "auto":
            return self.ckpt.latest()
        path = Path(resume)
        if (path / "checkpoints").is_dir():  # a run directory was given
            path = CheckpointManager(path / "checkpoints").latest() or path
        if not (path / "trainer_state.pt").exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")
        return path

    def _restore(self, path: Path) -> str | None:
        state = self.ckpt.load(
            path, self.model, self.optimizer, self.ema.state_dict() if self.ema else None
        )
        self.step = state["step"]
        self.epoch = state["epoch"]
        self.batches_in_epoch = state["batches_in_epoch"]
        self.best_value = state["best_value"]
        self.evals_without_improvement = state["evals_without_improvement"]
        self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(state["scaler"])
        if self.ema and state.get("ema_num_updates") is not None:
            self.ema.num_updates = state["ema_num_updates"]
        if (
            state.get("world_size") != self.world_size
            or state.get("batch_size") != self.cfg.dataloader.batch_size
        ):
            log.warning(
                "World size or batch size changed since the checkpoint; the data order "
                "will not exactly match an uninterrupted run (restarting the current epoch)."
            )
            self.batches_in_epoch = 0
        log.info(f"Resumed from {path} at step {self.step} (epoch {self.epoch})")
        return state.get("experiment_key")

    # ======================================================================== training
    def _setup_training(self) -> None:
        cfg = self.cfg
        if self.train_dataset is None:
            raise ValueError("Task.build_datasets() returned no training dataset")
        self.train_loader, self.train_sampler = build_train_loader(
            self.train_dataset, cfg, self.collate_fn
        )
        if len(self.train_loader) == 0:
            raise ValueError(
                "Training loader is empty: dataset smaller than batch_size * world_size?"
            )
        steps_per_epoch = len(self.train_loader) // cfg.trainer.grad_accum_steps
        candidates = [cfg.trainer.max_steps] if cfg.trainer.max_steps else []
        if cfg.trainer.max_epochs:
            candidates.append(cfg.trainer.max_epochs * steps_per_epoch)
        self.total_steps = min(candidates)

        self.optimizer = self._build_optimizer()
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, cfg.scheduler)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.precision == "fp16")
        if cfg.trainer.ema_decay:
            self.ema = EMA(self.unwrapped_model, cfg.trainer.ema_decay)

        resume_path = self._find_resume_checkpoint()
        experiment_key = None
        if resume_path is not None:
            experiment_key = self._restore(resume_path)
        elif cfg.checkpoint.init_from:
            log.info(f"Initialising weights from {cfg.checkpoint.init_from}")
            load_model_weights(
                cfg.checkpoint.init_from, self.model, strict=cfg.checkpoint.init_strict
            )
            if self.ema:  # start the average from the loaded weights
                self.ema = EMA(self.unwrapped_model, cfg.trainer.ema_decay)

        if is_main():
            self.run_dir.mkdir(parents=True, exist_ok=True)
            cfg_name = (
                "config.yaml" if resume_path is None else f"config.resume-{self.step:08d}.yaml"
            )
            save_config(cfg, self.run_dir / cfg_name)
            env = save_reproducibility_info(self.run_dir)
            if env["git"]["dirty"]:
                log.warning("Git tree is dirty; the diff is saved to git.diff in the run dir")

        self.tracker = build_tracker(cfg, self.run_dir, experiment_key)
        global_bs = cfg.dataloader.batch_size * self.world_size * cfg.trainer.grad_accum_steps
        self.tracker.log_params(
            {
                **_flatten(to_dict(cfg)),
                **self.param_counts,
                "run_dir": str(self.run_dir),
                "world_size": self.world_size,
                "global_batch_size": global_bs,
                "train_samples": len(self.train_dataset),
                "val_samples": len(self.val_set) if self.val_set else 0,
                "total_steps": self.total_steps,
            }
        )
        if cfg.logger.histograms:
            self.tracker.watch(self.unwrapped_model)
        log.info(
            f"Training for {self.total_steps} steps ({steps_per_epoch} steps/epoch, "
            f"global batch {global_bs}); {len(self.train_dataset)} train / "
            f"{len(self.val_set) if self.val_set else 0} val samples"
        )

    def fit(self) -> dict[str, float]:
        """Train until `total_steps` (or early stopping / preemption). Returns last val metrics."""
        self._setup_training()
        cfg = self.cfg
        self._train_iter = None
        self.train_metrics = MetricAccumulator(self.device)
        last_val: dict[str, float] = {}

        if cfg.val.at_start and self.step == 0 and self.val_set:
            self.validate(quick=True)

        self._reset_perf_counters()
        with self._preemption_handler():
            while self.step < self.total_steps:
                self._train_step()
                stop = self._stop_requested_everywhere()
                is_last = self.step >= self.total_steps

                if self.step % cfg.trainer.log_every == 0 or is_last:
                    self._log_train_metrics()
                run_full_val = bool(self.val_set) and (
                    (cfg.val.every and self.step % cfg.val.every == 0) or is_last
                )
                run_quick_val = (
                    bool(self.val_set)
                    and not run_full_val  # the full pass already covers the quick subset
                    and cfg.val.quick_every
                    and self.step % cfg.val.quick_every == 0
                )
                if run_quick_val:
                    self.validate(quick=True)

                improved, early_stop = False, False
                if run_full_val:
                    last_val = self.validate()
                    improved, early_stop = self._update_best(last_val)

                want_ckpt = (
                    (cfg.checkpoint.every and self.step % cfg.checkpoint.every == 0)
                    or is_last
                    or stop
                    or early_stop
                )
                if want_ckpt or (improved and cfg.checkpoint.save_best):
                    path = self.save_checkpoint()
                    if improved and cfg.checkpoint.save_best:
                        self.ckpt.mark_best(path, cfg.val.monitor, self.best_value, self.step)
                        if cfg.checkpoint.export_safetensors:
                            self._export("best")

                if stop:
                    log.warning(
                        f"Stop requested (preemption signal); checkpoint saved at step {self.step}"
                    )
                    break
                if early_stop:
                    log.info(f"Early stopping at step {self.step}")
                    break

        if self.step >= self.total_steps and cfg.checkpoint.export_safetensors:
            self._export("final")
        log.info(f"Training finished at step {self.step}")
        return last_val

    def _next_batch(self) -> Any:
        for _ in range(2):  # at most one epoch rollover per call
            if self._train_iter is None:
                start = self.batches_in_epoch * self.cfg.dataloader.batch_size
                self.train_sampler.set_epoch(self.epoch, start_index=start)
                seed_loader_for_epoch(self.train_loader, self.cfg.run.seed, self.epoch, self.rank)
                self._train_iter = iter(self.train_loader)
            try:
                batch = next(self._train_iter)
                self.batches_in_epoch += 1
                return batch
            except StopIteration:
                self._train_iter = None
                self.epoch += 1
                self.batches_in_epoch = 0
        raise RuntimeError("Training loader produced no batches for a whole epoch")

    def _train_step(self) -> None:
        cfg = self.cfg.trainer
        self.model.train()
        accum = cfg.grad_accum_steps
        for micro in range(accum):
            t_data = time.perf_counter()
            batch = move_to_device(self._next_batch(), self.device)
            self._perf["data_time"] += time.perf_counter() - t_data
            self._perf["samples"] += self.task.batch_size(batch)
            with self._grad_sync(micro == accum - 1):
                with self._autocast():
                    out = self.task.training_step(self.model, batch)
                loss = out["loss"]
                self.scaler.scale(loss / accum).backward()
            self.train_metrics.update(out)

        self.scaler.unscale_(self.optimizer)
        max_norm = cfg.grad_clip if cfg.grad_clip > 0 else math.inf
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        if hasattr(grad_norm, "full_tensor"):  # FSDP2 returns a DTensor
            grad_norm = grad_norm.full_tensor()

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)  # GradScaler skips non-finite steps itself
            self.scaler.update()
        elif torch.isfinite(grad_norm):  # grads are synced, so every rank agrees
            self.optimizer.step()
            self.bad_steps = 0
        else:
            self.bad_steps += 1
            log.warning(
                f"Non-finite gradient at step {self.step}; skipping update ({self.bad_steps} in a row)"
            )
            if self.bad_steps >= cfg.max_bad_steps:
                raise RuntimeError(f"{self.bad_steps} consecutive non-finite steps; aborting")
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        if self.ema and self.step >= cfg.ema_start_step:
            self.ema.update(self.unwrapped_model)
        self.train_metrics.update({"grad_norm": grad_norm})
        self.step += 1

    # ======================================================================== logging
    def _reset_perf_counters(self) -> None:
        self._perf = {"t0": time.perf_counter(), "step0": self.step, "data_time": 0.0, "samples": 0}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _log_train_metrics(self) -> None:
        metrics = {f"train/{k}": v for k, v in self.train_metrics.compute().items()}
        self.train_metrics.reset()
        elapsed = time.perf_counter() - self._perf["t0"]
        steps = max(1, self.step - self._perf["step0"])
        metrics.update(
            {
                "train/lr": self.optimizer.param_groups[0]["lr"],
                "train/epoch": self.epoch,
                "perf/step_time_s": elapsed / steps,
                "perf/samples_per_s": self._perf["samples"] * self.world_size / elapsed,
                "perf/data_wait_frac": self._perf["data_time"] / elapsed,
            }
        )
        if self.device.type == "cuda":
            metrics["perf/gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
        self.tracker.log_metrics(metrics, step=self.step, epoch=self.epoch)
        log.info(
            f"step {self.step}/{self.total_steps} | loss {metrics.get('train/loss', float('nan')):.4f} | "
            f"lr {metrics['train/lr']:.2e} | gnorm {metrics.get('train/grad_norm', float('nan')):.3f} | "
            f"{metrics['perf/samples_per_s']:.0f} samples/s"
        )
        self._reset_perf_counters()

    # ======================================================================== validation
    @torch.no_grad()
    def validate(self, quick: bool = False, use_ema: bool | None = None) -> dict[str, float]:
        """Evaluate on the fixed validation subset (or its quick prefix). Collective."""
        if self.val_set is None:
            return {}
        prefix = "val_quick" if quick else "val"
        loader = self.val_set.quick if quick else self.val_set.full
        use_ema = self.cfg.val.use_ema if use_ema is None else use_ema
        swap = (
            self.ema.swapped_in(self.unwrapped_model) if (self.ema and use_ema) else nullcontext()
        )
        acc = MetricAccumulator(self.device)
        t0 = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        visuals: dict[str, Any] = {}
        with swap:
            for batch in loader:
                batch = move_to_device(batch, self.device)
                with self._autocast():
                    out = self.task.validation_step(self.model, batch)
                acc.update(out, weight=self.task.batch_size(batch))
            if self.val_set.vis_batch is not None:
                with self._autocast():
                    visuals = self.task.visualize(
                        self.model, move_to_device(self.val_set.vis_batch, self.device)
                    )
        self.model.train(was_training)

        metrics = {f"{prefix}/{k}": v for k, v in acc.compute().items()}
        metrics[f"{prefix}/time_s"] = time.perf_counter() - t0
        self.tracker.log_metrics(metrics, step=self.step, epoch=self.epoch)
        for name, value in visuals.items():
            if isinstance(value, str):
                self.tracker.log_text(f"{prefix}/{name}", value, step=self.step)
            else:
                self.tracker.log_image(f"{prefix}/{name}", value, step=self.step)
        shown = " | ".join(f"{k} {v:.4f}" for k, v in metrics.items() if not k.endswith("time_s"))
        log.info(f"[{prefix}] step {self.step} | {shown}")
        return metrics

    def _update_best(self, metrics: dict[str, float]) -> tuple[bool, bool]:
        """Returns (improved, should_early_stop)."""
        vcfg = self.cfg.val
        if vcfg.monitor not in metrics:
            log.warning(
                f"val.monitor={vcfg.monitor!r} not in metrics {sorted(metrics)}; best tracking disabled"
            )
            return False, False
        value = metrics[vcfg.monitor]
        if self.best_value is None:
            improved = True
        elif vcfg.mode == "min":
            improved = value < self.best_value - vcfg.early_stopping_min_delta
        else:
            improved = value > self.best_value + vcfg.early_stopping_min_delta
        if improved:
            self.best_value = value
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
        self.tracker.log_metrics({f"{vcfg.monitor}_best": self.best_value}, step=self.step)
        patience = vcfg.early_stopping_patience
        return improved, bool(patience and self.evals_without_improvement >= patience)

    def _export(self, tag: str) -> None:
        """Export deployable weights (EMA if enabled) to <run_dir>/export/<tag>.safetensors."""
        swap = self.ema.swapped_in(self.unwrapped_model) if self.ema else nullcontext()
        with swap:
            path = export_safetensors(self.model, self.run_dir / "export" / f"{tag}.safetensors")
        if path is not None and self.cfg.logger.upload_checkpoints:
            self.tracker.log_asset(path, name=f"{tag}.safetensors")

    # ======================================================================== evaluation
    def evaluate(self, checkpoint: str | Path, use_ema: bool | None = None) -> dict[str, float]:
        """Load weights from a checkpoint dir / .safetensors and run full validation."""
        checkpoint = Path(checkpoint)
        meta_file = checkpoint / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        if use_ema is None:
            use_ema = self.cfg.val.use_ema and meta.get("has_ema", False)
        load_model_weights(checkpoint, self.model, strict=True, use_ema=bool(use_ema))
        log.info(f"Evaluating {checkpoint} (ema={bool(use_ema)})")
        self.step = meta.get("step", 0)
        return self.validate(use_ema=False)

    # ======================================================================== preemption
    @contextmanager
    def _preemption_handler(self):
        """SIGTERM/SIGUSR1 (sent by SLURM/torchrun before killing a job) -> save and exit."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return

        def handler(signum, frame):
            log.warning(f"Received signal {signal.Signals(signum).name}; will checkpoint and stop")
            self._stop_requested = True

        previous = {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGUSR1)}
        try:
            yield
        finally:
            for s, h in previous.items():
                signal.signal(s, h)

    def _stop_requested_everywhere(self) -> bool:
        if self.world_size == 1:
            return self._stop_requested
        flag = torch.tensor(float(self._stop_requested), device=self.device)
        return bool(all_reduce(flag, "max").item())

    def close(self) -> None:
        self.tracker.end()
        cleanup_distributed()


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
