"""Experiment tracking. Only rank 0 gets a real tracker; other ranks get `NullTracker`.

The `Tracker` interface is small on purpose so another backend (W&B, TensorBoard,
MLflow) can be added by implementing these methods.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from project_name.utils.dist import is_main
from project_name.utils.log import get_logger

log = get_logger(__name__)


def to_image_array(image: Any) -> np.ndarray:
    """Convert CHW/HWC float tensors or arrays (in [0,1]) or uint8 arrays to HWC uint8."""
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


class Tracker:
    experiment_key: str | None = None
    url: str | None = None

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(
        self, metrics: dict[str, float], step: int, epoch: int | None = None
    ) -> None: ...

    def log_image(self, name: str, image: Any, step: int) -> None: ...

    def log_text(self, name: str, text: str, step: int) -> None: ...

    def log_asset(self, path: str | Path, name: str | None = None) -> None: ...

    def watch(self, model: torch.nn.Module) -> None: ...

    def end(self) -> None: ...


class NullTracker(Tracker):
    pass


class JsonlTracker(Tracker):
    """Always-on local log: one JSON line per log call in <run_dir>/metrics.jsonl."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1)

    def log_metrics(self, metrics: dict[str, float], step: int, epoch: int | None = None) -> None:
        self._fh.write(
            json.dumps({"step": step, "epoch": epoch, "time": time.time(), **metrics}) + "\n"
        )

    def log_text(self, name: str, text: str, step: int) -> None:
        self._fh.write(json.dumps({"step": step, "text": {name: text}}) + "\n")

    def end(self) -> None:
        self._fh.close()


class MultiTracker(Tracker):
    def __init__(self, trackers: list[Tracker]):
        self.trackers = trackers

    @property
    def experiment_key(self) -> str | None:  # type: ignore[override]
        return next((t.experiment_key for t in self.trackers if t.experiment_key), None)

    @property
    def url(self) -> str | None:  # type: ignore[override]
        return next((t.url for t in self.trackers if t.url), None)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> None:
        for t in self.trackers:
            try:
                getattr(t, method)(*args, **kwargs)
            except Exception as e:  # a tracking hiccup must never kill a training run
                log.warning(f"{type(t).__name__}.{method} failed: {e}")

    def log_params(self, params):
        self._call("log_params", params)

    def log_metrics(self, metrics, step, epoch=None):
        self._call("log_metrics", metrics, step, epoch)

    def log_image(self, name, image, step):
        self._call("log_image", name, image, step)

    def log_text(self, name, text, step):
        self._call("log_text", name, text, step)

    def log_asset(self, path, name=None):
        self._call("log_asset", path, name)

    def watch(self, model):
        self._call("watch", model)

    def end(self):
        self._call("end")


def build_tracker(cfg: DictConfig, run_dir: Path, experiment_key: str | None = None) -> Tracker:
    if not is_main():
        return NullTracker()
    trackers: list[Tracker] = [JsonlTracker(run_dir / "metrics.jsonl")]
    if cfg.logger.backend == "comet":
        from project_name.tracking.comet import CometTracker

        trackers.append(CometTracker(cfg, run_dir, experiment_key))
    return MultiTracker(trackers)
