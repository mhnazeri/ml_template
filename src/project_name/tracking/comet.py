"""Comet ML backend.

Credentials come from the usual Comet places: COMET_API_KEY / COMET_WORKSPACE env vars
or ~/.comet.config (run `comet login` once).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import comet_ml
import torch
from omegaconf import DictConfig
from PIL import Image

from project_name.tracking.base import Tracker, to_image_array
from project_name.utils.log import get_logger

log = get_logger(__name__)


class CometTracker(Tracker):
    def __init__(self, cfg: DictConfig, run_dir: Path, experiment_key: str | None = None):
        lcfg = cfg.logger
        exp_config = comet_ml.ExperimentConfig(
            name=run_dir.name,
            tags=list(cfg.run.tags),
            log_code=lcfg.log_code,
            log_git_metadata=True,
            log_git_patch=True,
            log_env_details=lcfg.log_env,
            log_env_gpu=lcfg.log_env,
            log_env_cpu=lcfg.log_env,
            log_env_host=lcfg.log_env,
            auto_param_logging=False,
            auto_metric_logging=False,
            parse_args=False,
            auto_histogram_weight_logging=lcfg.histograms,
            auto_histogram_gradient_logging=lcfg.histograms,
            auto_histogram_activation_logging=False,
            offline_directory=str(run_dir / "comet"),
            display_summary_level=0,
        )
        kwargs = dict(
            project_name=lcfg.project,
            workspace=lcfg.workspace,
            online=lcfg.online,
            experiment_config=exp_config,
        )
        if experiment_key and lcfg.online:
            try:  # continue the same Comet experiment after a resume
                self.exp = comet_ml.start(experiment_key=experiment_key, mode="get", **kwargs)
                log.info(f"Continuing Comet experiment {experiment_key}")
            except Exception as e:
                log.warning(
                    f"Could not continue Comet experiment {experiment_key} ({e}); creating a new one"
                )
                self.exp = comet_ml.start(mode="create", **kwargs)
        else:
            self.exp = comet_ml.start(mode="create", **kwargs)
        self.experiment_key = self.exp.get_key()
        self.url = getattr(self.exp, "url", None)
        if self.url:
            log.info(f"Comet experiment: {self.url}")
        if cfg.run.notes:
            self.exp.log_other("notes", cfg.run.notes)

    def log_params(self, params: dict[str, Any]) -> None:
        self.exp.log_parameters(params)

    def log_metrics(self, metrics: dict[str, float], step: int, epoch: int | None = None) -> None:
        self.exp.log_metrics(metrics, step=step, epoch=epoch)

    def log_image(self, name: str, image: Any, step: int) -> None:
        # pass a PIL image: Comet's own numpy conversion would require matplotlib
        self.exp.log_image(Image.fromarray(to_image_array(image)), name=name, step=step)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.exp.log_text(text, step=step, metadata={"name": name})

    def log_asset(self, path: str | Path, name: str | None = None) -> None:
        self.exp.log_asset(str(path), file_name=name)

    def watch(self, model: torch.nn.Module) -> None:
        from comet_ml.integration.pytorch import watch

        watch(model)

    def end(self) -> None:
        self.exp.end()
