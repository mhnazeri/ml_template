"""Example task: behaviour cloning of action chunks from image + pose + language."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from project_name.tasks.base import Task


class BehaviorCloningTask(Task):
    def _predict(self, model: nn.Module, batch: dict) -> torch.Tensor:
        # always call model(...) (not model.submodule(...)) so DDP/FSDP hooks run
        return model(batch["image"], batch["pose"], batch["instruction_ids"])

    def _metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = pred.float()
        final_err = (pred[:, -1] - target[:, -1]).norm(dim=-1)
        return {
            "loss": F.mse_loss(pred, target),
            "final_err": final_err.mean(),
            "success": (final_err < self.cfg.task.get("success_radius", 0.1)).float().mean(),
        }

    def training_step(self, model: nn.Module, batch: dict) -> dict[str, torch.Tensor]:
        return self._metrics(self._predict(model, batch), batch["actions"])

    @torch.no_grad()
    def visualize(self, model: nn.Module, batch: dict) -> dict[str, Any]:
        """Draw ground-truth (green) and predicted (red) waypoints on each image."""
        pred = self._predict(model, batch).float().cpu().numpy()
        gt = batch["actions"].float().cpu().numpy()
        images = batch["image"].float().cpu().numpy()
        tiles = [_draw(img, g, p) for img, g, p in zip(images, gt, pred, strict=True)]
        return {
            "trajectories": np.concatenate(tiles, axis=1),  # HWC, side by side
            "instructions": "\n".join(
                f"{i}: {t}" for i, t in zip(batch["id"], batch["instruction"], strict=True)
            ),
        }


def _draw(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    img = np.transpose(image, (1, 2, 0)).copy()
    size = img.shape[0]
    for points, color in ((gt, (0.0, 1.0, 0.0)), (pred, (1.0, 0.0, 0.0))):
        for x, y in points[:, :2]:
            px = int(round((np.clip(x, -1, 1) + 1) / 2 * (size - 1)))
            py = int(round((np.clip(y, -1, 1) + 1) / 2 * (size - 1)))
            img[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2] = color
    return img
