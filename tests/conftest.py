import os

# Tests run on CPU so they are deterministic and work anywhere (set before torch import).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from project_name.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = ROOT / "configs" / "debug.yaml"

TINY = [
    "data.train.num_samples=256",
    "data.val.num_samples=64",
    "model.dim=32",
    "model.channels=[8,8,8]",
    "dataloader.batch_size=16",
    "dataloader.num_workers=0",
    "trainer.max_steps=12",
    "trainer.log_every=4",
    "trainer.precision=fp32",
    "scheduler.warmup_steps=2",
    "val.every=6",
    "val.quick_every=3",
    "val.quick_samples=16",
    "val.num_vis_samples=2",
    "checkpoint.every=6",
    "logger.backend=none",
]


@pytest.fixture
def make_cfg(tmp_path):
    def _make(*overrides: str, name: str = "test"):
        return load_config(
            DEBUG_CONFIG, [*TINY, f"run.root={tmp_path}", f"run.name={name}", *overrides]
        )

    return _make
