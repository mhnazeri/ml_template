"""Launched by test_distributed.py via torchrun (CPU / gloo)."""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from conftest import DEBUG_CONFIG, TINY  # noqa: E402

from project_name.config import load_config  # noqa: E402
from project_name.trainer import Trainer  # noqa: E402

if __name__ == "__main__":
    root, name, stop_at, *overrides = sys.argv[1:]
    cfg = load_config(DEBUG_CONFIG, [*TINY, f"run.root={root}", f"run.name={name}", *overrides])
    trainer = Trainer(cfg)
    if int(stop_at) > 0:
        original = trainer._train_step

        def step_then_maybe_stop():
            original()
            if trainer.step == int(stop_at) and trainer.rank == 0:  # only rank 0 gets the signal
                trainer._stop_requested = True

        trainer._train_step = step_then_maybe_stop
    trainer.fit()
    trainer.close()
