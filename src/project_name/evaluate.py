"""Evaluate a checkpoint on the (fixed) validation set using the run's saved config.

python -m project_name.evaluate --run runs/my-run                 # best (or latest) checkpoint
python -m project_name.evaluate --run runs/my-run --ckpt step_00002000
python -m project_name.evaluate --run runs/my-run --ckpt export/final.safetensors val.max_samples=null
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_name.config import load_config
from project_name.trainer import Trainer
from project_name.trainer.checkpoint import CheckpointManager
from project_name.utils.dist import is_main


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="run directory (contains config.yaml)")
    parser.add_argument(
        "--ckpt", default=None, help="checkpoint dir / .safetensors, relative to --run or absolute"
    )
    parser.add_argument(
        "--no-ema", action="store_true", help="evaluate raw weights even if EMA exists"
    )
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. data.val.split=test")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    overrides = ["logger.backend=none", *args.overrides]
    cfg = load_config(run_dir / "config.yaml", overrides)

    manager = CheckpointManager(run_dir / "checkpoints")
    if args.ckpt:
        ckpt = Path(args.ckpt)
        ckpt = ckpt if ckpt.is_absolute() or ckpt.exists() else run_dir / ckpt
        if not ckpt.exists() and (run_dir / "checkpoints" / args.ckpt).exists():
            ckpt = run_dir / "checkpoints" / args.ckpt
    else:
        best = manager.best()
        ckpt = manager.dir / best["name"] if best else manager.latest()
    if ckpt is None or not Path(ckpt).exists():
        raise FileNotFoundError(
            f"No checkpoint found (looked for {args.ckpt or 'best/latest'} in {run_dir})"
        )

    trainer = Trainer(cfg, run_dir=run_dir, read_only=True)
    try:
        metrics = trainer.evaluate(ckpt, use_ema=False if args.no_ema else None)
    finally:
        trainer.close()
    if is_main():
        out = run_dir / "eval" / f"{Path(ckpt).name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"checkpoint": str(ckpt), "metrics": metrics}, indent=2))
        print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
