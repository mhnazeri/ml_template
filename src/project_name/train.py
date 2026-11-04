"""Training entry point.

python -m project_name.train --config configs/example.yaml optim.lr=1e-3 run.name=my-run
torchrun --standalone --nproc-per-node=gpu -m project_name.train --config ...
"""

from __future__ import annotations

import argparse

import comet_ml  # noqa: F401  # Comet must be imported before torch to hook into it

from project_name.config import load_config
from project_name.trainer import Trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", "-c", required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. optim.lr=1e-3")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    trainer = Trainer(cfg)
    try:
        return trainer.fit()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
