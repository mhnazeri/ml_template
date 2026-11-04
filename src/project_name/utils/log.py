"""Rank-aware logging. Use `get_logger(__name__)` everywhere instead of print."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from project_name.utils.dist import get_rank

ROOT = "project_name"
_FMT = "%(asctime)s | %(levelname)s | %(message)s"


def get_logger(name: str = ROOT) -> logging.Logger:
    if not name.startswith(ROOT):
        name = f"{ROOT}.{name}"
    return logging.getLogger(name)


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    """Rank 0 logs at `level` to console (+file); other ranks only log warnings."""
    rank = get_rank()
    logger = logging.getLogger(ROOT)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level if rank == 0 else logging.WARNING)

    # wide console when piped to a file (e.g. SLURM logs) so lines don't wrap at 80 chars
    width = None if sys.stderr.isatty() else 160
    console = RichHandler(
        console=Console(stderr=True, width=width),
        show_path=False,
        rich_tracebacks=True,
        log_time_format="[%X]",
    )
    if rank != 0:
        console.setFormatter(logging.Formatter(f"[rank{rank}] %(message)s"))
    logger.addHandler(console)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file if rank == 0 else f"{log_file}.rank{rank}")
        fh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(fh)
