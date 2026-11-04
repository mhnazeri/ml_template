"""Capture everything needed to reproduce a run: git state, packages, hardware."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def git_info() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit.strip() if commit else None,
        "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty": bool(status and status.strip()),
    }


def env_info() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "argv": sys.argv,
    }
    info["git"] = git_info()
    return info


def save_reproducibility_info(run_dir: Path) -> dict:
    """Write env.json, packages.txt and git.diff (if dirty) into the run directory."""
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    info = env_info()
    (run_dir / "env.json").write_text(json.dumps(info, indent=2))
    pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    (run_dir / "packages.txt").write_text("\n".join(pkgs) + "\n")
    if info["git"]["dirty"]:
        diff = _git("diff", "HEAD")
        if diff:
            (run_dir / "git.diff").write_text(diff)
    return info
