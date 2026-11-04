"""Real multi-process runs (2 CPU processes, gloo) for DDP and FSDP2."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

pytestmark = pytest.mark.slow
WORKER = Path(__file__).with_name("dist_worker.py")


def torchrun(root, name, *overrides, stop_at=0, nproc=2):
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}",
        str(WORKER), str(root), name, str(stop_at), *overrides,
    ]  # fmt: skip
    env = {**os.environ, "PYTHONPATH": str(WORKER.parent), "OMP_NUM_THREADS": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]
    return Path(root) / name


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_resume_is_exact_under_multiprocess(tmp_path, strategy):
    common = [f"trainer.strategy={strategy}", "trainer.ema_decay=0.9", "trainer.max_steps=10"]
    straight = torchrun(tmp_path, "straight", *common)
    torchrun(tmp_path, "resumed", *common, stop_at=4)
    resumed = torchrun(tmp_path, "resumed", *common)

    a = load_file(straight / "export/final.safetensors")
    b = load_file(resumed / "export/final.safetensors")
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert (straight / "checkpoints/best.json").exists()
    assert "Stop requested" in (resumed / "train.log").read_text()
    # each rank wrote its RNG state
    assert len(list((resumed / "checkpoints").glob("step_*/rng_rank*.pt"))) >= 2


def test_fsdp_checkpoint_loads_in_single_process(tmp_path):
    """Sharded FSDP checkpoints can be evaluated without distributed (DCP reshards on load)."""
    run = torchrun(tmp_path, "fsdp", "trainer.strategy=fsdp", "trainer.max_steps=6")
    from conftest import DEBUG_CONFIG, TINY

    from project_name.config import load_config
    from project_name.trainer import Trainer

    cfg = load_config(DEBUG_CONFIG, [*TINY, f"run.root={tmp_path}", "run.name=fsdp"])
    trainer = Trainer(cfg, run_dir=run, read_only=True)
    metrics = trainer.evaluate(run / "checkpoints/step_00000006")
    trainer.close()
    assert metrics["val/loss"] > 0
