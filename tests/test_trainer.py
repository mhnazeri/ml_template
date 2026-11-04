import json

import pytest
import torch
from safetensors.torch import load_file

from project_name.trainer import Trainer


def _fit(cfg, stop_at: int | None = None) -> Trainer:
    trainer = Trainer(cfg)
    if stop_at is not None:  # simulate a preemption signal arriving at `stop_at`
        original = trainer._train_step

        def step_then_maybe_stop():
            original()
            if trainer.step == stop_at:
                trainer._stop_requested = True

        trainer._train_step = step_then_maybe_stop
    trainer.fit()
    trainer.close()
    return trainer


def _params(trainer):
    return {k: v.detach().clone() for k, v in trainer.unwrapped_model.state_dict().items()}


def test_training_reduces_loss(make_cfg):
    trainer = _fit(make_cfg("trainer.max_steps=30", "val.every=30"))
    lines = [
        json.loads(line) for line in (trainer.run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    losses = [x["train/loss"] for x in lines if "train/loss" in x]
    assert losses[-1] < losses[0]
    for name in (
        "config.yaml",
        "env.json",
        "packages.txt",
        "train.log",
        "export/final.safetensors",
    ):
        assert (trainer.run_dir / name).exists(), name


@pytest.mark.parametrize("ema", [None, 0.9])
def test_resume_after_preemption_is_exact(make_cfg, ema):
    """Stopping at step 5 and resuming must give bit-identical weights to an uninterrupted run.

    Step 5 is mid-epoch (16 batches/epoch) and between checkpoints, and the model has
    dropout, so this covers sampler position, RNG, optimizer, scheduler and EMA state.
    """
    extra = [f"trainer.ema_decay={ema}"] if ema else []
    straight = _fit(make_cfg(*extra, "trainer.max_steps=20", name="straight"))
    _fit(make_cfg(*extra, "trainer.max_steps=20", name="resumed"), stop_at=5)
    resumed = _fit(make_cfg(*extra, "trainer.max_steps=20", name="resumed"))
    assert resumed.step == straight.step == 20
    a, b = _params(straight), _params(resumed)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    if ema:
        for k in straight.ema.shadow:
            assert torch.equal(straight.ema.shadow[k], resumed.ema.shadow[k]), k
    va = load_file(straight.run_dir / "export/final.safetensors")
    vb = load_file(resumed.run_dir / "export/final.safetensors")
    assert all(torch.equal(va[k], vb[k]) for k in va)


def test_resume_never_refuses_existing_run(make_cfg):
    _fit(make_cfg(name="x"))
    with pytest.raises(FileExistsError):
        _fit(make_cfg("checkpoint.resume=never", name="x"))


def test_gradient_accumulation_matches_large_batch(make_cfg):
    """2 x 8 accumulated == 1 x 16 (no dropout; mean losses, equal micro-batch sizes).

    Uses SGD: Adam turns float-noise-level gradients (e.g. a conv bias right before
    GroupNorm, whose true gradient is ~0) into full-size updates, so weights wouldn't match.
    """
    common = [
        "model.dropout=0.0",
        "trainer.max_steps=3",
        "val.every=0",
        "val.quick_every=0",
        "optim.name=sgd",
        "optim.lr=0.1",
    ]
    big = _fit(make_cfg(*common, "dataloader.batch_size=16", name="big"))
    acc = _fit(
        make_cfg(*common, "dataloader.batch_size=8", "trainer.grad_accum_steps=2", name="acc")
    )
    a, b = _params(big), _params(acc)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-5)


def test_init_from_and_evaluate(make_cfg, tmp_path):
    src = _fit(make_cfg(name="src"))
    best = json.loads((src.run_dir / "checkpoints/best.json").read_text())

    # evaluate the best checkpoint: must reproduce the val metric recorded during training
    evaluator = Trainer(make_cfg(name="src"), run_dir=src.run_dir, read_only=True)
    metrics = evaluator.evaluate(src.run_dir / "checkpoints" / best["name"])
    assert metrics["val/loss"] == pytest.approx(best["value"], rel=1e-6)

    # fine-tune from exported weights: step counter starts at 0, weights are loaded
    ft = Trainer(
        make_cfg(f"checkpoint.init_from={src.run_dir}/export/final.safetensors", name="ft")
    )
    ft._setup_training()
    assert ft.step == 0
    ref = load_file(src.run_dir / "export/final.safetensors")
    for k, v in ft.unwrapped_model.state_dict().items():
        assert torch.equal(v, ref[k]), k
    ft.close()


def test_nonfinite_steps_are_skipped_then_abort(make_cfg):
    trainer = Trainer(make_cfg("trainer.max_bad_steps=3"))
    original = trainer.task.training_step
    trainer.task.training_step = lambda m, b: {
        **original(m, b),
        "loss": original(m, b)["loss"] * float("nan"),
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit()
    assert trainer.bad_steps == 3
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())  # no update applied
    trainer.close()
