import torch

from project_name.trainer.checkpoint import (
    CheckpointManager,
    export_safetensors,
    load_model_weights,
)


def _model():
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))


def test_save_load_roundtrip_and_pruning(tmp_path):
    torch.manual_seed(0)
    model = _model()
    opt = torch.optim.AdamW(model.parameters())
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    mgr = CheckpointManager(tmp_path, keep_last=2)
    paths = [mgr.save(s, model, opt, {"step": s}) for s in (1, 2)]
    mgr.mark_best(paths[0], "val/loss", 0.1, 1)
    mgr.save(3, model, opt, {"step": 3})
    mgr.save(4, model, opt, {"step": 4})
    names = [p.name for p in mgr.list_checkpoints()]
    assert names == ["step_00000001", "step_00000003", "step_00000004"]  # best kept
    assert mgr.latest().name == "step_00000004"

    other = _model()
    other_opt = torch.optim.AdamW(other.parameters())
    state = mgr.load(mgr.latest(), other, other_opt)
    assert state["step"] == 4
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        assert torch.equal(a, b)
    assert other_opt.state_dict()["state"][0]["step"] == 1


def test_incomplete_checkpoint_is_ignored(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model = _model()
    mgr.save(1, model, torch.optim.SGD(model.parameters(), lr=0.1), {"step": 1})
    (tmp_path / "step_00000002.tmp").mkdir()  # crashed mid-save
    (tmp_path / "step_00000003").mkdir()  # no trainer_state.pt
    (tmp_path / "latest").unlink()
    assert mgr.latest().name == "step_00000001"


def test_safetensors_export_roundtrip(tmp_path):
    model = _model()
    path = export_safetensors(model, tmp_path / "w.safetensors")
    other = _model()
    load_model_weights(path, other)
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        assert torch.equal(a, b)
