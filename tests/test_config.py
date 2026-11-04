import pytest
from conftest import DEBUG_CONFIG
from omegaconf.errors import ConfigKeyError, ValidationError

from project_name.config import load_config, safe_calc


def test_defaults_composition_and_overrides():
    cfg = load_config(DEBUG_CONFIG, ["optim.lr=0.5"])
    assert cfg.trainer.max_steps == 40  # from debug.yaml
    assert cfg.optim.weight_decay == 0.05  # inherited from example.yaml
    assert cfg.optim.lr == 0.5  # CLI override
    assert cfg.model.image_size == cfg.data.train.image_size  # interpolation resolved


def test_calc_resolver(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "trainer:\n  max_steps: 100\nscheduler:\n  warmup_steps: ${calc:${trainer.max_steps} // 20 + 1}\n"
    )
    assert load_config(p).scheduler.warmup_steps == 6


def test_from_yaml_resolver(tmp_path):
    (tmp_path / "other.yaml").write_text("a:\n  b: 7\n")
    p = tmp_path / "c.yaml"
    p.write_text(f"trainer:\n  max_steps: ${{from_yaml:{tmp_path}/other.yaml,a.b}}\n")
    assert load_config(p).trainer.max_steps == 7


@pytest.mark.parametrize("expr", ["__import__('os')", "().__class__", "open('x')", "a + 1"])
def test_calc_is_safe(expr):
    with pytest.raises(ValueError):
        safe_calc(expr)


def test_typo_in_typed_section_is_rejected():
    with pytest.raises(ConfigKeyError):
        load_config(DEBUG_CONFIG, ["trainer.max_step=5"])


def test_wrong_type_is_rejected():
    with pytest.raises(ValidationError):
        load_config(DEBUG_CONFIG, ["trainer.max_steps=abc"])


def test_invalid_choice_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        load_config(DEBUG_CONFIG, ["trainer.precision=fp8"])


def test_circular_defaults(tmp_path):
    (tmp_path / "a.yaml").write_text("defaults: [b.yaml]\n")
    (tmp_path / "b.yaml").write_text("defaults: [a.yaml]\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")
