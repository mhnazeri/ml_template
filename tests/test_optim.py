import pytest
from omegaconf import OmegaConf

from project_name.config import SchedulerConfig
from project_name.trainer.optim import lr_multiplier


def sched(**kw):
    return OmegaConf.structured(SchedulerConfig(**kw))


def test_warmup_then_cosine():
    cfg = sched(name="cosine", warmup_steps=10, min_lr_ratio=0.1)
    assert lr_multiplier(0, 100, cfg) == pytest.approx(0.1)
    assert lr_multiplier(9, 100, cfg) == pytest.approx(1.0)
    assert lr_multiplier(10, 100, cfg) == pytest.approx(1.0)
    assert lr_multiplier(100, 100, cfg) == pytest.approx(0.1)


def test_wsd_is_flat_then_decays():
    cfg = sched(name="wsd", warmup_steps=0, decay_ratio=0.2, min_lr_ratio=0.0)
    assert lr_multiplier(50, 100, cfg) == 1.0
    assert lr_multiplier(79, 100, cfg) == 1.0
    assert lr_multiplier(90, 100, cfg) == pytest.approx(0.5)
    assert lr_multiplier(100, 100, cfg) == pytest.approx(0.0)
