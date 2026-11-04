import torch

from project_name.data.samplers import (
    ResumableDistributedSampler,
    fixed_subset_indices,
    shard_evenly,
)
from project_name.data.synthetic import SyntheticRobotDataset


def test_sampler_is_deterministic_per_epoch():
    a = ResumableDistributedSampler(100, seed=3)
    b = ResumableDistributedSampler(100, seed=3)
    a.set_epoch(2)
    b.set_epoch(2)
    assert list(a) == list(b)
    b.set_epoch(3)
    assert list(a) != list(b)


def test_sampler_resume_skips_consumed():
    s = ResumableDistributedSampler(100, seed=0)
    s.set_epoch(1)
    full = list(s)
    s.set_epoch(1, start_index=30)
    assert list(s) == full[30:]
    assert len(s) == 70


def test_sampler_shards_are_disjoint_and_equal():
    shards = [list(ResumableDistributedSampler(103, rank=r, world_size=4)) for r in range(4)]
    assert len({len(x) for x in shards}) == 1
    flat = [i for s in shards for i in s]
    assert len(flat) == len(set(flat)) == 100  # drop_last truncates 3


def test_fixed_subset_independent_of_training_seed():
    torch.manual_seed(0)
    a = fixed_subset_indices(1000, 50, seed=0)
    torch.manual_seed(123)
    b = fixed_subset_indices(1000, 50, seed=0)
    assert a == b and len(a) == 50


def test_shard_evenly():
    shards = [shard_evenly(list(range(10)), r, 3) for r in range(3)]
    assert [len(s) for s in shards] == [3, 3, 3]


def test_synthetic_dataset_is_deterministic():
    ds = SyntheticRobotDataset(num_samples=10, horizon=4)
    a, b = ds[3], ds[3]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["actions"], b["actions"])
    assert a["actions"].shape == (4, 3) and a["pose"].shape == (7,)
    assert not torch.equal(ds[3]["image"], SyntheticRobotDataset(10, split="val")[3]["image"])
