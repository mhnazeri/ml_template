"""A synthetic multimodal robotics dataset used by the example task and the tests.

Each sample is a goal-reaching problem: an RGB image with a coloured block, the robot
end-effector pose (xyz + quaternion), a language instruction, and the target action
chunk (H future xyz waypoints). The waypoints depend on all three inputs, so a model
must use vision, proprioception and language to solve it.

Samples are a pure function of (split seed, index): no files, fully deterministic.
Replace this with your real dataset; keep returning a dict of tensors (+ strings).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

INSTRUCTIONS = {
    "reach the block": (0.0, 0.0),
    "hover left of the block": (-0.3, 0.0),
    "hover right of the block": (0.3, 0.0),
    "hover in front of the block": (0.0, 0.3),
}
_SPLIT_SEEDS = {"train": 0, "val": 1, "test": 2}


def encode_text(text: str, max_len: int) -> torch.Tensor:
    """Byte-level tokens (0 = padding). Swap for a real tokenizer when needed."""
    ids = [b + 1 for b in text.encode("utf-8")[:max_len]]
    return torch.tensor(ids + [0] * (max_len - len(ids)), dtype=torch.long)


class SyntheticRobotDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 10_000,
        split: str = "train",
        image_size: int = 64,
        horizon: int = 8,
        max_text_len: int = 32,
    ):
        self.num_samples = num_samples
        self.seed = _SPLIT_SEEDS[split]
        self.image_size = image_size
        self.horizon = horizon
        self.max_text_len = max_text_len
        self.instructions = list(INSTRUCTIONS)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng((self.seed, idx))
        s = self.image_size

        # image: noisy background with one coloured block; block centre is the goal
        image = rng.uniform(0.0, 0.15, size=(3, s, s)).astype(np.float32)
        half = s // 10
        cx, cy = rng.integers(half, s - half, size=2)
        color = rng.uniform(0.5, 1.0, size=3).astype(np.float32)
        image[:, cy - half : cy + half, cx - half : cx + half] = color[:, None, None]
        goal_xy = np.array([cx, cy], dtype=np.float32) / (s - 1) * 2 - 1  # pixels -> [-1, 1]

        # proprioception: end-effector position + random orientation quaternion
        position = rng.uniform(-1, 1, size=3).astype(np.float32)
        quat = rng.normal(size=4).astype(np.float32)
        quat /= np.linalg.norm(quat)

        instruction = self.instructions[rng.integers(len(self.instructions))]
        target = np.array([*(goal_xy + np.array(INSTRUCTIONS[instruction])), 0.2], dtype=np.float32)

        # action chunk: straight-line waypoints from the current position to the target
        alphas = np.arange(1, self.horizon + 1, dtype=np.float32)[:, None] / self.horizon
        actions = position[None] + alphas * (target - position)[None]

        return {
            "id": f"{list(_SPLIT_SEEDS)[self.seed]}-{idx:07d}",
            "image": torch.from_numpy(image),
            "pose": torch.from_numpy(np.concatenate([position, quat])),
            "instruction": instruction,
            "instruction_ids": encode_text(instruction, self.max_text_len),
            "actions": torch.from_numpy(actions),
        }
