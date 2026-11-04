[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ml_template

A PyTorch training template for research on multimodal data (vision, poses/odometry,
actions, text). It handles the engineering — multi-GPU, mixed precision, exact
resumption, fixed-sample validation, experiment tracking — so a new project only needs
a dataset, a model and a `Task` describing the loss and metrics.

**Highlights**

- **Step-based trainer**: gradient accumulation, bf16/fp16, `torch.compile`, warmup + cosine / linear / WSD schedules, EMA weights, gradient clipping, skipping of non-finite steps.
- **Multi-GPU**: single GPU, DDP or FSDP2 with one flag (`trainer.strategy`), launched with `torchrun`.
- **Exact resume**: model, optimizer, scheduler, EMA, sampler position and per-rank RNG are all checkpointed. Atomic saves, auto-resume and SIGTERM/SIGUSR1 handling for preemption. Tests check that an interrupted and resumed run is bit-identical to an uninterrupted one (single process, DDP and FSDP).
- **Stable validation**: the same fixed subset every time, a fast "quick" validation for early feedback, and fixed visualization samples.
- **Tracking**: Comet ML (online/offline, resumes the same experiment) plus a local `metrics.jsonl`; the resolved config, git commit/diff and package versions are saved with every run.
- **Typed configs**: YAML composition (`defaults:`), command-line overrides, safe arithmetic (`${calc:...}`), and typo/type checking.
- **Tooling**: uv + mise, ruff, pytest, pre-commit, CI, one-command project renaming / Copier.

## Quick start

```bash
mise install          # Python 3.12 + uv (or install uv yourself)
mise run setup        # uv sync + pre-commit hooks
comet login           # once; or set logger.backend=none

mise run train -- --config configs/debug.yaml        # ~10 s smoke test, no Comet
mise run train -- --config configs/example.yaml      # example multimodal BC task
mise run train-multi -- --config configs/example.yaml trainer.strategy=fsdp   # all GPUs
mise run eval -- --run runs/<run-name>
mise run test
```

## Start a new project

```bash
uvx copier copy --trust gh:<you>/ml_template my_project      # or:
git clone <this repo> my_project && cd my_project && mise run rename -- my_project
```

Then write your dataset, model and task, as described in the [tutorial](docs/tutorial.md).

## Layout

```
configs/                  example.yaml (documented), debug.yaml (composes example.yaml)
src/project_name/
  train.py, evaluate.py   entry points
  config.py               typed config schema + loader (all options documented here)
  tasks/                  base.py: the Task interface  |  behavior_cloning.py: example
  models/                 policy.py: example multimodal policy
  data/                   samplers/loaders (resumable, fixed-subset) | synthetic.py: example data
  trainer/                trainer loop, checkpointing, optim/schedules, EMA, metrics
  tracking/               Comet + JSONL trackers
  utils/                  distributed, logging, seeding, env capture, nn helpers
scripts/rename_project.py
tests/                    fast unit/integration tests + 2-process DDP/FSDP tests
```

Read **[docs/tutorial.md](docs/tutorial.md)** for the full guide.
