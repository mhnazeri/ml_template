# Tutorial

This guide walks from an empty project to multi-GPU training on your own data. The
running example is the included behavior-cloning task: image + end-effector pose +
language instruction → a chunk of future waypoints.

1. [Setup](#1-setup)
2. [Mental model](#2-mental-model)
3. [Configs](#3-configs)
4. [Your dataset](#4-your-dataset)
5. [Your model](#5-your-model)
6. [Your task](#6-your-task)
7. [Running](#7-running)
8. [Validation](#8-validation)
9. [Checkpoints, resuming, fine-tuning, evaluation](#9-checkpoints-resuming-fine-tuning-evaluation)
10. [Logging and what's in a run directory](#10-logging-and-whats-in-a-run-directory)
11. [Reproducibility](#11-reproducibility)
12. [Recipes](#12-recipes)
13. [Testing](#13-testing)

---

## 1. Setup

```bash
mise install            # Python + uv as pinned in mise.toml
mise run setup          # uv sync (incl. dev deps) + pre-commit install
comet login             # stores your API key in ~/.comet.config
```

Without mise: `uv sync && uv run pre-commit install`, then prefix commands with `uv run`.
`mise tasks` lists all tasks.

**New project.** Either `uvx copier copy --trust gh:<you>/ml_template my_project`, or
clone and run `mise run rename -- my_project` (add `--dry-run` to preview). Both
rename `src/project_name/` and every reference to it, then you run `uv sync`.

## 2. Mental model

```
             ┌────────────── Task (you write this) ──────────────┐
config ──▶   │ build_datasets()  build_model()  training_step()  │
             │ validation_step()  visualize()  param_groups() …  │
             └───────────────────────────────────────────────────┘
                                   ▲  called by
             ┌────────────── Trainer (don't edit) ───────────────┐
             │ devices, DDP/FSDP, precision, accumulation, clip,  │
             │ schedules, EMA, validation, checkpoints, logging   │
             └───────────────────────────────────────────────────┘
```

The **Task** holds everything specific to your problem, and the **Trainer** handles the
engineering. Datasets and models are plain PyTorch classes built from config via `_target_`.

## 3. Configs

Configs are YAML files validated against a typed schema in
[`config.py`](../src/project_name/config.py). **Every option and its default is
documented there.** A config is assembled in this order (later wins):

1. schema defaults → 2. files listed in `defaults:` → 3. the file itself → 4. command-line overrides

```yaml
# configs/my_exp.yaml
defaults:
  - example.yaml          # relative to this file; can be a list, nested arbitrarily

optim:
  lr: 5e-4
scheduler:
  warmup_steps: ${calc:${trainer.max_steps} // 20}   # safe arithmetic
data:
  root: ${from_yaml:configs/paths.yaml,datasets.my_robot}  # pull a value from another file
```

```bash
mise run train -- --config configs/my_exp.yaml optim.lr=1e-3 run.name=lr-sweep-1e-3
```

- **Typed sections** (`run`, `trainer`, `optim`, `scheduler`, `dataloader`, `val`,
  `checkpoint`, `logger`): typos and wrong types fail at startup, e.g.
  `trainer.max_step=5` → error.
- **Free-form sections** (`task`, `model`, `data`): anything goes; `_target_` names the
  class to build, and the other keys become its constructor arguments.
- The fully resolved config is saved to `<run_dir>/config.yaml` and logged to Comet.

## 4. Your dataset

Return a **dict** per sample: tensors, plus optional strings or ids (strings are collated into
lists and are handy for logging). Keep `__getitem__` deterministic given its index if you can.

```python
# src/project_name/data/episodes.py   (uses torchvision: `uv add torchvision`)
class EpisodeDataset(Dataset):
    """Windows of (obs, action-chunk) from recorded episodes stored as .npz files."""

    def __init__(self, root: str, split: str = "train", horizon: int = 8, image_size: int = 128):
        self.files = sorted(Path(root, split).glob("*.npz"))
        self.horizon = horizon
        # index = list of (episode, t) pairs, built once
        self.index = [
            (i, t)
            for i, f in enumerate(self.files)
            for t in range(np.load(f, mmap_mode="r")["actions"].shape[0] - horizon)
        ]
        self.transform = T.Compose([T.Resize(image_size), T.ConvertImageDtype(torch.float32)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, t = self.index[idx]
        d = np.load(self.files[ep], mmap_mode="r")
        return {
            "id": f"{self.files[ep].stem}:{t}",
            "image": self.transform(torch.from_numpy(d["rgb"][t]).permute(2, 0, 1)),
            "pose": torch.from_numpy(d["poses"][t]).float(),  # e.g. xyz + quat
            "instruction": str(d["instruction"]),
            "actions": torch.from_numpy(d["actions"][t : t + self.horizon]).float(),
        }
```

```yaml
data:
  train: {_target_: project_name.data.episodes.EpisodeDataset, root: /data/robot, split: train}
  val:   {_target_: project_name.data.episodes.EpisodeDataset, root: /data/robot, split: val}
```

Tips:
- **Variable-length data** (text tokens, point clouds, varying episode lengths):
  override `Task.collate_fn()` to pad and add masks.
- **Downloads / cache building**: `build_datasets` runs on every rank, wrapped in
  `main_process_first()`, so rank 0 builds caches before the others read them.
- **Tokenization**: do it in the dataset (in parallel workers) or in `collate_fn`.

## 5. Your model

A plain `nn.Module`; constructor arguments come from the `model:` config section.

```yaml
model:
  _target_: project_name.models.policy.MultimodalPolicy
  image_size: ${data.train.image_size}   # interpolate rather than duplicate
  dim: 256
```

The example [`policy.py`](../src/project_name/models/policy.py) uses one encoder per
modality → fusion → head. Using `transformers` or `timm` encoders works as usual: build
them inside your module.

## 6. Your task

Subclass `Task` and implement at least `training_step`. It receives the (possibly
DDP/FSDP-wrapped) model and a batch already on the device, and returns a dict of
**scalar** tensors that includes `"loss"`.

```python
class BehaviorCloningTask(Task):
    def training_step(self, model, batch):
        pred = model(batch["image"], batch["pose"], batch["instruction_ids"])
        loss = F.mse_loss(pred, batch["actions"])
        final_err = (pred[:, -1] - batch["actions"][:, -1]).norm(dim=-1).mean()
        return {"loss": loss, "final_err": final_err}  # all logged as train/*
```

Optional hooks:

| Hook | Default | Use it for |
|---|---|---|
| `validation_step(model, batch)` | = `training_step` | extra eval-only metrics (success rate, ATE/RPE for odometry, …) |
| `visualize(model, batch)` | `{}` | return `{name: image or str}` for the fixed visualization samples |
| `build_model()` / `build_datasets()` | instantiate from config | anything more complex than `_target_` |
| `collate_fn()` | `default_collate` | padding, masks, custom batching |
| `param_groups(model)` | decay / no-decay split | per-module LR (`"lr_scale"`), frozen parts |
| `fsdp_modules(model)` | `[]` | blocks to shard separately under FSDP (e.g. transformer layers) |
| `batch_size(batch)` | first tensor's dim 0 | weighting validation means correctly |

Rules that keep multi-GPU correct:
- Always call `model(...)`, **not** `model.encoder(...)`. Calling a submodule directly
  bypasses the DDP/FSDP hooks. If you need attributes, use
  `project_name.utils.nn.unwrap_model(model)`.
- Metric keys must be the same on every rank (they are reduced across ranks).
- `validation_step` returns batch **means**; the trainer weights them by batch size,
  so the reported value is the exact mean over the dataset.

Point the config at your task with `task._target_: project_name.tasks.my_task.MyTask`. Other
keys in `task:` are free for your own settings (read them via `self.cfg.task`).

## 7. Running

```bash
# one GPU (or CPU)
mise run train -- --config configs/example.yaml

# all GPUs on this node: DDP by default
mise run train-multi -- --config configs/example.yaml
NGPUS=2 mise run train-multi -- --config configs/example.yaml trainer.strategy=fsdp

# what train-multi runs under the hood:
uv run torchrun --standalone --nproc-per-node=gpu -m project_name.train --config ...
```

- **Batch size**: `dataloader.batch_size` is **per GPU, per micro-batch**. Global batch =
  `batch_size × num_gpus × trainer.grad_accum_steps` (logged as `global_batch_size`).
  Change the number of GPUs and adjust `grad_accum_steps` to keep it constant.
- **Strategy**: `auto` uses DDP when launched with more than one process. Use FSDP when
  the model plus optimizer state doesn't fit on one GPU. It shards parameters, gradients
  and optimizer state, and checkpoints are sharded too.
- **Precision**: `bf16` (default; no loss scaling needed), `fp16` (uses GradScaler; not
  with FSDP), `fp32`. `trainer.compile=true` enables `torch.compile`.
- **Length**: `trainer.max_steps` (optimizer steps) and/or `trainer.max_epochs`
  (whichever comes first). Logging, validation and checkpoint intervals are all in steps.
- **Debugging**: `configs/debug.yaml` (tiny data, no Comet, runs on CPU) and
  `trainer.detect_anomaly=true` for NaN hunting. Run `trainer.strategy=ddp|fsdp` under
  `torchrun --nproc-per-node=1` to debug distributed code paths on one GPU.
- **Multi-node (later)**: the code already uses `torchrun` env variables; launch with
  `torchrun --nnodes=N --rdzv-backend=c10d --rdzv-endpoint=<host>:29500 ...` (or `srun`).

## 8. Validation

Validation always uses the **same samples**, so curves are comparable across steps, runs
and resumes.

- `val.max_samples`: size of the fixed validation subset, chosen with `val.subset_seed`
  (independent of `run.seed`). `null` = the whole validation set.
- **Full validation** every `val.every` steps, and always at the end. Its metrics are logged as `val/*`.
  `val.monitor` + `val.mode` decide the **best** checkpoint and early stopping
  (`val.early_stopping_patience`, counted in full validations).
- **Quick validation** every `val.quick_every` steps on the first `val.quick_samples` of
  the fixed subset, logged as `val_quick/*`. It gives cheap, frequent feedback during long
  runs. Set `val.at_start=true` for a sanity check before training starts.
- **Visualizations**: `task.visualize` runs on the first `val.num_vis_samples` fixed
  samples at each validation, so you can watch the predictions on the same samples
  evolve in Comet's image panel.
- With EMA enabled (`trainer.ema_decay`), validation uses the EMA weights unless
  `val.use_ema=false`.

## 9. Checkpoints, resuming, fine-tuning, evaluation

```
runs/<name>/checkpoints/
  step_00001000/   model+optimizer+EMA (DCP, sharded under FSDP), trainer_state.pt, rng_rank*.pt, meta.json
  latest           newest complete checkpoint
  best.json        best checkpoint on val.monitor (protected from pruning)
runs/<name>/export/
  best.safetensors, final.safetensors   full deployable weights (EMA if enabled)
```

- **Saving**: every `checkpoint.every` steps, whenever validation improves (`save_best`), at the
  end, and on preemption. `keep_last` limits the number of rolling checkpoints. Saves are atomic.
- **Resuming**: give the run a name. Re-running the same command resumes automatically:
  ```bash
  mise run train -- --config configs/example.yaml run.name=bc-v1    # crashes / gets preempted
  mise run train -- --config configs/example.yaml run.name=bc-v1    # continues exactly
  ```
  `checkpoint.resume=auto|never|<path to checkpoint or run dir>`. Without `run.name`,
  each run gets a fresh timestamped name. The Comet experiment is continued too.
- **Preemption**: SIGTERM or SIGUSR1 (e.g. SLURM `--signal=USR1@120`) → the trainer
  finishes the current step, saves, and exits cleanly.
- **Fine-tuning / pretrain → finetune**: `checkpoint.init_from=runs/pretrain/export/final.safetensors`
  (or a checkpoint dir) loads **weights only**, with a fresh optimizer and step counter.
  `checkpoint.init_strict=false` allows partial loading (e.g. a new head).
- **Evaluation**:
  ```bash
  mise run eval -- --run runs/bc-v1                          # best (else latest) checkpoint
  mise run eval -- --run runs/bc-v1 --ckpt step_00002000 val.max_samples=null
  ```
  This uses the run's saved config (with your overrides) and writes `runs/bc-v1/eval/<ckpt>.json`.
  A sharded FSDP checkpoint can be evaluated on a single GPU.
- **Loading for inference** elsewhere:
  ```python
  from omegaconf import OmegaConf
  from safetensors.torch import load_file
  from project_name.utils.misc import instantiate

  cfg = OmegaConf.load("runs/bc-v1/config.yaml")
  model = instantiate(cfg.model)
  model.load_state_dict(load_file("runs/bc-v1/export/best.safetensors"))
  ```

## 10. Logging and what's in a run directory

| File | Content |
|---|---|
| `train.log` | the console log |
| `metrics.jsonl` | every logged metric (one JSON per line). Works without Comet, easy to load with pandas |
| `config.yaml` | the fully resolved config (`config.resume-<step>.yaml` for each resume) |
| `env.json`, `packages.txt`, `git.diff` | Python/torch/CUDA/GPU, exact package versions, commit and uncommitted changes |

Logged metrics: `train/*` (your metrics + `grad_norm`, `lr`, `epoch`), `val/*`,
`val_quick/*`, `<monitor>_best`, and `perf/*` (samples/s, step time, fraction of
time waiting for data, peak GPU memory). Only rank 0 logs.

**Comet**: `logger.project`, `logger.workspace`, `logger.online=false` (writes an offline
archive to `<run_dir>/comet/`; upload with `comet upload`), `logger.histograms=true` for
weight and gradient histograms, `logger.upload_checkpoints=true`. Use `logger.backend=none` to
turn Comet off. To add W&B or TensorBoard, implement the small `Tracker` interface
in `tracking/base.py`.

## 11. Reproducibility

- Every rank builds the model with the same seed (`run.seed`), then switches to
  `seed + rank` for dropout and augmentations.
- The data order depends only on `(run.seed, epoch)`. Resuming skips exactly the batches
  already consumed in the current epoch.
- Each checkpoint stores the RNG state of every rank. DataLoader worker seeds are a
  function of `(seed, epoch, rank)`, and numpy/random are seeded per worker (avoiding
  the classic "every worker produces the same augmentation" bug).
- `run.deterministic=true` enables deterministic algorithms (slower).
- **Limit:** with `num_workers > 0`, random augmentations after a **mid-epoch** resume
  differ from an uninterrupted run, because worker RNG streams restart. Everything else
  (data order, model, optimizer, dropout) is exact. For fully exact augmentations, derive
  the augmentation RNG from the sample, e.g. `np.random.default_rng((seed, epoch, idx))`.
- Changing the number of GPUs or `batch_size` between resumes is allowed. The current epoch
  then restarts, and a warning is logged.

## 12. Recipes

**Lower LR for a pretrained encoder, freeze another part**
```python
def param_groups(self, model):
    freeze(model.text_enc)  # utils.nn.freeze: requires_grad=False on all its params
    enc = [p for n, p in model.named_parameters() if n.startswith("image_enc")]
    rest = [
        p for n, p in model.named_parameters() if not n.startswith("image_enc") and p.requires_grad
    ]
    return [{"params": enc, "lr_scale": 0.1}, {"params": rest}]
```
With frozen or conditionally unused parameters under DDP, set
`trainer.ddp_find_unused_parameters=true`.

**FSDP for a transformer**: return the blocks so each one is a separate shard unit:
```python
def fsdp_modules(self, model):
    return list(model.blocks)
```

**Odometry / pose metrics**: compute them in `validation_step` (e.g. translation error,
rotation error in degrees), and return batch means.

**Multiple validation sets**: build them in your task, or run `evaluate.py` with
overrides such as `data.val.split=test`.

**Sweeps**: run a shell loop over overrides with distinct `run.name`s, and compare them in Comet.

**Learning-rate schedules**: `scheduler.name=cosine|linear|constant|wsd`.
`wsd` (warmup → stable → decay over the last `decay_ratio` of training) is useful for
long or continued pretraining: you can branch off a stable-phase checkpoint and
decay later.

## 13. Testing

```bash
mise run test        # fast suite (~5 s, CPU)
mise run test-all    # + real 2-process DDP and FSDP runs over gloo (~30 s)
mise run lint        # ruff
```

The key tests are in `tests/test_trainer.py` and `tests/test_distributed.py`. They
interrupt training mid-epoch with a simulated preemption signal, resume, and require
**bit-identical** final weights compared with an uninterrupted run. Keep them green
when you change the trainer. Add a test for your own task that runs
`configs/debug.yaml`-sized training on CPU.
