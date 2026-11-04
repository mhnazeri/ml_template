"""Configuration: typed schema + YAML composition + CLI overrides.

How a config is built (later wins):
  1. defaults of the typed schema below (`Config`)
  2. YAML files listed under `defaults:` in the config file (recursively, relative paths)
  3. the config file itself
  4. command-line overrides in dotlist form, e.g. `optim.lr=3e-4 trainer.max_steps=100`

The typed sections (run, trainer, optim, ...) are validated: a typo such as
`trainer.max_step` or a wrong type raises an error immediately. The free-form
sections (`task`, `model`, `data`) accept anything your code needs.

Custom resolvers available in YAML:
  ${calc:2 * ${dataloader.batch_size}}   safe arithmetic (+ - * / // % **, min, max, round)
  ${from_yaml:path/to/file.yaml,a.b}     read a value from another YAML file
  ${now:%Y%m%d-%H%M%S}                   current time (OmegaConf built-in)
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    name: str | None = None  # None -> "<prefix>-<timestamp>". Set it to make a run resumable.
    prefix: str = "run"  # used for auto-generated names
    root: str = "runs"  # all runs live under <root>/<name>
    seed: int = 42
    deterministic: bool = False  # bitwise reproducibility (slower)
    tf32: bool = True  # allow TF32 matmuls on Ampere+
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class TrainerConfig:
    max_steps: int | None = None  # optimizer steps; one of max_steps / max_epochs is required
    max_epochs: int | None = None
    grad_accum_steps: int = 1
    grad_clip: float = 1.0  # <= 0 disables clipping (grad norm is still logged)
    precision: str = "bf16"  # fp32 | bf16 | fp16
    strategy: str = "auto"  # auto | single | ddp | fsdp  (auto: ddp if world_size > 1)
    compile: bool = False  # torch.compile the model
    log_every: int = 10  # steps between train metric logs
    detect_anomaly: bool = False  # torch.autograd anomaly detection (very slow, debug only)
    max_bad_steps: int = 20  # abort after this many consecutive non-finite steps
    ddp_find_unused_parameters: bool = False
    fsdp_reshard_after_forward: bool = True
    ema_decay: float | None = None  # e.g. 0.999 enables an EMA of the weights
    ema_start_step: int = 0


@dataclass
class OptimConfig:
    name: str = "adamw"  # adamw | adam | sgd, or a fully-qualified class path
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    momentum: float = 0.9  # sgd only
    no_decay_on_bias_and_norm: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)  # extra optimizer arguments


@dataclass
class SchedulerConfig:
    name: str = "cosine"  # constant | cosine | linear | wsd (warmup-stable-decay)
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0  # final lr = lr * min_lr_ratio
    decay_ratio: float = 0.2  # wsd: fraction of training spent decaying at the end


@dataclass
class DataLoaderConfig:
    batch_size: int = 32  # per-GPU micro-batch; global = batch_size * world * grad_accum
    val_batch_size: int | None = None  # defaults to batch_size
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class ValConfig:
    every: int = 1000  # full validation every N steps (0 = only at the end)
    quick_every: int = 0  # quick (partial) validation every N steps (0 = off)
    quick_samples: int = 256  # quick validation uses the first N samples of the fixed subset
    max_samples: int | None = None  # fixed validation subset size (None = whole val set)
    subset_seed: int = 0  # independent of run.seed so every run validates the same samples
    at_start: bool = False  # run a quick validation before training (sanity check)
    monitor: str = "val/loss"  # metric used for "best" checkpoint and early stopping
    mode: str = "min"  # min | max
    num_vis_samples: int = 4  # fixed samples passed to task.visualize (0 = off)
    use_ema: bool = True  # validate with EMA weights when EMA is enabled
    early_stopping_patience: int = 0  # number of full validations without improvement; 0 = off
    early_stopping_min_delta: float = 0.0


@dataclass
class CheckpointConfig:
    every: int = 1000  # save every N steps (0 = only at the end / on preemption)
    keep_last: int = 3  # rolling checkpoints to keep (the best one is always kept)
    save_best: bool = True
    resume: str = "auto"  # auto (resume if run dir has a checkpoint) | never | <checkpoint path>
    init_from: str | None = (
        None  # load model weights only (fine-tuning), checkpoint dir or .safetensors
    )
    init_strict: bool = True
    export_safetensors: bool = True  # export final/best weights as .safetensors


@dataclass
class LoggerConfig:
    backend: str = "comet"  # comet | none   (metrics.jsonl is always written locally)
    project: str | None = None
    workspace: str | None = None
    online: bool = True  # False -> offline experiment archive in <run_dir>/comet
    log_code: bool = False
    log_env: bool = True
    histograms: bool = False  # weight/gradient histograms (expensive)
    upload_checkpoints: bool = False  # upload exported .safetensors to Comet


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    val: ValConfig = field(default_factory=ValConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    # free-form sections
    task: dict[str, Any] = field(
        default_factory=lambda: {"_target_": "project_name.tasks.base.Task"}
    )
    model: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Resolvers
# --------------------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"min": min, "max": max, "round": round, "int": int, "float": float, "abs": abs}


def safe_calc(expr: str) -> int | float:
    """Evaluate a numeric expression without `eval` (no names, attributes or imports)."""

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
        raise ValueError(f"Unsupported expression in ${{calc:...}}: {ast.dump(node)}")

    return _eval(ast.parse(str(expr), mode="eval"))


def _from_yaml(path: str, key: str | None = None) -> Any:
    cfg = OmegaConf.load(path)
    value = OmegaConf.select(cfg, key) if key else cfg
    if value is None:
        raise KeyError(f"Key '{key}' not found in {path}")
    return OmegaConf.to_container(value) if isinstance(value, DictConfig) else value


def register_resolvers() -> None:
    for name, fn in (("calc", safe_calc), ("from_yaml", _from_yaml)):
        if OmegaConf.has_resolver(name):
            continue
        if hasattr(OmegaConf, "register_resolver"):  # OmegaConf >= 2.4
            OmegaConf.register_resolver(name, fn, annotation_validation="off")
        else:
            OmegaConf.register_new_resolver(name, fn)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _load_with_defaults(path: Path, seen: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path.suffix not in {".yaml", ".yml"}:
        path = path.with_suffix(".yaml")
    if path in seen:
        raise ValueError(f"Circular `defaults` include: {' -> '.join(map(str, (*seen, path)))}")
    cfg = OmegaConf.load(path)
    defaults = cfg.pop("defaults", [])
    merged = OmegaConf.create()
    for d in defaults:
        merged = OmegaConf.merge(merged, _load_with_defaults(path.parent / d, (*seen, path)))
    return OmegaConf.merge(merged, cfg)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    """Load, compose, override, validate and resolve a config. Returns a read-only DictConfig."""
    register_resolvers()
    cfg = OmegaConf.structured(Config)
    if path is not None:
        cfg = OmegaConf.merge(cfg, _load_with_defaults(Path(path)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    validate(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def validate(cfg: DictConfig) -> None:
    t = cfg.trainer
    if t.max_steps is None and t.max_epochs is None:
        raise ValueError("Set trainer.max_steps or trainer.max_epochs")
    choices = {
        "trainer.precision": (t.precision, {"fp32", "bf16", "fp16"}),
        "trainer.strategy": (t.strategy, {"auto", "single", "ddp", "fsdp"}),
        "scheduler.name": (cfg.scheduler.name, {"constant", "cosine", "linear", "wsd"}),
        "val.mode": (cfg.val.mode, {"min", "max"}),
        "logger.backend": (cfg.logger.backend, {"comet", "none"}),
    }
    for key, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError(f"{key}={value!r} is not one of {sorted(allowed)}")
    if t.grad_accum_steps < 1:
        raise ValueError("trainer.grad_accum_steps must be >= 1")
    if t.precision == "fp16" and t.strategy == "fsdp":
        raise ValueError("fp16 + FSDP is not supported; use precision=bf16")


def save_config(cfg: DictConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
