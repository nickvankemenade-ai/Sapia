"""Configuration: dataclasses, presets, YAML files and CLI overrides.

Precedence (lowest to highest): preset defaults < ``--config file.yaml`` < CLI flags.
Every field of ``TrainConfig`` (and ``model.*``) is a CLI flag, e.g. ``--batch_size 8`` or
``--model.n_layer 4``. Fields documented as "auto" accept ``auto`` (or ``none``) on the CLI.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

PRECISIONS = ("bf16", "fp16", "fp32")
_PRECISION_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16", "bf-16": "bf16",
    "fp16": "fp16", "float16": "fp16", "half": "fp16", "16": "fp16",
    "fp32": "fp32", "float32": "fp32", "32": "fp32", "float": "fp32",
}


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration. Never worth retrying."""


def normalize_precision(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _PRECISION_ALIASES:
        raise ConfigError(f"precision must be one of {PRECISIONS}, got {value!r}")
    return _PRECISION_ALIASES[key]


def F(default: Any, help: str) -> Any:  # noqa: N802 - tiny field helper
    return field(default=default, metadata={"help": help})


@dataclass
class ModelConfig:
    vocab_size: int = F(49152, "Vocabulary size (SmolLM2 tokenizer, already a multiple of 64).")
    block_size: int = F(2048, "Context length in tokens.")
    n_layer: int = F(24, "Number of transformer blocks.")
    n_head: int = F(16, "Query heads (d_model must be divisible by n_head).")
    n_kv_head: int = F(4, "Key/value heads for grouped-query attention (n_head must be divisible by it; "
                          "n_kv_head == n_head is plain multi-head attention).")
    d_model: int = F(2048, "Residual stream width.")
    mlp_hidden: int = F(5632, "SwiGLU hidden width (2.75 * d_model, a multiple of 64).")
    rope_base: float = F(10000.0, "RoPE frequency base.")
    norm_eps: float = F(1e-6, "RMSNorm epsilon.")
    qk_norm: bool = F(True, "RMSNorm over each head's queries and keys (before RoPE).")
    z_loss: float = F(1e-4, "Weight of the z-loss, z_loss * logsumexp(logits)^2 (0 disables).")
    tie_embeddings: bool = F(True, "Share the input embedding and the output projection.")
    dropout: float = F(0.0, "Dropout probability (0 for pretraining).")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def validate(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_kv_head", "d_model", "mlp_hidden"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"model.{name} must be positive")
        if self.d_model % self.n_head != 0:
            raise ConfigError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        if self.head_dim % 2 != 0:
            raise ConfigError("head dimension (d_model / n_head) must be even for RoPE")
        if self.n_head % self.n_kv_head != 0:
            raise ConfigError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")
        if self.z_loss < 0:
            raise ConfigError("model.z_loss must be >= 0")

    def num_params(self) -> int:
        """Exact trainable parameter count (a tied embedding is counted once, no biases)."""
        d, hd = self.d_model, self.head_dim
        emb = self.vocab_size * d * (1 if self.tie_embeddings else 2)
        attn = d * (self.n_head + 2 * self.n_kv_head) * hd + d * d   # fused q/k/v + output projection
        mlp = 3 * d * self.mlp_hidden                                # gate, up, down
        norms = 2 * d + (2 * hd if self.qk_norm else 0)              # two RMSNorms (+ q and k norms)
        return emb + self.n_layer * (attn + mlp + norms) + d         # + final RMSNorm

    def flops_per_token(self) -> float:
        """Approximate forward+backward FLOPs per trained token.

        6*N for the weight matmuls (2N forward, 4N backward; the tied embedding counts
        because it is the output projection) plus the attention score/value matmuls,
        which have no parameters: 12 * n_layer * d_model * block_size. Grouped-query
        attention does not shrink the latter: the number of query heads is unchanged.
        """
        return 6.0 * self.num_params() + 12.0 * self.n_layer * self.d_model * self.block_size


@dataclass
class TrainConfig:
    preset: str = F("sapia", "Name of the preset this config started from (informational).")
    model: ModelConfig = field(default_factory=ModelConfig)

    # ---- paths
    data_dir: str = F("data/sapia-mix", "Directory with manifest.json and token shards.")
    out_dir: str = F("runs/sapia", "Run directory (checkpoints, metrics.jsonl, train.log).")

    # ---- training budget
    max_tokens: int = F(15_000_000_000, "Total training tokens; the LR schedule is defined over this. "
                                         "Set it from a measured throughput trial.")
    max_steps: Optional[int] = F(None, "Optional cap on optimizer steps (auto = derived from max_tokens).")
    max_hours: Optional[float] = F(None, "Wall-clock budget for this invocation; checkpoints and exits cleanly.")
    max_total_hours: Optional[float] = F(None, "Cumulative training-time budget across all resumes of this run.")

    # ---- batch / memory auto-config
    batch_size: Optional[int] = F(None, "Micro-batch size in sequences. auto = probe GPU memory.")
    grad_accum_steps: Optional[int] = F(None, "Micro-batches per optimizer step. auto = hit target_tokens_per_step.")
    target_tokens_per_step: int = F(1_048_576, "Target tokens per optimizer step (batch * accum * block_size).")
    max_batch_size: int = F(512, "Upper bound for the auto-probed micro-batch size.")
    vram_fraction: float = F(0.85, "Fraction of usable VRAM the probed peak may occupy (safety margin).")
    reuse_batch_plan: bool = F(True, "On resume, reuse the checkpoint's probed batch plan if the GPU matches.")

    # ---- optimizer / schedule
    learning_rate: float = F(3e-4, "Peak learning rate.")
    min_lr_ratio: float = F(0.1, "Final LR as a fraction of the peak (cosine floor).")
    warmup_tokens: int = F(400_000_000, "Linear warmup length, in tokens.")
    weight_decay: float = F(0.1, "AdamW weight decay, applied to >=2D parameters only.")
    beta1: float = F(0.9, "AdamW beta1.")
    beta2: float = F(0.95, "AdamW beta2.")
    adam_eps: float = F(1e-8, "AdamW epsilon.")
    grad_clip: float = F(1.0, "Global gradient-norm clip (0 disables).")

    # ---- precision / compilation
    precision: str = F("bf16", "bf16 (default, no GradScaler) | fp16 (autocast + GradScaler) | fp32.")
    compile: bool = F(True, "Use torch.compile (ignored on Windows, where Triton is unavailable).")

    # ---- data loading
    data_mode: str = F("auto", "auto | ram (load shards into memory) | memmap.")
    num_workers: Optional[int] = F(None, "DataLoader workers. auto = 0 for ram mode, a few for memmap.")
    prefetch_factor: Optional[int] = F(None, "Batches prefetched per worker. auto = 4.")

    # ---- evaluation / logging / checkpoints
    eval_interval: int = F(250, "Evaluate on the val split every N optimizer steps.")
    eval_tokens: int = F(1_048_576, "Val tokens per evaluation.")
    log_interval: int = F(10, "Log every N optimizer steps.")
    ckpt_interval: int = F(250, "Checkpoint every N optimizer steps.")
    ckpt_minutes: float = F(30.0, "Also checkpoint when this many minutes passed since the last one (0 disables).")
    keep_last: int = F(2, "Number of most recent checkpoints to keep.")
    metrics_seconds: float = F(10.0, "Rewrite training_metrics.json at most this often (also on every eval and at the end).")
    save_best: bool = F(True, "Also keep checkpoints/best.pt: weights of the lowest-val-loss model so far.")
    resume: str = F("auto", "auto (latest checkpoint in out_dir) | never | path to a checkpoint file.")

    # ---- robustness / misc
    max_nan_skips: int = F(10, "Abort after this many consecutive optimizer steps skipped for non-finite loss/grads.")
    seed: int = F(1337, "Random seed.")
    device: str = F("auto", "auto | cuda | cuda:N | cpu.")
    wandb: bool = F(False, "Log to Weights & Biases if the package is installed (never required).")
    peak_flops: Optional[float] = F(None, "GPU peak FLOP/s for the MFU estimate. auto = lookup by GPU name.")
    dry_run: bool = F(False, "Detect resources, plan batch/data settings, print them and exit without training.")

    def validate(self) -> None:
        self.model.validate()
        self.precision = normalize_precision(self.precision)
        if self.data_mode not in ("auto", "ram", "memmap"):
            raise ConfigError(f"data_mode must be auto|ram|memmap, got {self.data_mode!r}")
        if self.batch_size is not None and self.batch_size < 1:
            raise ConfigError("batch_size must be >= 1 (or auto)")
        if self.grad_accum_steps is not None and self.grad_accum_steps < 1:
            raise ConfigError("grad_accum_steps must be >= 1 (or auto)")
        if not 0.05 <= self.vram_fraction <= 1.0:
            raise ConfigError("vram_fraction must be in [0.05, 1.0]")
        if self.max_tokens <= 0 or self.target_tokens_per_step <= 0:
            raise ConfigError("max_tokens and target_tokens_per_step must be positive")
        if self.keep_last < 1:
            raise ConfigError("keep_last must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive")
        for name in ("eval_interval", "log_interval", "ckpt_interval", "max_nan_skips", "max_batch_size"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- presets

def _preset_sapia() -> TrainConfig:
    return TrainConfig()


def _preset_tiny() -> TrainConfig:
    """A few-million-parameter model for smoke tests on any GPU or CPU."""
    return TrainConfig(
        preset="tiny",
        model=ModelConfig(vocab_size=49152, block_size=128, n_layer=4, n_head=4, n_kv_head=2, d_model=128,
                          mlp_hidden=352),
        data_dir="data/tiny",
        out_dir="runs/tiny",
        max_tokens=1_600_000,
        compile=False,
        target_tokens_per_step=8192,
        max_batch_size=64,
        learning_rate=1e-3,
        warmup_tokens=80_000,
        eval_interval=50,
        eval_tokens=16_384,
        log_interval=10,
        ckpt_interval=50,
        ckpt_minutes=0.0,
    )


PRESETS = {"sapia": _preset_sapia, "tiny": _preset_tiny}


def get_preset(name: str) -> TrainConfig:
    if name not in PRESETS:
        raise ConfigError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]()


# ----------------------------------------------------------------------------- value parsing

def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def _parse_int(value: Any) -> int:
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("expected an integer, got a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
        return int(value)
    s = str(value).strip().replace("_", "").replace(",", "")
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)  # accepts 5e9
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
        if f != int(f):
            raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
        return int(f)


def _is_auto(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in ("auto", "none", "null", ""))


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def coerce_value(tp: Any, value: Any) -> Any:
    """Coerce a CLI string or YAML scalar to the dataclass field type ``tp``."""
    base, optional = _unwrap_optional(tp)
    if optional and _is_auto(value):
        return None
    if base is bool:
        return _parse_bool(value)
    if base is int:
        return _parse_int(value)
    if base is float:
        if isinstance(value, bool):
            raise argparse.ArgumentTypeError("expected a number, got a boolean")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if base is str:
        return str(value)
    return value


def _field_types(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def add_config_args(parser: argparse.ArgumentParser, cls: type = TrainConfig, prefix: str = "") -> None:
    """Add one ``--flag`` per dataclass field (nested dataclasses become ``--outer.inner``)."""
    hints = _field_types(cls)
    for f in fields(cls):
        tp = hints[f.name]
        if is_dataclass(tp):
            add_config_args(parser, tp, prefix=f"{prefix}{f.name}.")
            continue
        name = f"{prefix}{f.name}"
        flags = [f"--{name}"]
        dashed = name.replace("_", "-")
        if dashed != name:
            flags.append(f"--{dashed}")
        base, optional = _unwrap_optional(tp)
        default = f.default if f.default is not dataclasses.MISSING else None
        shown = "auto" if (optional and default is None) else default
        help_text = f"{f.metadata.get('help', '')} [preset default: {shown}]".strip().replace("%", "%%")
        kwargs: dict[str, Any] = dict(dest=name, default=argparse.SUPPRESS, help=help_text)
        if base is bool:
            parser.add_argument(*flags, nargs="?", const=True, type=_parse_bool, metavar="BOOL", **kwargs)
        else:
            parser.add_argument(*flags, type=(lambda v, _tp=tp: coerce_value(_tp, v)),
                                metavar="AUTO|" + base.__name__.upper() if optional else base.__name__.upper(),
                                **kwargs)


def apply_overrides(cfg: Any, overrides: dict[str, Any]) -> Any:
    """Apply ``{"a.b": value}`` style overrides in place, coercing to the field types."""
    for key, value in overrides.items():
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise ConfigError(f"unknown config key {key!r}")
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not is_dataclass(obj) or leaf not in {f.name for f in fields(obj)}:
            raise ConfigError(f"unknown config key {key!r}")
        tp = _field_types(type(obj))[leaf]
        if is_dataclass(tp):
            if not isinstance(value, dict):
                raise ConfigError(f"config key {key!r} expects a mapping")
            apply_overrides(getattr(obj, leaf), value)
            continue
        try:
            setattr(obj, leaf, coerce_value(tp, value))
        except argparse.ArgumentTypeError as e:
            raise ConfigError(f"bad value for {key!r}: {e}") from None
    return cfg


def load_yaml_overrides(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def build_config(preset: Optional[str] = None, config_path: Optional[str] = None,
                 overrides: Optional[dict[str, Any]] = None) -> TrainConfig:
    """Build a validated TrainConfig from a preset, an optional YAML file and overrides."""
    yaml_data = load_yaml_overrides(config_path) if config_path else {}
    preset_name = preset or yaml_data.get("preset") or "sapia"
    cfg = get_preset(preset_name)
    if yaml_data:
        apply_overrides(cfg, yaml_data)
    if overrides:
        apply_overrides(cfg, overrides)
    cfg.validate()
    return cfg


def config_from_dict(data: dict) -> TrainConfig:
    """Rebuild a TrainConfig from ``TrainConfig.to_dict()`` output (e.g. from a checkpoint)."""
    cfg = TrainConfig()
    known = {f.name for f in fields(TrainConfig)}
    apply_overrides(cfg, {k: copy.deepcopy(v) for k, v in data.items() if k in known})
    return cfg


def model_config_from_dict(data: dict) -> ModelConfig:
    known = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in data.items() if k in known})


def parse_train_args(argv: Optional[list[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        prog="python -m sapia.train",
        description="Pretrain a decoder-only GPT. Precedence: preset < --config YAML < CLI flags. "
                    "Mixed precision defaults to bf16; use --precision fp16 or fp32 to change it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML file with overrides (see configs/).")
    add_config_args(parser, TrainConfig)
    ns = vars(parser.parse_args(argv))
    config_path = ns.pop("config", None)
    preset = ns.pop("preset", None)
    cfg = build_config(preset=preset, config_path=config_path, overrides=ns)
    return cfg
