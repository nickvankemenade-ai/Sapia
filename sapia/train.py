"""Training entry point.

    python -m sapia.train --preset tiny  --data_dir data/sapia-mix-tiny --out_dir runs/tiny
    python -m sapia.train --config configs/sapia.yaml                       # the real run

``data_dir`` is a prepared mix (``python -m sapia.prepare_mix``). Training reads the main mix
until ``(1 - anneal_fraction) * max_tokens`` tokens, then the anneal mix, while the
warmup-stable-decay schedule decays the learning rate over the same final stretch.
Rerunning the same command resumes from the newest checkpoint in ``out_dir``.
See ``python -m sapia.train --help`` for every option.
"""
from __future__ import annotations

import json
import math
import os
import platform
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .autoconfig import (AutoConfigError, BatchPlan, device_signature, free_cuda_memory, gpu_memory_budget,
                         heuristic_cpu_batch_size, is_oom_error, make_gpu_probe, plan_accumulation,
                         search_batch_size)
from .checkpoint import (capture_rng_state, clean_stale_tmp, list_checkpoints, load_checkpoint,
                         load_latest_checkpoint, read_best_meta, restore_rng_state, save_best_checkpoint,
                         save_checkpoint, strip_compile_prefix)
from .config import ConfigError, TrainConfig, parse_train_args
from .data import DataError, MixtureLoader, iter_eval_batches, load_manifest, mixture_val_sets
from .metrics_report import REPORT_NAME, TrainingMetrics
from .model import GPT
from .notify import Notifier
from .precision import PrecisionPlan, resolve_precision
from .resources import Resources, detect_resources
from .utils import (GIB, append_jsonl, atomic_write_bytes, atomic_write_json, close_logger, fmt_bytes, fmt_count,
                    fmt_duration, get_logger, read_json, read_jsonl, seed_everything)

METRICS_NAME = "metrics.jsonl"
MAX_OOM_RECOVERIES = 5

# Approximate peak dense FLOP/s with 16-bit tensor-core math, used only for the MFU estimate.
PEAK_FLOPS = [
    ("GH200", 989e12), ("H100", 989e12), ("H200", 989e12), ("A100", 312e12), ("L40S", 362e12), ("A10", 125e12), ("L4", 121e12),
    ("T4", 65e12), ("V100", 125e12), ("RTX 4090", 330e12), ("RTX 3090", 142e12),
]


class NonFiniteLossError(RuntimeError):
    """Training aborted because loss/gradients stayed non-finite. Not worth retrying as-is."""


def lookup_peak_flops(gpu_name: Optional[str]) -> Optional[float]:
    if not gpu_name:
        return None
    for key, value in PEAK_FLOPS:
        if key.lower() in gpu_name.lower():
            return value
    return None


def anneal_start(cfg: TrainConfig, horizon_tokens: int) -> int:
    """Token count at which the anneal phase (and the WSD decay) begins."""
    return int(horizon_tokens * (1.0 - cfg.anneal_fraction))


def lr_at(tokens_after_step: int, cfg: TrainConfig, horizon_tokens: int) -> float:
    """Learning rate as a function of tokens processed.

    ``wsd`` (default): linear warmup, constant peak, then a linear decay to the floor over the
    final ``anneal_fraction`` of the horizon. Because the decay only starts near the end,
    ``max_tokens`` can still be changed mid-run (before the decay) without harm.
    ``cosine``: linear warmup, then cosine decay over the whole horizon.
    Defining the schedule over tokens (not steps) keeps it identical when the batch size
    or accumulation changes between resumes.
    """
    peak = cfg.learning_rate
    floor = peak * cfg.min_lr_ratio
    warmup = min(cfg.warmup_tokens, horizon_tokens)
    if warmup > 0 and tokens_after_step < warmup:
        return peak * tokens_after_step / warmup
    if tokens_after_step >= horizon_tokens:
        return floor
    if cfg.lr_schedule == "wsd":
        start = max(warmup, anneal_start(cfg, horizon_tokens))
        if tokens_after_step < start:
            return peak
        progress = (tokens_after_step - start) / max(1, horizon_tokens - start)
        return peak + (floor - peak) * progress
    progress = (tokens_after_step - warmup) / max(1, horizon_tokens - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("device=cuda requested but CUDA is not available")
        if device.index is None:  # always carry an explicit index (torch.cuda.set_device needs one)
            device = torch.device("cuda", torch.cuda.current_device())
        elif device.index >= torch.cuda.device_count():
            raise ConfigError(f"device {name} requested but only {torch.cuda.device_count()} GPU(s) are visible")
    return device


@torch.no_grad()
def evaluate(model: GPT, val_sets: dict, batch_size: int, eval_tokens: int, device: torch.device,
             precision: PrecisionPlan) -> tuple[Optional[float], dict[str, float]]:
    """Cross-entropy per source on fixed val windows, plus the share-weighted mean.

    Each source gets ``eval_tokens * share`` tokens (at least one window). Returns
    ``(weighted mean or None, {source: loss})``.
    """
    if not val_sets:
        return None, {}
    was_training = model.training
    model.eval()
    per_source: dict[str, float] = {}
    for name, (share, shards) in val_sets.items():
        max_windows = max(1, int(eval_tokens * share) // shards.block_size)
        total, count = 0.0, 0
        for arr in iter_eval_batches(shards, batch_size, max_windows):
            buf = torch.from_numpy(arr).to(device)
            with precision.autocast(device.type):
                out = model(buf[:, :-1], buf[:, 1:])
            total += float(out.ce.item()) * arr.shape[0]
            count += arr.shape[0]
        per_source[name] = total / max(1, count)
    model.train(was_training)
    shares = {n: val_sets[n][0] for n in per_source}
    weighted = sum(per_source[n] * shares[n] for n in per_source) / max(1e-12, sum(shares.values()))
    return weighted, per_source


# ----------------------------------------------------------------------------- batch planning

def plan_batch(cfg: TrainConfig, model: GPT, device: torch.device, res: Resources, precision: PrecisionPlan,
               n_train_windows: int, ckpt: Optional[dict], log: Callable[[str], None]) -> BatchPlan:
    """Choose the micro-batch size and accumulation steps (config > checkpoint > probe)."""
    block = cfg.model.block_size
    target = cfg.target_tokens_per_step
    seqs_per_step = max(1, int(round(target / block)))
    gpu = res.gpus[device.index or 0] if (device.type == "cuda" and res.gpus) else None
    sig = device_signature(device, gpu)

    def finish(batch: int, accum: int, max_fit: Optional[int], source: str, reason: str,
               peak: Optional[int] = None, budget: Optional[int] = None) -> BatchPlan:
        return BatchPlan(batch_size=batch, grad_accum_steps=accum, tokens_per_step=batch * accum * block,
                         max_fit_batch_size=max_fit, source=source, reason=reason, device_signature=sig,
                         peak_bytes=peak, budget_bytes=budget)

    if cfg.batch_size is not None:
        batch = min(cfg.batch_size, n_train_windows)
        accum = cfg.grad_accum_steps or max(1, int(round(seqs_per_step / batch)))
        return finish(batch, accum, None, "config", f"batch_size={batch} set explicitly")

    # The micro-batch never needs to exceed the sequences in one optimizer step.
    cap = max(1, min(cfg.max_batch_size, seqs_per_step, n_train_windows))
    peak = budget = None

    prev = (ckpt or {}).get("batch_plan") if cfg.reuse_batch_plan else None
    reusable = bool(
        prev and prev.get("max_fit_batch_size") and prev.get("device_signature") == sig
        and (ckpt or {}).get("precision") == precision.mode
        and (ckpt or {}).get("train_config", {}).get("vram_fraction") == cfg.vram_fraction
    )
    if reusable and device.type == "cuda":
        budget = gpu_memory_budget(device, cfg.vram_fraction)
        # Reuse only if the peak measured back then still fits what is free right now.
        reusable = prev.get("peak_bytes") is not None and prev["peak_bytes"] <= budget

    if reusable:
        max_fit = min(int(prev["max_fit_batch_size"]), cap)
        peak = prev.get("peak_bytes")
        source = "checkpoint"
        reason = f"reusing the batch size probed earlier on the same GPU (max fit {max_fit})"
    elif device.type == "cuda":
        budget = gpu_memory_budget(device, cfg.vram_fraction)
        log(f"auto batch size: probing real forward+backward passes (cap {cap}, budget {fmt_bytes(budget)} = "
            f"{cfg.vram_fraction:.0%} of usable VRAM, precision {precision.mode})")
        probe, cleanup = make_gpu_probe(model, block, cfg.model.vocab_size, device, precision)
        try:
            result = search_batch_size(probe, budget, cap, log=log)
        finally:
            cleanup()
        max_fit, peak = result.batch_size, result.peak_bytes
        source = "probe"
        reason = (f"largest probed batch size within budget is {max_fit} "
                  f"(peak {fmt_bytes(peak)} of {fmt_bytes(budget)}, {len(result.probes)} probes)")
    else:
        max_fit = min(cap, heuristic_cpu_batch_size(cfg.model, res.mem_available_bytes, cap))
        source = "cpu-heuristic"
        reason = f"CPU heuristic from {fmt_bytes(res.mem_available_bytes)} available RAM -> {max_fit}"

    batch, accum = plan_accumulation(max_fit, block, target, fixed_accum=cfg.grad_accum_steps)
    return finish(batch, accum, max_fit, source, reason, peak, budget)


# ----------------------------------------------------------------------------- signals

class _StopFlag:
    def __init__(self) -> None:
        self.requested = False
        self.signal_name: Optional[str] = None


def _install_signal_handlers(stop: _StopFlag) -> dict:
    """First SIGINT/SIGTERM requests a graceful stop (checkpoint + exit); a second one aborts."""
    previous: dict = {}
    if threading.current_thread() is not threading.main_thread():
        return previous

    def handler(signum, _frame):
        if stop.requested:
            raise KeyboardInterrupt
        stop.requested = True
        try:
            stop.signal_name = signal.Signals(signum).name
        except ValueError:
            stop.signal_name = str(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_signal_handlers(previous: dict) -> None:
    for sig, old in previous.items():
        try:
            signal.signal(sig, old)
        except (ValueError, OSError):
            pass


def _oom_recovery_enabled(device: torch.device) -> bool:
    """Out-of-memory back-off only makes sense for CUDA allocations."""
    return device.type == "cuda"


def _process_rss() -> Optional[int]:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _truncate_metrics(path: Path, max_step: int) -> None:
    """Drop metric records newer than the resumed checkpoint so steps are never logged twice."""
    if not path.exists():
        return
    records = [r for r in read_jsonl(path) if r.get("step", 0) <= max_step]
    atomic_write_bytes(path, "".join(json.dumps(r) + "\n" for r in records).encode("utf-8"))


# ----------------------------------------------------------------------------- main training function

def train(cfg: TrainConfig, *, after_checkpoint: Optional[Callable[[Path], None]] = None,
          after_metrics: Optional[Callable[[Path], None]] = None,
          resources: Optional[Resources] = None, _stop_after_steps: Optional[int] = None) -> dict:
    """Train (or resume) according to ``cfg`` and return a summary dict.

    ``summary["status"]`` is one of:
      complete          the token/step budget was reached
      time_limit        ``max_hours`` for this invocation was reached (resume to continue)
      total_time_limit  ``max_total_hours`` across all resumes was reached
      interrupted       SIGINT/SIGTERM received; a checkpoint was written
      dry_run           ``dry_run`` was set; nothing was trained

    ``after_checkpoint(path)`` is called after every checkpoint (Modal commits its Volume there).
    ``after_metrics(path)`` is called whenever ``training_metrics.json`` was rewritten.
    ``_stop_after_steps`` is a test hook that simulates a stop signal after N steps.
    """
    cfg.validate()
    if cfg.tpu:
        from .train_tpu import train_tpu

        return train_tpu(cfg, _stop_after_steps=_stop_after_steps)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("sapia.train", out_dir / "train.log")
    try:
        return _train(cfg, out_dir, logger.info, after_checkpoint, after_metrics, resources, _stop_after_steps)
    finally:
        close_logger(logger)


def _train(cfg: TrainConfig, out_dir: Path, log: Callable[[str], None],
           after_checkpoint: Optional[Callable[[Path], None]],
           after_metrics: Optional[Callable[[Path], None]], resources: Optional[Resources],
           stop_after_steps: Optional[int] = None) -> dict:
    ckpt_dir = out_dir / "checkpoints"
    metrics_path = out_dir / METRICS_NAME
    block = cfg.model.block_size

    log("=" * 78)
    log(f"Sapia training | preset {cfg.preset} | torch {torch.__version__} | python {platform.python_version()} "
        f"| {platform.system()}")

    # ---- resources, device, precision
    res = resources if resources is not None else detect_resources()
    log("detected resources:")
    for line in res.summary_lines():
        log(f"  {line}")
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    precision = resolve_precision(cfg.precision, device.type)
    log(f"device: {device} | precision requested {precision.requested} -> using {precision.mode} "
        f"(GradScaler {'on' if precision.use_scaler else 'off'})")
    log(("WARNING: " if precision.fell_back else "precision: ") + precision.reason)

    # ---- data: a prepared mix with a main and an anneal phase
    mix_path = Path(cfg.data_dir) / "mix.json"
    if not mix_path.exists():
        raise DataError(f"no mix.json in {cfg.data_dir}; run python -m sapia.prepare_mix first")
    mix = read_json(mix_path)
    data_mode = "ram" if cfg.data_mode == "ram" else "memmap"
    phases = [p for p in ("main", "anneal") if p in mix["phases"]]
    if "main" not in phases:
        raise DataError(f"{mix_path} has no main phase")
    first_dir = Path(cfg.data_dir) / mix["phases"]["main"]["sources"][0]["dir"]
    manifest = load_manifest(first_dir, verify=False)
    if manifest["vocab_size"] > cfg.model.vocab_size:
        raise ConfigError(f"dataset vocab_size {manifest['vocab_size']} exceeds model.vocab_size {cfg.model.vocab_size}")
    for phase in phases:
        for src in mix["phases"][phase]["sources"]:
            status = "ready" if src["prepared"] else ("PARTIAL" if src["train_tokens"] else "MISSING")
            log(f"data {phase:6s} {src['name']:18s} share {src['share']:6.1%} | {fmt_count(src['train_tokens'])} of "
                f"{fmt_count(src['target_tokens'])} tokens [{status}]")
    ctx_main = cfg.main_block_size or block

    def phase_ctx(phase_name: str) -> int:
        """Training context per phase: main_block_size for the main mix, the full block_size after."""
        return block if phase_name == "anneal" else ctx_main

    log(f"context: {ctx_main} tokens in the main phase, {block} in the anneal phase")
    val_sets = mixture_val_sets(cfg.data_dir, "main", ctx_main, mode=data_mode)
    if not val_sets:
        log("WARNING: no usable validation data; val loss will not be reported")
    probe_loader = MixtureLoader(cfg.data_dir, "main", block, 1, cfg.seed, mode=data_mode)
    n_train_windows = sum(s.n_windows for s in probe_loader.shards)   # in full-length (block_size) windows
    n_train_tokens = sum(s.n_tokens for s in probe_loader.shards)
    probe_loader.close()

    # ---- model (allocated directly on the target device: no CPU-RAM copy of the weights)
    seed_everything(cfg.seed)
    with torch.device(device):
        model = GPT(cfg.model)
    n_params = model.num_params()
    log(f"model: {n_params:,} parameters ({n_params / 1e6:.1f}M) | {cfg.model.n_layer} layers, d_model "
        f"{cfg.model.d_model}, {cfg.model.n_head} heads, SwiGLU hidden {cfg.model.mlp_hidden}, context {block}, "
        f"vocab {cfg.model.vocab_size}")

    # ---- resume
    ckpt: Optional[dict] = None
    ckpt_path: Optional[Path] = None
    if cfg.resume == "auto":
        clean_stale_tmp(ckpt_dir)
        ckpt, ckpt_path = load_latest_checkpoint(ckpt_dir, log=log)
    elif cfg.resume == "never":
        if list_checkpoints(ckpt_dir):
            raise ConfigError(f"resume=never but {ckpt_dir} already contains checkpoints; "
                              "use a new --out_dir or delete them")
    elif Path(cfg.resume).is_dir():
        # A folder (e.g. the previous Kaggle session's output): the newest checkpoint in it.
        d = Path(cfg.resume)
        d = d if list_checkpoints(d) else d / "checkpoints"
        ckpt, ckpt_path = load_latest_checkpoint(d, log=log)
        if ckpt is None:
            raise ConfigError(f"no checkpoints found in {cfg.resume}")
    else:
        ckpt_path = Path(cfg.resume)
        ckpt = load_checkpoint(ckpt_path)
    if ckpt is not None:
        saved_model_cfg = ckpt.get("model_config", {})
        current = asdict(cfg.model)
        diffs = {k: (saved_model_cfg.get(k), v) for k, v in current.items() if saved_model_cfg.get(k) != v}
        if diffs:
            raise ConfigError(f"checkpoint {ckpt_path} was trained with a different model config "
                              f"(checkpoint vs now): {diffs}. Use a new --out_dir or matching settings.")
        model.load_state_dict(strip_compile_prefix(ckpt["model"]))
        log(f"resuming from {ckpt_path} (step {ckpt['step']}, {ckpt.get('tokens_seen', 0):,} tokens seen)")
    else:
        log("starting a new run (no checkpoint found)" if cfg.resume == "auto" else "starting a new run")

    # ---- batch plan (before the optimizer exists, so probing sees the true free memory)
    bplan = plan_batch(cfg, model, device, res, precision, n_train_windows, ckpt, log)
    log(f"batch plan [{bplan.source}]: micro-batch {bplan.batch_size} x accum {bplan.grad_accum_steps} x "
        f"context {block} = {bplan.tokens_per_step:,} tokens/optimizer step "
        f"(target {cfg.target_tokens_per_step:,}) | {bplan.reason}")

    step = int(ckpt["step"]) if ckpt else 0
    tokens_seen = int(ckpt.get("tokens_seen", 0)) if ckpt else 0
    horizon = cfg.max_tokens
    if cfg.max_steps is not None:
        horizon = min(horizon, tokens_seen + max(0, cfg.max_steps - step) * bplan.tokens_per_step)
    if cfg.warmup_tokens > horizon:
        log(f"WARNING: warmup_tokens ({cfg.warmup_tokens:,}) exceeds the training horizon ({horizon:,}); "
            "warmup is shortened to the horizon")
    est_steps = max(0, math.ceil((horizon - tokens_seen) / bplan.tokens_per_step))
    switch = anneal_start(cfg, horizon) if "anneal" in phases else horizon
    log(f"schedule: {horizon:,} total tokens, about {est_steps:,} optimizer steps remaining, {cfg.lr_schedule} lr "
        f"(peak {cfg.learning_rate:g}, floor {cfg.learning_rate * cfg.min_lr_ratio:g}), warmup "
        f"{min(cfg.warmup_tokens, horizon):,} tokens, anneal phase from {switch:,} tokens")

    if cfg.dry_run:
        log("dry_run set: exiting before training")
        return {"status": "dry_run", "step": step, "tokens_seen": tokens_seen, "n_params": n_params,
                "batch_size": bplan.batch_size, "grad_accum_steps": bplan.grad_accum_steps,
                "tokens_per_step": bplan.tokens_per_step, "precision": precision.mode,
                "data_mode": data_mode, "out_dir": str(out_dir)}

    # ---- optimizer, scaler, restored state
    optimizer = model.configure_optimizer(cfg.learning_rate, cfg.weight_decay, (cfg.beta1, cfg.beta2),
                                          cfg.adam_eps, device.type, optimizer=cfg.optimizer,
                                          momentum=cfg.muon_momentum)
    log(f"optimizer: {cfg.optimizer} | " + ", ".join(
        f"{g.get('name', i)}: {sum(p.numel() for p in g['params']):,} params" for i, g in enumerate(optimizer.param_groups)))
    fused = optimizer.defaults.get("fused", None)
    scaler = precision.make_scaler()
    loader_states: dict = {}
    best_val = float("inf")
    last_val: Optional[float] = None
    train_time_prev = 0.0
    if ckpt is not None:
        saved_opt = ckpt.get("train_config", {}).get("optimizer", "adamw")
        if saved_opt != cfg.optimizer:
            raise ConfigError(f"the checkpoint was trained with optimizer={saved_opt!r}; resume with the same "
                              f"optimizer or start a new --out_dir")
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:
            # Current settings win over the saved hyperparameters; fused must match this device.
            group["weight_decay"] = 0.0 if group.get("name") == "adamw_no_decay" else cfg.weight_decay
            group["betas"] = (cfg.beta1, cfg.beta2)
            group["eps"] = cfg.adam_eps
            if group.get("use_muon"):
                group["momentum"] = cfg.muon_momentum
            if fused is not None:
                group["fused"] = fused
        if precision.use_scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        elif precision.use_scaler:
            log("note: the checkpoint has no GradScaler state (it was not trained in fp16); starting a fresh scaler")
        loader_states = dict(ckpt.get("loader_states") or {})
        best_val = float(ckpt.get("best_val_loss", best_val))
        # best.pt may be newer than the resumed checkpoint (an eval ran after the last periodic
        # save); never let a worse model overwrite it.
        best_meta = read_best_meta(ckpt_dir)
        if best_meta is not None and float(best_meta["val_loss"]) < best_val:
            best_val = float(best_meta["val_loss"])
        last_val = ckpt.get("last_val_loss")
        train_time_prev = float(ckpt.get("train_time_s", 0.0))
        restore_rng_state(ckpt.get("rng"), log)
        _truncate_metrics(metrics_path, step)
    ckpt = None  # release the CPU copy of weights and optimizer state
    free_cuda_memory()

    atomic_write_json(out_dir / "config.json", {
        "train_config": cfg.to_dict(), "n_params": n_params, "precision": precision.mode,
        "batch_plan": bplan.to_dict(), "data_mode": data_mode, "resources": res.to_dict(),
        "torch": torch.__version__, "started_unix": time.time(),
    })

    train_model = model
    compiled = False
    if cfg.compile:
        if platform.system() == "Windows":
            log("WARNING: --compile ignored on Windows (torch.compile needs Triton, which is unavailable there)")
        else:
            log("compiling the model with torch.compile (the first steps will be slow)")
            train_model = torch.compile(model)
            compiled = True

    notify = Notifier(cfg.ntfy_topic, run=out_dir.name, log=log)
    wandb_run = None
    wandb_ready = bool(os.environ.get("WANDB_API_KEY")) or (
        Path.home().joinpath(".netrc").exists() and "api.wandb.ai" in Path.home().joinpath(".netrc").read_text())
    if cfg.wandb and not wandb_ready:
        # wandb.init would stop and ask for a key on the terminal, stalling an unattended run.
        log("WARNING: wandb=true but no WANDB_API_KEY / wandb login found; W&B logging is off for this run")
    if cfg.wandb and wandb_ready:
        try:
            import wandb

            run_id_path = out_dir / "wandb_run_id.txt"
            run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
            wandb_run = wandb.init(project=cfg.wandb_project, name=out_dir.name, id=run_id, config=cfg.to_dict(),
                                   resume="allow")
            run_id_path.write_text(wandb_run.id)
        except Exception as exc:  # noqa: BLE001 - wandb is strictly optional
            log(f"WARNING: wandb logging disabled ({type(exc).__name__}: {exc})")

    gpu_name = res.gpus[device.index or 0].name if (device.type == "cuda" and res.gpus) else None
    peak_flops = cfg.peak_flops or lookup_peak_flops(gpu_name)
    flops_per_token = cfg.model.flops_per_token(ctx_main)
    params = [p for p in model.parameters() if p.requires_grad]

    # ---- training_metrics.json (single-document summary, read by the dashboard)
    report = TrainingMetrics(out_dir / REPORT_NAME, {
        "preset": cfg.preset, "out_dir": str(out_dir), "data_dir": cfg.data_dir, "n_params": n_params,
        "model": asdict(cfg.model), "precision": precision.mode, "device": str(device), "gpu": gpu_name,
        "peak_flops": peak_flops, "batch_size": bplan.batch_size, "grad_accum_steps": bplan.grad_accum_steps,
        "tokens_per_step": bplan.tokens_per_step, "horizon_tokens": horizon, "learning_rate": cfg.learning_rate,
        "warmup_tokens": cfg.warmup_tokens, "eval_interval": cfg.eval_interval, "log_interval": cfg.log_interval,
        "train_tokens_available": n_train_tokens, "started_unix": time.time(),
        "torch": torch.__version__,
    }, existing=read_jsonl(metrics_path) if step > 0 else (), min_interval=cfg.metrics_seconds)
    report.set_progress(step=step, tokens_seen=tokens_seen, horizon_tokens=horizon)

    def write_report(force: bool = False) -> None:
        try:
            written = report.write(force=force)
        except Exception as exc:  # noqa: BLE001 - metrics must never stop training
            log(f"WARNING: writing {REPORT_NAME} failed: {type(exc).__name__}: {exc}")
            return
        if written and after_metrics is not None:
            try:
                after_metrics(report.path)
            except Exception as exc:  # noqa: BLE001
                log(f"WARNING: after_metrics hook failed: {type(exc).__name__}: {exc}")

    write_report(force=True)

    def phase_for(tokens: int) -> str:
        return "anneal" if ("anneal" in phases and tokens >= anneal_start(cfg, horizon)) else "main"

    def make_loader(phase: str) -> MixtureLoader:
        # The micro-batch is planned in full-length sequences; shorter contexts take more of them,
        # so the tokens per micro-batch (and per optimizer step) stay the same.
        ctx = phase_ctx(phase)
        return MixtureLoader(cfg.data_dir, phase, ctx, bplan.batch_size * (block // ctx), cfg.seed, mode=data_mode,
                             state=loader_states.get(phase))

    phase = phase_for(tokens_seen)
    loader = make_loader(phase)
    flops_per_token = cfg.model.flops_per_token(phase_ctx(phase))
    log(f"data phase: {phase} ({', '.join(f'{n} {w:.1%}' for n, w in zip(loader.names, loader.weights))})")
    t_loop = time.time()
    t_loop_pc = time.perf_counter()  # high-resolution clock for the time limits (time.time is coarse on Windows)

    def train_time() -> float:
        return train_time_prev + (time.time() - t_loop)

    last_ckpt_step = step
    last_ckpt_time = time.time()
    last_ckpt_path: Optional[Path] = ckpt_path

    def save(reason: str) -> Path:
        nonlocal last_ckpt_step, last_ckpt_time, last_ckpt_path
        t = time.time()
        state = {
            "model": strip_compile_prefix(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if precision.use_scaler else None,
            "tokens_seen": tokens_seen,
            "loader_states": {**loader_states, phase: loader.state()},
            "phase": phase,
            "rng": capture_rng_state(),
            "model_config": asdict(cfg.model),
            "train_config": cfg.to_dict(),
            "batch_plan": bplan.to_dict(),
            "precision": precision.mode,
            "tokenizer": manifest["tokenizer"],
            "eot_token": manifest["eot_token"],
            "best_val_loss": best_val,
            "last_val_loss": last_val,
            "train_time_s": train_time(),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
        }
        path = save_checkpoint(ckpt_dir, step, state, keep_last=cfg.keep_last)
        last_ckpt_step, last_ckpt_time, last_ckpt_path = step, time.time(), path
        log(f"checkpoint saved: {path.name} ({fmt_bytes(path.stat().st_size)}, {time.time() - t:.1f}s) [{reason}]")
        if after_checkpoint is not None:
            try:
                after_checkpoint(path)
            except Exception as exc:  # noqa: BLE001 - a failing hook must not kill training
                log(f"WARNING: after_checkpoint hook failed: {type(exc).__name__}: {exc}")
        return path

    def save_best(val_loss: float) -> None:
        t = time.time()
        state = {
            "model": strip_compile_prefix(model.state_dict()),
            "tokens_seen": tokens_seen,
            "model_config": asdict(cfg.model),
            "train_config": cfg.to_dict(),
            "precision": precision.mode,
            "tokenizer": manifest["tokenizer"],
            "eot_token": manifest["eot_token"],
            "best_val_loss": val_loss,
            "last_val_loss": val_loss,
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
        }
        try:
            path = save_best_checkpoint(ckpt_dir, step, val_loss, state)
        except Exception as exc:  # noqa: BLE001 - a failed best save must not kill training
            log(f"WARNING: saving best.pt failed: {type(exc).__name__}: {exc}")
            return
        log(f"best model saved: {path.name} (val loss {val_loss:.4f}, step {step}, "
            f"{fmt_bytes(path.stat().st_size)}, {time.time() - t:.1f}s)")
        if after_checkpoint is not None:
            try:
                after_checkpoint(path)
            except Exception as exc:  # noqa: BLE001
                log(f"WARNING: after_checkpoint hook failed: {type(exc).__name__}: {exc}")

    def run_eval(tag: str) -> Optional[float]:
        nonlocal best_val, last_val
        t = time.time()
        val_loss, per_source = evaluate(model, val_sets, bplan.batch_size, cfg.eval_tokens, device, precision)
        if val_loss is None:
            return None
        last_val = val_loss
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        if improved and cfg.save_best:
            save_best(val_loss)
        log(f"eval  step {step:>7d} | val loss {val_loss:.4f} (best {best_val:.4f}) | ppl {math.exp(min(val_loss, 20)):.1f} "
            f"| {time.time() - t:.1f}s [{tag}] | " + ", ".join(f"{n} {v:.3f}" for n, v in per_source.items()))
        eval_record = {"kind": "eval", "step": step, "tokens_seen": tokens_seen,
                       "val_loss": val_loss, "best_val_loss": best_val, "time_unix": time.time(),
                       "val_by_source": per_source}
        append_jsonl(metrics_path, eval_record)
        report.add(eval_record)
        write_report(force=True)
        if wandb_run is not None:
            wandb_run.log({"val/loss": val_loss, **{f"val_source/{n}": v for n, v in per_source.items()}},
                          step=step)
        return val_loss

    # ---- training loop
    stop = _StopFlag()
    previous_handlers = _install_signal_handlers(stop)
    status = "complete"
    first_loss: Optional[float] = None
    last_loss: Optional[float] = None
    window_losses: list[float] = []
    window_tokens = 0
    window_t0 = time.time()
    consecutive_bad = 0
    skipped_total = 0
    oom_recoveries = 0
    steps_this_run = 0
    last_eval_step = -1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    log(f"training starts at step {step}, {tokens_seen:,} tokens seen, phase {phase}")
    notify.send(f"Training {'resumed' if step else 'started'} at step {step} "
                f"({fmt_count(tokens_seen)}/{fmt_count(horizon)} tokens, {phase} phase).")
    next_milestone = (tokens_seen // cfg.notify_every_tokens + 1) * cfg.notify_every_tokens

    try:
        while True:
            if tokens_seen >= horizon or (cfg.max_steps is not None and step >= cfg.max_steps):
                status = "complete"
                break
            if stop_after_steps is not None and steps_this_run >= stop_after_steps and not stop.requested:
                stop.requested, stop.signal_name = True, "test hook"
            if stop.requested:
                status = "interrupted"
                break
            elapsed = time.perf_counter() - t_loop_pc
            # The cumulative cap is checked first: when both limits are hit together the run
            # must end for good ("total_time_limit") rather than ask to be resumed.
            if cfg.max_total_hours is not None and train_time_prev + elapsed >= cfg.max_total_hours * 3600:
                status = "total_time_limit"
                break
            if cfg.max_hours is not None and elapsed >= cfg.max_hours * 3600:
                status = "time_limit"
                break

            wanted = phase_for(tokens_seen)
            if wanted != phase:
                loader_states[phase] = loader.state()
                loader.close()
                phase = wanted
                loader = make_loader(phase)
                flops_per_token = cfg.model.flops_per_token(phase_ctx(phase))
                log(f"data phase: {phase} from {tokens_seen:,} tokens "
                    f"({', '.join(f'{n} {w:.1%}' for n, w in zip(loader.names, loader.weights))})")
                notify.send(f"Switched to the {phase} phase at {fmt_count(tokens_seen)} tokens; lr now decays.")

            lr = lr_at(tokens_seen + bplan.tokens_per_step, cfg, horizon)
            for group in optimizer.param_groups:
                group["lr"] = lr

            accum = bplan.grad_accum_steps
            step_loss = 0.0
            bad_reason: Optional[str] = None
            grad_norm = float("nan")
            step_start_state = loader.state()
            new_state = step_start_state
            step_ce = 0.0
            oom = False
            aborted = False
            try:
                for _micro in range(accum):
                    if stop.requested:
                        aborted = True
                        break
                    tokens, _draws = next(loader)
                    new_state = loader.state()
                    if bad_reason is not None:
                        continue  # keep consuming so every step uses exactly `accum` batches
                    buf = tokens.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else tokens
                    x, y = buf[:, :-1], buf[:, 1:]
                    try:
                        with precision.autocast(device.type):
                            out = train_model(x, y)
                    except Exception as exc:  # noqa: BLE001
                        if not (compiled and steps_this_run == 0 and not is_oom_error(exc)):
                            raise
                        # torch.compile can fail on new platforms (e.g. ARM); fall back to eager mode.
                        log(f"WARNING: torch.compile failed ({type(exc).__name__}: {str(exc)[:300]}); "
                            "continuing without compile")
                        notify.send("torch.compile failed; continuing without it (slower).")
                        train_model, compiled = model, False
                        with precision.autocast(device.type):
                            out = train_model(x, y)
                    loss = out.loss
                    step_ce += float(out.ce.item()) / accum
                    loss_value = float(loss.item())
                    if not math.isfinite(loss_value):
                        bad_reason = f"non-finite loss ({loss_value})"
                        continue
                    step_loss += loss_value / accum
                    scaler.scale(loss / accum).backward()
                if not aborted and bad_reason is None:
                    scaler.unscale_(optimizer)
                    max_norm = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
                    if math.isfinite(grad_norm):
                        scaler.step(optimizer)
                        scaler.update()
                    elif precision.use_scaler:
                        # fp16 overflow in the scaled gradients: GradScaler skips the update
                        # and lowers its scale. Normal now and then, a problem if it persists.
                        scaler.step(optimizer)
                        scaler.update()
                        bad_reason = f"non-finite gradients (GradScaler scale lowered to {scaler.get_scale():g})"
                    else:
                        bad_reason = "non-finite gradient norm"
            except Exception as exc:  # noqa: BLE001 - only OOM is handled; everything else propagates
                if not (_oom_recovery_enabled(device) and is_oom_error(exc)):
                    raise
                oom = True
            optimizer.zero_grad(set_to_none=True)

            if aborted:
                status = "interrupted"
                break

            if oom:
                # Out of memory despite the probe (fragmentation, another process, ...): shrink
                # the micro-batch, keep the tokens per step, and redo this step from the same data.
                oom_recoveries += 1
                loader_states[phase] = step_start_state   # redo this step from the same data
                loader.close()
                try:
                    getattr(scaler, "_per_optimizer_states", {}).clear()
                except Exception:
                    pass
                free_cuda_memory()
                if bplan.batch_size <= 1 or oom_recoveries > MAX_OOM_RECOVERIES:
                    raise AutoConfigError(
                        f"CUDA out of memory at micro-batch {bplan.batch_size} after {oom_recoveries} recoveries. "
                        "Free GPU memory, lower --vram_fraction, or set --batch_size manually."
                    )
                new_max = max(1, int(bplan.batch_size * 0.8))
                batch, accum2 = plan_accumulation(new_max, block, cfg.target_tokens_per_step,
                                                  fixed_accum=cfg.grad_accum_steps)
                log(f"WARNING: CUDA out of memory at micro-batch {bplan.batch_size}; backing off to "
                    f"{batch} x accum {accum2} and retrying the step (recovery {oom_recoveries}/{MAX_OOM_RECOVERIES})")
                bplan = BatchPlan(batch_size=batch, grad_accum_steps=accum2, tokens_per_step=batch * accum2 * block,
                                  max_fit_batch_size=new_max, source="oom-backoff",
                                  reason=f"reduced after an out-of-memory error (was {bplan.batch_size})",
                                  device_signature=bplan.device_signature, peak_bytes=None,
                                  budget_bytes=bplan.budget_bytes)
                loader = make_loader(phase)
                continue

            # The step's data was consumed whether or not the update was applied.
            step += 1
            tokens_seen += bplan.tokens_per_step
            window_tokens += bplan.tokens_per_step
            loader_states[phase] = new_state

            if bad_reason is not None:
                consecutive_bad += 1
                skipped_total += 1
                log(f"WARNING: step {step} skipped: {bad_reason} "
                    f"[{consecutive_bad}/{cfg.max_nan_skips} consecutive]")
                if consecutive_bad >= cfg.max_nan_skips:
                    notify.send(f"Training stopped: {consecutive_bad} non-finite steps in a row at step {step}.",
                                priority="high")
                    raise NonFiniteLossError(
                        f"{consecutive_bad} consecutive optimizer steps were skipped ({bad_reason}); aborting at "
                        f"step {step}. No update was applied from these steps and the last checkpoint is intact. "
                        f"Things to try: --precision bf16 or fp32 (currently {precision.mode}), a lower "
                        "--learning_rate, a longer --warmup_tokens, or a smaller --grad_clip."
                    )
            else:
                consecutive_bad = 0
                last_loss = step_loss
                window_losses.append(step_ce)
                if first_loss is None:
                    first_loss = step_loss

            steps_this_run += 1
            if step % cfg.log_interval == 0 or steps_this_run == 1:
                now = time.time()
                dt = max(1e-9, now - window_t0)
                tok_per_sec = window_tokens / dt
                mfu = (flops_per_token * tok_per_sec / peak_flops) if peak_flops else None
                mean_loss = float(np.mean(window_losses)) if window_losses else float("nan")
                record = {
                    "kind": "train", "step": step, "tokens_seen": tokens_seen, "phase": phase,
                    "loss": mean_loss, "loss_with_z": step_loss, "lr": lr, "grad_norm": grad_norm, "tok_per_sec": tok_per_sec, "mfu": mfu,
                    "skipped_steps": skipped_total, "train_time_s": train_time(), "time_unix": now,
                    "batch_size": bplan.batch_size, "grad_accum_steps": bplan.grad_accum_steps,
                }
                mem = ""
                if device.type == "cuda":
                    alloc = torch.cuda.max_memory_allocated(device)
                    reserved = torch.cuda.memory_reserved(device)
                    record["gpu_mem_peak_alloc_gb"] = alloc / GIB
                    record["gpu_mem_reserved_gb"] = reserved / GIB
                    mem = f" | vram {alloc / GIB:.2f}G peak, {reserved / GIB:.2f}G reserved"
                rss = _process_rss()
                if rss is not None:
                    record["ram_rss_gb"] = rss / GIB
                    mem += f" | ram {rss / GIB:.2f}G"
                if precision.use_scaler:
                    record["scaler_scale"] = scaler.get_scale()
                append_jsonl(metrics_path, record)
                remaining = max(0, horizon - tokens_seen)
                eta = remaining / tok_per_sec if tok_per_sec > 0 else float("inf")
                report.add(record)
                report.set_progress(step=step, tokens_seen=tokens_seen, horizon_tokens=horizon,
                                    tok_per_sec=tok_per_sec, mfu=mfu, eta_s=eta, train_time_s=train_time(),
                                    batch_size=bplan.batch_size, grad_accum_steps=bplan.grad_accum_steps)
                write_report()
                log(f"step {step:>7d} | loss {mean_loss:.4f} | lr {lr:.2e} | gnorm {grad_norm:.2f} | "
                    f"{fmt_count(tok_per_sec)} tok/s"
                    + (f" | mfu {100 * mfu:.1f}%" if mfu is not None else "")
                    + mem
                    + f" | {fmt_count(tokens_seen)}/{fmt_count(horizon)} tok | eta {fmt_duration(eta)}")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": mean_loss, "train/loss_with_z": step_loss, "train/lr": lr,
                                   "train/tok_per_sec": tok_per_sec, "train/grad_norm": grad_norm,
                                   "train/mfu": mfu or 0.0, "train/tokens": tokens_seen,
                                   "train/anneal": float(phase == "anneal")}, step=step)
                window_losses, window_tokens, window_t0 = [], 0, time.time()

            t_side = time.time()
            if step % cfg.eval_interval == 0:
                run_eval("periodic")
                last_eval_step = step
            due_by_time = cfg.ckpt_minutes > 0 and (time.time() - last_ckpt_time) >= cfg.ckpt_minutes * 60
            if step % cfg.ckpt_interval == 0 or due_by_time:
                save("periodic" if not due_by_time else "time-based")
            if tokens_seen >= next_milestone:
                next_milestone += cfg.notify_every_tokens
                notify.send(f"{fmt_count(tokens_seen)}/{fmt_count(horizon)} tokens, step {step}, train loss "
                            f"{last_loss if last_loss is None else round(last_loss, 3)}, val loss "
                            f"{last_val if last_val is None else round(last_val, 3)}.")
            # Keep evaluation / checkpoint time out of the throughput window.
            window_t0 += time.time() - t_side
    except BaseException as exc:
        if not isinstance(exc, KeyboardInterrupt):
            notify.send(f"Training CRASHED at step {step}: {type(exc).__name__}: {str(exc)[:200]}", priority="high")
        try:
            report.finish("failed", error=f"{type(exc).__name__}: {exc}", step=step, tokens_seen=tokens_seen,
                          train_time_s=train_time())
            if after_metrics is not None:
                after_metrics(report.path)
        except Exception:  # noqa: BLE001 - never mask the original error
            pass
        raise
    finally:
        _restore_signal_handlers(previous_handlers)
        loader.close()

    # ---- wrap up
    if status == "interrupted":
        log(f"stop requested ({stop.signal_name}); saving a checkpoint at step {step}")
    elif status in ("time_limit", "total_time_limit"):
        log(f"{status.replace('_', ' ')} reached at step {step}; saving a checkpoint")
    if status == "complete" and steps_this_run == 0:
        log("nothing to do: this run already reached its token/step budget")
    if status == "complete" and last_eval_step != step:
        run_eval("final")
    if step != last_ckpt_step or last_ckpt_path is None:
        save(status)
    for _name, (_share, shards) in val_sets.items():
        shards.close()
    final_path = None
    if status == "complete":
        from .export import export_weights

        final_path = export_weights(last_ckpt_path, out_dir / "final.pt")
        log(f"final weights exported: {final_path} ({fmt_bytes(final_path.stat().st_size)})")
    notify.send(f"Training {status} at step {step}: {fmt_count(tokens_seen)} tokens, val loss "
                f"{last_val if last_val is None else round(last_val, 3)}."
                + (f" Weights: {final_path}" if final_path else ""), priority="high" if status == "complete" else "default")
    if wandb_run is not None:
        wandb_run.finish()

    summary = {
        "status": status, "step": step, "tokens_seen": tokens_seen, "horizon_tokens": horizon,
        "first_loss": first_loss, "last_loss": last_loss, "val_loss": last_val,
        "best_val_loss": best_val if math.isfinite(best_val) else None,
        "n_params": n_params, "precision": precision.mode, "batch_size": bplan.batch_size,
        "grad_accum_steps": bplan.grad_accum_steps, "tokens_per_step": bplan.tokens_per_step,
        "batch_plan_source": bplan.source, "data_mode": data_mode, "final_weights": str(final_path) if final_path else None,
        "skipped_steps": skipped_total, "train_time_s": train_time(),
        "checkpoint": str(last_ckpt_path) if last_ckpt_path else None, "out_dir": str(out_dir),
    }
    try:
        report.finish(status, step=step, tokens_seen=tokens_seen, horizon_tokens=horizon,
                      train_time_s=summary["train_time_s"], eta_s=0.0 if status == "complete" else None)
        if after_metrics is not None:
            after_metrics(report.path)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: writing {REPORT_NAME} failed: {type(exc).__name__}: {exc}")
    summary["metrics_report"] = str(report.path)
    log(f"finished: status={status}, step={step}, tokens={tokens_seen:,}, last train loss "
        f"{last_loss if last_loss is None else round(last_loss, 4)}, val loss "
        f"{last_val if last_val is None else round(last_val, 4)}, train time {fmt_duration(summary['train_time_s'])}")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    try:
        cfg = parse_train_args(argv)
        summary = train(cfg)
    except (ConfigError, DataError, AutoConfigError, NonFiniteLossError) as exc:
        print(f"\nERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    return 0 if summary["status"] in ("complete", "dry_run") else 3


if __name__ == "__main__":
    sys.exit(main())
