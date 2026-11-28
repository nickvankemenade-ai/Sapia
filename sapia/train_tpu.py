"""Training on TPUs (e.g. Kaggle's TPU v5e-8) with PyTorch/XLA.

    python -m sapia.train --tpu --config configs/sapia.yaml --data_dir /tmp/sapia-mix \\
        --out_dir /kaggle/working/run --resume /kaggle/input/<previous-session>/run/checkpoints \\
        --max_hours 8.3

How it uses the hardware
------------------------
* **SPMD, one process, all chips.** Every parameter (and, through XLA's sharding
  propagation, its gradient and optimizer state) is split across the chips along its first
  dimension, and every batch along the batch dimension: fully sharded data parallelism done
  by the XLA compiler. A 1B model's ~9 GB of training state becomes ~1.1 GB per chip.
* **Whole-step compilation and fusion.** Nothing executes until ``sync()``: each micro-batch
  (forward + backward) and each optimizer step (clipping + Muon/AdamW) is traced into one
  graph that XLA compiles once and fuses (norms, RoPE, SwiGLU, softmax, the optimizer maths).
  Shapes never change within a phase, so after the first steps nothing recompiles; the
  anneal phase (longer context) compiles once more.
* **Flash attention.** With ``tpu_flash`` the attention runs in the Pallas TPU flash kernel
  (head size padded to a multiple of 128 when needed). If the kernel fails to compile on the
  first step, training continues with XLA's fused attention.
* **No host round trips in the hot loop.** Loss and gradient norm stay on the TPU and are
  read only every ``log_interval`` steps.

Stopping and continuing
-----------------------
Kaggle sessions end after ~9-12 hours. Pass ``--max_hours`` a little below the limit (8.3 for
a 9-hour session): the run saves a checkpoint and exits cleanly when it is reached, and also
saves every ``ckpt_minutes``. The next session resumes with ``--resume <folder or file>``
(the previous session's checkpoints, attached as a Kaggle input): weights, optimizer state,
the data position in both phases, the RNG state and the token count all continue exactly.
Checkpoints are written as ordinary CPU PyTorch files, so they also load on CUDA or CPU.

The loop is written against a tiny backend interface so its logic can be tested on CPU.
"""
from __future__ import annotations

import json
import math
import os
import platform
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from . import model as model_mod
from .checkpoint import (capture_rng_state, clean_stale_tmp, list_checkpoints, load_checkpoint,
                         load_latest_checkpoint, restore_rng_state, save_checkpoint, strip_compile_prefix)
from .config import ConfigError, TrainConfig
from .data import DataError, MixtureLoader, load_manifest, mixture_val_sets
from .model import GPT
from .notify import Notifier
from .utils import (GIB, append_jsonl, atomic_write_json, close_logger, fmt_bytes, fmt_count, fmt_duration,
                    get_logger, read_json, read_jsonl, seed_everything)

TPU_V5E_PEAK_BF16 = 197e12       # per chip, dense bf16


# ----------------------------------------------------------------------------- backends

class CPUBackend:
    """Stand-in used by the tests: same loop, no XLA."""

    name = "cpu"

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.num_devices = 1
        self.peak_flops = None

    def sync(self) -> None:
        pass

    def shard_batch(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def shard_params(self, model: torch.nn.Module, log: Callable[[str], None]) -> None:
        pass

    def autocast(self):
        return torch.autocast("cpu", enabled=False)

    def shard_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    def attention(self, flash: bool, log: Callable[[str], None]):
        return None

    def describe(self) -> str:
        return "CPU backend (test mode)"


class XLABackend:
    """PyTorch/XLA on TPU with SPMD: one process drives every chip."""

    name = "xla"

    def __init__(self) -> None:
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.spmd as xs
            import torch_xla.runtime as xr
        except ImportError as exc:  # pragma: no cover - only on machines without torch_xla
            raise ConfigError("--tpu needs PyTorch/XLA. On Kaggle (TPU VM) install it with:\n"
                              "  pip install 'torch~=2.6.0' 'torch_xla[tpu]~=2.6.0' -f "
                              "https://storage.googleapis.com/libtpu-releases/index.html") from exc
        self.torch_xla, self.xm, self.xs, self.xr = torch_xla, xm, xs, xr
        xr.use_spmd()
        self.device = xm.xla_device()
        self.num_devices = xr.global_runtime_device_count()
        self.mesh = xs.Mesh(np.arange(self.num_devices), (self.num_devices, 1), ("fsdp", "model"))
        try:
            xs.set_global_mesh(self.mesh)
        except AttributeError:  # older torch_xla
            pass
        try:
            kind = self.xm.xla_device_hw(self.device)
        except Exception:  # noqa: BLE001
            kind = "TPU"
        self.kind = kind
        self.peak_flops = TPU_V5E_PEAK_BF16 * self.num_devices

    def sync(self) -> None:
        self.xm.mark_step()

    def shard_batch(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(self.device)
        self.xs.mark_sharding(t, self.mesh, ("fsdp",) + (None,) * (t.dim() - 1))
        return t

    def shard_params(self, model: torch.nn.Module, log: Callable[[str], None]) -> None:
        """Split every parameter along dim 0 across the chips (replicate the few that don't divide)."""
        n, sharded, replicated = self.num_devices, 0, 0
        for name, p in model.named_parameters():
            if p.dim() >= 1 and p.shape[0] % n == 0 and p.numel() >= 4096:
                self.xs.mark_sharding(p, self.mesh, ("fsdp",) + (None,) * (p.dim() - 1))
                sharded += p.numel()
            else:
                replicated += p.numel()
        log(f"sharding: {sharded:,} parameters split over {n} chips, {replicated:,} replicated")

    def autocast(self):
        return torch.autocast("xla", dtype=torch.bfloat16)

    def shard_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        """Give loaded optimizer state (momentum, Adam moments) the same split as its parameter.

        Without this, state restored from a checkpoint would be replicated on every chip.
        """
        n = self.num_devices
        for p, st in optimizer.state.items():
            for v in st.values():
                if torch.is_tensor(v) and v.shape == p.shape and v.dim() >= 1 and v.shape[0] % n == 0                         and v.numel() >= 4096:
                    self.xs.mark_sharding(v, self.mesh, ("fsdp",) + (None,) * (v.dim() - 1))

    def attention(self, flash: bool, log: Callable[[str], None]):
        """Causal attention for training: Pallas flash kernel, or XLA's fused attention."""
        import torch.nn.functional as F

        def repeat_kv(q, k, v):
            H, KV = q.size(1), k.size(1)
            if KV != H:
                k = k.repeat_interleave(H // KV, dim=1)
                v = v.repeat_interleave(H // KV, dim=1)
            return k, v

        def xla_attention(q, k, v):
            k, v = repeat_kv(q, k, v)
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if not flash:
            log("attention: XLA fused attention")
            return xla_attention
        try:
            from torch_xla.experimental.custom_kernel import flash_attention
        except ImportError:
            log("attention: Pallas flash kernel unavailable in this torch_xla; using XLA fused attention")
            return xla_attention
        mesh = self.mesh

        def flash_attn(q, k, v):
            k, v = repeat_kv(q, k, v)
            hd = q.size(-1)
            pad = (-hd) % 128                      # the TPU kernel wants a multiple of 128
            if pad:
                q, k, v = (torch.nn.functional.pad(t, (0, pad)) for t in (q, k, v))
            y = flash_attention(q, k, v, causal=True, sm_scale=1.0 / math.sqrt(hd),
                                partition_spec=("fsdp", None, None, None), mesh=mesh)
            return y[..., :hd] if pad else y

        log("attention: Pallas flash attention kernel (falls back to XLA attention if it fails to compile)")
        flash_attn.fallback = xla_attention
        return flash_attn

    def describe(self) -> str:
        return f"{self.num_devices} x {self.kind} (SPMD, fully sharded), torch_xla {self.torch_xla.__version__}"


# ----------------------------------------------------------------------------- helpers

def to_cpu(obj):
    """Copy a (nested) state to CPU tensors so checkpoints are plain PyTorch files."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def plan_tpu_batch(cfg: TrainConfig, num_devices: int) -> dict:
    """Micro-batch and accumulation for a fixed number of tokens per chip per micro-batch.

    Tokens per micro-batch (and so per optimizer step) are the same in both phases: the
    main phase packs more, shorter sequences.
    """
    block = cfg.model.block_size
    ctx_main = cfg.main_block_size or block
    per_chip = cfg.tpu_tokens_per_chip
    if per_chip % block:
        raise ConfigError(f"tpu_tokens_per_chip ({per_chip}) must be a multiple of block_size ({block})")
    micro_tokens = per_chip * num_devices
    accum = max(1, round(cfg.target_tokens_per_step / micro_tokens))
    return {"micro_tokens": micro_tokens, "accum": accum, "tokens_per_step": micro_tokens * accum,
            "seqs": {"main": micro_tokens // ctx_main, "anneal": micro_tokens // block},
            "ctx": {"main": ctx_main, "anneal": block}}


def find_resume(cfg: TrainConfig, ckpt_dir: Path, log: Callable[[str], None]) -> tuple[Optional[dict], Optional[Path]]:
    """resume=auto: newest checkpoint in out_dir; a folder: newest checkpoint in it; a file: that file."""
    if cfg.resume == "never":
        if list_checkpoints(ckpt_dir):
            raise ConfigError(f"resume=never but {ckpt_dir} already contains checkpoints")
        return None, None
    if cfg.resume == "auto":
        clean_stale_tmp(ckpt_dir)
        return load_latest_checkpoint(ckpt_dir, log=log)
    p = Path(cfg.resume)
    if p.is_dir():
        for d in (p, p / "checkpoints"):
            if list_checkpoints(d):
                return load_latest_checkpoint(d, log=log)
        raise ConfigError(f"no checkpoints found in {p}")
    return load_checkpoint(p), p


# ----------------------------------------------------------------------------- training

def train_tpu(cfg: TrainConfig, backend=None, _stop_after_steps: Optional[int] = None) -> dict:
    cfg.validate()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("sapia.train_tpu", out_dir / "train.log")
    try:
        return _train_tpu(cfg, out_dir, logger.info, backend or XLABackend(), _stop_after_steps)
    finally:
        close_logger(logger)


def _train_tpu(cfg: TrainConfig, out_dir: Path, log: Callable[[str], None], be,
               stop_after_steps: Optional[int]) -> dict:
    from .train import anneal_start, lr_at

    ckpt_dir = out_dir / "checkpoints"
    metrics_path = out_dir / "metrics.jsonl"
    block = cfg.model.block_size
    log("=" * 78)
    log(f"Sapia TPU training | preset {cfg.preset} | {be.describe()} | torch {torch.__version__} | "
        f"python {platform.python_version()}")

    # ---- data
    mix_path = Path(cfg.data_dir) / "mix.json"
    if not mix_path.exists():
        raise DataError(f"no mix.json in {cfg.data_dir} (on Kaggle: python -m sapia.kaggle merge ...)")
    mix = read_json(mix_path)
    phases = [p for p in ("main", "anneal") if p in mix["phases"]]
    first = Path(cfg.data_dir) / mix["phases"]["main"]["sources"][0]["dir"]
    manifest = load_manifest(first, verify=False)
    if manifest["vocab_size"] > cfg.model.vocab_size:
        raise ConfigError(f"dataset vocab_size {manifest['vocab_size']} exceeds model.vocab_size")
    plan = plan_tpu_batch(cfg, be.num_devices)
    log(f"batch: {plan['seqs']['main']} x {plan['ctx']['main']} (main) / {plan['seqs']['anneal']} x "
        f"{plan['ctx']['anneal']} (anneal) tokens per micro-batch over {be.num_devices} chip(s), accum "
        f"{plan['accum']} = {plan['tokens_per_step']:,} tokens/step")
    val_sets = mixture_val_sets(cfg.data_dir, "main", plan["ctx"]["main"], mode="memmap")

    # ---- model (built on CPU, loaded, then moved and sharded)
    seed_everything(cfg.seed)
    model = GPT(cfg.model)
    n_params = model.num_params()
    log(f"model: {n_params:,} parameters | {cfg.model.n_layer} layers x {cfg.model.d_model}, "
        f"{cfg.model.n_head}/{cfg.model.n_kv_head} heads, optimizer {cfg.optimizer}")
    ckpt, ckpt_path = find_resume(cfg, ckpt_dir, log)
    if ckpt is not None:
        saved = ckpt.get("model_config", {})
        diffs = {k: (saved.get(k), v) for k, v in asdict(cfg.model).items() if saved.get(k) != v}
        if diffs:
            raise ConfigError(f"checkpoint {ckpt_path} has a different model config: {diffs}")
        if ckpt.get("train_config", {}).get("optimizer", "adamw") != cfg.optimizer:
            raise ConfigError("the checkpoint was trained with another optimizer")
        model.load_state_dict(strip_compile_prefix(ckpt["model"]))
        log(f"resuming from {ckpt_path} (step {ckpt['step']}, {ckpt.get('tokens_seen', 0):,} tokens)")
    else:
        log("starting a new run")
    model.to(be.device)
    be.shard_params(model, log)
    attn = be.attention(cfg.tpu_flash, log)
    model_mod.ATTENTION_OVERRIDE = attn

    optimizer = model.configure_optimizer(cfg.learning_rate, cfg.weight_decay, (cfg.beta1, cfg.beta2),
                                          cfg.adam_eps, be.device.type, optimizer=cfg.optimizer,
                                          momentum=cfg.muon_momentum)
    step = int(ckpt["step"]) if ckpt else 0
    tokens_seen = int(ckpt.get("tokens_seen", 0)) if ckpt else 0
    loader_states: dict = dict(ckpt.get("loader_states") or {}) if ckpt else {}
    best_val = float(ckpt.get("best_val_loss", float("inf"))) if ckpt else float("inf")
    last_val = ckpt.get("last_val_loss") if ckpt else None
    train_time_prev = float(ckpt.get("train_time_s", 0.0)) if ckpt else 0.0
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        be.shard_optimizer_state(optimizer)
        restore_rng_state(ckpt.get("rng"), log)
        if metrics_path.exists():
            recs = [r for r in read_jsonl(metrics_path) if r.get("step", 0) <= step]
            metrics_path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    ckpt = None

    horizon = cfg.max_tokens
    switch = anneal_start(cfg, horizon) if "anneal" in phases else horizon
    log(f"schedule: {horizon:,} tokens, {cfg.lr_schedule} lr peak {cfg.learning_rate:g}, anneal from {switch:,}")
    atomic_write_json(out_dir / "config.json", {"train_config": cfg.to_dict(), "n_params": n_params,
                                                "backend": be.describe(), "plan": plan, "torch": torch.__version__})
    notify = Notifier(cfg.ntfy_topic, run=out_dir.name, log=log)
    pinger = notify.every(cfg.notify_minutes)

    def phase_for(tokens: int) -> str:
        return "anneal" if tokens >= switch and "anneal" in phases else "main"

    def make_loader(ph: str) -> MixtureLoader:
        loader = MixtureLoader(cfg.data_dir, ph, plan["ctx"][ph], plan["seqs"][ph], cfg.seed,
                               state=loader_states.get(ph), phase_tokens=switch if ph == "main" else horizon - switch,
                               max_epochs=cfg.max_source_epochs)
        log(f"data mix ({ph}): {loader.describe()}")
        return loader

    phase = phase_for(tokens_seen)
    loader = make_loader(phase)
    params = [p for p in model.parameters() if p.requires_grad]

    def save(reason: str) -> Path:
        be.sync()
        t = time.time()
        state = {
            "model": to_cpu(strip_compile_prefix(model.state_dict())),
            "optimizer": to_cpu(optimizer.state_dict()),
            "scaler": None, "step": step, "tokens_seen": tokens_seen,
            "loader_states": {**loader_states, phase: loader.state()}, "phase": phase,
            "rng": capture_rng_state(), "model_config": asdict(cfg.model), "train_config": cfg.to_dict(),
            "precision": "bf16", "tokenizer": manifest["tokenizer"], "eot_token": manifest["eot_token"],
            "best_val_loss": best_val, "last_val_loss": last_val, "train_time_s": train_time(),
            "versions": {"torch": torch.__version__, "backend": be.describe()},
        }
        path = save_checkpoint(ckpt_dir, step, state, keep_last=cfg.keep_last)
        log(f"checkpoint saved: {path.name} ({fmt_bytes(path.stat().st_size)}, {time.time() - t:.1f}s) [{reason}]")
        return path

    @torch.no_grad()
    def evaluate() -> tuple[Optional[float], dict]:
        if not val_sets:
            return None, {}
        model.eval()
        per = {}
        bsz = plan["seqs"]["main"]
        for name, (share, shards) in val_sets.items():
            n = (min(shards.n_windows, max(bsz, int(cfg.eval_tokens * share) // shards.block_size)) // bsz) * bsz
            if n == 0:
                continue
            total = None
            for start in range(0, n, bsz):          # fixed batch shape: compiled once
                buf = be.shard_batch(torch.from_numpy(shards.batch(np.arange(start, start + bsz))))
                with be.autocast():
                    ce = model(buf[:, :-1], buf[:, 1:]).ce
                total = ce if total is None else total + ce
                be.sync()
            per[name] = float((total / (n // bsz)).item())
        model.train()
        shares = {n: val_sets[n][0] for n in per}
        return (sum(per[n] * shares[n] for n in per) / max(1e-12, sum(shares.values())) if per else None), per

    # ---- stop handling
    stop = {"requested": False}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: stop.update(requested=True))
            except (ValueError, OSError):
                pass

    t_loop = time.time()

    def train_time() -> float:
        return train_time_prev + (time.time() - t_loop)

    status, steps_run = "complete", 0
    last_ckpt = time.time()
    window_t0, window_tokens = time.time(), 0
    loss_window = None
    flops_tok = cfg.model.flops_per_token(plan["ctx"][phase])
    next_milestone = (tokens_seen // cfg.notify_every_tokens + 1) * cfg.notify_every_tokens
    model.train()
    log(f"training starts at step {step}, {tokens_seen:,} tokens, phase {phase}")
    notify.send(f"TPU training {'resumed' if step else 'started'} at step {step} ({fmt_count(tokens_seen)} tokens).")

    try:
        while True:
            if tokens_seen >= horizon or (cfg.max_steps is not None and step >= cfg.max_steps):
                break
            elapsed = time.time() - t_loop
            if stop["requested"] or (stop_after_steps is not None and steps_run >= stop_after_steps):
                status = "interrupted"
                break
            if cfg.max_total_hours is not None and train_time() >= cfg.max_total_hours * 3600:
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
                flops_tok = cfg.model.flops_per_token(plan["ctx"][phase])
                log(f"data phase: {phase} from {tokens_seen:,} tokens (context {plan['ctx'][phase]})")
                notify.send(f"Switched to the {phase} phase at {fmt_count(tokens_seen)} tokens.")

            lr = lr_at(tokens_seen + plan["tokens_per_step"], cfg, horizon)
            for group in optimizer.param_groups:
                group["lr"] = lr
            step_loss = None
            for _ in range(plan["accum"]):
                tokens, _draws = next(loader)
                buf = be.shard_batch(tokens)
                try:
                    with be.autocast():
                        out = model(buf[:, :-1], buf[:, 1:])
                    (out.loss / plan["accum"]).backward()
                    be.sync()
                except Exception as exc:  # noqa: BLE001
                    fallback = getattr(model_mod.ATTENTION_OVERRIDE, "fallback", None)
                    if steps_run > 0 or fallback is None:
                        raise
                    log(f"WARNING: flash attention failed to compile ({type(exc).__name__}: {str(exc)[:300]}); "
                        "using XLA fused attention")
                    model_mod.ATTENTION_OVERRIDE = fallback
                    optimizer.zero_grad(set_to_none=True)
                    with be.autocast():
                        out = model(buf[:, :-1], buf[:, 1:])
                    (out.loss / plan["accum"]).backward()
                    be.sync()
                ce = out.ce / plan["accum"]
                step_loss = ce if step_loss is None else step_loss + ce
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            be.sync()
            step += 1
            steps_run += 1
            tokens_seen += plan["tokens_per_step"]
            window_tokens += plan["tokens_per_step"]
            loader_states[phase] = loader.state()
            loss_window = step_loss.detach() if loss_window is None else loss_window + step_loss.detach()

            if step % cfg.log_interval == 0 or steps_run == 1:
                n_win = cfg.log_interval if steps_run > 1 else 1
                loss_val = float(loss_window.item()) / max(1, min(n_win, steps_run))
                gn = float(grad_norm.item())
                loss_window = None
                dt = max(1e-9, time.time() - window_t0)
                tps = window_tokens / dt
                mfu = flops_tok * tps / be.peak_flops if be.peak_flops else None
                if not math.isfinite(loss_val):
                    notify.send(f"TPU training stopped: non-finite loss at step {step}.", priority="high")
                    raise RuntimeError(f"non-finite loss at step {step}; resume from the last checkpoint with a "
                                       "lower --learning_rate")
                rec = {"kind": "train", "step": step, "tokens_seen": tokens_seen, "phase": phase, "loss": loss_val,
                       "lr": lr, "grad_norm": gn, "tok_per_sec": tps, "mfu": mfu, "train_time_s": train_time(),
                       "time_unix": time.time()}
                append_jsonl(metrics_path, rec)
                pinger.maybe(step, tokens_seen, horizon, loss_val, tps, last_val, mfu)
                eta = (horizon - tokens_seen) / tps if tps > 0 else float("inf")
                log(f"step {step:>7d} | loss {loss_val:.4f} | lr {lr:.2e} | gnorm {gn:.2f} | {fmt_count(tps)} tok/s"
                    + (f" | mfu {100 * mfu:.1f}%" if mfu else "")
                    + f" | {fmt_count(tokens_seen)}/{fmt_count(horizon)} tok | eta {fmt_duration(eta)}")
                window_t0, window_tokens = time.time(), 0

            t_side = time.time()
            if step % cfg.eval_interval == 0:
                val, per = evaluate()
                if val is not None:
                    last_val, best_val = val, min(best_val, val)
                    log(f"eval  step {step:>7d} | val loss {val:.4f} (best {best_val:.4f}) | "
                        + ", ".join(f"{n} {v:.3f}" for n, v in per.items()))
                    append_jsonl(metrics_path, {"kind": "eval", "step": step, "tokens_seen": tokens_seen,
                                                "val_loss": val, "val_by_source": per, "time_unix": time.time()})
            due = cfg.ckpt_minutes > 0 and time.time() - last_ckpt >= cfg.ckpt_minutes * 60
            if step % cfg.ckpt_interval == 0 or due:
                save("periodic" if not due else "time-based")
                last_ckpt = time.time()
            if tokens_seen >= next_milestone:
                next_milestone += cfg.notify_every_tokens
                notify.send(f"{fmt_count(tokens_seen)}/{fmt_count(horizon)} tokens, step {step}, val "
                            f"{last_val if last_val is None else round(last_val, 3)}.")
            window_t0 += time.time() - t_side
    except BaseException as exc:
        if not isinstance(exc, KeyboardInterrupt):
            notify.send(f"TPU training CRASHED at step {step}: {type(exc).__name__}: {str(exc)[:200]}", priority="high")
        raise
    finally:
        model_mod.ATTENTION_OVERRIDE = None

    if status == "complete" and steps_run:
        val, per = evaluate()
        if val is not None:
            last_val, best_val = val, min(best_val, val)
            log(f"eval  step {step:>7d} | val loss {val:.4f} [final]")
    path = save(status)
    final = None
    if status == "complete":
        from .export import export_weights

        final = export_weights(path, out_dir / "final.pt")
        log(f"final weights exported: {final} ({fmt_bytes(final.stat().st_size)})")
    loader.close()
    msg = (f"TPU session ended ({status}) at step {step}, {fmt_count(tokens_seen)}/{fmt_count(horizon)} tokens, "
           f"val {last_val if last_val is None else round(last_val, 3)}.")
    log(msg)
    notify.send(msg + (" Start the next session from this output." if status != "complete" else " Done!"),
                priority="high" if status == "complete" else "default")
    return {"status": status, "step": step, "tokens_seen": tokens_seen, "val_loss": last_val, "checkpoint": str(path),
            "final_weights": str(final) if final else None, "tokens_per_step": plan["tokens_per_step"],
            "n_params": n_params}
