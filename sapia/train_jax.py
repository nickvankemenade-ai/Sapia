"""Training on TPUs with JAX (``python -m sapia.train --tpu-jax``), e.g. Kaggle's TPU v5e-8.

    python -m sapia.train --tpu-jax --config configs/sapia.yaml --data_dir /tmp/sapia-mix \\
        --out_dir /kaggle/working/run --resume /kaggle/input/<previous-session>/run/checkpoints \\
        --max_hours 8.3

Same model, optimizer, data order, schedule and checkpoint format as ``--tpu`` (PyTorch/XLA),
``--gpu`` and ``--cpu``: a run can stop in one mode and continue in another.

How it uses the hardware
------------------------
* **One compiled function per optimizer step.** Gradient accumulation (a ``lax.scan`` over the
  micro-batches), clipping and the Muon/AdamW update are a single ``jax.jit`` program; XLA
  fuses it end to end. The 28 transformer blocks are one ``lax.scan`` too, so compiling
  takes one block's worth of time. The step's inputs are donated: no second copy of the
  weights or optimizer state is ever allocated.
* **Fully sharded over all chips.** Weights, gradients and optimizer state are split across
  the chips (FSDP through XLA's GSPMD partitioner); each batch is split along the batch axis.
* **Flash attention.** On TPU the Pallas flash-attention kernel runs per chip (head size
  padded to 128); if it fails to compile, the step is rebuilt with XLA's fused attention.
* **Asynchronous.** The host prepares the next batch while the TPU runs the current step;
  loss and gradient norm are fetched only every ``log_interval`` steps.

Never losing progress
---------------------
* Checkpoints every ``ckpt_interval`` steps and every ``ckpt_minutes``, on ``--max_hours``,
  on SIGTERM/SIGINT, and (when the weights are still finite) after a crash.
* Each file is written atomically, read back and checksummed; older checkpoints are deleted
  only after that. Resuming verifies the checksum and falls back to an older file if needed.
* The data position, optimizer state, token count, best val loss and training time are all
  in the checkpoint, so the next session continues exactly.
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

from .config import ConfigError, TrainConfig
from .data import DataError, MixtureLoader, load_manifest, mixture_val_sets
from .notify import Notifier
from .utils import (append_jsonl, atomic_write_json, close_logger, fmt_bytes, fmt_count, fmt_duration, get_logger,
                    read_json, read_jsonl, seed_everything)

# libtpu flags (as used by MaxText on v5e) that let XLA overlap collectives with compute.
TPU_PERF_FLAGS = ("--xla_tpu_enable_data_parallel_all_reduce_opt=true "
                  "--xla_tpu_data_parallel_opt_different_sized_ops=true "
                  "--xla_tpu_enable_async_collective_fusion=true "
                  "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
                  "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
                  "--xla_tpu_overlap_compute_collective_tc=true "
                  "--xla_enable_async_all_gather=true")

# Dense bf16 peak per JAX device, by device_kind (lower case, substring match).
PEAK_FLOPS = (("v6", 918e12), ("v5p", 459e12), ("v5 lite", 197e12), ("v5e", 197e12), ("v4", 275e12),
              ("v3", 61.5e12), ("v2", 22.5e12))


def _peak_flops(kind: str) -> Optional[float]:
    kind = kind.lower()
    for key, value in PEAK_FLOPS:
        if key in kind:
            return value
    return None


def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - only without jax
        raise ConfigError("--tpu-jax needs JAX. On a Kaggle TPU VM: pip install -U 'jax[tpu]' -f "
                          "https://storage.googleapis.com/jax-releases/libtpu_releases.html") from exc
    return jax, jnp


def _shard_map():
    try:
        from jax import shard_map             # newer JAX
    except ImportError:
        from jax.experimental.shard_map import shard_map
    return shard_map


def _smap(f, mesh, in_specs, out_specs):
    """shard_map without replication checks (the argument was renamed across JAX versions)."""
    shard_map = _shard_map()
    try:
        return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
    except TypeError:
        return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)


def _fsdp_axis(spec) -> Optional[int]:
    return next((i for i, s in enumerate(tuple(spec)) if s == "fsdp"), None)


def train_jax(cfg: TrainConfig, _stop_after_steps: Optional[int] = None, _allow_cpu: bool = False,
              _bench: bool = False) -> dict:
    """Train (or resume). ``_bench``: time the steps only (no evals, no checkpoints)."""
    cfg.validate()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.tpu_xla_flags and "jax" not in __import__("sys").modules:
        current = os.environ.get("LIBTPU_INIT_ARGS", "")
        if "async_collective_fusion" not in current:
            os.environ["LIBTPU_INIT_ARGS"] = (current + " " + TPU_PERF_FLAGS).strip()
    logger = get_logger("sapia.train_jax", out_dir / "train.log")
    try:
        return _train_jax(cfg, out_dir, logger.info, _stop_after_steps, _allow_cpu, _bench)
    finally:
        close_logger(logger)


def _param_spec(key: str, shape: tuple, n: int):
    """FSDP split: the first evenly divisible matrix axis (never the stacked layer axis)."""
    from jax.sharding import PartitionSpec as P

    if n == 1:
        return P()
    if key.startswith("blocks."):
        if len(shape) == 3:
            for ax in (1, 2):
                if shape[ax] % n == 0:
                    return P(*[("fsdp" if i == ax else None) for i in range(3)])
        return P()                                         # stacked norms: tiny, replicated
    if len(shape) == 2:
        for ax in (0, 1):
            if shape[ax] % n == 0:
                return P(*[("fsdp" if i == ax else None) for i in range(2)])
    return P()


def _make_attention(jax, jnp, use_flash: bool, mesh, on_tpu: bool, log: Callable[[str], None]):
    """(attention for sharded global arrays, attention for per-device arrays, flash on?)."""
    from . import jax_model as jm

    if not use_flash or not on_tpu:
        log("attention: XLA fused attention")
        return jm.xla_attention, jm.xla_attention, False
    try:
        from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
        from jax.sharding import PartitionSpec as P
    except ImportError as exc:
        log(f"attention: Pallas flash kernel unavailable ({exc}); using XLA fused attention")
        return jm.xla_attention, jm.xla_attention, False

    def wrap(kernel_call):
        def attn(q, k, v):                                 # (B, T, H, D) in, (B, T, H, D) out
            H, KV, hd = q.shape[2], k.shape[2], q.shape[3]
            if KV != H:
                k = jnp.repeat(k, H // KV, axis=2)
                v = jnp.repeat(v, H // KV, axis=2)
            q, k, v = (jnp.swapaxes(t, 1, 2) for t in (q, k, v))
            pad = (-hd) % 128                              # the kernel wants a multiple of 128
            if pad:
                q, k, v = (jnp.pad(t, ((0, 0), (0, 0), (0, 0), (0, pad))) for t in (q, k, v))
            y = kernel_call(q, k, v, 1.0 / math.sqrt(hd))
            if pad:
                y = y[..., :hd]
            return jnp.swapaxes(y, 1, 2)
        return attn

    local = lambda q, k, v, scale: flash_attention(q, k, v, causal=True, sm_scale=scale)

    def sharded(q, k, v, scale):
        fn = _smap(lambda a, b, c: local(a, b, c, scale), mesh, (P("fsdp"),) * 3, P("fsdp"))
        return fn(q, k, v)

    log("attention: Pallas flash attention kernel (falls back to XLA attention if it fails to compile)")
    return wrap(sharded), wrap(local), True


def _train_jax(cfg: TrainConfig, out_dir: Path, log: Callable[[str], None], stop_after_steps: Optional[int],
               allow_cpu: bool, bench: bool = False) -> dict:
    jax, jnp = _import_jax()
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from . import jax_model as jm
    from .checkpoint import capture_rng_state, restore_rng_state, save_checkpoint
    from .train import anneal_start, lr_at
    from .train_tpu import find_resume, plan_tpu_batch

    ckpt_dir = out_dir / "checkpoints"
    metrics_path = out_dir / "metrics.jsonl"
    mc = cfg.model

    devices = jax.devices()
    platform_name = devices[0].platform
    on_tpu = platform_name == "tpu"
    if not on_tpu and not allow_cpu:
        raise ConfigError(f"--tpu-jax found no TPU (JAX sees {len(devices)} {platform_name} device(s)). On Kaggle, "
                          "choose Accelerator: TPU v5e-8 and install jax[tpu] (see docs/KAGGLE.md).")
    n_dev = len(devices)
    kind = getattr(devices[0], "device_kind", platform_name)
    per_dev = _peak_flops(kind) if on_tpu else None
    peak = per_dev * n_dev if per_dev else None
    describe = f"{n_dev} x {kind} (JAX {jax.__version__}, fully sharded)"
    mesh = Mesh(np.array(devices), ("fsdp",))
    log("=" * 78)
    log(f"Sapia JAX training | preset {cfg.preset} | {describe} | python {platform.python_version()}")

    # ---- data
    mix_path = Path(cfg.data_dir) / "mix.json"
    if not mix_path.exists():
        raise DataError(f"no mix.json in {cfg.data_dir} (on Kaggle: python -m sapia.kaggle merge ...)")
    mix = read_json(mix_path)
    phases = [p for p in ("main", "anneal") if p in mix["phases"]]
    first = Path(cfg.data_dir) / mix["phases"]["main"]["sources"][0]["dir"]
    manifest = load_manifest(first, verify=False)
    if manifest["vocab_size"] > mc.vocab_size:
        raise ConfigError(f"dataset vocab_size {manifest['vocab_size']} exceeds model.vocab_size")
    plan = plan_tpu_batch(cfg, n_dev)
    for ph in ("main", "anneal"):
        if plan["seqs"][ph] % n_dev:
            raise ConfigError(f"{plan['seqs'][ph]} sequences per micro-batch ({ph}) do not split over {n_dev} "
                              "devices; raise --tpu_tokens_per_chip")
    log(f"batch: {plan['seqs']['main']} x {plan['ctx']['main']} (main) / {plan['seqs']['anneal']} x "
        f"{plan['ctx']['anneal']} (anneal) tokens per micro-batch over {n_dev} device(s), accum "
        f"{plan['accum']} = {plan['tokens_per_step']:,} tokens/step")
    val_sets = mixture_val_sets(cfg.data_dir, "main", plan["ctx"]["main"], mode="memmap")

    # ---- weights and optimizer state (host numpy first, then sharded onto the devices)
    n_params = mc.num_params()
    log(f"model: {n_params:,} parameters | {mc.n_layer} layers x {mc.d_model}, {mc.n_head}/{mc.n_kv_head} heads, "
        f"optimizer {cfg.optimizer} | sharding {cfg.tpu_sharding}, remat {cfg.tpu_remat}")
    groups, layout = jm.optimizer_layout(mc, cfg.optimizer, cfg.learning_rate, cfg.weight_decay,
                                         (cfg.beta1, cfg.beta2), cfg.adam_eps, cfg.muon_momentum)
    ckpt, ckpt_path = find_resume(cfg, ckpt_dir, log)
    if ckpt is not None:
        saved = ckpt.get("model_config", {})
        diffs = {k: (saved.get(k), v) for k, v in asdict(mc).items() if saved.get(k) != v}
        if diffs:
            raise ConfigError(f"checkpoint {ckpt_path} has a different model config: {diffs}")
        if ckpt.get("train_config", {}).get("optimizer", "adamw") != cfg.optimizer:
            raise ConfigError("the checkpoint was trained with another optimizer")
        from .checkpoint import strip_compile_prefix

        host_params = jm.params_from_torch(strip_compile_prefix(ckpt["model"]), mc)
        host_opt = jm.opt_state_from_torch(ckpt["optimizer"], host_params, mc, layout, cfg.optimizer)
        origin = ckpt.get("versions", {}).get("backend", "unknown backend")
        log(f"resuming from {ckpt_path} (step {ckpt['step']}, {ckpt.get('tokens_seen', 0):,} tokens; saved by {origin})")
    else:
        host_params = jm.init_params(mc, cfg.seed)
        host_opt = jm.init_opt_state(host_params, cfg.optimizer)
        log("starting a new run")
    seed_everything(cfg.seed)

    p_sh = {k: NamedSharding(mesh, _param_spec(k, v.shape, n_dev)) for k, v in host_params.items()}
    o_sh = {"mu": {k: p_sh[k] for k in host_opt["mu"]}, "m": {k: p_sh[k] for k in host_opt["m"]},
            "v": {k: p_sh[k] for k in host_opt["v"]}, "step": NamedSharding(mesh, P())}
    rep = NamedSharding(mesh, P())
    batch_sh = NamedSharding(mesh, P(None, "fsdp", None))
    eval_sh = NamedSharding(mesh, P("fsdp", None))
    params = jax.device_put(host_params, p_sh)
    opt = jax.device_put(host_opt, o_sh)
    n_split = sum(int(np.prod(v.shape)) for k, v in host_params.items() if p_sh[k].spec != P())
    log(f"sharding: {n_split:,} parameter values split over {n_dev} device(s), the rest replicated")
    del host_params, host_opt

    step = int(ckpt["step"]) if ckpt else 0
    tokens_seen = int(ckpt.get("tokens_seen", 0)) if ckpt else 0
    loader_states: dict = dict(ckpt.get("loader_states") or {}) if ckpt else {}
    best_val = float(ckpt.get("best_val_loss", float("inf"))) if ckpt else float("inf")
    last_val = ckpt.get("last_val_loss") if ckpt else None
    train_time_prev = float(ckpt.get("train_time_s", 0.0)) if ckpt else 0.0
    if ckpt is not None:
        restore_rng_state(ckpt.get("rng"), log)
        if metrics_path.exists():
            recs = [r for r in read_jsonl(metrics_path) if r.get("step", 0) <= step]
            metrics_path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    ckpt = None

    # ---- compiled functions
    dtype = jnp.float32 if cfg.precision == "fp32" else jnp.bfloat16
    if cfg.precision == "fp16":
        log("note: fp16 is not used with JAX; computing in bf16")

    specs = {k: s.spec for k, s in p_sh.items()}
    f32 = jnp.float32

    def compute_copy(p):
        """bf16 copies of the matrices that go through the MXU (as PyTorch autocast); the rest stays f32."""
        return {k: (v.astype(dtype) if k in jm.HIDDEN else v) for k, v in p.items()}

    def build(attn_global, attn_local):
        def loss_with(attn):
            return lambda p, toks: jm.forward_loss(p, toks, mc, attn, dtype, cfg.tpu_remat)

        loss_global = loss_with(attn_global)
        grad_global = jax.value_and_grad(loss_global, has_aux=True)
        grad_local = jax.value_and_grad(loss_with(attn_local), has_aux=True)

        def scan_accumulate(grad_fn, cp, batch, zeros, reduce):
            accum = batch.shape[0]

            def micro(carry, toks):
                acc, ce_acc = carry
                (_, ce), g = grad_fn(cp, toks)
                g = reduce(g)
                acc = jax.tree_util.tree_map(lambda a, b: a + b.astype(f32) / accum, acc, g)
                return (acc, ce_acc + ce / accum), None

            (acc, ce), _ = jax.lax.scan(micro, (zeros, jnp.zeros((), f32)), batch)
            return acc, ce

        def scatter(g):
            """Per-device partial gradients -> this device's shard of the mean gradient."""
            out = {}
            for k, x in g.items():
                ax = _fsdp_axis(specs[k])
                x = jax.lax.psum(x, "fsdp") if ax is None else jax.lax.psum_scatter(
                    x, "fsdp", scatter_dimension=ax, tiled=True)
                out[k] = x / n_dev
            return out

        def local_shape(k, shape):
            ax = _fsdp_axis(specs[k])
            return tuple(d // n_dev if i == ax else d for i, d in enumerate(shape))

        def gradients(p, batch):
            mode = cfg.tpu_sharding
            if mode == "fsdp":            # GSPMD: sharded weights gathered by XLA inside every micro-batch
                zeros = jax.lax.with_sharding_constraint(jax.tree_util.tree_map(lambda v: jnp.zeros(v.shape, f32), p),
                                                         p_sh)
                return scan_accumulate(grad_global, p, batch, zeros, lambda g: g)
            # zero2 / dp: bf16 weights gathered once per step, then explicit per-device code.
            cp = jax.lax.with_sharding_constraint(compute_copy(p), {k: rep for k in p})

            def per_device(cp, b):
                if mode == "zero2":       # reduce-scatter every micro-batch, accumulate the shard
                    zeros = {k: jnp.zeros(local_shape(k, v.shape), f32) for k, v in cp.items()}
                    acc, ce = scan_accumulate(grad_local, cp, b, zeros, scatter)
                else:                     # dp: accumulate locally, one reduce-scatter per step
                    zeros = {k: jnp.zeros(v.shape, f32) for k, v in cp.items()}
                    acc, ce = scan_accumulate(grad_local, cp, b, zeros, lambda g: g)
                    acc = scatter(acc)
                return acc, jax.lax.pmean(ce, "fsdp")

            fn = _smap(per_device, mesh, ({k: P() for k in p}, P(None, "fsdp", None)), (specs, P()))
            return fn(cp, batch)

        def train_step(p, o, batch, lr):
            grads, ce = gradients(p, batch)
            p, o, gnorm = jm.apply_updates(p, grads, o, lr, mc, cfg.optimizer, cfg.weight_decay,
                                           (cfg.beta1, cfg.beta2), cfg.adam_eps, cfg.muon_momentum, cfg.grad_clip)
            return p, o, {"ce": ce, "grad_norm": gnorm}

        step_fn = jax.jit(train_step, in_shardings=(p_sh, o_sh, batch_sh, rep), out_shardings=(p_sh, o_sh, rep),
                          donate_argnums=(0, 1))
        eval_fn = jax.jit(lambda p, toks: loss_global(p, toks)[1], in_shardings=(p_sh, eval_sh), out_shardings=rep)
        return step_fn, eval_fn

    attn_global, attn_local, flash_on = _make_attention(jax, jnp, cfg.tpu_flash, mesh, on_tpu, log)
    step_fn, eval_fn = build(attn_global, attn_local)
    compiled: dict = {}                                    # batch shape -> compiled step

    horizon = cfg.max_tokens
    switch = anneal_start(cfg, horizon) if "anneal" in phases else horizon
    log(f"schedule: {horizon:,} tokens, {cfg.lr_schedule} lr peak {cfg.learning_rate:g}, anneal from {switch:,}")
    atomic_write_json(out_dir / "config.json", {"train_config": cfg.to_dict(), "n_params": n_params,
                                                "backend": describe, "plan": plan, "jax": jax.__version__})
    notify = Notifier(cfg.ntfy_topic, run=out_dir.name, log=log)

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
    t_loop = time.time()

    def train_time() -> float:
        return train_time_prev + (time.time() - t_loop)

    def save(reason: str) -> Path:
        t = time.time()
        host_p = jax.device_get(params)
        host_o = jax.device_get(opt)
        state = {
            "model": jm.params_to_torch(host_p, mc),
            "optimizer": jm.opt_state_to_torch(host_o, mc, groups, layout, cfg.optimizer),
            "scaler": None, "step": step, "tokens_seen": tokens_seen,
            "loader_states": {**loader_states, phase: loader.state()}, "phase": phase,
            "rng": capture_rng_state(), "model_config": asdict(mc), "train_config": cfg.to_dict(),
            "precision": "bf16" if dtype == jnp.bfloat16 else "fp32", "tokenizer": manifest["tokenizer"],
            "eot_token": manifest["eot_token"], "best_val_loss": best_val, "last_val_loss": last_val,
            "train_time_s": train_time(), "versions": {"jax": jax.__version__, "backend": describe},
        }
        del host_p, host_o
        path = save_checkpoint(ckpt_dir, step, state, keep_last=cfg.keep_last)
        log(f"checkpoint saved: {path.name} ({fmt_bytes(path.stat().st_size)}, {time.time() - t:.1f}s, verified) "
            f"[{reason}]")
        return path

    def evaluate() -> tuple[Optional[float], dict]:
        if not val_sets:
            return None, {}
        per = {}
        bsz = plan["seqs"]["main"]
        for name, (share, shards) in val_sets.items():
            n = (min(shards.n_windows, max(bsz, int(cfg.eval_tokens * share) // shards.block_size)) // bsz) * bsz
            if n == 0:
                continue
            total = 0.0
            for start in range(0, n, bsz):
                toks = jax.device_put(shards.batch(np.arange(start, start + bsz)).astype(np.int32), eval_sh)
                total = total + eval_fn(params, toks)
            per[name] = float(total) / (n // bsz)
        shares = {n: val_sets[n][0] for n in per}
        return (sum(per[n] * shares[n] for n in per) / max(1e-12, sum(shares.values())) if per else None), per

    def next_step_batch():
        rows = [loader.next_batch()[0] for _ in range(plan["accum"])]
        return jax.device_put(np.stack(rows).astype(np.int32), batch_sh)

    # ---- stop handling
    stop = {"requested": False}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: stop.update(requested=True))
            except (ValueError, OSError):
                pass

    status, steps_run = "complete", 0
    last_ckpt = time.time()
    window_t0, window_tokens = time.time(), 0
    ce_window, metrics = None, None
    flops_tok = mc.flops_per_token(plan["ctx"][phase])
    next_milestone = (tokens_seen // cfg.notify_every_tokens + 1) * cfg.notify_every_tokens
    log(f"training starts at step {step}, {tokens_seen:,} tokens, phase {phase}")
    notify.send(f"JAX TPU training {'resumed' if step else 'started'} at step {step} ({fmt_count(tokens_seen)} tokens).")

    try:
        while True:
            if tokens_seen >= horizon or (cfg.max_steps is not None and step >= cfg.max_steps):
                break
            if stop["requested"] or (stop_after_steps is not None and steps_run >= stop_after_steps):
                status = "interrupted"
                break
            if cfg.max_total_hours is not None and train_time() >= cfg.max_total_hours * 3600:
                status = "total_time_limit"
                break
            if cfg.max_hours is not None and time.time() - t_loop >= cfg.max_hours * 3600:
                status = "time_limit"
                break
            wanted = phase_for(tokens_seen)
            if wanted != phase:
                loader_states[phase] = loader.state()
                loader.close()
                phase = wanted
                loader = make_loader(phase)
                flops_tok = mc.flops_per_token(plan["ctx"][phase])
                log(f"data phase: {phase} from {tokens_seen:,} tokens (context {plan['ctx'][phase]})")
                notify.send(f"Switched to the {phase} phase at {fmt_count(tokens_seen)} tokens.")

            lr = jnp.asarray(lr_at(tokens_seen + plan["tokens_per_step"], cfg, horizon), jnp.float32)
            batch = next_step_batch()
            shape = tuple(batch.shape)
            if shape not in compiled:
                t_c = time.time()
                log(f"compiling the training step for batch {shape} (takes a few minutes on TPU)...")
                try:
                    compiled[shape] = step_fn.lower(params, opt, batch, lr).compile()
                except Exception as exc:  # noqa: BLE001
                    if not flash_on:
                        raise
                    log(f"WARNING: flash attention failed to compile ({type(exc).__name__}: {str(exc)[:300]}); "
                        "using XLA fused attention")
                    flash_on = False
                    compiled.clear()
                    step_fn, eval_fn = build(jm.xla_attention, jm.xla_attention)
                    compiled[shape] = step_fn.lower(params, opt, batch, lr).compile()
                log(f"compiled in {time.time() - t_c:.0f}s")
                window_t0 += time.time() - t_c
            params, opt, metrics = compiled[shape](params, opt, batch, lr)
            step += 1
            steps_run += 1
            tokens_seen += plan["tokens_per_step"]
            window_tokens += plan["tokens_per_step"]
            loader_states[phase] = loader.state()
            ce_window = metrics["ce"] if ce_window is None else ce_window + metrics["ce"]

            if step % cfg.log_interval == 0 or steps_run == 1:
                n_win = cfg.log_interval if steps_run > 1 else 1
                loss_val = float(ce_window) / max(1, min(n_win, steps_run))
                gn = float(metrics["grad_norm"])
                ce_window = None
                dt = max(1e-9, time.time() - window_t0)
                tps = window_tokens / dt
                mfu = flops_tok * tps / peak if peak else None
                if not math.isfinite(loss_val):
                    notify.send(f"JAX training stopped: non-finite loss at step {step}.", priority="high")
                    metrics = None                     # never save these weights
                    raise RuntimeError(f"non-finite loss at step {step}; resume from the last checkpoint with a "
                                       "lower --learning_rate")
                lr_f = float(lr)
                append_jsonl(metrics_path, {"kind": "train", "step": step, "tokens_seen": tokens_seen, "phase": phase,
                                            "loss": loss_val, "lr": lr_f, "grad_norm": gn, "tok_per_sec": tps,
                                            "mfu": mfu, "train_time_s": train_time(), "time_unix": time.time()})
                eta = (horizon - tokens_seen) / tps if tps > 0 else float("inf")
                log(f"step {step:>7d} | loss {loss_val:.4f} | lr {lr_f:.2e} | gnorm {gn:.2f} | {fmt_count(tps)} tok/s"
                    + (f" | mfu {100 * mfu:.1f}%" if mfu else "")
                    + f" | {fmt_count(tokens_seen)}/{fmt_count(horizon)} tok | eta {fmt_duration(eta)}")
                window_t0, window_tokens = time.time(), 0

            if bench:
                continue
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
                save("time-based" if due else "periodic")
                last_ckpt = time.time()
            if tokens_seen >= next_milestone:
                next_milestone += cfg.notify_every_tokens
                notify.send(f"{fmt_count(tokens_seen)}/{fmt_count(horizon)} tokens, step {step}, val "
                            f"{last_val if last_val is None else round(last_val, 3)}.")
            window_t0 += time.time() - t_side
    except BaseException as exc:
        if bench:
            raise
        if not isinstance(exc, KeyboardInterrupt):
            notify.send(f"JAX training CRASHED at step {step}: {type(exc).__name__}: {str(exc)[:200]}", priority="high")
        # Keep the progress since the last checkpoint if the weights are intact and finite.
        if metrics is not None and steps_run:
            try:
                if math.isfinite(float(metrics["ce"])):
                    save(f"emergency after {type(exc).__name__}")
            except Exception as save_exc:  # noqa: BLE001
                log(f"emergency checkpoint not possible ({type(save_exc).__name__}: {str(save_exc)[:200]})")
        raise

    if bench:
        loader.close()
        recs = [r for r in read_jsonl(metrics_path) if r.get("kind") == "train"] if metrics_path.exists() else []
        steady = [r for r in recs if r["step"] > 2] or recs      # skip the first, still-warming steps
        mean = lambda key: (sum(r[key] for r in steady) / len(steady)) if steady and steady[0].get(key) else None
        return {"status": "bench", "step": step, "tok_per_sec": mean("tok_per_sec"), "mfu": mean("mfu"),
                "loss": steady[-1]["loss"] if steady else None, "tokens_per_step": plan["tokens_per_step"]}
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
    msg = (f"JAX session ended ({status}) at step {step}, {fmt_count(tokens_seen)}/{fmt_count(horizon)} tokens, "
           f"val {last_val if last_val is None else round(last_val, 3)}.")
    log(msg)
    notify.send(msg + (" Start the next session from this output." if status != "complete" else " Done!"),
                priority="high" if status == "complete" else "default")
    return {"status": status, "step": step, "tokens_seen": tokens_seen, "val_loss": last_val, "checkpoint": str(path),
            "final_weights": str(final) if final else None, "tokens_per_step": plan["tokens_per_step"],
            "n_params": n_params}
