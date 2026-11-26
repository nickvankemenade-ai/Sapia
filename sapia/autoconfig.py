"""Memory-aware automatic configuration.

Given the detected ``Resources`` this module decides:

* the micro-batch size, by probing real forward+backward passes on the GPU;
* gradient-accumulation steps, to hit a target number of tokens per optimizer step;
* how training data is held (all shards in RAM vs ``np.memmap``) and how many
  DataLoader workers / prefetched batches to use;
* data-preparation parallelism and shard size.

Every decision returns a small "plan" object carrying a human-readable ``reason`` that the
caller logs, and every decision can be overridden from the config / CLI.
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from .resources import Resources
from .utils import GIB, MIB, fmt_bytes


class AutoConfigError(RuntimeError):
    """Automatic configuration is impossible (e.g. batch size 1 does not fit)."""


# ----------------------------------------------------------------------------- batch-size search

@dataclass
class ProbeResult:
    batch_size: int
    oom: bool            # an out-of-memory error was raised
    peak_bytes: int      # peak bytes the process held on the device during the probe (0 if oom)
    seconds: float = 0.0

    def fits(self, budget_bytes: int) -> bool:
        return (not self.oom) and self.peak_bytes <= budget_bytes


@dataclass
class BatchSearchResult:
    batch_size: int
    budget_bytes: int
    peak_bytes: int                      # measured peak at the chosen batch size
    probes: list[ProbeResult] = field(default_factory=list)


def search_batch_size(probe: Callable[[int], ProbeResult], budget_bytes: int, max_batch: int,
                      max_probes: int = 16, log: Callable[[str], None] = lambda s: None) -> BatchSearchResult:
    """Find the largest batch size in [1, max_batch] whose measured peak memory fits the budget.

    Strategy: probe 1 and a small second point, fit ``peak = fixed + per_sample * b``
    through the two largest batch sizes known to fit, jump to the predicted maximum and
    verify it with a real probe. Once a probe fails, the search only narrows the bracket
    between the largest size that fits and the smallest that does not: by interpolating
    on the measured peaks when the failure was "peak above budget", by bisection when it
    was a real out-of-memory error (which gives no measurement).

    Extrapolating instead of doubling until OOM keeps the number of probes low and
    normally avoids triggering a real OOM at all. That matters on Windows, where the
    driver may silently spill into slow shared memory instead of raising; such a probe
    simply measures a peak above the budget and is rejected.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    probes: list[ProbeResult] = []

    def attempt(b: int) -> ProbeResult:
        r = probe(b)
        probes.append(r)
        status = "OOM" if r.oom else f"peak {fmt_bytes(r.peak_bytes)}"
        verdict = "fits" if r.fits(budget_bytes) else "too big"
        log(f"  probe batch_size={b:<5d} {status:<22s} budget {fmt_bytes(budget_bytes)} -> {verdict}")
        return r

    first = attempt(1)
    if not first.fits(budget_bytes):
        raise AutoConfigError(
            "even batch_size=1 does not fit in GPU memory "
            f"({'out of memory' if first.oom else 'peak ' + fmt_bytes(first.peak_bytes)}, "
            f"budget {fmt_bytes(budget_bytes)}). Use a smaller model or block_size, free GPU memory, "
            "or raise --vram_fraction."
        )

    lo, lo_peak = 1, first.peak_bytes           # largest batch size known to fit
    prev: tuple[int, int] = (1, first.peak_bytes)
    hi = max_batch + 1                           # smallest batch size known not to fit (exclusive bound)
    hi_peak: Optional[int] = None                # its measured peak, when the failure was not an OOM
    failed = False
    cand = min(max_batch, 4)

    while len(probes) < max_probes and lo < cand < hi:
        r = attempt(cand)
        fit = r.fits(budget_bytes)
        if fit:
            prev, (lo, lo_peak) = (lo, lo_peak), (cand, r.peak_bytes)
        else:
            failed = True
            hi, hi_peak = cand, (None if r.oom else r.peak_bytes)
        if hi - lo <= 1:
            break
        tolerance = max(1, int(0.03 * lo))       # stop once within ~3% of what is verified
        if not failed:
            per_sample = (lo_peak - prev[1]) / max(1, lo - prev[0])
            nxt = hi - 1 if per_sample <= 0 else lo + int((budget_bytes - lo_peak) // per_sample)
            nxt = min(nxt, hi - 1)
            if nxt - lo < tolerance:
                break
            cand = nxt
        elif hi_peak is not None and hi_peak > lo_peak:
            slope = (hi_peak - lo_peak) / (hi - lo)
            nxt = lo + int((budget_bytes - lo_peak) // slope)
            if fit and nxt - lo < tolerance:
                break
            cand = max(lo + 1, min(hi - 1, nxt))
        else:
            cand = (lo + hi) // 2

    return BatchSearchResult(batch_size=lo, budget_bytes=int(budget_bytes), peak_bytes=int(lo_peak), probes=probes)


def is_oom_error(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "alloc_failed" in msg or "cuda error: out of memory" in msg
    return False


def free_cuda_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_gpu_probe(model, block_size: int, vocab_size: int, device, precision,
                   micro_steps: int = 2, optimizer_steps: int = 2,
                   reserve_optimizer_state: bool = True) -> tuple[Callable[[int], ProbeResult], Callable[[], None]]:
    """Build ``probe(batch_size)`` running real forward+backward passes, and a cleanup function.

    * Uses the real model, autocast mode and GradScaler, so the measurement includes
      everything a training step allocates.
    * Runs ``micro_steps`` passes without zeroing gradients in between, as gradient
      accumulation does (the second pass allocates activations while gradients exist),
      and repeats that for ``optimizer_steps`` cycles with a ``zero_grad`` between them,
      so the measurement includes the allocator fragmentation of steady-state training.
    * AdamW keeps two extra tensors per parameter that do not exist before the first
      optimizer step; equivalent "ballast" tensors are held during probing so the model
      weights are never touched by a throwaway optimizer.
    * Token ids come from a private generator, so the global RNG stream is not consumed.
    """
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    ballast: list = []
    if reserve_optimizer_state:
        for p in params:
            ballast.append(torch.empty_like(p))
            ballast.append(torch.empty_like(p))
    scaler = precision.make_scaler()
    gen = torch.Generator().manual_seed(0)
    was_training = model.training
    model.train()

    def run(batch_size: int) -> None:
        tokens = torch.randint(0, vocab_size, (batch_size, block_size + 1), generator=gen).to(device)
        x, y = tokens[:, :-1], tokens[:, 1:]
        for _cycle in range(optimizer_steps):
            for _ in range(micro_steps):
                with precision.autocast(device.type):
                    _, loss = model(x, y)
                scaler.scale(loss / micro_steps).backward()
            del loss
            model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def probe(batch_size: int) -> ProbeResult:
        free_cuda_memory()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        oom = False
        try:
            run(batch_size)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            if not is_oom_error(exc):
                model.zero_grad(set_to_none=True)
                raise
            oom = True
        # The exception (and the frames holding its tensors) is released here, outside
        # the except block, so the cache can really be emptied.
        # Peak *reserved* memory: what the process really claimed from the device, i.e.
        # allocated tensors plus allocator overhead. It is >= the peak allocated bytes.
        peak = 0 if oom else int(max(torch.cuda.max_memory_reserved(device),
                                     torch.cuda.max_memory_allocated(device)))
        model.zero_grad(set_to_none=True)
        free_cuda_memory()
        return ProbeResult(batch_size=batch_size, oom=oom, peak_bytes=peak, seconds=time.time() - t0)

    def cleanup() -> None:
        ballast.clear()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        free_cuda_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    return probe, cleanup


def gpu_memory_budget(device, vram_fraction: float) -> int:
    """Bytes the probed peak may reach: a fraction of what this process can use on the GPU.

    Usable memory = currently free device memory + what this process already holds
    (weights, allocator cache). Other processes' usage is thereby excluded.
    """
    import torch

    free, _total = torch.cuda.mem_get_info(device)
    usable = int(free) + int(torch.cuda.memory_reserved(device))
    return int(usable * vram_fraction)


def heuristic_cpu_batch_size(model_cfg, mem_available_bytes: int, max_batch: int) -> int:
    """Rough CPU batch size from available RAM (no probing on CPU; it is too slow to matter)."""
    # fp32 activations kept for backward: ~(6*d_model + 3*mlp_hidden) floats per token per
    # layer, plus logits and their gradient-side copies (~3 * vocab floats per token).
    per_token = model_cfg.n_layer * (6 * model_cfg.d_model + 3 * model_cfg.mlp_hidden) + 3 * model_cfg.vocab_size
    per_sample = 4 * per_token * model_cfg.block_size
    budget = 0.4 * max(0, mem_available_bytes)
    return int(max(1, min(max_batch, 32, budget // max(1, per_sample))))


# ----------------------------------------------------------------------------- batch plan

@dataclass
class BatchPlan:
    batch_size: int                 # micro-batch size, in sequences
    grad_accum_steps: int
    tokens_per_step: int
    max_fit_batch_size: Optional[int]   # largest batch size the probe verified (None if not probed)
    source: str                     # "probe" | "config" | "checkpoint" | "cpu-heuristic"
    reason: str
    device_signature: str = ""
    peak_bytes: Optional[int] = None
    budget_bytes: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def plan_accumulation(max_micro_batch: int, block_size: int, target_tokens_per_step: int,
                      fixed_accum: Optional[int] = None) -> tuple[int, int]:
    """Choose (batch_size, grad_accum_steps) with batch_size <= max_micro_batch.

    The number of sequences per optimizer step is ``round(target / block_size)``; it is
    split into the fewest micro-batches that fit, and the micro-batch is then shrunk so the
    product lands as close to the target as possible.
    """
    if max_micro_batch < 1:
        raise ValueError("max_micro_batch must be >= 1")
    seqs = max(1, int(round(target_tokens_per_step / block_size)))
    if fixed_accum is not None:
        accum = max(1, int(fixed_accum))
    else:
        accum = max(1, math.ceil(seqs / max_micro_batch))
    batch = max(1, min(max_micro_batch, math.ceil(seqs / accum)))
    return batch, accum


def device_signature(device, gpu) -> str:
    if gpu is None or device.type != "cuda":
        return "cpu"
    return f"{gpu.name}|{gpu.total_bytes}"


# ----------------------------------------------------------------------------- data loading plan

@dataclass
class DataPlan:
    mode: str                 # "ram" | "memmap"
    num_workers: int
    prefetch_factor: Optional[int]
    pin_memory: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def plan_data_loading(data_bytes: int, res: Resources, *, mode: str = "auto",
                      num_workers: Optional[int] = None, prefetch_factor: Optional[int] = None,
                      on_cuda: bool = True, process_reserve_bytes: int = 6 * GIB,
                      ram_fraction: float = 0.6, small_data_bytes: int = 256 * MIB) -> DataPlan:
    """Decide whether the training shards are read into RAM or memory-mapped.

    RAM is preferred (no page-fault stalls, and fast even when the dataset lives on a
    network volume) when the data fits in ``ram_fraction`` of the effective available
    memory after reserving ``process_reserve_bytes`` for the trainer itself (Python,
    CUDA context, checkpoint serialization). "Effective" means the container limit when
    there is one, not the host's RAM.
    """
    available = res.mem_available_bytes
    reasons: list[str] = []
    if mode == "auto":
        headroom = available - process_reserve_bytes
        if data_bytes <= small_data_bytes and data_bytes * 2 <= available:
            mode = "ram"
            reasons.append(f"dataset is small ({fmt_bytes(data_bytes)})")
        elif headroom > 0 and data_bytes <= ram_fraction * headroom:
            mode = "ram"
            reasons.append(
                f"dataset {fmt_bytes(data_bytes)} <= {ram_fraction:.0%} of "
                f"({fmt_bytes(available)} available - {fmt_bytes(process_reserve_bytes)} reserved)"
            )
        else:
            mode = "memmap"
            reasons.append(
                f"dataset {fmt_bytes(data_bytes)} does not fit comfortably in "
                f"{fmt_bytes(available)} available RAM"
            )
    else:
        reasons.append(f"data_mode={mode} set explicitly")

    if num_workers is None:
        if mode == "ram":
            num_workers = 0
            reasons.append("0 workers (in-RAM batches are assembled in well under a millisecond each)")
        else:
            num_workers = max(1, min(4, res.cpu_usable - 1)) if res.cpu_usable > 1 else 0
            reasons.append(f"{num_workers} workers to hide memmap page faults ({res.cpu_usable} usable CPUs)")
    else:
        reasons.append(f"num_workers={num_workers} set explicitly")

    if num_workers > 0:
        prefetch = prefetch_factor if prefetch_factor is not None else 4
    else:
        prefetch = None
    return DataPlan(mode=mode, num_workers=int(num_workers), prefetch_factor=prefetch,
                    pin_memory=bool(on_cuda), reason="; ".join(reasons))


# ----------------------------------------------------------------------------- data-prep plan

DEFAULT_SHARD_TOKENS = 100_000_000
MIN_SHARD_TOKENS = 1_000_000


@dataclass
class PrepPlan:
    num_workers: int           # worker processes (one source file each)
    threads_per_worker: int    # tokenizer threads inside each worker
    shard_tokens: int          # tokens per output shard (also the per-worker buffer size)
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# Fixed per-worker memory, independent of the shard size. Measured: a single worker
# streaming one fineweb-edu parquet file peaks at ~2.0 GiB RSS on Windows whether
# shard_tokens is 5M or 20M (imports + tokenizer are only ~135 MiB; the rest is
# pyarrow/datasets reading whole parquet row groups). On Modal (Linux, gVisor) 14 workers
# with 100M-token shards exhausted a 48 GiB limit, i.e. ~3.4 GiB per worker.
PREP_WORKER_BASE_BYTES = 3 * GIB


def prep_worker_bytes(shard_tokens: int) -> int:
    """Estimated peak RSS of one data-prep worker.

    A fixed ~3 GiB (interpreter, libraries, tokenizer, parquet row-group buffers and
    allocator retention), plus one shard of uint16 tokens buffered as per-document
    arrays and a second copy while it is concatenated for writing.
    """
    return PREP_WORKER_BASE_BYTES + 2 * 2 * shard_tokens


def plan_data_prep(res: Resources, n_files: int, *, shard_tokens: Optional[int] = None,
                   num_workers: Optional[int] = None, threads_per_worker: Optional[int] = None,
                   target_tokens: Optional[int] = None, ram_fraction: float = 0.6) -> PrepPlan:
    budget = int(res.mem_available_bytes * ram_fraction)
    reasons: list[str] = []

    if shard_tokens is None:
        shard = DEFAULT_SHARD_TOKENS
        if target_tokens is not None:
            # Keep at least a handful of shards for small targets.
            shard = min(shard, max(MIN_SHARD_TOKENS, target_tokens // 8))
        # Only the token buffers scale with the shard size; the fixed part cannot shrink.
        while shard > MIN_SHARD_TOKENS and prep_worker_bytes(shard) > max(budget, prep_worker_bytes(MIN_SHARD_TOKENS)):
            shard //= 2
        shard = max(MIN_SHARD_TOKENS, shard)
        reasons.append(f"shard_tokens={shard:,} (buffer budget {fmt_bytes(budget)})")
    else:
        shard = int(shard_tokens)
        reasons.append(f"shard_tokens={shard:,} set explicitly")
    if budget < prep_worker_bytes(shard):
        reasons.append(f"WARNING: memory budget {fmt_bytes(budget)} is below one worker's "
                       f"~{fmt_bytes(prep_worker_bytes(shard))}; running a single worker anyway")

    if num_workers is None:
        by_mem = max(1, budget // prep_worker_bytes(shard))
        workers = int(max(1, min(n_files, res.cpu_usable, by_mem)))
        reasons.append(
            f"{workers} workers = min({n_files} source files, {res.cpu_usable} usable CPUs, "
            f"{by_mem} by memory at ~{fmt_bytes(prep_worker_bytes(shard))}/worker)"
        )
    else:
        workers = max(1, int(num_workers))
        reasons.append(f"num_workers={workers} set explicitly")

    if threads_per_worker is None:
        threads = max(1, res.cpu_usable // workers)
        reasons.append(f"{threads} tokenizer threads per worker")
    else:
        threads = max(1, int(threads_per_worker))

    return PrepPlan(num_workers=workers, threads_per_worker=threads, shard_tokens=shard, reason="; ".join(reasons))
