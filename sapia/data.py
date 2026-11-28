"""Token shards on disk and the deterministic, resumable training loader.

On-disk layout of a prepared dataset directory::

    manifest.json                  # tokenizer, counts, ordered shard list per split
    train_00000_00000.bin          # raw little-endian uint16 token ids, no header
    val_00000_00000.bin
    prep_state/                    # resume state of the data-prep job (see prepare_data.py)

Training order: each shard is cut into non-overlapping windows of ``block_size`` tokens
(the target is the window shifted by one, so a window reads ``block_size + 1`` tokens).
Windows are visited in a permutation that depends only on ``(seed, epoch)``. The loader's
position is ``(epoch, cursor)`` with ``cursor`` counted in windows, so a checkpoint can
resume exactly, even with a different batch size.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .utils import atomic_write_json, read_json, replace_with_retry

MANIFEST_NAME = "manifest.json"
SHARD_DTYPE = np.dtype("<u2")
MANIFEST_VERSION = 1


class DataError(RuntimeError):
    """The dataset on disk is missing, incomplete or inconsistent."""


# ----------------------------------------------------------------------------- shard IO

def write_shard_atomic(path: str | os.PathLike, tokens: np.ndarray) -> int:
    """Write ``tokens`` as raw uint16 to ``path`` via a temp file + rename. Returns the token count."""
    path = Path(path)
    arr = np.ascontiguousarray(tokens, dtype=SHARD_DTYPE)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        arr.tofile(fh)
        try:
            fh.flush()
            os.fsync(fh.fileno())
        except OSError:
            pass
    replace_with_retry(tmp, path)
    return int(arr.size)


def read_shard(path: str | os.PathLike, mmap: bool = False) -> np.ndarray:
    if mmap:
        return np.memmap(path, dtype=SHARD_DTYPE, mode="r")
    return np.fromfile(path, dtype=SHARD_DTYPE)


def write_manifest(data_dir: str | os.PathLike, manifest: dict) -> None:
    atomic_write_json(Path(data_dir) / MANIFEST_NAME, manifest)


def load_manifest(data_dir: str | os.PathLike, verify: bool = True) -> dict:
    """Load manifest.json and (optionally) check every shard exists with the expected size."""
    data_dir = Path(data_dir)
    path = data_dir / MANIFEST_NAME
    if not path.exists():
        raise DataError(
            f"no {MANIFEST_NAME} in {data_dir}. Run data preparation first, e.g.\n"
            f"  python -m sapia.prepare_mix --preset tiny"
        )
    manifest = read_json(path)
    if manifest.get("version") != MANIFEST_VERSION:
        raise DataError(f"{path}: unsupported manifest version {manifest.get('version')!r}")
    if verify:
        for split, shards in manifest["shards"].items():
            for shard in shards:
                p = data_dir / shard["name"]
                if not p.exists():
                    raise DataError(f"shard listed in the manifest is missing: {p}")
                expected = shard["tokens"] * SHARD_DTYPE.itemsize
                actual = p.stat().st_size
                if actual != expected:
                    raise DataError(f"{p}: size {actual} bytes, manifest expects {expected} ({split} split)")
    return manifest


def split_bytes(manifest: dict, split: str) -> int:
    return sum(s["tokens"] for s in manifest["shards"].get(split, [])) * SHARD_DTYPE.itemsize


# ----------------------------------------------------------------------------- windowed access

class ShardSet:
    """Random access to fixed-size token windows over the shards of one split.

    ``mode="ram"`` reads every shard into memory; ``mode="memmap"`` maps them lazily.
    Instances are picklable for DataLoader workers: a worker always re-opens the shards
    as memmaps, so in-RAM arrays are never copied to worker processes.
    """

    def __init__(self, data_dir: str | os.PathLike, split: str, block_size: int,
                 mode: str = "memmap", manifest: Optional[dict] = None):
        if mode not in ("ram", "memmap"):
            raise ValueError(f"mode must be 'ram' or 'memmap', got {mode!r}")
        self.data_dir = str(data_dir)
        self.split = split
        self.block_size = int(block_size)
        self.mode = mode
        manifest = manifest if manifest is not None else load_manifest(data_dir)
        shards = manifest["shards"].get(split, [])
        self.names = [s["name"] for s in shards]
        self.tokens = np.array([s["tokens"] for s in shards], dtype=np.int64)
        # A window needs block_size + 1 tokens (inputs plus the shifted targets).
        self.windows = np.maximum(self.tokens - 1, 0) // self.block_size
        self.cum_windows = np.concatenate(([0], np.cumsum(self.windows))).astype(np.int64)
        self.n_tokens = int(self.tokens.sum())
        self.n_windows = int(self.cum_windows[-1])
        self._arrays: Optional[list[np.ndarray]] = None

    # -- lifecycle
    def _open(self, mode: str) -> None:
        arrays = []
        for name in self.names:
            arrays.append(read_shard(Path(self.data_dir) / name, mmap=(mode == "memmap")))
        self._arrays = arrays

    def _ensure_open(self) -> list[np.ndarray]:
        if self._arrays is None:
            self._open(self.mode)
        assert self._arrays is not None
        return self._arrays

    def load(self) -> "ShardSet":
        """Open (and for ram mode, read) all shards now rather than on first access."""
        self._ensure_open()
        return self

    def close(self) -> None:
        self._arrays = None
        gc.collect()  # drop memmap handles promptly (matters on Windows)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = None
        state["mode"] = "memmap"
        return state

    # -- access
    def __len__(self) -> int:
        return self.n_windows

    def batch(self, window_ids: np.ndarray) -> np.ndarray:
        """Return an int64 array (len(window_ids), block_size + 1) for the given global window ids."""
        arrays = self._ensure_open()
        ids = np.asarray(window_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.n_windows):
            raise IndexError(f"window id out of range [0, {self.n_windows})")
        T = self.block_size
        out = np.empty((ids.size, T + 1), dtype=np.int64)
        shard_idx = np.searchsorted(self.cum_windows, ids, side="right") - 1
        starts = (ids - self.cum_windows[shard_idx]) * T
        for row, (s, start) in enumerate(zip(shard_idx.tolist(), starts.tolist())):
            out[row] = arrays[s][start:start + T + 1]
        return out

    def __getitem__(self, key):
        """DataLoader entry point: ``key = (epoch, cursor_after, window_ids)``."""
        import torch

        epoch, cursor_after, ids = key
        return torch.from_numpy(self.batch(ids)), int(epoch), int(cursor_after)


def epoch_permutation(n_windows: int, seed: int, epoch: int) -> np.ndarray:
    """Deterministic visiting order for one epoch; depends only on (n_windows, seed, epoch)."""
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(epoch)])
    return rng.permutation(n_windows)


class WindowSampler:
    """Infinite stream of ``(epoch, cursor_after, window_ids)`` keys starting at a saved position.

    The trailing ``n_windows % batch_size`` windows of an epoch are dropped.
    """

    def __init__(self, n_windows: int, batch_size: int, seed: int, epoch: int = 0, cursor: int = 0):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if n_windows < batch_size:
            raise DataError(
                f"the training split has only {n_windows} windows but batch_size is {batch_size}; "
                "prepare more data, or lower batch_size / block_size"
            )
        self.n_windows = int(n_windows)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.cursor = int(cursor)

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        epoch, cursor, B = self.epoch, self.cursor, self.batch_size
        while True:
            perm = epoch_permutation(self.n_windows, self.seed, epoch)
            while cursor + B <= self.n_windows:
                ids = perm[cursor:cursor + B]
                cursor += B
                yield epoch, cursor, ids
            epoch += 1
            cursor = 0


class TrainLoader:
    """Iterates training batches and tracks the position of the last batch handed out.

    Yields ``(tokens, epoch, cursor_after)`` where ``tokens`` is an int64 tensor of shape
    (batch_size, block_size + 1): inputs are ``tokens[:, :-1]``, targets ``tokens[:, 1:]``.
    With ``num_workers == 0`` batches are assembled synchronously in the main process;
    with workers, a ``torch.utils.data.DataLoader`` prefetches them (workers use memmaps).
    """

    def __init__(self, shards: ShardSet, batch_size: int, seed: int, epoch: int = 0, cursor: int = 0,
                 num_workers: int = 0, prefetch_factor: Optional[int] = None, pin_memory: bool = False):
        self.shards = shards
        self.sampler = WindowSampler(shards.n_windows, batch_size, seed, epoch, cursor)
        self.num_workers = int(num_workers)
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self._loader = None
        self._iter = None

    def _make_iter(self):
        import torch

        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor or 4
            kwargs["persistent_workers"] = True
        self._loader = torch.utils.data.DataLoader(
            self.shards, sampler=self.sampler, batch_size=None, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, **kwargs,
        )
        return iter(self._loader)

    def __iter__(self) -> "TrainLoader":
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = self._make_iter()
        tokens, epoch, cursor = next(self._iter)
        return tokens, int(epoch), int(cursor)

    def close(self) -> None:
        it, self._iter, self._loader = self._iter, None, None
        if it is not None and hasattr(it, "_shutdown_workers"):
            try:
                it._shutdown_workers()
            except Exception:
                pass
        del it
        gc.collect()


def iter_eval_batches(shards: ShardSet, batch_size: int, max_windows: int) -> Iterator[np.ndarray]:
    """Fixed, sequential evaluation batches: the first ``max_windows`` windows of the split."""
    n = min(shards.n_windows, max_windows)
    for start in range(0, n, batch_size):
        yield shards.batch(np.arange(start, min(n, start + batch_size), dtype=np.int64))


# ----------------------------------------------------------------------------- the pretraining mix

def _mix_sources(mix_dir: str | os.PathLike, phase: str) -> list[dict]:
    path = Path(mix_dir) / "mix.json"
    if not path.exists():
        raise DataError(f"no mix.json in {mix_dir}; run python -m sapia.prepare_mix first")
    mix = read_json(path)
    if phase not in mix.get("phases", {}):
        raise DataError(f"{path} has no phase {phase!r} (has {sorted(mix.get('phases', {}))})")
    return mix["phases"][phase]["sources"]


def cap_repeats(weights: np.ndarray, available: np.ndarray, phase_tokens: float, max_epochs: float) -> np.ndarray:
    """Limit each source's share so it is read at most ``max_epochs`` times over the phase.

    A source prepared short of its target (e.g. it ran out of documents) would otherwise be
    repeated many times to keep its share. Its share is lowered to ``max_epochs * available /
    phase_tokens`` and the freed share goes to the other sources in proportion to theirs.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    allowed = max_epochs * np.asarray(available, dtype=np.float64) / float(phase_tokens)
    capped = np.zeros(len(w), dtype=bool)
    for _ in range(len(w)):
        over = ~capped & (w > allowed + 1e-12)
        if not over.any():
            break
        excess = float((w[over] - allowed[over]).sum())
        w[over] = allowed[over]
        capped |= over
        free = ~capped
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


class MixtureLoader:
    """Training batches drawn from several prepared sources in proportion to their shares.

    Every row of a batch picks its source with a random draw seeded by ``(seed, phase, batch
    index)``, then takes that source's next window (each source walks its own windows in a
    per-epoch permutation, like :class:`WindowSampler`). Since every window holds
    ``block_size`` tokens, row shares are token shares. A source that runs out starts its next
    epoch, so its share holds even if it was prepared short.

    The position is ``{"phase", "batch", "positions": {source: [epoch, cursor]}}``; restoring
    it continues exactly where a checkpoint left off.

    With ``phase_tokens`` and ``max_epochs``, no source is read more than ``max_epochs`` times
    over the phase (see :func:`cap_repeats`).
    """

    def __init__(self, mix_dir: str | os.PathLike, phase: str, block_size: int, batch_size: int, seed: int,
                 mode: str = "memmap", state: Optional[dict] = None, phase_tokens: Optional[float] = None,
                 max_epochs: Optional[float] = None):
        self.phase = phase
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.names: list[str] = []
        self.shards: list[ShardSet] = []
        shares = []
        for src in _mix_sources(mix_dir, phase):
            d = Path(mix_dir) / src["dir"]
            if not (d / MANIFEST_NAME).exists() or src["share"] <= 0:
                continue
            ss = ShardSet(d, "train", block_size, mode=mode)
            if ss.n_windows == 0:
                continue
            self.names.append(src["name"])
            self.shards.append(ss)
            shares.append(float(src["share"]))
        if not self.shards:
            raise DataError(f"no prepared sources with data for phase {phase!r} in {mix_dir}")
        w = np.asarray(shares, dtype=np.float64)
        self.weights = w / w.sum()
        self.capped: list[str] = []
        if phase_tokens and max_epochs:
            available = np.array([s.n_windows * block_size for s in self.shards], dtype=np.float64)
            before = self.weights
            self.weights = cap_repeats(self.weights, available, phase_tokens, max_epochs)
            self.capped = [n for n, b, a in zip(self.names, before, self.weights) if a < b - 1e-9]
        self.missing_share = 1.0 - float(sum(shares)) if sum(shares) < 1.0 else 0.0
        self.batch = 0
        self.positions = {n: [0, 0] for n in self.names}
        self._perm_cache: dict[tuple[int, int], np.ndarray] = {}
        if state is not None:
            self.load_state(state)

    def describe(self) -> str:
        """The effective mix, e.g. ``fineweb_edu_top 33.1%, stackexchange_top 2.2% (capped)``."""
        return ", ".join(f"{n} {100 * w:.1f}%" + (" (capped)" if n in self.capped else "")
                         for n, w in zip(self.names, self.weights))

    # -- position
    def state(self) -> dict:
        return {"phase": self.phase, "batch": self.batch, "positions": {k: list(v) for k, v in self.positions.items()}}

    def load_state(self, state: dict) -> None:
        if state.get("phase") != self.phase:
            raise DataError(f"loader state is for phase {state.get('phase')!r}, not {self.phase!r}")
        self.batch = int(state["batch"])
        for name, pos in state["positions"].items():
            if name in self.positions:
                self.positions[name] = [int(pos[0]), int(pos[1])]

    # -- sampling
    def _perm(self, i: int, epoch: int) -> np.ndarray:
        key = (i, epoch)
        if key not in self._perm_cache:
            for k in [k for k in self._perm_cache if k[0] == i]:
                del self._perm_cache[k]             # keep one epoch per source
            seed = (self.seed * 1_000_003 + i * 7919) & 0xFFFFFFFF
            self._perm_cache[key] = epoch_permutation(self.shards[i].n_windows, seed, epoch)
        return self._perm_cache[key]

    def _take(self, i: int, count: int) -> np.ndarray:
        name, n = self.names[i], self.shards[i].n_windows
        epoch, cursor = self.positions[name]
        out = []
        while count > 0:
            if cursor >= n:
                epoch, cursor = epoch + 1, 0
            k = min(count, n - cursor)
            out.append(self._perm(i, epoch)[cursor:cursor + k])
            cursor += k
            count -= k
        self.positions[name] = [epoch, cursor]
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def source_draws(self, batch_index: int) -> np.ndarray:
        phase_id = zlib_crc(self.phase)
        rng = np.random.default_rng([self.seed & 0xFFFFFFFF, phase_id, int(batch_index)])
        return rng.choice(len(self.shards), size=self.batch_size, p=self.weights)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (tokens int64 (B, block_size + 1), source index per row) and advance."""
        draws = self.source_draws(self.batch)
        T = self.shards[0].block_size
        out = np.empty((self.batch_size, T + 1), dtype=np.int64)
        for i in np.unique(draws):
            rows = np.nonzero(draws == i)[0]
            out[rows] = self.shards[i].batch(self._take(int(i), rows.size))
        self.batch += 1
        return out, draws

    def __iter__(self) -> "MixtureLoader":
        return self

    def __next__(self):
        import torch

        tokens, draws = self.next_batch()
        return torch.from_numpy(tokens), torch.from_numpy(draws)

    def close(self) -> None:
        for s in self.shards:
            s.close()


def zlib_crc(text: str) -> int:
    import zlib

    return zlib.crc32(text.encode("utf-8"))


def mixture_val_sets(mix_dir: str | os.PathLike, phase: str, block_size: int,
                     mode: str = "memmap") -> dict[str, tuple[float, ShardSet]]:
    """Validation windows per source: ``{name: (share, ShardSet)}`` (sources without val data are left out)."""
    out = {}
    for src in _mix_sources(mix_dir, phase):
        d = Path(mix_dir) / src["dir"]
        if not (d / MANIFEST_NAME).exists():
            continue
        ss = ShardSet(d, "val", block_size, mode=mode)
        if ss.n_windows > 0:
            out[src["name"]] = (float(src["share"]), ss)
    return out
