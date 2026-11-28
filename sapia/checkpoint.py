"""Checkpointing: atomic writes, keep-last-N, latest-valid lookup, best model, RNG capture.

The checkpoints folder always holds the newest ``ckpt_<step>.pt`` files (full training
state, used for resuming; ``keep_last`` >= 1) and, once an eval has run, ``best.pt``
(weights of the lowest-val-loss model so far) with its ``best.json`` sidecar.

A checkpoint file ``ckpt_<step>.pt`` only ever appears under its final name after it has
been completely written and flushed (temp file + ``os.replace``), so a crash or
preemption during a save can never leave a truncated "latest" checkpoint. Leftover
``*.tmp`` files are ignored and cleaned up.

Every checkpoint also gets a ``ckpt_<step>.pt.sha256`` sidecar (size and SHA-256 of the
finished file, computed by reading it back). Loading verifies it, so a file damaged while
being copied between Kaggle sessions is detected and the next older checkpoint is used
instead. Older checkpoints are only pruned after the new one is written and verified.

The format is the same for every backend (CPU, GPU, PyTorch/XLA TPU, JAX TPU): plain CPU
PyTorch tensors with PyTorch parameter names and optimizer state, so a run can continue on
any of them.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from .utils import replace_with_retry

CKPT_RE = re.compile(r"^ckpt_(\d+)\.pt$")
CKPT_FORMAT = 1
SUM_SUFFIX = ".sha256"
BEST_NAME = "best.pt"
BEST_META_NAME = "best.json"


def checkpoint_path(ckpt_dir: str | os.PathLike, step: int) -> Path:
    return Path(ckpt_dir) / f"ckpt_{int(step):09d}.pt"


def list_checkpoints(ckpt_dir: str | os.PathLike) -> list[tuple[int, Path]]:
    """All finished checkpoints as (step, path), oldest first."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        return []
    found = []
    for p in d.iterdir():
        m = CKPT_RE.match(p.name)
        if m and p.is_file():
            found.append((int(m.group(1)), p))
    return sorted(found)


def clean_stale_tmp(ckpt_dir: str | os.PathLike) -> None:
    d = Path(ckpt_dir)
    if not d.is_dir():
        return
    for p in [*d.glob("ckpt_*.pt.tmp"), *d.glob("ckpt_*.pt" + SUM_SUFFIX + ".tmp"), *d.glob(BEST_NAME + ".tmp"),
              *d.glob(BEST_META_NAME + ".tmp")]:
        try:
            p.unlink()
        except OSError:
            pass


def _atomic_torch_save(state: dict, final: Path) -> None:
    """Write ``state`` to ``final`` via a fsynced temp file + atomic rename."""
    tmp = final.with_name(final.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(state, fh)
        try:
            fh.flush()
            os.fsync(fh.fileno())
        except OSError:
            pass
    replace_with_retry(tmp, final)


def file_digest(path: str | os.PathLike, chunk: int = 16 << 20) -> tuple[int, str]:
    """(size in bytes, SHA-256 hex) of a file, read in large chunks."""
    h, size = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            size += len(block)
    return size, h.hexdigest()


def write_checksum(path: str | os.PathLike) -> dict:
    """Read the finished file back and record its size and hash in ``<file>.sha256``."""
    path = Path(path)
    size, digest = file_digest(path)
    meta = {"file": path.name, "bytes": size, "sha256": digest}
    side = path.with_name(path.name + SUM_SUFFIX)
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    replace_with_retry(tmp, side)
    return meta


def verify_checkpoint(path: str | os.PathLike) -> tuple[bool, str]:
    """Check a checkpoint against its sidecar. (True, reason) if it matches or has no sidecar."""
    path = Path(path)
    side = path.with_name(path.name + SUM_SUFFIX)
    if not side.is_file():
        return True, "no checksum file (older checkpoint)"
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return True, f"checksum file unreadable ({exc}); not verified"
    if not path.is_file():
        return False, "file missing"
    if path.stat().st_size != meta.get("bytes"):
        return False, f"size {path.stat().st_size:,} != recorded {meta.get('bytes'):,} bytes (truncated copy?)"
    if file_digest(path)[1] != meta.get("sha256"):
        return False, "SHA-256 mismatch (corrupted file)"
    return True, "checksum ok"


def save_checkpoint(ckpt_dir: str | os.PathLike, step: int, state: dict, keep_last: int = 2,
                    verify: bool = True) -> Path:
    """Atomically write ``state`` as the checkpoint for ``step``, checksum it, then prune old ones.

    Older checkpoints are deleted only after the new file is complete and its checksum is
    written, so at every moment at least one good checkpoint exists.
    """
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    final = checkpoint_path(d, step)
    state = dict(state)
    state["format"] = CKPT_FORMAT
    state["step"] = int(step)
    _atomic_torch_save(state, final)
    if verify:
        write_checksum(final)
    prune_checkpoints(d, keep_last)
    return final


# ----------------------------------------------------------------------------- best model

def best_checkpoint_path(ckpt_dir: str | os.PathLike) -> Path:
    return Path(ckpt_dir) / BEST_NAME


def save_best_checkpoint(ckpt_dir: str | os.PathLike, step: int, val_loss: float, state: dict) -> Path:
    """Atomically write the best-so-far model as ``best.pt`` plus a small ``best.json`` sidecar.

    ``best.pt`` is weights-only (no optimizer state), never matches ``ckpt_*.pt``, so it is
    neither pruned by keep-last-N nor picked up for resuming. ``best.json`` lets a resumed
    run know the best val loss without loading the weights.
    """
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    final = best_checkpoint_path(d)
    state = dict(state)
    state["format"] = CKPT_FORMAT
    state["step"] = int(step)
    state["val_loss"] = float(val_loss)
    _atomic_torch_save(state, final)
    meta = {"step": int(step), "val_loss": float(val_loss), "file": final.name}
    tmp_json = d / (BEST_META_NAME + ".tmp")
    tmp_json.write_text(json.dumps(meta), encoding="utf-8")
    replace_with_retry(tmp_json, d / BEST_META_NAME)
    return final


def read_best_meta(ckpt_dir: str | os.PathLike) -> Optional[dict]:
    """The ``best.json`` sidecar as a dict, or None if missing/unreadable or best.pt is gone."""
    d = Path(ckpt_dir)
    try:
        meta = json.loads((d / BEST_META_NAME).read_text(encoding="utf-8"))
        float(meta["val_loss"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not best_checkpoint_path(d).is_file():
        return None
    return meta


def prune_checkpoints(ckpt_dir: str | os.PathLike, keep_last: int) -> list[Path]:
    """Delete all but the ``keep_last`` newest checkpoints. Returns the deleted paths."""
    ckpts = list_checkpoints(ckpt_dir)
    removed = []
    for _step, path in ckpts[:-keep_last] if keep_last > 0 else []:
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            pass
        try:
            path.with_name(path.name + SUM_SUFFIX).unlink(missing_ok=True)
        except OSError:
            pass
    return removed


def load_checkpoint(path: str | os.PathLike, map_location: Any = "cpu") -> dict:
    """Load a checkpoint written by ``save_checkpoint``.

    ``weights_only=False`` is required because checkpoints carry Python/NumPy RNG state
    and config dicts. Only load checkpoints you created yourself.
    """
    ok, why = verify_checkpoint(path)
    if not ok:
        raise ValueError(f"{path} failed verification: {why}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or "model" not in state or "step" not in state:
        raise ValueError(f"{path} is not a Sapia checkpoint")
    return state


def load_latest_checkpoint(ckpt_dir: str | os.PathLike, map_location: Any = "cpu",
                           log: Callable[[str], None] = lambda s: None) -> tuple[Optional[dict], Optional[Path]]:
    """Load the newest readable checkpoint, falling back to older ones if one is corrupt."""
    for step, path in reversed(list_checkpoints(ckpt_dir)):
        try:
            return load_checkpoint(path, map_location=map_location), path
        except Exception as exc:  # noqa: BLE001 - any unreadable file is skipped
            log(f"WARNING: checkpoint {path.name} is unreadable ({type(exc).__name__}: {exc}); trying an older one")
    return None, None


# ----------------------------------------------------------------------------- RNG state

def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def restore_rng_state(state: Optional[dict], log: Callable[[str], None] = lambda s: None) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            if len(state["cuda"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # noqa: BLE001 - RNG restore is best effort
        log(f"WARNING: could not fully restore RNG state ({type(exc).__name__}: {exc})")


def strip_compile_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix torch.compile adds to parameter names."""
    prefix = "_orig_mod."
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
