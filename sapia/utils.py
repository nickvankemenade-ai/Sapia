"""Small shared helpers: atomic file writes, logging, formatting, seeding."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

GIB = 1024**3
MIB = 1024**2


def fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


def fmt_count(n: float) -> str:
    n = float(n)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.2f}{suffix}"
    return f"{n:.0f}"


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _fsync_quiet(fh) -> None:
    try:
        fh.flush()
        os.fsync(fh.fileno())
    except OSError:
        # Some network filesystems reject fsync; the rename below is still atomic.
        pass


def replace_with_retry(src: str | os.PathLike, dst: str | os.PathLike, attempts: int = 10) -> None:
    """os.replace with a short retry loop.

    On Windows a destination that is briefly held open by another process (antivirus,
    indexer) makes os.replace raise PermissionError; retrying a few times is enough.
    """
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        _fsync_quiet(fh)
    replace_with_retry(tmp, path)


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=False).encode("utf-8"))


def read_json(path: str | os.PathLike, attempts: int = 10) -> Any:
    """Read a JSON file, retrying briefly on Windows sharing violations.

    On Windows, opening a file at the instant another process ``os.replace``-s it raises
    PermissionError (seen when the prep progress loop reads a worker's state file).
    """
    for i in range(attempts):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line from a hard kill; ignore it.
                continue
    return out


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "sapia", log_file: Optional[str | os.PathLike] = None) -> logging.Logger:
    """Return a logger writing to stdout and optionally to a file.

    Handlers are replaced on every call so repeated runs in one process (tests, notebooks)
    do not duplicate output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logger(logger)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
