"""Data preparation for one source: stream it, format, tokenize, write uint16 shards + manifest.

    python -m sapia.prepare_data --preset tiny --out_dir data/tiny        # ~0.5M tokens
    python -m sapia.prepare_mix  --total_tokens 15e9                      # the whole Sapia mix

Design
------
* The corpus is a list of *source files* (the parquet files of a Hugging Face dataset matching
  a glob, pinned to a commit, or local parquet/jsonl/txt files), optionally thinned with
  ``exclude`` and ``file_mod``/``file_keep``. Each source file is an independent unit of work
  handled by one worker process that streams it with ``datasets`` (streaming mode, only the
  columns the formatter needs), turns rows into documents with a formatter from
  ``sapia.sources`` (which may skip rows or insert special tokens), tokenizes batches of
  documents, appends the end-of-text token to every document and writes shards.
* Every ``val_every``-th document of each file goes to the validation split.
* Resumability: a worker commits a (train shard, val shard, per-file state) triple.
  Shards are written to a temp name and renamed; the per-file state JSON (documents
  consumed, shards written) is replaced atomically *after* the shards exist. A crash at
  any point therefore loses at most the uncommitted buffer, and a rerun continues from
  ``docs_done`` without duplicating or dropping a document. Only shards referenced by a
  state file ever reach the manifest; stray files from a crash are deleted on restart.
* The token target is a budget shared by all workers. The final shard is cut at a
  document boundary so the train split ends within one document of the target, and a
  later run with a larger target simply continues.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import glob
import json
import multiprocessing as mp
import os
import platform
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .autoconfig import plan_data_prep
from .config import ConfigError, F, add_config_args, apply_overrides
from .data import MANIFEST_VERSION, SHARD_DTYPE, write_manifest, write_shard_atomic
from .resources import Resources, detect_resources, read_cgroup_memory
from .sources import Special, columns_for, make_doc_fn
from .tokenizer import get_tokenizer
from .utils import atomic_write_json, fmt_count, fmt_duration, read_json

STATE_DIR = "prep_state"
SOURCE_FILE = "source.json"


class PrepError(RuntimeError):
    """Data preparation failed in a way that needs attention (state on disk stays resumable)."""


class PrepInterrupted(RuntimeError):
    """Deliberate interruption (fault injection in tests); never retried."""


@dataclass
class PrepConfig:
    out_dir: str = F("data/fineweb-edu", "Output directory for shards, manifest.json and resume state.")
    dataset: str = F("HuggingFaceFW/fineweb-edu", "Hugging Face dataset repo (parquet-backed).")
    files: str = F("sample/10BT/*.parquet", "Glob of the files to read inside the dataset repo.")
    revision: Optional[str] = F(None, "Dataset revision; auto = resolve the current commit once and pin it.")
    data_files: Optional[str] = F(None, "Comma-separated local paths/globs or URLs (parquet, jsonl, txt) to use "
                                        "instead of a hub dataset.")
    exclude: Optional[str] = F(None, "Regex on source file paths; matching files are skipped.")
    file_mod: int = F(1, "Keep source file i only if i % file_mod is in file_keep.")
    file_keep: str = F("0", "Comma-separated residues for file_mod.")
    formatter: str = F("text", "Row-to-document formatter (see sapia.sources.FORMATTERS).")
    formatter_args: str = F("{}", "JSON object of formatter arguments, e.g. '{\"min_int_score\": 4}'.")
    extra_columns: str = F("", "Comma-separated extra columns to read (e.g. int_score for score filters).")
    key_column: Optional[str] = F(None, "Column used for bucket hashing (auto = the text).")
    bucket: str = F("0,100", "Keep documents whose crc32(key) % 100 is in [lo, hi).")
    tokenizer: str = F("smollm2", "smollm2 (SmolLM2 BPE) | bytes (debug).")
    target_tokens: int = F(5_000_000_000, "Train-split token target.")
    val_every: int = F(500, "Every N-th document of each source file goes to the val split (0 = no val split).")
    shard_tokens: Optional[int] = F(None, "Tokens per shard. auto = 100M, reduced to fit RAM.")
    num_workers: Optional[int] = F(None, "Worker processes. auto = min(files, CPUs, RAM budget).")
    threads_per_worker: Optional[int] = F(None, "Tokenizer threads per worker. auto = CPUs / workers.")
    read_batch_docs: int = F(256, "Documents per streaming/tokenization batch.")
    max_files: Optional[int] = F(None, "Use only the first N source files.")
    max_retries: int = F(5, "Retries per source file on transient (network) errors, with exponential backoff.")
    progress_seconds: float = F(30.0, "Seconds between progress lines.")
    stall_minutes: float = F(15.0, "With worker processes: exit (code 75, resumable) if no tokens arrive for this "
                                   "long (0 disables).")

    def bucket_range(self) -> tuple[int, int]:
        lo, hi = (int(x) for x in self.bucket.split(","))
        return lo, hi

    def keep_residues(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.file_keep.split(",") if x.strip())

    def formatter_kwargs(self) -> dict:
        return json.loads(self.formatter_args or "{}")

    def columns(self) -> list[str]:
        extra = [c.strip() for c in self.extra_columns.split(",") if c.strip()]
        return columns_for(self.formatter, self.key_column, extra)

    def validate(self) -> None:
        if self.target_tokens <= 0:
            raise ConfigError("target_tokens must be positive")
        try:
            lo, hi = self.bucket_range()
        except ValueError:
            raise ConfigError(f"bucket must be 'lo,hi', got {self.bucket!r}") from None
        if not 0 <= lo < hi <= 100:
            raise ConfigError("bucket must satisfy 0 <= lo < hi <= 100")
        if self.file_mod < 1 or not self.keep_residues():
            raise ConfigError("file_mod must be >= 1 and file_keep non-empty")
        try:
            if not isinstance(self.formatter_kwargs(), dict):
                raise ValueError
        except ValueError:
            raise ConfigError(f"formatter_args must be a JSON object, got {self.formatter_args!r}") from None
        try:
            make_doc_fn(self.formatter, self.formatter_kwargs())
        except ValueError as e:
            raise ConfigError(str(e)) from None
        if self.val_every < 0:
            raise ConfigError("val_every must be >= 0")
        if self.read_batch_docs < 1:
            raise ConfigError("read_batch_docs must be >= 1")
        if self.shard_tokens is not None and self.shard_tokens < 1:
            raise ConfigError("shard_tokens must be >= 1 (or auto)")


def _prep_default() -> PrepConfig:
    return PrepConfig()


def _prep_tiny() -> PrepConfig:
    """About 0.5M tokens from the first row group of the first source file (a few MB of download)."""
    return PrepConfig(out_dir="data/tiny", target_tokens=500_000, val_every=20, shard_tokens=200_000,
                      max_files=1, read_batch_docs=64, progress_seconds=5.0)


PREP_PRESETS = {"fineweb-edu": _prep_default, "tiny": _prep_tiny}


# ----------------------------------------------------------------------------- shared token budget

class _LocalCounter:
    """In-process stand-in for ``multiprocessing.Value``."""

    def __init__(self, value: int = 0):
        self.value = value
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


class TokenBudget:
    """Train tokens committed so far, against the target; shared by all workers."""

    def __init__(self, target: int, counter):
        self.target = int(target)
        self.counter = counter

    def total(self) -> int:
        with self.counter.get_lock():
            return int(self.counter.value)

    def remaining(self) -> int:
        return max(0, self.target - self.total())

    def reserve_prefix(self, doc_lengths: np.ndarray) -> tuple[int, int]:
        """Reserve tokens for the longest prefix of documents that is still wanted.

        Returns (number of documents granted, their token count). The last granted
        document may cross the target, so the overshoot is less than one document.
        """
        with self.counter.get_lock():
            remaining = self.target - int(self.counter.value)
            if remaining <= 0 or len(doc_lengths) == 0:
                return 0, 0
            cum = np.cumsum(np.asarray(doc_lengths, dtype=np.int64))
            k = min(int(np.searchsorted(cum, remaining, side="left")) + 1, len(cum))
            granted = int(cum[k - 1])
            self.counter.value = int(self.counter.value) + granted
            return k, granted

    def release(self, tokens: int) -> None:
        with self.counter.get_lock():
            self.counter.value = max(0, int(self.counter.value) - int(tokens))


STALL_EXIT_CODE = 75   # "temporary failure": rerun the same command to resume


def start_stall_watchdog(heartbeat, stall_minutes: float, log: Callable[[str], None],
                         stop: threading.Event) -> Optional[threading.Thread]:
    """Exit the whole job if the workers stop reading documents for ``stall_minutes``.

    ``heartbeat`` is a shared counter every worker bumps after each batch of rows it reads
    (also rows its filters skip, so heavy filtering never looks like a stall). A worker can hang
    without raising: a network read that never returns, or a worker killed while holding the
    token counter's lock, which then blocks everyone, including this process's progress logging.
    The watchdog runs on its own thread and reads the heartbeat without any lock, so it still
    works in exactly those cases. On a stall it kills the worker processes and exits with code
    75; committed work is on disk, so rerunning the same command resumes.
    """
    if not stall_minutes or stall_minutes <= 0:
        return None
    raw = heartbeat.get_obj()

    def watch() -> None:
        last, since = raw.value, time.monotonic()
        while not stop.wait(min(30.0, stall_minutes * 60 / 4)):
            now = raw.value
            if now != last:
                last, since = now, time.monotonic()
                continue
            if time.monotonic() - since >= stall_minutes * 60:
                log(f"[prep] STALLED: no documents read for {stall_minutes:g} minutes; stopping the "
                    f"workers and exiting with code {STALL_EXIT_CODE}. Rerun the same command to resume.")
                for child in mp.active_children():
                    try:
                        child.kill()
                    except Exception:  # noqa: BLE001
                        pass
                os._exit(STALL_EXIT_CODE)

    thread = threading.Thread(target=watch, name="prep-stall-watchdog", daemon=True)
    thread.start()
    return thread


_WORKER_BUDGET: Optional[TokenBudget] = None
_WORKER_HEARTBEAT = None


def _init_worker(counter, target: int, heartbeat=None) -> None:
    global _WORKER_BUDGET, _WORKER_HEARTBEAT
    _WORKER_BUDGET = TokenBudget(target, counter)
    _WORKER_HEARTBEAT = heartbeat.get_obj() if heartbeat is not None else None


# ----------------------------------------------------------------------------- per-file work

@dataclass
class FileTask:
    file_idx: int
    source: dict              # {"path": ..., "format": "parquet" | "json" | "text"}
    out_dir: str
    tokenizer: str
    formatter: str
    formatter_args: dict
    columns: list
    key_column: Optional[str]
    bucket: tuple
    val_every: int
    shard_tokens: int
    threads: int
    read_batch_docs: int
    max_retries: int
    # Fault injection for tests: raise PrepInterrupted on the N-th commit of this
    # invocation, either "after_commit" or "before_state" (shards written, state not).
    fail_after_commits: Optional[int] = None
    fail_mode: str = "after_commit"


def file_state_path(out_dir: str | os.PathLike, file_idx: int) -> Path:
    return Path(out_dir) / STATE_DIR / f"file_{file_idx:05d}.json"


def load_file_state(out_dir: str | os.PathLike, file_idx: int, source_path: str) -> dict:
    path = file_state_path(out_dir, file_idx)
    if path.exists():
        state = read_json(path)
        if state.get("source") != source_path:
            raise PrepError(f"{path} belongs to a different source file ({state.get('source')!r})")
        return state
    return {"file_idx": file_idx, "source": source_path, "docs_done": 0, "next_shard": 0, "done": False,
            "train_tokens": 0, "val_tokens": 0, "shards": []}


def shard_name(split: str, file_idx: int, shard_idx: int) -> str:
    return f"{split}_{file_idx:05d}_{shard_idx:05d}.bin"


def detect_format(path: str) -> str:
    lower = path.lower()
    for suffix in (".gz", ".zst", ".bz2", ".xz"):
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]
    if lower.endswith(".parquet"):
        return "parquet"
    if lower.endswith(".jsonl") or lower.endswith(".json") or lower.endswith(".ndjson"):
        return "json"
    if lower.endswith(".txt"):
        return "text"
    raise PrepError(f"cannot infer the format of {path!r} (expected .parquet, .jsonl/.json or .txt)")


def open_stream(source: dict, columns: list[str]):
    """Open one source file as a ``datasets`` streaming IterableDataset."""
    from datasets import load_dataset

    kwargs = {}
    if source["format"] == "parquet":
        kwargs["columns"] = list(columns)  # read only the needed columns from the parquet file
    return load_dataset(source["format"], data_files=[source["path"]], split="train", streaming=True, **kwargs)


def special_ids(tok) -> dict[str, int]:
    ids = {"eot": tok.eot_token}
    for name in ("im_start", "im_end"):
        tid = getattr(tok, f"{name}_token", None)
        if tid is not None:
            ids[name] = tid
    return ids


def encode_docs(docs: list, tok, threads: int) -> list[Optional[list[int]]]:
    """Tokenize formatted documents (str, list of str/Special, or None) in one batch call."""
    texts: list[str] = []
    for doc in docs:
        if isinstance(doc, str):
            texts.append(doc)
        elif isinstance(doc, list):
            texts.extend(p for p in doc if isinstance(p, str))
    encoded = iter(tok.encode_batch(texts, threads)) if texts else iter(())
    specials = special_ids(tok)
    out: list[Optional[list[int]]] = []
    for doc in docs:
        if doc is None or (isinstance(doc, str) and not doc):
            out.append(None)
        elif isinstance(doc, str):
            out.append(next(encoded))
        else:
            ids: list[int] = []
            for piece in doc:
                if isinstance(piece, Special):
                    if piece.name not in specials:
                        raise PrepError(f"tokenizer {tok.name!r} has no special token {piece.name!r}")
                    ids.append(specials[piece.name])
                else:
                    ids.extend(next(encoded))
            out.append(ids)
    return out


# ----------------------------------------------------------------------------- memory back-pressure

MEM_CHECK_SECONDS = 2.0      # how often a worker looks at the container's memory
MEM_FLUSH_FRACTION = 0.85    # above this share of the cgroup limit: commit the buffer early
MEM_WAIT_FRACTION = 0.93     # above this share even with an empty buffer: pause reading briefly
MEM_MAX_WAIT_S = 30.0


def memory_pressure() -> Optional[float]:
    """Non-reclaimable cgroup memory use as a fraction of the cgroup limit (None without a limit).

    This is the number the container gets killed on (Modal: the hard memory limit), so it
    is checked at runtime on top of the up-front estimate in ``plan_data_prep``.
    """
    cg = read_cgroup_memory()
    if cg.limit_bytes is None or cg.usage_bytes is None:
        return None
    used = max(0, cg.usage_bytes - (cg.inactive_file_bytes or 0))
    return used / cg.limit_bytes


def release_memory() -> None:
    """Hand freed memory back to the OS (Python GC + pyarrow's allocator)."""
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _limit_arrow_retention() -> None:
    """Make jemalloc (pyarrow's default allocator on Linux) return freed memory immediately."""
    try:
        import pyarrow as pa
        pa.jemalloc_set_decay_ms(0)
    except Exception:  # noqa: BLE001 - not built with jemalloc (e.g. Windows)
        pass


_NON_RETRYABLE = (PrepInterrupted, PrepError, ConfigError, KeyError, TypeError, AssertionError, MemoryError)


def process_file(task: FileTask, budget: Optional[TokenBudget] = None,
                 on_commit: Optional[Callable[[dict], None]] = None) -> dict:
    """Process one source file until it is exhausted or the token budget is used up.

    Returns the per-file state. Safe to call again after any interruption.
    """
    budget = budget if budget is not None else _WORKER_BUDGET
    if budget is None:
        raise PrepError("process_file needs a TokenBudget")
    state = load_file_state(task.out_dir, task.file_idx, task.source["path"])
    if state["done"] or budget.remaining() <= 0:
        return state
    tok = get_tokenizer(task.tokenizer)
    _limit_arrow_retention()
    failures = 0
    while True:
        try:
            _stream_file(task, state, tok, budget, on_commit)
            return state
        except _NON_RETRYABLE:
            raise
        except Exception as exc:  # noqa: BLE001 - network / IO errors are retried from the last commit
            failures += 1
            if failures > task.max_retries:
                raise PrepError(
                    f"source file {task.file_idx} ({task.source['path']}) failed {failures} times; "
                    f"last error: {type(exc).__name__}: {exc}. Progress is saved; rerun to resume."
                ) from exc
            wait = min(60.0, 2.0 ** failures)
            print(f"[prep] file {task.file_idx}: {type(exc).__name__}: {exc} -> retry {failures}/"
                  f"{task.max_retries} in {wait:.0f}s (resuming from the last commit)", flush=True)
            time.sleep(wait)
            state = load_file_state(task.out_dir, task.file_idx, task.source["path"])


def _stream_file(task: FileTask, state: dict, tok, budget: TokenBudget,
                 on_commit: Optional[Callable[[dict], None]]) -> None:
    ds = open_stream(task.source, task.columns)
    doc_fn = make_doc_fn(task.formatter, task.formatter_args, tuple(task.bucket), task.key_column)
    if state["docs_done"]:
        ds = ds.skip(state["docs_done"])

    eot = tok.eot_token
    out_dir = Path(task.out_dir)
    train_docs: list[np.ndarray] = []
    train_idx: list[int] = []      # source-document index of each buffered train doc
    val_docs: list[np.ndarray] = []
    val_idx: list[int] = []
    n_train = 0
    doc_idx = int(state["docs_done"])  # index of the next document to read from the source file
    commits = 0
    last_mem_check = time.monotonic()

    def flush(final: bool) -> bool:
        """Commit buffered documents. Returns True when the worker should stop (budget used up)."""
        nonlocal train_docs, train_idx, val_docs, val_idx, n_train, commits
        lengths = np.fromiter((a.size for a in train_docs), dtype=np.int64, count=len(train_docs))
        k, granted = budget.reserve_prefix(lengths)
        truncated = k < len(train_docs)
        if truncated and k == 0:
            return True  # nothing more is wanted; leave the state untouched
        cut_doc = (train_idx[k - 1] + 1) if truncated else doc_idx
        try:
            shard_idx = int(state["next_shard"])
            entries = []
            if k > 0:
                name = shard_name("train", task.file_idx, shard_idx)
                tokens = write_shard_atomic(out_dir / name, np.concatenate(train_docs[:k]))
                entries.append({"name": name, "split": "train", "tokens": tokens, "docs": k})
            kept_val = [a for a, i in zip(val_docs, val_idx) if i < cut_doc]
            if kept_val:
                name = shard_name("val", task.file_idx, shard_idx)
                tokens = write_shard_atomic(out_dir / name, np.concatenate(kept_val))
                entries.append({"name": name, "split": "val", "tokens": tokens, "docs": len(kept_val)})
            commits += 1
            if task.fail_after_commits and commits >= task.fail_after_commits and task.fail_mode == "before_state":
                raise PrepInterrupted("injected failure between shard write and state update")
            state["shards"].extend(entries)
            state["docs_done"] = int(cut_doc)
            state["next_shard"] = shard_idx + 1
            state["train_tokens"] = sum(e["tokens"] for e in state["shards"] if e["split"] == "train")
            state["val_tokens"] = sum(e["tokens"] for e in state["shards"] if e["split"] == "val")
            state["done"] = bool(final and not truncated)
            atomic_write_json(file_state_path(task.out_dir, task.file_idx), state)
        except BaseException:
            budget.release(granted)
            raise
        train_docs, train_idx, val_docs, val_idx, n_train = [], [], [], [], 0
        if on_commit is not None:
            on_commit(state)
        if task.fail_after_commits and commits >= task.fail_after_commits:
            raise PrepInterrupted("injected failure after a commit")
        return truncated

    for batch in ds.iter(batch_size=task.read_batch_docs):
        keys = list(batch)
        n_rows = len(batch[keys[0]]) if keys else 0
        rows = [{k: batch[k][i] for k in keys} for i in range(n_rows)]
        if _WORKER_HEARTBEAT is not None:
            _WORKER_HEARTBEAT.value += n_rows        # lock-free on purpose (see start_stall_watchdog)
        for ids in encode_docs([doc_fn(r) for r in rows], tok, task.threads):
            if ids:
                arr = np.empty(len(ids) + 1, dtype=SHARD_DTYPE)
                arr[:-1] = ids
                arr[-1] = eot
                if task.val_every > 0 and doc_idx % task.val_every == 0:
                    val_docs.append(arr)
                    val_idx.append(doc_idx)
                else:
                    train_docs.append(arr)
                    train_idx.append(doc_idx)
                    n_train += arr.size
            doc_idx += 1  # skipped / empty rows still count, so docs_done matches ds.skip()
        if n_train >= min(task.shard_tokens, budget.remaining()):
            if flush(final=False):
                return
            release_memory()
        elif time.monotonic() - last_mem_check >= MEM_CHECK_SECONDS:
            last_mem_check = time.monotonic()
            pressure = memory_pressure()
            if pressure is not None and pressure >= MEM_FLUSH_FRACTION and train_docs:
                print(f"[prep] file {task.file_idx}: container memory at {pressure:.0%} of its limit; "
                      f"committing a {n_train:,}-token shard early", flush=True)
                if flush(final=False):
                    return
                release_memory()
                pressure = memory_pressure()
            waited = 0.0
            while pressure is not None and pressure >= MEM_WAIT_FRACTION and waited < MEM_MAX_WAIT_S:
                time.sleep(1.0)  # let the other workers commit and free their buffers
                waited += 1.0
                pressure = memory_pressure()
    flush(final=True)


# ----------------------------------------------------------------------------- source resolution

def resolve_sources(cfg: PrepConfig) -> dict:
    """Resolve the ordered list of source files (one network call for hub datasets)."""
    if cfg.data_files:
        paths: list[str] = []
        for pattern in [p.strip() for p in cfg.data_files.split(",") if p.strip()]:
            if "://" in pattern:
                paths.append(pattern)
                continue
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise PrepError(f"data_files pattern matched nothing: {pattern!r}")
            paths.extend(str(Path(m).resolve()) for m in matches)
        origin = {"kind": "files"}
    else:
        from huggingface_hub import HfApi, HfFileSystem

        revision = cfg.revision or HfApi().dataset_info(cfg.dataset).sha
        root = f"datasets/{cfg.dataset}@{revision}/"
        matches = sorted(HfFileSystem().glob(root + cfg.files))
        paths = ["hf://" + m for m in matches]
        if not paths:
            raise PrepError(f"no files match {cfg.files!r} in dataset {cfg.dataset!r} at {revision}")
        origin = {"kind": "hub", "dataset": cfg.dataset, "files": cfg.files, "revision": revision}
    if cfg.exclude:
        rx = re.compile(cfg.exclude)
        paths = [p for p in paths if not rx.search(p)]
    keep = set(cfg.keep_residues())
    paths = [p for i, p in enumerate(paths) if i % cfg.file_mod in keep]
    if cfg.max_files is not None:
        paths = paths[: cfg.max_files]
    if not paths:
        raise PrepError("no source files found")
    return {
        **origin,
        **_pinned_settings(cfg),
        "max_files": cfg.max_files,
        "files": [{"path": p, "format": detect_format(p)} for p in paths],
    }


def _pinned_settings(cfg: PrepConfig) -> dict:
    """Settings that decide which documents exist and how they are tokenized; fixed per out_dir."""
    return {
        "tokenizer": cfg.tokenizer,
        "formatter": cfg.formatter,
        "formatter_args": cfg.formatter_kwargs(),
        "key_column": cfg.key_column,
        "bucket": list(cfg.bucket_range()),
        "val_every": cfg.val_every,
    }


def load_or_create_source(cfg: PrepConfig, out_dir: Path, log: Callable[[str], None]) -> dict:
    """Pin the source list on the first run; verify it is unchanged on every later run."""
    path = out_dir / STATE_DIR / SOURCE_FILE
    if path.exists():
        source = read_json(path)
        for key, wanted in _pinned_settings(cfg).items():
            if source.get(key) != wanted:
                raise PrepError(
                    f"{cfg.out_dir} was prepared with {key}={source.get(key)!r} but this run asks for "
                    f"{wanted!r}. Use a new --out_dir (or delete the directory) to change it."
                )
        if cfg.max_files != source.get("max_files"):
            log(f"[prep] note: keeping the {len(source['files'])} source files pinned by the first run "
                f"(it used max_files={source.get('max_files')}; max_files={cfg.max_files} is ignored)")
        log(f"[prep] resuming: {len(source['files'])} pinned source files from {path}")
        return source
    source = resolve_sources(cfg)
    atomic_write_json(path, source)
    return source


# ----------------------------------------------------------------------------- manifest

def build_manifest(cfg: PrepConfig, source: dict, states: list[dict], complete: bool, tok) -> dict:
    shards: dict[str, list[dict]] = {"train": [], "val": []}
    for state in sorted(states, key=lambda s: s["file_idx"]):
        for entry in state["shards"]:
            shards[entry["split"]].append({"name": entry["name"], "tokens": entry["tokens"], "docs": entry["docs"]})
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("datasets", "tokenizers"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    return {
        "version": MANIFEST_VERSION,
        "complete": bool(complete),
        "tokenizer": tok.name,
        "vocab_size": int(tok.vocab_size),
        "eot_token": int(tok.eot_token),
        "dtype": "uint16",
        "byteorder": "little",
        "target_tokens": int(cfg.target_tokens),
        "train_tokens": sum(s["tokens"] for s in shards["train"]),
        "val_tokens": sum(s["tokens"] for s in shards["val"]),
        "train_docs": sum(s["docs"] for s in shards["train"]),
        "val_docs": sum(s["docs"] for s in shards["val"]),
        "val_every": int(cfg.val_every),
        "source": {k: v for k, v in source.items() if k != "files"},
        "source_files": [
            {"path": f["path"], "docs_done": st["docs_done"], "exhausted": st["done"]}
            for f, st in zip(source["files"], sorted(states, key=lambda s: s["file_idx"]))
        ],
        "shards": shards,
        "updated_unix": time.time(),
        "versions": versions,
    }


def clean_orphans(out_dir: Path, states: list[dict]) -> list[str]:
    """Delete temp files and shards that no state file references (left by a crash mid-commit)."""
    referenced = {e["name"] for s in states for e in s["shards"]}
    removed = []
    for p in list(out_dir.glob("*.bin.tmp")) + list(out_dir.glob("*.json.tmp")) + \
            list((out_dir / STATE_DIR).glob("*.tmp")):
        try:
            p.unlink()
            removed.append(p.name)
        except OSError:
            pass
    for p in list(out_dir.glob("train_*.bin")) + list(out_dir.glob("val_*.bin")):
        if p.name not in referenced:
            try:
                p.unlink()
                removed.append(p.name)
            except OSError:
                pass
    return removed


# ----------------------------------------------------------------------------- coordinator

def run_prepare(cfg: PrepConfig, *, resources: Optional[Resources] = None,
                on_progress: Optional[Callable[[dict], None]] = None,
                log: Callable[[str], None] = lambda s: print(s, flush=True),
                _fail_after_commits: Optional[int] = None, _fail_mode: str = "after_commit") -> dict:
    """Run (or resume) data preparation and return the final manifest.

    ``on_progress(manifest)`` is called after the manifest is refreshed (used on Modal to
    commit the Volume). Arguments starting with an underscore inject faults for tests.
    """
    cfg.validate()
    out_dir = Path(cfg.out_dir)
    (out_dir / STATE_DIR).mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer(cfg.tokenizer)  # validates the name and warms any on-disk tokenizer cache
    source = load_or_create_source(cfg, out_dir, log)
    files = source["files"]

    def read_states() -> list[dict]:
        return [load_file_state(out_dir, i, f["path"]) for i, f in enumerate(files)]

    states = read_states()
    removed = clean_orphans(out_dir, states)
    if removed:
        log(f"[prep] removed {len(removed)} stray file(s) from an interrupted run")
    done_tokens = sum(s["train_tokens"] for s in states)
    pending = [i for i, s in enumerate(states) if not s["done"]]

    def refresh_manifest(complete: bool) -> dict:
        manifest = build_manifest(cfg, source, read_states(), complete, tok)
        write_manifest(out_dir, manifest)
        if on_progress is not None:
            try:
                on_progress(manifest)
            except Exception as exc:  # noqa: BLE001 - progress hooks must not kill the job
                log(f"[prep] WARNING: on_progress hook failed: {type(exc).__name__}: {exc}")
        return manifest

    if done_tokens >= cfg.target_tokens or not pending:
        manifest = refresh_manifest(complete=True)
        why = "target reached" if done_tokens >= cfg.target_tokens else "all source files exhausted"
        log(f"[prep] nothing to do ({why}): {manifest['train_tokens']:,} train / {manifest['val_tokens']:,} val tokens")
        return manifest

    res = resources if resources is not None else detect_resources(with_gpu=False)
    plan = plan_data_prep(res, len(pending), shard_tokens=cfg.shard_tokens, num_workers=cfg.num_workers,
                          threads_per_worker=cfg.threads_per_worker,
                          target_tokens=cfg.target_tokens - done_tokens)
    log("[prep] resources:")
    for line in res.summary_lines():
        log(f"[prep]   {line}")
    log(f"[prep] plan: {plan.reason}")
    log(f"[prep] target {cfg.target_tokens:,} train tokens; already have {done_tokens:,}; "
        f"{len(pending)} of {len(files)} source files pending; tokenizer {tok.name}")

    def make_task(i: int) -> FileTask:
        return FileTask(
            file_idx=i, source=files[i], out_dir=str(out_dir), tokenizer=cfg.tokenizer,
            formatter=cfg.formatter, formatter_args=cfg.formatter_kwargs(), columns=cfg.columns(),
            key_column=cfg.key_column, bucket=cfg.bucket_range(),
            val_every=cfg.val_every, shard_tokens=plan.shard_tokens,
            threads=plan.threads_per_worker, read_batch_docs=cfg.read_batch_docs, max_retries=cfg.max_retries,
            fail_after_commits=_fail_after_commits, fail_mode=_fail_mode,
        )

    t0 = time.time()
    last_log = [t0]

    def progress(total: int, force: bool = False) -> None:
        now = time.time()
        if not force and now - last_log[0] < cfg.progress_seconds:
            return
        last_log[0] = now
        new = total - done_tokens
        rate = new / max(1e-9, now - t0)
        eta = (cfg.target_tokens - total) / rate if rate > 0 else float("inf")
        log(f"[prep] {total:,} / {cfg.target_tokens:,} train tokens ({100 * total / cfg.target_tokens:.1f}%) | "
            f"{fmt_count(rate)} tok/s | elapsed {fmt_duration(now - t0)} | "
            f"eta {fmt_duration(eta) if eta != float('inf') else 'n/a'}")
        refresh_manifest(complete=False)

    errors: list[BaseException] = []
    if plan.num_workers <= 1:
        budget = TokenBudget(cfg.target_tokens, _LocalCounter(done_tokens))
        for i in pending:
            if budget.remaining() <= 0:
                break
            process_file(make_task(i), budget, on_commit=lambda _s: progress(budget.total()))
    else:
        ctx = mp.get_context("spawn")  # same start method everywhere; safe with threads in the parent
        counter = ctx.Value("q", done_tokens)
        budget = TokenBudget(cfg.target_tokens, counter)
        heartbeat = ctx.Value("q", 0)
        stop_watchdog = threading.Event()
        start_stall_watchdog(heartbeat, cfg.stall_minutes, log, stop_watchdog)
        with cf.ProcessPoolExecutor(max_workers=plan.num_workers, mp_context=ctx, initializer=_init_worker,
                                    initargs=(counter, cfg.target_tokens, heartbeat)) as pool:
            futures = {pool.submit(process_file, make_task(i)): i for i in pending}
            not_done = set(futures)
            try:
                while not_done:
                    finished, not_done = cf.wait(not_done, timeout=min(5.0, cfg.progress_seconds),
                                                 return_when=cf.FIRST_COMPLETED)
                    for fut in finished:
                        exc = fut.exception()
                        if exc is not None:
                            errors.append(exc)
                            log(f"[prep] ERROR in source file {futures[fut]}: {type(exc).__name__}: {exc}")
                    if errors:
                        for fut in not_done:
                            fut.cancel()  # files not started yet are skipped; running ones finish their work
                    progress(budget.total())
            except BaseException:
                for fut in not_done:
                    fut.cancel()
                raise
            finally:
                stop_watchdog.set()

    states = read_states()
    total = sum(s["train_tokens"] for s in states)
    complete = (total >= cfg.target_tokens or all(s["done"] for s in states)) and not errors
    manifest = refresh_manifest(complete=complete)
    clean_orphans(out_dir, states)
    log(f"[prep] {'done' if complete else 'stopped'}: {manifest['train_tokens']:,} train tokens in "
        f"{len(manifest['shards']['train'])} shards, {manifest['val_tokens']:,} val tokens in "
        f"{len(manifest['shards']['val'])} shards, {fmt_duration(time.time() - t0)} elapsed")
    if errors:
        raise PrepError(f"{len(errors)} source file(s) failed; first error: "
                        f"{type(errors[0]).__name__}: {errors[0]}. Progress is saved; rerun to resume.") from errors[0]
    if total < cfg.target_tokens:
        log(f"[prep] WARNING: the source was exhausted at {total:,} tokens, below the target "
            f"{cfg.target_tokens:,}")
    return manifest


# ----------------------------------------------------------------------------- CLI

def parse_prep_args(argv: Optional[list[str]] = None) -> PrepConfig:
    parser = argparse.ArgumentParser(
        prog="python -m sapia.prepare_data",
        description="Stream one source from the Hugging Face hub (or local files), format and tokenize it and "
                    "write resumable uint16 shards. Rerun the same command to resume after an interruption. "
                    "For the full Sapia mix use python -m sapia.prepare_mix.",
    )
    parser.add_argument("--preset", default="fineweb-edu", choices=sorted(PREP_PRESETS),
                        help="Starting point for all options below.")
    add_config_args(parser, PrepConfig)
    ns = vars(parser.parse_args(argv))
    cfg = PREP_PRESETS[ns.pop("preset")]()
    apply_overrides(cfg, ns)
    cfg.validate()
    return cfg


def main(argv: Optional[list[str]] = None) -> None:
    cfg = parse_prep_args(argv)
    print(f"[prep] config: {asdict(cfg)}", flush=True)
    run_prepare(cfg)


if __name__ == "__main__":
    main()
