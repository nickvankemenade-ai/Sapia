"""Shared fixtures. All tests are offline: data comes from small local parquet files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sapia.config import ModelConfig, TrainConfig  # noqa: E402
from sapia.data import write_manifest, write_shard_atomic  # noqa: E402


@pytest.fixture
def small_model_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=320, block_size=32, n_layer=2, n_head=2, n_kv_head=1, d_model=48, mlp_hidden=96)


def make_docs(n: int, seed: int = 0) -> list[str]:
    """Structured, highly predictable text so a tiny model's loss drops within a few steps."""
    rng = np.random.default_rng(seed)
    animals = ["cat", "dog", "fox", "owl", "bee", "elk"]
    docs = []
    for i in range(n):
        k = int(rng.integers(2, 6))
        a = animals[int(rng.integers(len(animals)))]
        docs.append(f"doc {i:05d}: " + " ".join(f"the {a} counts {j} and {j + 1}." for j in range(k)))
    return docs


def write_parquet(path: Path, docs: list[str], row_group_size: int = 50) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"text": docs, "id": [f"id-{i}" for i in range(len(docs))]})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


@pytest.fixture
def parquet_corpus(tmp_path: Path):
    """Three local parquet files of 200 documents each. Returns (glob pattern, list of per-file docs)."""
    src = tmp_path / "corpus"
    src.mkdir()
    per_file = []
    for f in range(3):
        docs = make_docs(200, seed=f)
        docs = [d.replace("doc ", f"file {f} doc ", 1) for d in docs]
        write_parquet(src / f"part_{f:03d}.parquet", docs)
        per_file.append(docs)
    return str(src / "*.parquet"), per_file


def write_token_dataset(data_dir: Path, train: list[np.ndarray], val: list[np.ndarray],
                        vocab_size: int = 259, tokenizer: str = "bytes", complete: bool = True) -> dict:
    """Write shards + manifest directly (bypassing data prep) for loader/trainer tests."""
    data_dir.mkdir(parents=True, exist_ok=True)
    shards = {"train": [], "val": []}
    for split, arrays in (("train", train), ("val", val)):
        for i, arr in enumerate(arrays):
            name = f"{split}_00000_{i:05d}.bin"
            write_shard_atomic(data_dir / name, arr)
            shards[split].append({"name": name, "tokens": int(arr.size), "docs": 1})
    manifest = {
        "version": 1, "complete": complete, "tokenizer": tokenizer, "vocab_size": vocab_size,
        "eot_token": 256, "dtype": "uint16", "byteorder": "little",
        "train_tokens": sum(s["tokens"] for s in shards["train"]),
        "val_tokens": sum(s["tokens"] for s in shards["val"]),
        "shards": shards,
    }
    write_manifest(data_dir, manifest)
    return manifest


def pattern_tokens(n: int, period: int = 17, offset: int = 0) -> np.ndarray:
    """A periodic token stream: trivially learnable, so loss must fall fast."""
    return ((np.arange(n) + offset) % period + 3).astype(np.uint16)


@pytest.fixture
def pattern_dataset(tmp_path: Path) -> Path:
    """A prepared mix: one main source and one anneal source of periodic tokens."""
    import json

    data_dir = tmp_path / "data"
    write_token_dataset(data_dir / "main" / "pattern",
                        train=[pattern_tokens(6000), pattern_tokens(5000, offset=5)],
                        val=[pattern_tokens(1500, offset=9)])
    write_token_dataset(data_dir / "anneal" / "pattern_late",
                        train=[pattern_tokens(4000, offset=2)], val=[pattern_tokens(800, offset=4)])
    src = lambda name, phase: {"name": name, "share": 1.0, "dir": f"{phase}/{name}", "target_tokens": 1,
                               "prepared": True, "train_tokens": 1, "val_tokens": 1}
    mix = {"version": 1, "total_tokens": 1, "anneal_fraction": 0.1, "tokenizer": "bytes",
           "phases": {"main": {"tokens": 1, "sources": [src("pattern", "main")]},
                      "anneal": {"tokens": 1, "sources": [src("pattern_late", "anneal")]}}}
    (data_dir / "mix.json").write_text(json.dumps(mix), encoding="utf-8")
    return data_dir


def small_train_cfg(model_cfg: ModelConfig, data_dir: Path, out_dir: Path, **overrides) -> TrainConfig:
    cfg = TrainConfig(
        preset="test", model=model_cfg, data_dir=str(data_dir), out_dir=str(out_dir),
        max_tokens=16 * 32 * 12, target_tokens_per_step=16 * 32, batch_size=16, grad_accum_steps=1,
        learning_rate=3e-3, warmup_tokens=16 * 32 * 2, eval_interval=6, eval_tokens=32 * 16,
        log_interval=2, ckpt_interval=6, ckpt_minutes=0.0, device="cpu", seed=7, compile=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
