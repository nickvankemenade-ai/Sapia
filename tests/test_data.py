"""Shard format, loader determinism/resume, and the resumable data-prep pipeline (offline)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from sapia.data import (DataError, ShardSet, TrainLoader, WindowSampler, epoch_permutation, iter_eval_batches,
                          load_manifest, read_shard, write_shard_atomic)
from sapia.prepare_data import (PrepConfig, PrepError, PrepInterrupted, TokenBudget, _LocalCounter,
                                  run_prepare)
from sapia.resources import detect_resources
from sapia.tokenizer import ByteTokenizer, get_tokenizer

from conftest import make_docs, pattern_tokens, write_parquet, write_token_dataset

EOT = ByteTokenizer.eot_token


# ----------------------------------------------------------------------------- shards

def test_shard_round_trip(tmp_path):
    tokens = np.random.default_rng(0).integers(0, 65536, size=10_001, dtype=np.uint16)
    path = tmp_path / "train_00000_00000.bin"
    assert write_shard_atomic(path, tokens) == tokens.size
    assert path.stat().st_size == 2 * tokens.size
    assert not list(tmp_path.glob("*.tmp")), "no temp file may remain"
    assert np.array_equal(read_shard(path), tokens)
    mm = read_shard(path, mmap=True)
    assert isinstance(mm, np.memmap) and np.array_equal(np.asarray(mm), tokens)
    del mm


def test_shard_is_little_endian_uint16(tmp_path):
    path = tmp_path / "x.bin"
    write_shard_atomic(path, np.array([1, 258, 50256], dtype=np.int64))
    assert path.read_bytes() == bytes([1, 0, 2, 1, 0x50, 0xC4])


def test_manifest_verification(tmp_path):
    with pytest.raises(DataError, match="Run data preparation first"):
        load_manifest(tmp_path)
    write_token_dataset(tmp_path, [pattern_tokens(500)], [pattern_tokens(100)])
    manifest = load_manifest(tmp_path)
    assert manifest["train_tokens"] == 500
    (tmp_path / "train_00000_00000.bin").write_bytes(b"\x00" * 10)      # truncated shard
    with pytest.raises(DataError, match="size"):
        load_manifest(tmp_path)
    (tmp_path / "train_00000_00000.bin").unlink()
    with pytest.raises(DataError, match="missing"):
        load_manifest(tmp_path)


@pytest.mark.parametrize("mode", ["ram", "memmap"])
def test_shardset_windows(tmp_path, mode):
    a, b = np.arange(0, 103, dtype=np.uint16), np.arange(1000, 1045, dtype=np.uint16)
    write_token_dataset(tmp_path, [a, b], [a[:30]])
    shards = ShardSet(tmp_path, "train", block_size=10, mode=mode)
    # 103 tokens -> (103-1)//10 = 10 windows; 45 tokens -> 4 windows
    assert shards.n_windows == 14 and shards.n_tokens == 148
    batch = shards.batch(np.array([0, 9, 10, 13]))
    assert batch.shape == (4, 11) and batch.dtype == np.int64
    assert np.array_equal(batch[0], np.arange(0, 11))
    assert np.array_equal(batch[1], np.arange(90, 101))
    assert np.array_equal(batch[2], np.arange(1000, 1011))
    assert np.array_equal(batch[3], np.arange(1030, 1041))
    with pytest.raises(IndexError):
        shards.batch(np.array([14]))
    shards.close()


def test_ram_and_memmap_modes_agree_and_pickle_as_memmap(tmp_path):
    write_token_dataset(tmp_path, [pattern_tokens(999), pattern_tokens(777, offset=3)], [pattern_tokens(50)])
    ram = ShardSet(tmp_path, "train", 16, mode="ram").load()
    mm = ShardSet(tmp_path, "train", 16, mode="memmap").load()
    ids = np.arange(ram.n_windows)
    assert np.array_equal(ram.batch(ids), mm.batch(ids))
    clone = pickle.loads(pickle.dumps(ram))            # what a DataLoader worker receives
    assert clone.mode == "memmap" and clone._arrays is None
    assert len(pickle.dumps(ram)) < 5000, "in-RAM arrays must not be pickled into workers"
    assert np.array_equal(clone.batch(ids), ram.batch(ids))
    for s in (ram, mm, clone):
        s.close()


# ----------------------------------------------------------------------------- sampling order / resume

def test_epoch_permutation_is_deterministic():
    p = epoch_permutation(1000, seed=3, epoch=0)
    assert sorted(p.tolist()) == list(range(1000))
    assert np.array_equal(p, epoch_permutation(1000, seed=3, epoch=0))
    assert not np.array_equal(p, epoch_permutation(1000, seed=3, epoch=1))
    assert not np.array_equal(p, epoch_permutation(1000, seed=4, epoch=0))


def test_sampler_visits_each_window_once_per_epoch():
    sampler = WindowSampler(n_windows=103, batch_size=10, seed=1)
    it = iter(sampler)
    seen, positions = [], []
    for _ in range(10):                                 # 10 full batches = one epoch (3 windows dropped)
        epoch, cursor, ids = next(it)
        assert epoch == 0
        seen.extend(ids.tolist())
        positions.append(cursor)
    assert len(seen) == len(set(seen)) == 100
    assert positions == list(range(10, 101, 10))
    epoch, cursor, ids = next(it)
    assert (epoch, cursor) == (1, 10)


def test_sampler_resumes_exactly_even_with_another_batch_size():
    def windows(batch_size, epoch, cursor, n):
        it = iter(WindowSampler(200, batch_size, seed=5, epoch=epoch, cursor=cursor))
        out = []
        while len(out) < n:
            e, c, ids = next(it)
            out.extend(ids.tolist())
        return out[:n], (e, c)

    straight, _ = windows(8, 0, 0, 160)
    first, (e, c) = windows(8, 0, 0, 64)
    assert (e, c) == (0, 64)
    rest, _ = windows(8, e, c, 96)
    assert first + rest == straight
    rest_other_batch, _ = windows(16, e, c, 96)         # e.g. the GPU changed between runs
    assert first + rest_other_batch == straight


def test_sampler_rejects_too_small_datasets():
    with pytest.raises(DataError, match="only 3 windows"):
        WindowSampler(3, 8, seed=0)


def test_train_loader_tracks_position_and_resumes(tmp_path):
    write_token_dataset(tmp_path, [pattern_tokens(3000), pattern_tokens(2000, offset=4)], [pattern_tokens(100)])
    shards = ShardSet(tmp_path, "train", 16, mode="ram")
    loader = TrainLoader(shards, batch_size=8, seed=11)
    batches = [next(loader) for _ in range(6)]
    tokens, epoch, cursor = batches[3]
    assert tokens.shape == (8, 17) and str(tokens.dtype) == "torch.int64"
    assert (epoch, cursor) == (0, 32)
    resumed = TrainLoader(shards, batch_size=8, seed=11, epoch=epoch, cursor=cursor)
    for expected in batches[4:]:
        got = next(resumed)
        assert np.array_equal(got[0].numpy(), expected[0].numpy()) and got[1:] == expected[1:]
    loader.close(), resumed.close(), shards.close()


def test_train_loader_with_worker_process_matches_inline(tmp_path):
    """The memmap + DataLoader-worker path (used when the dataset does not fit in RAM)."""
    write_token_dataset(tmp_path, [pattern_tokens(3000), pattern_tokens(2000, offset=4)], [pattern_tokens(100)])
    inline = TrainLoader(ShardSet(tmp_path, "train", 16, mode="memmap"), batch_size=8, seed=11)
    worker = TrainLoader(ShardSet(tmp_path, "train", 16, mode="memmap"), batch_size=8, seed=11,
                         num_workers=1, prefetch_factor=2)
    try:
        for _ in range(5):
            a, b = next(inline), next(worker)
            assert np.array_equal(a[0].numpy(), b[0].numpy()) and a[1:] == b[1:]
    finally:
        inline.close(), worker.close()
        inline.shards.close(), worker.shards.close()


def test_eval_batches_are_fixed_and_sequential(tmp_path):
    write_token_dataset(tmp_path, [pattern_tokens(400)], [np.arange(0, 101, dtype=np.uint16)])
    val = ShardSet(tmp_path, "val", 10, mode="ram")
    batches = list(iter_eval_batches(val, batch_size=4, max_windows=7))
    assert [b.shape[0] for b in batches] == [4, 3]
    assert np.array_equal(batches[0][0], np.arange(0, 11)) and np.array_equal(batches[1][-1], np.arange(60, 71))
    val.close()


# ----------------------------------------------------------------------------- token budget

def test_token_budget_cuts_at_a_document_boundary():
    budget = TokenBudget(100, _LocalCounter(0))
    assert budget.reserve_prefix(np.array([30, 30])) == (2, 60)
    assert budget.remaining() == 40
    # 25 + 25 = 50 >= 40: the document that crosses the target is still granted, the third is not
    assert budget.reserve_prefix(np.array([25, 25, 25])) == (2, 50)
    assert budget.total() == 110 and budget.remaining() == 0
    assert budget.reserve_prefix(np.array([5])) == (0, 0)
    budget.release(50)
    assert budget.remaining() == 40
    assert budget.reserve_prefix(np.array([], dtype=np.int64)) == (0, 0)


# ----------------------------------------------------------------------------- data-prep pipeline

def _prep_cfg(out_dir: Path, pattern: str, **kw) -> PrepConfig:
    base = dict(out_dir=str(out_dir), data_files=pattern, tokenizer="bytes", target_tokens=10**9,
                val_every=10, shard_tokens=4000, num_workers=1, threads_per_worker=1, read_batch_docs=16,
                progress_seconds=0.0, max_retries=0)
    base.update(kw)
    return PrepConfig(**base)


def _quiet(_msg: str) -> None:
    pass


def _split_docs(data_dir: Path, split: str) -> list[str]:
    """Decode a split back into documents (each one is terminated by EOT)."""
    manifest = load_manifest(data_dir)
    tok = get_tokenizer("bytes")
    docs = []
    for shard in manifest["shards"][split]:
        arr = read_shard(data_dir / shard["name"])
        assert arr[-1] == EOT, "every shard ends at a document boundary"
        ends = np.flatnonzero(arr == EOT)
        assert len(ends) == shard["docs"]
        start = 0
        for end in ends:
            docs.append(tok.decode(arr[start:end]))
            start = end + 1
    return docs


def _all_bytes(data_dir: Path) -> dict[str, bytes]:
    manifest = load_manifest(data_dir)
    return {s["name"]: (data_dir / s["name"]).read_bytes() for split in ("train", "val") for s in manifest["shards"][split]}


def test_prep_writes_every_document_exactly_once(tmp_path, parquet_corpus):
    pattern, per_file = parquet_corpus
    out = tmp_path / "out"
    manifest = run_prepare(_prep_cfg(out, pattern), log=_quiet)
    assert manifest["complete"] is True                      # source exhausted below the (huge) target
    assert manifest["tokenizer"] == "bytes" and manifest["vocab_size"] == 259 and manifest["eot_token"] == EOT
    expected_val = [d for docs in per_file for i, d in enumerate(docs) if i % 10 == 0]
    expected_train = [d for docs in per_file for i, d in enumerate(docs) if i % 10 != 0]
    assert _split_docs(out, "train") == expected_train        # same documents, same order, none twice
    assert _split_docs(out, "val") == expected_val
    assert manifest["train_docs"] == 540 and manifest["val_docs"] == 60
    assert manifest["train_tokens"] == sum(len(d.encode()) + 1 for d in expected_train)
    assert len(manifest["shards"]["train"]) > 3, "several shards per file expected"
    assert all(f["exhausted"] and f["docs_done"] == 200 for f in manifest["source_files"])
    assert not list(out.glob("*.tmp"))


def test_prep_commits_early_under_memory_pressure(tmp_path, parquet_corpus, monkeypatch):
    """High container memory makes workers commit small shards early, without losing documents."""
    import sapia.prepare_data as prep
    pattern, per_file = parquet_corpus
    monkeypatch.setattr(prep, "MEM_CHECK_SECONDS", 0.0)
    monkeypatch.setattr(prep, "memory_pressure", lambda: 0.88)   # above flush (0.85), below wait (0.93)
    out = tmp_path / "out"
    manifest = run_prepare(_prep_cfg(out, pattern, shard_tokens=10**9), log=_quiet)
    assert manifest["complete"] is True
    # With a 1e9-token shard size, only the pressure check can produce more than one shard per file.
    assert len(manifest["shards"]["train"]) > len(per_file)
    expected_train = [d for docs in per_file for i, d in enumerate(docs) if i % 10 != 0]
    expected_val = [d for docs in per_file for i, d in enumerate(docs) if i % 10 == 0]
    assert _split_docs(out, "train") == expected_train
    assert _split_docs(out, "val") == expected_val


def test_memory_pressure_reads_the_cgroup(monkeypatch):
    import sapia.prepare_data as prep
    from sapia.resources import CgroupMemory
    monkeypatch.setattr(prep, "read_cgroup_memory",
                        lambda: CgroupMemory(version=1, limit_bytes=100, usage_bytes=90, inactive_file_bytes=10))
    assert prep.memory_pressure() == pytest.approx(0.8)
    monkeypatch.setattr(prep, "read_cgroup_memory", lambda: CgroupMemory())
    assert prep.memory_pressure() is None                       # no limit (e.g. Windows): guard is off


def test_prep_stops_at_the_token_target_and_can_be_extended(tmp_path, parquet_corpus):
    pattern, per_file = parquet_corpus
    out = tmp_path / "out"
    longest = max(len(d.encode()) + 1 for docs in per_file for d in docs)
    manifest = run_prepare(_prep_cfg(out, pattern, target_tokens=9000), log=_quiet)
    assert manifest["complete"] is True
    assert 9000 <= manifest["train_tokens"] < 9000 + longest
    first_docs = _split_docs(out, "train")

    # rerunning with the same target changes nothing
    before = _all_bytes(out)
    again = run_prepare(_prep_cfg(out, pattern, target_tokens=9000), log=_quiet)
    assert again["train_tokens"] == manifest["train_tokens"] and _all_bytes(out) == before

    # raising the target continues after the last document instead of starting over
    bigger = run_prepare(_prep_cfg(out, pattern, target_tokens=30000), log=_quiet)
    assert 30000 <= bigger["train_tokens"] < 30000 + longest
    docs = _split_docs(out, "train")
    assert docs[: len(first_docs)] == first_docs, "the earlier documents are kept as they were"
    assert len(docs) > len(first_docs) and len(docs) == len(set(docs)), "no duplicated documents"
    expected_train = [d for f_docs in per_file for i, d in enumerate(f_docs) if i % 10 != 0]
    assert sorted(docs) == sorted(expected_train[: len(docs)]), "a contiguous prefix of the corpus, nothing skipped"


@pytest.mark.parametrize("fail_mode", ["after_commit", "before_state"])
def test_prep_resume_after_interruption_is_lossless(tmp_path, parquet_corpus, fail_mode):
    pattern, _ = parquet_corpus
    reference = tmp_path / "reference"
    run_prepare(_prep_cfg(reference, pattern, target_tokens=40000), log=_quiet)

    out = tmp_path / "interrupted"
    cfg = _prep_cfg(out, pattern, target_tokens=40000)
    interruptions = 0
    for _ in range(50):
        try:
            # every invocation dies at its 2nd commit (either right after it, or after
            # writing the shards but before recording them in the state file)
            run_prepare(cfg, log=_quiet, _fail_after_commits=2, _fail_mode=fail_mode)
            break
        except PrepInterrupted:
            interruptions += 1
    else:
        pytest.fail("data prep never finished")
    assert interruptions >= 3, "the scenario must really have been interrupted several times"

    ref_manifest, manifest = load_manifest(reference), load_manifest(out)
    assert manifest["complete"] is True
    assert manifest["train_tokens"] == ref_manifest["train_tokens"]
    assert manifest["val_tokens"] == ref_manifest["val_tokens"]
    assert _split_docs(out, "train") == _split_docs(reference, "train")
    assert _split_docs(out, "val") == _split_docs(reference, "val")
    referenced = {s["name"] for split in ("train", "val") for s in manifest["shards"][split]}
    on_disk = {p.name for p in out.glob("*.bin")}
    assert on_disk == referenced, "no orphan shards may survive"
    assert not list(out.glob("*.tmp"))


def test_prep_with_worker_processes(tmp_path, parquet_corpus):
    pattern, per_file = parquet_corpus
    out = tmp_path / "out"
    manifest = run_prepare(_prep_cfg(out, pattern, num_workers=2, target_tokens=10**9), log=_quiet)
    assert manifest["complete"] is True
    expected_train = [d for docs in per_file for i, d in enumerate(docs) if i % 10 != 0]
    expected_val = [d for docs in per_file for i, d in enumerate(docs) if i % 10 == 0]
    assert _split_docs(out, "train") == expected_train
    assert _split_docs(out, "val") == expected_val


def test_prep_worker_processes_share_the_token_budget(tmp_path, parquet_corpus):
    pattern, per_file = parquet_corpus
    out = tmp_path / "out"
    longest = max(len(d.encode()) + 1 for docs in per_file for d in docs)
    manifest = run_prepare(_prep_cfg(out, pattern, num_workers=3, target_tokens=20000, shard_tokens=2000), log=_quiet)
    assert manifest["complete"] is True
    assert 20000 <= manifest["train_tokens"] < 20000 + longest
    docs = _split_docs(out, "train")
    assert len(docs) == len(set(docs))
    all_train = {d for f_docs in per_file for i, d in enumerate(f_docs) if i % 10 != 0}
    assert set(docs) <= all_train


def test_prep_refuses_to_mix_tokenizers(tmp_path, parquet_corpus):
    pattern, _ = parquet_corpus
    out = tmp_path / "out"
    run_prepare(_prep_cfg(out, pattern, target_tokens=5000), log=_quiet)
    with pytest.raises(PrepError, match="formatter_args"):
        run_prepare(_prep_cfg(out, pattern, target_tokens=5000, formatter_args='{"min_int_score": 4}'), log=_quiet)
    with pytest.raises(PrepError, match="val_every"):
        run_prepare(_prep_cfg(out, pattern, target_tokens=5000, val_every=3), log=_quiet)


def test_prep_handles_empty_rows_and_jsonl(tmp_path):
    import json

    docs = ["alpha", "", "beta", "gamma", "delta"]
    src = tmp_path / "docs.jsonl"
    src.write_text("\n".join(json.dumps({"text": d}) for d in docs) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    manifest = run_prepare(_prep_cfg(out, str(src), val_every=0), log=_quiet)
    assert _split_docs(out, "train") == ["alpha", "beta", "gamma", "delta"]
    assert manifest["val_tokens"] == 0 and manifest["shards"]["val"] == []
    assert manifest["source_files"][0]["docs_done"] == 5      # the empty row still counts as consumed


def test_prep_auto_plan_uses_detected_resources(tmp_path, parquet_corpus):
    pattern, _ = parquet_corpus
    lines: list[str] = []
    res = detect_resources(with_gpu=False)
    run_prepare(_prep_cfg(tmp_path / "out", pattern, target_tokens=3000, num_workers=None, shard_tokens=None,
                          threads_per_worker=None, max_files=1), resources=res, log=lines.append)
    text = "\n".join(lines)
    assert "effective RAM" in text and "plan:" in text and "shard_tokens=" in text


def test_prep_on_progress_hook_is_called(tmp_path, parquet_corpus):
    pattern, _ = parquet_corpus
    seen: list[int] = []
    run_prepare(_prep_cfg(tmp_path / "out", pattern, target_tokens=12000), log=_quiet,
                on_progress=lambda m: seen.append(m["train_tokens"]))
    assert len(seen) >= 2 and seen == sorted(seen) and seen[-1] >= 12000


def _run_watchdog_script(stop_early: bool) -> int:
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(f"""
        import multiprocessing as mp, threading, time, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from sapia.prepare_data import start_stall_watchdog
        hb = mp.get_context("spawn").Value("q", 0)
        stop = threading.Event()
        start_stall_watchdog(hb, 0.01, print, stop)       # 0.6 s without a heartbeat = stalled
        if {stop_early}:
            stop.set()
        time.sleep(3)
        print("still alive")
    """)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60).returncode


def test_stall_watchdog_exits_resumably_when_workers_hang():
    from sapia.prepare_data import STALL_EXIT_CODE

    assert _run_watchdog_script(stop_early=False) == STALL_EXIT_CODE
    assert _run_watchdog_script(stop_early=True) == 0          # stopped after a normal finish: no exit


def test_heartbeat_keeps_the_watchdog_quiet_during_parallel_prep(tmp_path, parquet_corpus):
    pattern, _ = parquet_corpus
    manifest = run_prepare(_prep_cfg(tmp_path / "out", pattern, num_workers=2, stall_minutes=5.0), log=_quiet)
    assert manifest["complete"] is True
