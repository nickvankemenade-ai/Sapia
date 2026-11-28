"""Checkpoint file handling and exact training resume."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from sapia.checkpoint import (best_checkpoint_path, capture_rng_state, checkpoint_path, clean_stale_tmp,
                                list_checkpoints, load_checkpoint, load_latest_checkpoint, read_best_meta,
                                restore_rng_state, save_best_checkpoint, save_checkpoint, strip_compile_prefix)
from sapia.config import ConfigError
from sapia.model import GPT
from sapia.sample import find_checkpoint, load_model
from sapia.train import train
from sapia.utils import read_jsonl

from conftest import small_train_cfg


def _state(model: GPT, extra: float = 0.0) -> dict:
    return {"model": model.state_dict(), "tokens_seen": 123, "note": extra}


def test_save_and_load_round_trip(tmp_path, small_model_cfg):
    model = GPT(small_model_cfg)
    opt = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), 1e-8, "cpu")
    idx = torch.randint(0, small_model_cfg.vocab_size, (2, 8))
    model(idx, idx)[1].backward()
    opt.step()
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": None, "tokens_seen": 4096,
             "loader_states": {"epoch": 2, "cursor": 77}, "rng": capture_rng_state()}
    path = save_checkpoint(tmp_path, 42, state)
    assert path.name == "ckpt_000000042.pt" and path == checkpoint_path(tmp_path, 42)
    loaded = load_checkpoint(path)
    assert loaded["step"] == 42 and loaded["tokens_seen"] == 4096 and loaded["loader_states"] == {"epoch": 2, "cursor": 77}
    clone = GPT(small_model_cfg)
    clone.load_state_dict(loaded["model"])
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), clone.state_dict().values()))
    assert clone.lm_head.weight is clone.tok_emb.weight
    opt2 = clone.configure_optimizer(1e-3, 0.1, (0.9, 0.95), 1e-8, "cpu")
    opt2.load_state_dict(loaded["optimizer"])
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys()
    assert all(torch.equal(s1[k]["exp_avg_sq"], s2[k]["exp_avg_sq"]) for k in s1)


def test_writes_are_atomic_and_stale_temp_files_are_ignored(tmp_path, small_model_cfg):
    model = GPT(small_model_cfg)
    save_checkpoint(tmp_path, 10, _state(model))
    assert not list(tmp_path.glob("*.tmp"))
    # A crash mid-save leaves only a .tmp file, which must never be picked up as a checkpoint.
    (tmp_path / "ckpt_000000020.pt.tmp").write_bytes(b"partial write")
    assert [s for s, _ in list_checkpoints(tmp_path)] == [10]
    state, path = load_latest_checkpoint(tmp_path)
    assert state["step"] == 10 and path.name == "ckpt_000000010.pt"
    clean_stale_tmp(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_newest_checkpoint_falls_back_to_the_previous_one(tmp_path, small_model_cfg):
    model = GPT(small_model_cfg)
    save_checkpoint(tmp_path, 10, _state(model), keep_last=5)
    save_checkpoint(tmp_path, 20, _state(model), keep_last=5)
    checkpoint_path(tmp_path, 20).write_bytes(b"this is not a checkpoint")
    warnings: list[str] = []
    state, path = load_latest_checkpoint(tmp_path, log=warnings.append)
    assert state["step"] == 10 and path == checkpoint_path(tmp_path, 10)
    assert warnings and "unreadable" in warnings[0]
    assert load_latest_checkpoint(tmp_path / "empty") == (None, None)


def test_keep_last_n(tmp_path, small_model_cfg):
    model = GPT(small_model_cfg)
    for step in (5, 10, 15, 20, 25):
        save_checkpoint(tmp_path, step, _state(model), keep_last=2)
    assert [s for s, _ in list_checkpoints(tmp_path)] == [20, 25]


def test_rng_state_round_trip():
    random.seed(1), np.random.seed(1), torch.manual_seed(1)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.random(), np.random.rand(), torch.rand(3)             # advance all streams
    restore_rng_state(state)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_strip_compile_prefix():
    assert strip_compile_prefix({"_orig_mod.a.weight": 1, "b": 2}) == {"a.weight": 1, "b": 2}


def test_load_rejects_foreign_files(tmp_path):
    path = tmp_path / "ckpt_000000001.pt"
    torch.save({"something": "else"}, path)
    with pytest.raises(ValueError, match="not a Sapia checkpoint"):
        load_checkpoint(path)


# ----------------------------------------------------------------------------- trainer-level resume

def _final_weights(run_dir) -> dict:
    state, _ = load_latest_checkpoint(run_dir / "checkpoints")
    return state


def test_interrupted_run_resumes_to_the_same_result(tmp_path, small_model_cfg, pattern_dataset):
    """12 steps in one go == 5 steps, interruption, then the remaining 7 after a resume."""
    straight = train(small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "straight"))
    assert (straight["status"], straight["step"]) == ("complete", 12)

    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "resumed")
    part1 = train(cfg, _stop_after_steps=5)
    assert (part1["status"], part1["step"]) == ("interrupted", 5)
    assert [s for s, _ in list_checkpoints(tmp_path / "resumed" / "checkpoints")] == [5]
    part2 = train(cfg)
    assert (part2["status"], part2["step"]) == ("complete", 12)
    assert part2["tokens_seen"] == straight["tokens_seen"]

    a, b = _final_weights(tmp_path / "straight"), _final_weights(tmp_path / "resumed")
    assert a["step"] == b["step"] == 12
    assert a["loader_states"] == b["loader_states"]
    for key in a["model"]:
        assert torch.allclose(a["model"][key], b["model"][key], atol=1e-6, rtol=1e-5), key
    assert part2["val_loss"] == pytest.approx(straight["val_loss"], abs=1e-5)

    # metrics: every logged step appears exactly once, in order, across the interruption
    steps = [r["step"] for r in read_jsonl(tmp_path / "resumed" / "metrics.jsonl") if r["kind"] == "train"]
    assert steps == sorted(set(steps)) and steps[-1] == 12


def test_checkpoint_contents(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run")
    train(cfg)
    state = _final_weights(tmp_path / "run")
    for key in ("model", "optimizer", "scaler", "step", "tokens_seen", "loader_states", "rng",
                "model_config", "train_config", "batch_plan", "precision", "tokenizer", "best_val_loss",
                "train_time_s", "versions"):
        assert key in state, key
    assert state["precision"] == "fp32"          # bf16 requested, CPU run -> fp32
    assert state["scaler"] is None
    assert state["train_config"]["precision"] == "bf16"
    assert state["tokens_seen"] == 12 * 16 * 32
    assert set(state["rng"]) >= {"python", "numpy", "torch"}
    assert state["batch_plan"]["batch_size"] == 16
    # keep_last=2 with checkpoints at steps 6 and 12
    assert [s for s, _ in list_checkpoints(tmp_path / "run" / "checkpoints")] == [6, 12]


def test_resume_of_a_finished_run_is_a_noop(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run")
    first = train(cfg)
    before = _final_weights(tmp_path / "run")
    again = train(cfg)
    assert (again["status"], again["step"]) == ("complete", first["step"])
    after = _final_weights(tmp_path / "run")
    assert all(torch.equal(before["model"][k], after["model"][k]) for k in before["model"])


def test_resume_rejects_a_different_model_config(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run")
    train(cfg, _stop_after_steps=2)
    cfg.model.n_layer = 3
    with pytest.raises(ConfigError, match="different model config"):
        train(cfg)


def test_resume_never_refuses_to_overwrite(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run")
    train(cfg, _stop_after_steps=2)
    cfg.resume = "never"
    with pytest.raises(ConfigError, match="already contains checkpoints"):
        train(cfg)


# ----------------------------------------------------------------------------- best model

def test_best_checkpoint_is_separate_from_keep_last(tmp_path, small_model_cfg):
    model = GPT(small_model_cfg)
    best = save_best_checkpoint(tmp_path, 7, 3.25, _state(model))
    for step in (10, 20, 30):
        save_checkpoint(tmp_path, step, _state(model), keep_last=1)
    assert best == best_checkpoint_path(tmp_path) and best.is_file()          # never pruned
    assert [s for s, _ in list_checkpoints(tmp_path)] == [30]                 # never used for resume
    assert load_latest_checkpoint(tmp_path)[0]["step"] == 30
    loaded = load_checkpoint(best)
    assert loaded["step"] == 7 and loaded["val_loss"] == 3.25
    assert read_best_meta(tmp_path) == {"step": 7, "val_loss": 3.25, "file": "best.pt"}
    best.unlink()
    assert read_best_meta(tmp_path) is None                                   # sidecar without weights


def test_training_keeps_latest_and_best(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run")
    out = train(cfg)
    ckpt_dir = tmp_path / "run" / "checkpoints"
    meta = read_best_meta(ckpt_dir)
    assert meta is not None and meta["val_loss"] == pytest.approx(out["best_val_loss"])
    best = load_checkpoint(best_checkpoint_path(ckpt_dir))
    assert "optimizer" not in best and best["val_loss"] == pytest.approx(out["best_val_loss"])
    assert list_checkpoints(ckpt_dir)[-1][0] == out["step"]                   # latest still there
    model, _meta = load_model(find_checkpoint(str(tmp_path / "run"), best=True), torch.device("cpu"))
    assert model is not None


def test_resume_never_overwrites_a_better_best(tmp_path, small_model_cfg, pattern_dataset):
    """best.pt written after the last periodic checkpoint must survive a resume."""
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run")
    train(cfg, _stop_after_steps=5)
    ckpt_dir = tmp_path / "run" / "checkpoints"
    model = GPT(small_model_cfg)
    save_best_checkpoint(ckpt_dir, 99, 0.001, _state(model))                  # unbeatable best
    train(cfg)
    assert read_best_meta(ckpt_dir)["val_loss"] == 0.001
    assert load_checkpoint(best_checkpoint_path(ckpt_dir))["step"] == 99


def test_save_best_can_be_disabled(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run")
    cfg.save_best = False
    train(cfg)
    assert not best_checkpoint_path(tmp_path / "run" / "checkpoints").exists()


def test_resume_from_explicit_path(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "a")
    train(cfg, _stop_after_steps=4)
    ckpt = checkpoint_path(tmp_path / "a" / "checkpoints", 4)
    cfg2 = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "b", resume=str(ckpt))
    out = train(cfg2)
    assert (out["status"], out["step"]) == ("complete", 12)
    assert list_checkpoints(tmp_path / "b" / "checkpoints")[-1][0] == 12


def test_checksum_detects_damage_and_falls_back(tmp_path):
    from sapia.checkpoint import SUM_SUFFIX, load_checkpoint, load_latest_checkpoint, save_checkpoint, verify_checkpoint

    d = tmp_path / "ck"
    old = save_checkpoint(d, 10, {"model": {"w": torch.ones(64)}}, keep_last=3)
    new = save_checkpoint(d, 20, {"model": {"w": torch.full((64,), 2.0)}}, keep_last=3)
    assert (d / (new.name + SUM_SUFFIX)).exists() and verify_checkpoint(new)[0]
    data = bytearray(new.read_bytes())
    data[len(data) // 2] ^= 0xFF                                   # one flipped byte
    new.write_bytes(bytes(data))
    assert not verify_checkpoint(new)[0]
    with pytest.raises(ValueError):
        load_checkpoint(new)
    warnings = []
    state, path = load_latest_checkpoint(d, log=warnings.append)
    assert path == old and state["step"] == 10 and warnings
    new.write_bytes(bytes(data[:100]))                              # a truncated copy
    assert "size" in verify_checkpoint(new)[1]


def test_prune_removes_checksum_files(tmp_path):
    from sapia.checkpoint import SUM_SUFFIX, save_checkpoint

    d = tmp_path / "ck"
    for step in (1, 2, 3):
        save_checkpoint(d, step, {"model": {}}, keep_last=1)
    assert sorted(p.name for p in d.iterdir()) == ["ckpt_000000003.pt", "ckpt_000000003.pt" + SUM_SUFFIX]
