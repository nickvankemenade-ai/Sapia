"""End-to-end: data prep -> training -> resume -> sampling, plus robustness paths."""
from __future__ import annotations

import math

import pytest
import torch

from sapia import train as train_mod
from sapia.autoconfig import AutoConfigError
from sapia.checkpoint import list_checkpoints, load_latest_checkpoint
from sapia.config import ModelConfig
from sapia.model import GPT
from sapia.prepare_mix import MixConfig, run_mix
from sapia.sources import SourceSpec
from sapia.sample import find_checkpoint, generate_text, load_model
from sapia.tokenizer import get_tokenizer
from sapia.train import NonFiniteLossError, lr_at, train
from sapia.utils import read_json, read_jsonl

from conftest import small_train_cfg


def test_end_to_end_prep_train_resume_sample(tmp_path, parquet_corpus):
    """The whole pipeline on CPU: the loss must go down, a resume must continue, sampling must work."""
    pattern, _ = parquet_corpus
    data_dir = tmp_path / "data"
    phases = {"main": [SourceSpec("corpus", 1.0, "", pattern)],
              "anneal": [SourceSpec("late", 1.0, "", pattern, file_mod=3, file_keep=(2,))]}
    mix = run_mix(MixConfig(out_dir=str(data_dir), total_tokens=60_000, anneal_fraction=0.1, tokenizer="bytes",
                            val_every=10, shard_tokens=8000, num_workers=1), phases, log=lambda s: None)
    assert mix["phases"]["main"]["sources"][0]["train_tokens"] > 40_000

    model_cfg = ModelConfig(vocab_size=320, block_size=48, n_layer=2, n_head=2, n_kv_head=1, d_model=64,
                            mlp_hidden=128)
    tokens_per_step = 16 * 48
    cfg = small_train_cfg(
        model_cfg, data_dir, tmp_path / "run", max_tokens=tokens_per_step * 40, target_tokens_per_step=tokens_per_step,
        batch_size=None, grad_accum_steps=None, max_batch_size=16, warmup_tokens=tokens_per_step * 4,
        eval_interval=20, eval_tokens=48 * 32, log_interval=5, ckpt_interval=20, main_block_size=24,
    )
    first = train(cfg, _stop_after_steps=20)
    assert (first["status"], first["step"]) == ("interrupted", 20)
    assert first["batch_size"] * first["grad_accum_steps"] == 16
    assert first["first_loss"] == pytest.approx(math.log(320), abs=0.5)

    second = train(cfg)                                    # automatic resume from the newest checkpoint
    assert (second["status"], second["step"]) == ("complete", 40)
    assert second["last_loss"] < first["first_loss"] - 1.5, "training loss must clearly go down"
    assert second["val_loss"] < first["first_loss"] - 1.5, "validation loss must clearly go down"

    run_dir = tmp_path / "run"
    records = read_jsonl(run_dir / "metrics.jsonl")
    train_recs = [r for r in records if r["kind"] == "train"]
    eval_recs = [r for r in records if r["kind"] == "eval"]
    assert [r["step"] for r in train_recs] == sorted({r["step"] for r in train_recs})
    assert train_recs[0]["loss"] > train_recs[-1]["loss"]
    for key in ("step", "tokens_seen", "loss", "lr", "grad_norm", "tok_per_sec", "mfu", "ram_rss_gb", "train_time_s"):
        assert key in train_recs[-1], key
    assert train_recs[-1]["tok_per_sec"] > 0
    assert [r["step"] for r in eval_recs] == [20, 40]
    assert eval_recs[-1]["val_loss"] < eval_recs[0]["val_loss"]
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert log.count("detected resources") == 2
    assert "data phase: anneal from" in log                       # switched to the anneal mix near the end
    assert "context: 24 tokens in the main phase, 48 in the anneal phase" in log
    assert "optimizer: muon" in log
    assert {r.get("phase") for r in train_recs} == {"main", "anneal"}
    assert second["final_weights"] and (run_dir / "final.pt").exists()
    final_model, final_meta = load_model(run_dir / "final.pt", torch.device("cpu"))
    assert final_meta["step"] == 40
    config = read_json(run_dir / "config.json")
    assert first["batch_plan_source"] == "cpu-heuristic" and second["batch_plan_source"] == "checkpoint"
    assert config["precision"] == "fp32" and config["batch_plan"]["batch_size"] == first["batch_size"]
    assert config["resources"]["cpu_usable"] >= 1

    # sampling from the run directory
    ckpt = find_checkpoint(str(run_dir))
    assert ckpt.name == "ckpt_000000040.pt"
    model, meta = load_model(ckpt, torch.device("cpu"))
    assert meta["tokenizer"] == "bytes" and meta["step"] == 40
    text = generate_text(model, get_tokenizer("bytes"), "the cat counts ", max_new_tokens=24, temperature=0.0)
    assert text.startswith("the cat counts ") and len(text) > len("the cat counts ")


def test_gradient_accumulation_matches_big_batch(tmp_path, small_model_cfg, pattern_dataset):
    """4 micro-batches of 4 must take the same optimizer steps as 1 batch of 16 (same data order).

    Checked with AdamW: Muon's bf16 Newton-Schulz step amplifies float rounding differences
    between the two summation orders to ~1e-4, which is expected and harmless.
    """
    big = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "big", max_tokens=16 * 32 * 4,
                          eval_interval=100, ckpt_interval=100, optimizer="adamw")
    acc = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "acc", max_tokens=16 * 32 * 4,
                          eval_interval=100, ckpt_interval=100, batch_size=4, grad_accum_steps=4, optimizer="adamw")
    a, b = train(big), train(acc)
    assert a["step"] == b["step"] == 4 and a["tokens_seen"] == b["tokens_seen"]
    wa = load_latest_checkpoint(tmp_path / "big" / "checkpoints")[0]["model"]
    wb = load_latest_checkpoint(tmp_path / "acc" / "checkpoints")[0]["model"]
    for key in wa:
        assert torch.allclose(wa[key], wb[key], atol=2e-5, rtol=1e-4), key


def test_wsd_schedule_warms_up_holds_then_decays(small_model_cfg, tmp_path):
    cfg = small_train_cfg(small_model_cfg, tmp_path, tmp_path, learning_rate=1.0, warmup_tokens=1000)
    cfg.lr_schedule, cfg.min_lr_ratio, cfg.anneal_fraction = "wsd", 0.0, 0.1
    horizon = 20_000
    assert lr_at(500, cfg, horizon) == pytest.approx(0.5)
    assert lr_at(1000, cfg, horizon) == lr_at(17_999, cfg, horizon) == pytest.approx(1.0)   # stable phase
    assert lr_at(19_000, cfg, horizon) == pytest.approx(0.5)                                # linear decay
    assert lr_at(20_000, cfg, horizon) == pytest.approx(0.0)


def test_lr_schedule_is_defined_over_tokens(small_model_cfg, tmp_path):
    cfg = small_train_cfg(small_model_cfg, tmp_path, tmp_path, learning_rate=1.0, warmup_tokens=1000)
    cfg.min_lr_ratio = 0.1
    cfg.lr_schedule = "cosine"
    horizon = 11_000
    assert lr_at(100, cfg, horizon) == pytest.approx(0.1)             # linear warmup
    assert lr_at(1000, cfg, horizon) == pytest.approx(1.0)            # peak at the end of warmup
    assert lr_at(6000, cfg, horizon) == pytest.approx(0.55)           # cosine midpoint
    assert lr_at(11_000, cfg, horizon) == pytest.approx(0.1)          # floor
    assert lr_at(50_000, cfg, horizon) == pytest.approx(0.1)
    values = [lr_at(t, cfg, horizon) for t in range(1000, 11_001, 500)]
    assert values == sorted(values, reverse=True)


def test_dry_run_plans_without_training(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run", dry_run=True)
    out = train(cfg)
    assert out["status"] == "dry_run" and out["batch_size"] == 16 and out["data_mode"] == "memmap"
    assert not list_checkpoints(tmp_path / "run" / "checkpoints")


def test_time_limits_checkpoint_and_stop(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run", max_hours=1e-12)
    out = train(cfg)
    assert out["status"] == "time_limit" and out["step"] == 0
    assert [s for s, _ in list_checkpoints(tmp_path / "run" / "checkpoints")] == [0]
    cfg.max_hours = None
    cfg.max_total_hours = 1e-12
    assert train(cfg)["status"] == "total_time_limit"
    cfg.max_total_hours = None
    assert train(cfg)["status"] == "complete"


def test_data_mode_and_worker_overrides_are_respected(tmp_path, small_model_cfg, pattern_dataset):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run", data_mode="memmap", num_workers=0,
                          max_tokens=16 * 32 * 2)
    out = train(cfg)
    assert out["status"] == "complete" and out["data_mode"] == "memmap"


def test_vocab_mismatch_is_reported(tmp_path, pattern_dataset):
    tiny_vocab = ModelConfig(vocab_size=100, block_size=32, n_layer=1, n_head=2, n_kv_head=1, d_model=32,
                             mlp_hidden=64)
    with pytest.raises(train_mod.ConfigError, match="vocab_size"):
        train(small_train_cfg(tiny_vocab, pattern_dataset, tmp_path / "run"))


class _NaNAfter(GPT):
    """A model whose loss becomes NaN after a number of training forward passes."""

    healthy_calls = 3

    def forward(self, idx, targets=None, **kw):
        out = super().forward(idx, targets, **kw)
        if targets is not None and self.training:
            self._calls = getattr(self, "_calls", 0) + 1
            if self._calls > self.healthy_calls:
                out = out._replace(loss=out.loss * float("nan"))
        return out


def test_non_finite_loss_skips_updates_then_aborts_cleanly(tmp_path, small_model_cfg, pattern_dataset, monkeypatch):
    monkeypatch.setattr(train_mod, "GPT", _NaNAfter)
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run", max_nan_skips=3, ckpt_interval=2,
                          eval_interval=100)
    with pytest.raises(NonFiniteLossError, match="consecutive optimizer steps were skipped") as err:
        train(cfg)
    assert "--precision" in str(err.value) and "--learning_rate" in str(err.value)
    # steps 1-3 were healthy (checkpoint at step 2), steps 4-6 were skipped, then the run aborted
    state, _ = load_latest_checkpoint(tmp_path / "run" / "checkpoints")
    assert state["step"] in (2, 4)
    assert all(torch.isfinite(v).all() for v in state["model"].values()), "no NaN may ever reach a checkpoint"
    log = (tmp_path / "run" / "train.log").read_text(encoding="utf-8")
    assert log.count("skipped: non-finite loss") == 3


def test_single_bad_step_is_skipped_and_training_continues(tmp_path, small_model_cfg, pattern_dataset, monkeypatch):
    class OneBad(GPT):
        def forward(self, idx, targets=None, **kw):
            out = super().forward(idx, targets, **kw)
            if targets is not None and self.training:
                self._calls = getattr(self, "_calls", 0) + 1
                if self._calls == 3:
                    out = out._replace(loss=out.loss * float("inf"))
            return out

    monkeypatch.setattr(train_mod, "GPT", OneBad)
    out = train(small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run"))
    assert out["status"] == "complete" and out["skipped_steps"] == 1 and out["step"] == 12
    state, _ = load_latest_checkpoint(tmp_path / "run" / "checkpoints")
    assert all(torch.isfinite(v).all() for v in state["model"].values())


def test_oom_during_training_backs_off_the_batch_size(tmp_path, small_model_cfg, pattern_dataset, monkeypatch):
    """An out-of-memory error in a training step shrinks the micro-batch and keeps the tokens per step."""
    class OomOnBigBatch(GPT):
        def forward(self, idx, targets=None, **kw):
            if targets is not None and self.training and idx.shape[0] > 10:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return super().forward(idx, targets, **kw)

    monkeypatch.setattr(train_mod, "GPT", OomOnBigBatch)
    # the recovery path is CUDA-only; enable it for this CPU run so the logic can be tested anywhere
    monkeypatch.setattr(train_mod, "_oom_recovery_enabled", lambda device: True)
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run", max_tokens=16 * 32 * 4,
                          grad_accum_steps=None, eval_interval=100, ckpt_interval=100)
    out = train(cfg)
    assert out["status"] == "complete"
    assert out["batch_size"] <= 10 and out["batch_size"] * out["grad_accum_steps"] == 16
    assert out["batch_plan_source"] == "oom-backoff"
    log = (tmp_path / "run" / "train.log").read_text(encoding="utf-8")
    assert "out of memory at micro-batch 16" in log

    class AlwaysOom(GPT):
        def forward(self, idx, targets=None, **kw):
            if targets is not None and self.training:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return super().forward(idx, targets, **kw)

    monkeypatch.setattr(train_mod, "GPT", AlwaysOom)
    with pytest.raises(AutoConfigError, match="out of memory"):
        train(small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run2", grad_accum_steps=None))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("requested", ["bf16", "fp16", "fp32"])
def test_gpu_training_with_auto_batch_size(tmp_path, small_model_cfg, pattern_dataset, requested):
    cfg = small_train_cfg(small_model_cfg, pattern_dataset, tmp_path / "run", device="cuda", precision=requested,
                          batch_size=None, grad_accum_steps=None, max_tokens=16 * 32 * 30,
                          warmup_tokens=16 * 32 * 3, eval_interval=15, ckpt_interval=15)
    out = train(cfg, _stop_after_steps=15)
    assert out["status"] == "interrupted" and out["batch_plan_source"] == "probe"
    assert out["batch_size"] == 16 and out["grad_accum_steps"] == 1     # tiny model: the cap is reached
    assert out["precision"] in ((requested,) if requested != "bf16" else ("bf16", "fp16"))
    state, _ = load_latest_checkpoint(tmp_path / "run" / "checkpoints")
    assert (state["scaler"] is not None) == (out["precision"] == "fp16")
    assert state["batch_plan"]["max_fit_batch_size"] == 16 and state["batch_plan"]["peak_bytes"] > 0

    resumed = train(cfg)
    assert resumed["status"] == "complete" and resumed["step"] == 30
    assert resumed["batch_plan_source"] == "checkpoint", "the probed plan is reused on the same GPU"
    assert resumed["last_loss"] < out["first_loss"] - 1.0
    recs = [r for r in read_jsonl(tmp_path / "run" / "metrics.jsonl") if r["kind"] == "train"]
    assert recs[-1]["gpu_mem_peak_alloc_gb"] > 0
    assert ("scaler_scale" in recs[-1]) == (out["precision"] == "fp16")
