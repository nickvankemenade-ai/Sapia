"""The TPU training loop's logic, run on CPU through the test backend (no torch_xla needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sapia import model as model_mod
from sapia.checkpoint import list_checkpoints, load_latest_checkpoint
from sapia.config import ConfigError, ModelConfig
from sapia.data import MixtureLoader
from sapia.kaggle import find_mixes, latest_checkpoint, merge_mixes
from sapia.model import GPT
from sapia.sample import load_model
from sapia.train_tpu import CPUBackend, plan_tpu_batch, train_tpu

from conftest import pattern_tokens, small_train_cfg, write_token_dataset


@pytest.fixture
def tpu_cfg(tmp_path, pattern_dataset):
    def make(out: str, **kw):
        mc = ModelConfig(vocab_size=320, block_size=32, n_layer=2, n_head=2, n_kv_head=1, d_model=48, mlp_hidden=96)
        base = dict(tpu=True, main_block_size=16, tpu_tokens_per_chip=64, target_tokens_per_step=256,
                    max_tokens=256 * 12, warmup_tokens=256 * 2, eval_interval=6, ckpt_interval=6,
                    log_interval=2, eval_tokens=32 * 8, learning_rate=3e-3)
        base.update(kw)
        return small_train_cfg(mc, pattern_dataset, tmp_path / out, **base)
    return make


def test_batch_plan_keeps_tokens_per_step_across_phases(tpu_cfg):
    plan = plan_tpu_batch(tpu_cfg("x"), num_devices=1)
    assert plan["seqs"] == {"main": 4, "anneal": 2} and plan["ctx"] == {"main": 16, "anneal": 32}
    assert plan["accum"] == 4 and plan["tokens_per_step"] == 256
    big = plan_tpu_batch(tpu_cfg("y", tpu_tokens_per_chip=2048, target_tokens_per_step=524_288,
                                 model=ModelConfig(block_size=2048), main_block_size=1024), num_devices=8)
    assert big["seqs"] == {"main": 16, "anneal": 8} and big["accum"] == 32 and big["tokens_per_step"] == 524_288
    with pytest.raises(ConfigError):
        plan_tpu_batch(tpu_cfg("z", tpu_tokens_per_chip=48), num_devices=1)   # not a multiple of block_size


def test_full_run_learns_switches_phase_and_exports(tpu_cfg, tmp_path):
    out = train_tpu(tpu_cfg("run"), backend=CPUBackend())
    assert out["status"] == "complete" and out["step"] == 12 and out["tokens_seen"] == 256 * 12
    log = (tmp_path / "run" / "train.log").read_text(encoding="utf-8")
    assert "data phase: anneal from" in log and "(context 32)" in log
    recs = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in recs if r["kind"] == "train"]
    assert train[-1]["loss"] < train[0]["loss"] - 1.0
    assert any(r["kind"] == "eval" for r in recs)
    model, meta = load_model(tmp_path / "run" / "final.pt", torch.device("cpu"))     # plain CPU file
    assert meta["step"] == 12


def test_sessions_can_stop_and_continue_exactly(tpu_cfg, tmp_path):
    """Kaggle-style: session 1 stops early, session 2 resumes from session 1's checkpoint folder."""
    straight = train_tpu(tpu_cfg("straight"), backend=CPUBackend())
    s1 = train_tpu(tpu_cfg("session1"), backend=CPUBackend(), _stop_after_steps=5)
    assert s1["status"] == "interrupted" and s1["step"] == 5
    s2 = train_tpu(tpu_cfg("session2", resume=str(tmp_path / "session1" / "checkpoints")), backend=CPUBackend())
    assert s2["status"] == "complete" and s2["step"] == 12 and s2["tokens_seen"] == straight["tokens_seen"]
    a = load_latest_checkpoint(tmp_path / "straight" / "checkpoints")[0]
    b = load_latest_checkpoint(tmp_path / "session2" / "checkpoints")[0]
    assert a["loader_states"] == b["loader_states"]
    for key in a["model"]:
        assert torch.allclose(a["model"][key], b["model"][key], atol=1e-5), key


def test_time_limit_saves_and_exits(tpu_cfg, tmp_path):
    out = train_tpu(tpu_cfg("run", max_hours=1e-12), backend=CPUBackend())
    assert out["status"] == "time_limit"
    assert list_checkpoints(tmp_path / "run" / "checkpoints")


def test_train_dispatches_to_tpu_loop(tpu_cfg, monkeypatch):
    from sapia import train as train_mod, train_tpu as tpu_mod

    seen = {}
    monkeypatch.setattr(tpu_mod, "train_tpu", lambda cfg, _stop_after_steps=None: seen.setdefault("cfg", cfg) and {})
    train_mod.train(tpu_cfg("run"))
    assert seen["cfg"].tpu is True


def test_attention_override_is_used_and_matches():
    torch.manual_seed(0)
    m = GPT(ModelConfig(vocab_size=97, block_size=32, n_layer=2, n_head=4, n_kv_head=2, d_model=32,
                        mlp_hidden=64)).eval()
    idx = torch.randint(0, 97, (2, 16))
    ref = m(idx, all_logits=True).logits
    calls = []

    def plain(q, k, v):
        calls.append(q.shape)
        k, v = (t.repeat_interleave(q.size(1) // k.size(1), dim=1) for t in (k, v))
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    model_mod.ATTENTION_OVERRIDE = plain
    try:
        got = m(idx, all_logits=True).logits
        cached = m.generate(idx[:, :4], 3, temperature=0.0)           # the KV-cache path never uses it
    finally:
        model_mod.ATTENTION_OVERRIDE = None
    assert len(calls) >= 2 and torch.allclose(got, ref, atol=1e-5)
    assert cached.shape == (2, 7)


def _write_mix(root: Path, sources: dict[str, str]) -> Path:
    """A prep-notebook-like output: root/sapia-mix with mix.json and the given {name: phase} sources."""
    mix_dir = root / "sapia-mix"
    phases = {"main": {"tokens": 1, "sources": []}, "anneal": {"tokens": 1, "sources": []}}
    for name, phase in sources.items():
        write_token_dataset(mix_dir / phase / name, train=[pattern_tokens(3000)], val=[pattern_tokens(500)])
        phases[phase]["sources"].append({"name": name, "share": 0.5, "dir": f"{phase}/{name}", "target_tokens": 3000,
                                         "prepared": True, "train_tokens": 3000, "val_tokens": 500})
    (mix_dir / "mix.json").write_text(json.dumps({"version": 1, "total_tokens": 1, "anneal_fraction": 0.1,
                                                  "tokenizer": "bytes", "phases": phases}), encoding="utf-8")
    return mix_dir


def test_kaggle_merge_joins_notebook_outputs(tmp_path):
    inputs = tmp_path / "input"
    _write_mix(inputs / "sapia-prep-1", {"fineweb_edu": "main"})
    _write_mix(inputs / "sapia-prep-2", {"dclm": "main", "ultrachat": "anneal"})
    dirs = find_mixes(inputs)
    assert len(dirs) == 2
    merged = merge_mixes(dirs, tmp_path / "merged", log=lambda s: None)
    names = [s["name"] for s in merged["phases"]["main"]["sources"]]
    assert names == ["fineweb_edu", "dclm"]
    assert all(Path(s["dir"]).is_absolute() for s in merged["phases"]["main"]["sources"])
    loader = MixtureLoader(tmp_path / "merged", "main", block_size=16, batch_size=8, seed=0)
    tokens, draws = loader.next_batch()
    assert tokens.shape == (8, 17) and set(loader.names) == {"fineweb_edu", "dclm"}


def test_kaggle_latest_checkpoint_picks_highest_step(tmp_path):
    for name, step in (("session-1", 300), ("session-2", 900)):
        d = tmp_path / "input" / name / "run" / "checkpoints"
        d.mkdir(parents=True)
        torch.save({"model": {}, "step": step}, d / f"ckpt_{step:09d}.pt")
    assert latest_checkpoint(tmp_path / "input") == tmp_path / "input" / "session-2" / "run" / "checkpoints"
    assert latest_checkpoint(tmp_path / "nothing-here") is None


def test_kaggle_merge_leaves_out_a_phase_without_data(tmp_path):
    inputs = tmp_path / "input"
    _write_mix(inputs / "sapia-prep-1", {"fineweb_edu": "main"})
    merged = merge_mixes(find_mixes(inputs), tmp_path / "merged", log=lambda s: None)
    assert list(merged["phases"]) == ["main"]


def test_progress_pinger_sends_loss_and_average_mfu(monkeypatch):
    from sapia import notify as notify_mod

    sent = []
    n = notify_mod.Notifier("test-topic", run="run")
    monkeypatch.setattr(n, "send", lambda msg, priority="default": sent.append((msg, priority)))
    clock = [1000.0]
    monkeypatch.setattr(notify_mod.time, "time", lambda: clock[0])
    p = n.every(15)
    assert not p.maybe(10, 0, 16_000_000_000, 4.2, 100_000, mfu=0.30)                 # baseline, too soon
    # 1M tokens at 100K tok/s (10 s, MFU 30%), then 3M tokens at 300K tok/s (10 s, MFU 50%)
    assert not p.maybe(20, 1_000_000, 16_000_000_000, 4.0, 100_000, mfu=0.30)
    clock[0] += 15 * 60 + 1
    assert p.maybe(30, 4_000_000, 16_000_000_000, 3.141, 300_000, val=3.2, mfu=0.50)
    msg, prio = sent[-1]
    assert "loss 3.141" in msg and "val 3.200" in msg and "4.00M/16.00B" in msg
    assert "MFU 40.0% (avg since last message)" in msg and "200.00K tok/s avg" in msg   # time-weighted
    assert prio == "low" and not p.maybe(31, 4_100_000, 16_000_000_000, 3.1, 300_000, mfu=0.5)
    assert not notify_mod.Notifier(None).every(15).maybe(1, 1, 2, 1.0, 1.0)             # no topic: nothing sent


@pytest.fixture
def kaggle_env(tmp_path, monkeypatch):
    """A fake Kaggle machine: one attached data dataset, an empty working dir, no network."""
    from sapia import kaggle as kg

    inputs, work = tmp_path / "input", tmp_path / "working"
    _write_mix(inputs / "sapia-data-1", {"fineweb_edu": "main"})
    work.mkdir()
    monkeypatch.setattr(kg, "INPUT_ROOT", inputs)
    monkeypatch.setattr(kg, "WORK_ROOT", work)
    monkeypatch.setattr(kg, "load_secrets", lambda log=print: [])
    monkeypatch.setattr(kg, "ensure_packages", lambda backend, log=print: None)
    prev = tmp_path / "previous"
    monkeypatch.setattr(kg, "Path", lambda p: prev if p == "/tmp/sapia-previous" else (tmp_path / "mix" if p == "/tmp/sapia-mix" else Path(p)))
    calls = []
    monkeypatch.setattr(kg.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})())
    return kg, inputs, work, prev, calls


def _fake_ckpt(run_dir: Path, step: int) -> None:
    d = run_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {}, "step": step}, d / f"ckpt_{step:09d}.pt")
    (run_dir / "metrics.jsonl").write_text('{"kind": "train", "step": %d}\n' % step)


def test_kaggle_session_resumes_from_its_previous_output(kaggle_env, monkeypatch):
    kg, inputs, work, prev, calls = kaggle_env
    monkeypatch.setattr(kg, "fetch_previous_output", lambda k, dest, log=print: _fake_ckpt(dest / "run", 900) or True)
    _fake_ckpt(inputs / "old-version" / "run", 300)                  # an older attached version loses
    assert kg.run_session(2.0, log=lambda s: None) == 0
    cmd = calls[-1]
    assert cmd[cmd.index("--resume") + 1] == str(prev / "run" / "checkpoints")
    assert cmd[cmd.index("--max_hours") + 1] == "2.0" and cmd[cmd.index("--backend") + 1] == "tpu-jax"
    assert "16e9" in cmd and (work / "run" / "metrics.jsonl").exists()          # history carried over
    assert json.loads((work / "run" / "session.json").read_text())["resumed_step"] == 900


def test_kaggle_session_first_run_and_refusals(kaggle_env, monkeypatch):
    kg, inputs, work, prev, calls = kaggle_env
    quiet = lambda s: None
    monkeypatch.setattr(kg, "fetch_previous_output", lambda k, dest, log=print: True)   # API works, nothing yet
    assert kg.run_session(2.0, log=quiet) == 0 and calls[-1][calls[-1].index("--resume") + 1] == "auto"
    calls.clear()
    monkeypatch.setattr(kg, "fetch_previous_output", lambda k, dest, log=print: None)   # no API secrets
    assert kg.run_session(2.0, log=quiet) == 2 and not calls                          # can't tell: refuse
    assert kg.run_session(2.0, fresh=True, log=quiet) == 0
    calls.clear()

    def broken(k, dest, log=print):
        raise RuntimeError("401 unauthorized")
    monkeypatch.setattr(kg, "fetch_previous_output", broken)
    assert kg.run_session(2.0, fresh=True, log=quiet) == 2 and not calls              # API error: always refuse

    def failed_last_time(k, dest, log=print):                         # last session started at 900, saved nothing
        (dest / "run").mkdir(parents=True, exist_ok=True)
        (dest / "run" / "session.json").write_text('{"resumed_step": 900}')
        return True
    monkeypatch.setattr(kg, "fetch_previous_output", failed_last_time)
    assert kg.run_session(2.0, log=quiet) == 2 and not calls
