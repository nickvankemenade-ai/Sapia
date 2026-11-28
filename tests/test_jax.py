"""The JAX backend (--tpu-jax), run on CPU: parity with PyTorch and checkpoints across modes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from sapia import jax_model as jm  # noqa: E402
from sapia.checkpoint import load_latest_checkpoint  # noqa: E402
from sapia.config import ModelConfig  # noqa: E402
from sapia.model import GPT  # noqa: E402
from sapia.train_jax import train_jax  # noqa: E402
from sapia.train_tpu import CPUBackend, train_tpu  # noqa: E402

from conftest import small_train_cfg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MC = ModelConfig(vocab_size=97, block_size=32, n_layer=3, n_head=4, n_kv_head=2, d_model=64, mlp_hidden=128)


def _torch_setup(optimizer="muon", seed=0):
    torch.manual_seed(seed)
    m = GPT(MC)
    tok = torch.randint(0, MC.vocab_size, (4, 17))
    opt = m.configure_optimizer(1e-2, 0.1, (0.9, 0.95), 1e-8, "cpu", optimizer=optimizer)
    return m, tok, opt


def _jax_steps(p0, tok, optimizer, n):
    params = {k: jnp.asarray(v) for k, v in p0.items()}
    st = jax.tree_util.tree_map(jnp.asarray, jm.init_opt_state(p0, optimizer))
    grad = jax.grad(lambda pp: jm.forward_loss(pp, jnp.asarray(tok.numpy()), MC, dtype=jnp.float32)[0])
    for _ in range(n):
        params, st, _ = jm.apply_updates(params, grad(params), st, 1e-2, MC, optimizer, 0.1, (0.9, 0.95), 1e-8,
                                         0.95, 1.0)
    return params, st


def _torch_steps(m, tok, opt, n):
    for _ in range(n):
        m(tok[:, :-1], tok[:, 1:]).loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()


def test_forward_and_gradients_match_pytorch():
    m, tok, _ = _torch_setup()
    out = m(tok[:, :-1], tok[:, 1:])
    out.loss.backward()
    p = jm.params_from_torch(m.state_dict(), MC)
    loss_fn = lambda pp: jm.forward_loss(pp, jnp.asarray(tok.numpy()), MC, dtype=jnp.float32)
    (loss, ce), g = jax.value_and_grad(loss_fn, has_aux=True)({k: jnp.asarray(v) for k, v in p.items()})
    assert abs(float(loss) - out.loss.item()) < 1e-5 and abs(float(ce) - out.ce.item()) < 1e-5
    tg = jm.params_from_torch({n: q.grad for n, q in m.named_parameters()} | {"lm_head.weight": m.tok_emb.weight.grad}, MC)
    for k in tg:
        assert np.abs(tg[k] - np.asarray(g[k])).max() <= 1e-5 * np.abs(tg[k]).max() + 1e-9, k


@pytest.mark.parametrize("optimizer,tol", [("adamw", 0.01), ("muon", 0.1)])
def test_optimizer_steps_match_pytorch(optimizer, tol):
    """AdamW agrees closely; Muon within bf16 Newton-Schulz rounding (fp32 maths is identical)."""
    m, tok, opt = _torch_setup(optimizer)
    p0 = jm.params_from_torch(m.state_dict(), MC)
    _torch_steps(m, tok, opt, 3)
    params, _ = _jax_steps(p0, tok, optimizer, 3)
    after = jm.params_from_torch(m.state_dict(), MC)
    for k in after:
        dt, dj = after[k] - p0[k], np.asarray(params[k]) - p0[k]
        assert np.abs(dt - dj).mean() <= tol * np.abs(dt).mean(), k


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
def test_optimizer_state_converts_both_ways(optimizer):
    m, tok, opt = _torch_setup(optimizer)
    _torch_steps(m, tok, opt, 2)
    groups, layout = jm.optimizer_layout(MC, optimizer, 1e-2, 0.1, (0.9, 0.95), 1e-8, 0.95)
    params = jm.params_from_torch(m.state_dict(), MC)
    st = jm.opt_state_from_torch(opt.state_dict(), params, MC, layout, optimizer)
    assert int(st["step"]) == 2
    back = jm.opt_state_to_torch(st, MC, groups, layout, optimizer)
    fresh = GPT(MC)
    fresh.load_state_dict(jm.params_to_torch(params, MC))
    opt2 = fresh.configure_optimizer(1e-2, 0.1, (0.9, 0.95), 1e-8, "cpu", optimizer=optimizer)
    opt2.load_state_dict(back)                                       # PyTorch accepts it
    a, b = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert a.keys() == b.keys()
    for i in a:
        for name, v in a[i].items():
            assert torch.allclose(torch.as_tensor(v).float(), torch.as_tensor(b[i][name]).float()), (i, name)
    _torch_steps(m, tok, opt, 1)
    _torch_steps(fresh, tok, opt2, 1)                                # and continues identically
    for (n1, p1), (_, p2) in zip(m.named_parameters(), fresh.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), n1


@pytest.fixture
def cfg_for(tmp_path, pattern_dataset):
    def make(out: str, **kw):
        mc = ModelConfig(vocab_size=320, block_size=32, n_layer=2, n_head=2, n_kv_head=1, d_model=48, mlp_hidden=96)
        base = dict(main_block_size=16, tpu_tokens_per_chip=64, target_tokens_per_step=256, max_tokens=256 * 12,
                    warmup_tokens=256 * 2, eval_interval=6, ckpt_interval=6, log_interval=2, eval_tokens=32 * 8,
                    learning_rate=3e-3, precision="fp32", backend="tpu-jax")
        base.update(kw)
        return small_train_cfg(mc, pattern_dataset, tmp_path / out, **base)
    return make


def _losses(run: Path) -> list[float]:
    recs = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    return [r["loss"] for r in recs if r["kind"] == "train"]


def test_jax_run_learns_switches_phase_and_exports(cfg_for, tmp_path):
    out = train_jax(cfg_for("run"), _allow_cpu=True)
    assert out["status"] == "complete" and out["step"] == 12 and out["tokens_seen"] == 256 * 12
    log = (tmp_path / "run" / "train.log").read_text(encoding="utf-8")
    assert "data phase: anneal" in log and "verified" in log
    losses = _losses(tmp_path / "run")
    assert losses[-1] < losses[0] - 1.0
    from sapia.sample import load_model
    model, meta = load_model(tmp_path / "run" / "final.pt", torch.device("cpu"))     # a plain PyTorch file
    assert meta["step"] == 12


def test_jax_sessions_stop_and_continue_exactly(cfg_for, tmp_path):
    train_jax(cfg_for("straight"), _allow_cpu=True)
    s1 = train_jax(cfg_for("s1"), _allow_cpu=True, _stop_after_steps=5)
    assert s1["status"] == "interrupted" and s1["step"] == 5
    s2 = train_jax(cfg_for("s2", resume=str(tmp_path / "s1")), _allow_cpu=True)
    assert s2["status"] == "complete" and s2["step"] == 12
    a = load_latest_checkpoint(tmp_path / "straight" / "checkpoints")[0]
    b = load_latest_checkpoint(tmp_path / "s2" / "checkpoints")[0]
    assert a["loader_states"] == b["loader_states"] and a["tokens_seen"] == b["tokens_seen"]
    for key in a["model"]:
        assert torch.allclose(a["model"][key], b["model"][key], atol=1e-5), key


@pytest.mark.parametrize("first,second", [("torch", "jax"), ("jax", "torch")])
def test_checkpoints_continue_across_modes(cfg_for, tmp_path, first, second):
    """Stop after 5 steps in one mode, finish in the other: same data, same schedule, close weights."""
    run = {"torch": lambda cfg, **kw: train_tpu(cfg, backend=CPUBackend(), **kw),
           "jax": lambda cfg, **kw: train_jax(cfg, _allow_cpu=True, **kw)}
    straight = run["torch"](cfg_for("straight", backend="tpu"))
    s1 = run[first](cfg_for("s1", backend="tpu" if first == "torch" else "tpu-jax"), _stop_after_steps=5)
    assert s1["step"] == 5
    s2 = run[second](cfg_for("s2", backend="tpu" if second == "torch" else "tpu-jax", resume=str(tmp_path / "s1")))
    assert s2["status"] == "complete" and s2["step"] == straight["step"] == 12
    assert s2["tokens_seen"] == straight["tokens_seen"]
    a = load_latest_checkpoint(tmp_path / "straight" / "checkpoints")[0]
    b = load_latest_checkpoint(tmp_path / "s2" / "checkpoints")[0]
    assert a["loader_states"] == b["loader_states"]                 # exactly the same data was used
    log = (tmp_path / "s2" / "train.log").read_text(encoding="utf-8")
    assert "resuming from" in log
    # The second half ran in a different framework (and Muon's bf16 rounding differs between
    # them), so the weights are close rather than identical.
    for key in a["model"]:
        ref = a["model"][key]
        assert (ref - b["model"][key]).abs().mean() <= 0.1 * ref.abs().mean() + 1e-6, key
    assert abs(_losses(tmp_path / "s2")[-1] - _losses(tmp_path / "straight")[-1]) < 0.25


def test_sharding_modes_agree_on_several_devices(cfg_for, tmp_path):
    """Four simulated CPU devices (the 8-chip TPU code path): fsdp, zero2 and dp, with different
    remat settings, must all compute the same training run. AdamW, so that no bf16 Newton-Schulz
    step amplifies summation-order rounding: the gradients themselves must agree."""
    runs = {}
    for sharding, remat in (("fsdp", "none"), ("zero2", "dots"), ("dp", "full")):
        name = f"{sharding}-{remat}"
        cfg = cfg_for(name, max_tokens=256 * 4, eval_interval=4, ckpt_interval=4, tpu_sharding=sharding,
                      tpu_remat=remat, optimizer="adamw")
        cfg_json = tmp_path / f"{name}.json"
        cfg_json.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
        script = textwrap.dedent(f"""
            import json, sys
            sys.path.insert(0, {str(REPO)!r})
            from sapia.config import config_from_dict
            from sapia.train_jax import train_jax
            cfg = config_from_dict(json.load(open({str(cfg_json)!r})))
            print("RESULT", json.dumps(train_jax(cfg, _allow_cpu=True)))
        """)
        env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_PLATFORMS="cpu")
        r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
        result = json.loads(r.stdout.split("RESULT", 1)[1])
        assert result["status"] == "complete" and result["step"] == 4
        log = (tmp_path / name / "train.log").read_text(encoding="utf-8")
        assert "4 x" in log and "split over 4 device(s)" in log and f"sharding {sharding}" in log
        runs[name] = load_latest_checkpoint(tmp_path / name / "checkpoints")[0]
    ref = runs.pop("fsdp-none")
    for name, ck in runs.items():
        assert ck["loader_states"] == ref["loader_states"]
        for key in ref["model"]:
            assert torch.allclose(ck["model"][key], ref["model"][key], atol=1e-5), (name, key)


def test_bench_mode_reports_speed_without_saving(cfg_for, tmp_path):
    out = train_jax(cfg_for("bench", max_steps=6, log_interval=2), _allow_cpu=True, _bench=True)
    assert out["status"] == "bench" and out["step"] == 6 and out["tok_per_sec"] > 0
    assert not (tmp_path / "bench" / "checkpoints").exists()


def test_param_sharding_rule():
    from jax.sharding import PartitionSpec as P

    from sapia.train_jax import _param_spec

    assert _param_spec("blocks.qkv", (28, 2688, 1792), 8) == P(None, "fsdp", None)
    assert _param_spec("blocks.norm1", (28, 1792), 8) == P()
    assert _param_spec("tok_emb", (49152, 1792), 8) == P("fsdp", None)
    assert _param_spec("norm_f", (1792,), 8) == P()
    assert _param_spec("tok_emb", (49152, 1792), 1) == P()
