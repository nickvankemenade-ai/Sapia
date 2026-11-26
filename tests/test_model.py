from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from sapia import model as model_mod
from sapia.config import ConfigError, ModelConfig, get_preset
from sapia.model import GPT, RMSNorm, apply_rope, build_rope_cache


def small_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=97, block_size=32, n_layer=2, n_head=4, n_kv_head=2, d_model=32, mlp_hidden=64)
    base.update(kw)
    return ModelConfig(**base)


def small_model(seed: int = 0, **kw) -> GPT:
    torch.manual_seed(seed)
    return GPT(small_cfg(**kw)).eval()


# ----------------------------------------------------------------------------- building blocks

def test_rmsnorm_matches_formula_and_is_fp16_safe():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 16) * 3.0
    norm = RMSNorm(16)
    expected = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6)
    assert torch.allclose(norm(x), expected, atol=1e-6)
    y = RMSNorm(16)(torch.full((2, 4, 16), 300.0, dtype=torch.float16))
    assert torch.isfinite(y).all()


def test_rope_preserves_norm_and_depends_only_on_relative_position():
    torch.manual_seed(0)
    cos, sin = build_rope_cache(128, 32)
    x = torch.randn(2, 4, 64, 32)
    assert torch.allclose(apply_rope(x, cos[:64], sin[:64]).norm(dim=-1), x.norm(dim=-1), atol=1e-4)

    q, k = torch.randn(32), torch.randn(32)

    def rot(v, p):
        return apply_rope(v.view(1, 1, 1, -1), cos[p:p + 1], sin[p:p + 1]).view(-1)

    base = float(rot(q, 5) @ rot(k, 2))
    for shift in (1, 40, 100):
        assert float(rot(q, 5 + shift) @ rot(k, 2 + shift)) == pytest.approx(base, abs=1e-4)


# ----------------------------------------------------------------------------- parameter counts

def test_sapia_preset_has_the_documented_shape_and_size():
    cfg = get_preset("sapia").model
    assert (cfg.n_layer, cfg.d_model, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_hidden) == \
        (24, 2048, 16, 4, 128, 5632)
    assert (cfg.vocab_size, cfg.block_size, cfg.qk_norm, cfg.tie_embeddings) == (49152, 2048, True, True)
    assert cfg.num_params() == 1_182_900_224


def test_full_size_model_matches_formula_on_meta_device():
    cfg = get_preset("sapia").model
    with torch.device("meta"):
        model = GPT(cfg)
    assert model.num_params() == cfg.num_params() == 1_182_900_224


@pytest.mark.parametrize("kw", [{}, {"n_kv_head": 4}, {"n_kv_head": 1}, {"qk_norm": False},
                                {"tie_embeddings": False}])
def test_param_formula_matches_module(kw):
    cfg = small_cfg(**kw)
    assert GPT(cfg).num_params() == cfg.num_params()


def test_embeddings_are_tied_by_default_and_untied_on_request():
    m = small_model()
    assert m.lm_head.weight is m.tok_emb.weight
    u = small_model(tie_embeddings=False)
    assert u.lm_head.weight is not u.tok_emb.weight


def test_validation_rejects_bad_head_counts():
    with pytest.raises(ConfigError):
        small_cfg(n_kv_head=3).validate()          # 4 query heads not divisible by 3
    with pytest.raises(ConfigError):
        small_cfg(n_head=5).validate()             # 32 not divisible by 5


# ----------------------------------------------------------------------------- forward / loss

def test_forward_shapes_and_initial_loss_near_uniform():
    m = small_model()
    idx = torch.randint(0, 97, (3, 16))
    out = m(idx, torch.randint(0, 97, (3, 16)))   # random targets (tied weights favour the input token)
    assert out.logits.shape == (3, 16, 97)
    assert abs(float(out.ce) - math.log(97)) < 0.3
    assert m(idx).logits.shape == (3, 1, 97)
    assert m(idx, all_logits=True).logits.shape == (3, 16, 97)


def test_z_loss_is_added_and_matches_formula():
    m = small_model(z_loss=1e-2)
    idx = torch.randint(0, 97, (2, 16))
    tgt = idx.clone()
    tgt[0, :5] = -1                                   # ignored positions are excluded
    out = m(idx, tgt)
    flat = out.logits.view(-1, 97).float()
    valid = tgt.reshape(-1) != -1
    z = 1e-2 * torch.logsumexp(flat, dim=-1)[valid].pow(2).mean()
    ce = F.cross_entropy(flat, tgt.reshape(-1), ignore_index=-1)
    assert torch.allclose(out.z, z, atol=1e-6)
    assert torch.allclose(out.ce, ce, atol=1e-6)
    assert torch.allclose(out.loss, ce + z, atol=1e-6)

    off = small_model(z_loss=0.0)(idx, tgt)
    assert off.z is None and torch.allclose(off.loss, off.ce)


def test_attention_is_causal():
    m = small_model()
    a = torch.randint(0, 97, (1, 20))
    b = a.clone()
    b[0, 12:] = (b[0, 12:] + 1) % 97
    la, lb = m(a, all_logits=True).logits, m(b, all_logits=True).logits
    assert torch.allclose(la[:, :12], lb[:, :12], atol=1e-5)
    assert not torch.allclose(la[:, 12:], lb[:, 12:], atol=1e-3)


def test_sequences_longer_than_block_size_are_rejected():
    with pytest.raises(ValueError):
        small_model()(torch.zeros(1, 33, dtype=torch.long))


# ----------------------------------------------------------------------------- grouped-query attention

def _reference_attention(attn, x, cos, sin):
    """Plain multi-head attention with each K/V head copied to its group of query heads."""
    B, T, _ = x.shape
    H, KV, hd = attn.n_head, attn.n_kv_head, attn.head_dim
    q, k, v = attn.qkv(x).split([H * hd, KV * hd, KV * hd], dim=-1)
    q, k, v = (t.view(B, T, -1, hd).transpose(1, 2) for t in (q, k, v))
    if attn.q_norm is not None:
        q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    k, v = k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    y = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v
    return attn.proj(y.transpose(1, 2).reshape(B, T, -1))


@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
def test_gqa_matches_explicit_reference(n_kv_head):
    m = small_model(n_kv_head=n_kv_head)
    attn = m.blocks[0].attn
    x = torch.randn(2, 10, 32)
    cos, sin = m.rope_cos[:10], m.rope_sin[:10]
    assert torch.allclose(attn(x, cos, sin), _reference_attention(attn, x, cos, sin), atol=1e-5)


def test_native_gqa_and_repeat_fallback_agree(monkeypatch):
    if not model_mod._SDPA_HAS_GQA:
        pytest.skip("this PyTorch has no native enable_gqa")
    m = small_model()
    idx = torch.randint(0, 97, (2, 12))
    native = m(idx, all_logits=True).logits
    monkeypatch.setattr(model_mod, "_SDPA_HAS_GQA", False)
    fallback = m(idx, all_logits=True).logits
    assert torch.allclose(native, fallback, atol=1e-5)


def test_qk_norm_normalises_queries_and_keys():
    m = small_model()
    attn = m.blocks[0].attn
    assert attn.q_norm is not None and attn.k_norm is not None
    assert small_model(qk_norm=False).blocks[0].attn.q_norm is None


# ----------------------------------------------------------------------------- KV cache

def test_cached_decoding_matches_full_forward():
    m = small_model()
    idx = torch.randint(0, 97, (2, 20))
    full = m(idx, all_logits=True).logits

    cache = m.new_cache(2)
    steps = [m(idx[:, :8], cache=cache, all_logits=True).logits]          # prefill
    steps.append(m(idx[:, 8:13], cache=cache, all_logits=True).logits)    # chunk after a prefix
    for t in range(13, 20):                                                # one token at a time
        steps.append(m(idx[:, t:t + 1], cache=cache).logits)
    assert cache.pos == 20
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)


def test_cache_overflow_is_rejected():
    m = small_model()
    cache = m.new_cache(1)
    m(torch.zeros(1, 30, dtype=torch.long), cache=cache)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 3, dtype=torch.long), cache=cache)


def test_generate_with_and_without_cache_agree_greedily():
    m = small_model()
    prompt = torch.randint(0, 97, (2, 5))
    a = m.generate(prompt, 20, temperature=0.0, use_cache=True)
    b = m.generate(prompt, 20, temperature=0.0, use_cache=False)
    assert a.shape == (2, 25) and torch.equal(a, b)


def test_generate_past_block_size_keeps_going():
    m = small_model()
    out = m.generate(torch.randint(0, 97, (1, 30)), 40, temperature=0.8, top_k=10, top_p=0.9)
    assert out.shape == (1, 70)
    assert int(out.max()) < 97


def test_generate_stops_at_stop_token():
    m = small_model()
    prompt = torch.randint(0, 97, (1, 4))
    first = int(m.generate(prompt, 1, temperature=0.0)[0, -1])
    assert m.generate(prompt, 10, temperature=0.0, stop_token=first).shape == (1, 5)


# ----------------------------------------------------------------------------- training

def test_a_few_optimizer_steps_overfit_one_batch():
    torch.manual_seed(0)
    m = GPT(small_cfg()).train()
    opt = m.configure_optimizer(3e-3, 0.1, (0.9, 0.95), 1e-8, "cpu")
    idx = torch.randint(0, 97, (4, 32))
    first = None
    for _ in range(60):
        loss = m(idx[:, :-1], idx[:, 1:]).loss
        first = first if first is not None else float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first


def test_weight_decay_only_on_matrices():
    m = small_model()
    opt = m.configure_optimizer(1e-3, 0.1, (0.9, 0.95), 1e-8, "cpu")
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and decay["weight_decay"] == 0.1
    assert all(p.dim() < 2 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_autocast_forward_backward_on_gpu():
    torch.manual_seed(0)
    m = GPT(small_cfg()).cuda().train()
    idx = torch.randint(0, 97, (2, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(idx[:, :-1], idx[:, 1:])
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
