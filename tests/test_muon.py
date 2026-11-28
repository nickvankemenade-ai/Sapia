from __future__ import annotations

import math

import pytest
import torch

from sapia.config import ModelConfig
from sapia.model import GPT
from sapia.muon import MuonAdamW, newton_schulz


def small_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(ModelConfig(vocab_size=97, block_size=32, n_layer=2, n_head=4, n_kv_head=2, d_model=32,
                           mlp_hidden=64))


@pytest.mark.parametrize("shape", [(64, 64), (32, 96), (96, 32)])
def test_newton_schulz_pushes_singular_values_towards_one(shape):
    torch.manual_seed(0)
    g = torch.randn(*shape) * torch.logspace(-1, 0, shape[1])        # 10x spread in column scales
    s = torch.linalg.svdvals(newton_schulz(g).float())
    # 5 quintic steps map singular values into roughly [0.7, 1.2]; only the very smallest lag behind
    assert s.max() < 1.3 and s.median() > 0.6, s
    assert newton_schulz(g).shape == g.shape


def test_muon_update_has_adamw_like_rms():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.zeros(128, 64))
    opt = MuonAdamW([{"params": [w], "use_muon": True}], lr=1.0, nesterov=False)
    w.grad = torch.randn(128, 64)
    opt.step()
    rms = w.detach().pow(2).mean().sqrt().item()
    # an orthogonal 128x64 update has RMS sqrt(64/(128*64)); scaled by 0.2*sqrt(128) -> ~0.2
    assert 0.12 < rms < 0.25, rms


def test_fused_qkv_is_orthogonalised_per_projection():
    m = small_model()
    qkv = m.blocks[0].attn.qkv.weight
    assert qkv.muon_split == [32, 16, 16]
    opt = MuonAdamW([{"params": [qkv], "use_muon": True}], lr=1.0, nesterov=False)
    before = qkv.detach().clone()
    torch.manual_seed(1)
    qkv.grad = torch.randn_like(qkv)
    opt.step()
    delta = (before - qkv.detach())
    for part in delta.split(qkv.muon_split, dim=0):
        s = torch.linalg.svdvals(part / MuonAdamW.muon_scale(*part.shape))
        assert s.max() < 1.3 and s.min() > 0.5


def test_muon_and_adamw_groups_cover_the_model():
    m = small_model()
    opt = m.configure_optimizer(1e-3, 0.1, (0.9, 0.95), 1e-8, "cpu", optimizer="muon")
    names = [g["name"] for g in opt.param_groups]
    assert names == ["muon", "adamw_no_decay"]                          # tied model: no separate head group
    muon = opt.param_groups[0]
    assert muon["use_muon"] and all(p.dim() == 2 for p in muon["params"])
    assert not any(p is m.tok_emb.weight for p in muon["params"])        # the embedding stays on AdamW
    assert len(muon["params"]) == 2 * 5                                   # qkv, proj, gate, up, down per layer


def test_muon_trains_a_small_model_and_resumes_exactly():
    torch.manual_seed(0)
    idx = torch.randint(0, 97, (4, 32))

    def run(steps, model=None, opt=None):
        model = model or small_model()
        opt = opt or model.configure_optimizer(3e-3, 0.1, (0.9, 0.95), 1e-8, "cpu", optimizer="muon")
        losses = []
        for _ in range(steps):
            loss = model(idx[:, :-1], idx[:, 1:]).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        return model, opt, losses

    _, _, straight = run(40)
    assert straight[-1] < 0.5 * straight[0]

    m, opt, first = run(20)
    state = {"model": m.state_dict(), "opt": opt.state_dict()}
    m2 = small_model(seed=123)
    m2.load_state_dict(state["model"])
    opt2 = m2.configure_optimizer(3e-3, 0.1, (0.9, 0.95), 1e-8, "cpu", optimizer="muon")
    opt2.load_state_dict(state["opt"])
    _, _, second = run(20, m2, opt2)
    assert first + second == pytest.approx(straight, rel=1e-4, abs=1e-5)


def test_muon_rejects_non_matrix_parameters():
    with pytest.raises(ValueError):
        MuonAdamW([{"params": [torch.nn.Parameter(torch.zeros(4))], "use_muon": True}])
