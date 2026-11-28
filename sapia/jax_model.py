"""Sapia in JAX: the same transformer and optimizer as ``sapia.model`` / ``sapia.muon``.

Used by ``python -m sapia.train --tpu-jax`` (see ``sapia.train_jax``). Everything here mirrors
the PyTorch code operation for operation, so a checkpoint moves between the two without loss:

* **Parameters** are a flat dict. The 28 blocks are stacked along a leading layer axis (one
  ``lax.scan`` over layers: XLA compiles one block instead of 28). Matrices keep PyTorch's
  ``(out, in)`` layout, so conversion is a reshape and nothing else.
* **Precision** matches PyTorch autocast: float32 weights and residual stream, matmuls in
  bfloat16, norms with float32 statistics, logits and loss in float32.
* **Optimizer**: Muon (Nesterov momentum, quintic Newton-Schulz in bfloat16, per-projection
  for the fused q/k/v, update scaled by ``0.2 * sqrt(max(rows, cols))``, decoupled weight decay)
  for the hidden matrices and AdamW for the rest, exactly as ``sapia.muon.MuonAdamW``; or AdamW
  everywhere. Its state converts to and from PyTorch's ``optimizer.state_dict()``.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp

from .config import ModelConfig

# Per-block parameters: flat key -> PyTorch name inside ``blocks.<i>.``
BLOCK_PARAMS = {
    "blocks.norm1": "norm1.weight",
    "blocks.qkv": "attn.qkv.weight",
    "blocks.proj": "attn.proj.weight",
    "blocks.q_norm": "attn.q_norm.weight",
    "blocks.k_norm": "attn.k_norm.weight",
    "blocks.norm2": "norm2.weight",
    "blocks.gate": "mlp.gate.weight",
    "blocks.up": "mlp.up.weight",
    "blocks.down": "mlp.down.weight",
}
TOP_PARAMS = {"tok_emb": "tok_emb.weight", "norm_f": "norm_f.weight", "lm_head": "lm_head.weight"}
HIDDEN = ("blocks.qkv", "blocks.proj", "blocks.gate", "blocks.up", "blocks.down")   # the Muon matrices
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def param_keys(cfg: ModelConfig) -> list[str]:
    keys = ["tok_emb"] + [k for k in BLOCK_PARAMS if cfg.qk_norm or k not in ("blocks.q_norm", "blocks.k_norm")]
    keys.append("norm_f")
    if not cfg.tie_embeddings:
        keys.append("lm_head")
    return keys


def torch_name_to_key(name: str) -> tuple[str, Optional[int]]:
    """``blocks.3.attn.qkv.weight`` -> (``blocks.qkv``, 3); ``tok_emb.weight`` -> (``tok_emb``, None)."""
    m = _BLOCK_RE.match(name)
    if m:
        inner = {v: k for k, v in BLOCK_PARAMS.items()}
        return inner[m.group(2)], int(m.group(1))
    top = {v: k for k, v in TOP_PARAMS.items()}
    return top[name], None


# ----------------------------------------------------------------------------- conversion

def _tensor(arr: np.ndarray):
    """A CPU tensor viewing ``arr`` (no copy). Arrays from ``jax.device_get`` are read-only, which
    PyTorch warns about; checkpoints only ever read them, so the warning is silenced."""
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _np(t) -> np.ndarray:
    if hasattr(t, "detach"):                     # a torch tensor: copy, never share its memory
        return t.detach().to("cpu").float().numpy().copy()
    return np.array(t, dtype=np.float32)


def params_from_torch(state_dict: dict, cfg: ModelConfig) -> dict[str, np.ndarray]:
    """PyTorch ``model.state_dict()`` -> flat dict of float32 numpy arrays (blocks stacked)."""
    out: dict[str, np.ndarray] = {}
    for key in param_keys(cfg):
        if key.startswith("blocks."):
            out[key] = np.stack([_np(state_dict[f"blocks.{i}.{BLOCK_PARAMS[key]}"]) for i in range(cfg.n_layer)])
        else:
            out[key] = _np(state_dict[TOP_PARAMS[key]])
    return out


def params_to_torch(params: dict, cfg: ModelConfig) -> dict:
    """Flat (numpy) params -> a PyTorch ``state_dict`` (CPU float32 tensors, tied head shared)."""
    import torch

    sd = {}
    for key in param_keys(cfg):
        arr = np.asarray(params[key], dtype=np.float32)
        if key.startswith("blocks."):
            for i in range(cfg.n_layer):
                sd[f"blocks.{i}.{BLOCK_PARAMS[key]}"] = _tensor(arr[i])
        else:
            sd[TOP_PARAMS[key]] = _tensor(arr)
    if cfg.tie_embeddings:
        sd["lm_head.weight"] = sd["tok_emb.weight"]
    return sd


def init_params(cfg: ModelConfig, seed: int) -> dict[str, np.ndarray]:
    """Initial weights, drawn by the PyTorch model with the same seed (identical start in every mode)."""
    import torch

    from .model import GPT
    from .utils import seed_everything

    seed_everything(seed)
    model = GPT(cfg)
    return params_from_torch(model.state_dict(), cfg)


def optimizer_layout(cfg: ModelConfig, optimizer: str, lr: float, weight_decay: float,
                     betas: tuple[float, float], eps: float, momentum: float) -> tuple[list[dict], dict[int, str]]:
    """PyTorch optimizer ``param_groups`` (as in ``state_dict()``) and {state index: parameter name}.

    Built from the real PyTorch model and optimizer on the ``meta`` device (no memory), so the
    layout always matches what ``sapia.train`` / ``sapia.train_tpu`` produce.
    """
    import torch

    from .model import GPT

    with torch.device("meta"):
        model = GPT(cfg)
    opt = model.configure_optimizer(lr, weight_decay, betas, eps, "cpu", optimizer=optimizer, momentum=momentum)
    names = {id(p): n for n, p in model.named_parameters()}
    sd = opt.state_dict()
    index_to_name = {}
    for live, saved in zip(opt.param_groups, sd["param_groups"]):
        for p, i in zip(live["params"], saved["params"]):
            index_to_name[i] = names[id(p)]
    return sd["param_groups"], index_to_name


def is_muon(key: str, optimizer: str) -> bool:
    return optimizer == "muon" and key in HIDDEN


def decays(key: str) -> bool:
    """Weight decay on the hidden matrices and an untied head; never the embedding or norms."""
    return key in HIDDEN or key == "lm_head"


def init_opt_state(params: dict, optimizer: str) -> dict:
    """{"mu": Muon momentum, "m"/"v": Adam moments, "step": Adam step} with zeros."""
    z = lambda a: np.zeros(a.shape, np.float32)
    return {"mu": {k: z(a) for k, a in params.items() if is_muon(k, optimizer)},
            "m": {k: z(a) for k, a in params.items() if not is_muon(k, optimizer)},
            "v": {k: z(a) for k, a in params.items() if not is_muon(k, optimizer)},
            "step": np.zeros((), np.int32)}


def opt_state_from_torch(opt_sd: dict, params: dict, cfg: ModelConfig, layout: dict[int, str],
                         optimizer: str) -> dict:
    """PyTorch ``optimizer.state_dict()`` (MuonAdamW or AdamW) -> JAX optimizer state."""
    state = init_opt_state(params, optimizer)
    step = 0
    for idx, st in opt_sd.get("state", {}).items():
        key, layer = torch_name_to_key(layout[int(idx)])
        if cfg.tie_embeddings and key == "lm_head":
            key = "tok_emb"

        def put(group: str, value) -> None:
            arr = _np(value)
            if layer is None:
                state[group][key] = arr
            else:
                state[group][key][layer] = arr

        if "momentum_buffer" in st:
            put("mu", st["momentum_buffer"])
        if "exp_avg" in st:
            put("m", st["exp_avg"])
            put("v", st["exp_avg_sq"])
            step = max(step, int(float(st["step"])))
    state["step"] = np.asarray(step, np.int32)
    return state


def opt_state_to_torch(state: dict, cfg: ModelConfig, param_groups: list[dict], layout: dict[int, str],
                       optimizer: str) -> dict:
    """JAX optimizer state -> a PyTorch ``optimizer.state_dict()`` that MuonAdamW/AdamW can load."""
    import copy

    import torch

    step = int(np.asarray(state["step"]))
    out = {}
    for idx, name in layout.items():
        key, layer = torch_name_to_key(name)
        pick = (lambda g: np.asarray(state[g][key])) if layer is None else (lambda g: np.asarray(state[g][key][layer]))
        t = lambda g: _tensor(pick(g))
        if key in state["mu"]:
            out[idx] = {"momentum_buffer": t("mu")}
        elif step > 0:
            out[idx] = {"step": step if optimizer == "muon" else torch.tensor(float(step)),
                        "exp_avg": t("m"), "exp_avg_sq": t("v")}
    return {"state": out, "param_groups": copy.deepcopy(param_groups)}


# ----------------------------------------------------------------------------- model

def rms_norm(x, w, eps: float):
    xf = x.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (normed * w.astype(jnp.float32)).astype(x.dtype)


def linear(x, w, dtype):
    """``x @ w.T`` with PyTorch's (out, in) weight layout, computed in ``dtype``."""
    return jnp.einsum("...i,oi->...o", x.astype(dtype), w.astype(dtype))


def rope_tables(T: int, head_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = np.outer(np.arange(T, dtype=np.float32), inv_freq)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """x: (B, T, H, D); cos/sin: (T, D/2). Rotates the pairs (x[i], x[i + D/2])."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[None, :, None, :].astype(x.dtype)
    s = sin[None, :, None, :].astype(x.dtype)
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def xla_attention(q, k, v):
    """Causal grouped-query attention, (B, T, H, D) layout, XLA's fused implementation."""
    return jax.nn.dot_product_attention(q, k, v, is_causal=True)


AttentionFn = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def forward_loss(params: dict, tokens, cfg: ModelConfig, attention: AttentionFn = xla_attention,
                 dtype=jnp.bfloat16, remat: str = "none"):
    """Mean cross-entropy (+ z-loss) of next-token prediction on ``tokens`` (B, T + 1).

    Returns ``(loss, ce)``: ``loss`` is what is differentiated (ce + z-loss), ``ce`` is the
    plain cross-entropy that is logged.
    """
    idx, targets = tokens[:, :-1], tokens[:, 1:]
    B, T = idx.shape
    H, KV, hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    cos, sin = rope_tables(T, hd, cfg.rope_base)
    eps = cfg.norm_eps

    def block(x, p):
        h = rms_norm(x, p["blocks.norm1"], eps)
        qkv = linear(h, p["blocks.qkv"], dtype)
        q, k, v = jnp.split(qkv, [H * hd, (H + KV) * hd], axis=-1)
        q = q.reshape(B, T, H, hd)
        k = k.reshape(B, T, KV, hd)
        v = v.reshape(B, T, KV, hd)
        if cfg.qk_norm:
            q, k = rms_norm(q, p["blocks.q_norm"], eps), rms_norm(k, p["blocks.k_norm"], eps)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = attention(q, k, v).reshape(B, T, H * hd)
        x = x + linear(y, p["blocks.proj"], dtype)
        h = rms_norm(x, p["blocks.norm2"], eps)
        m = jax.nn.silu(linear(h, p["blocks.gate"], dtype)) * linear(h, p["blocks.up"], dtype)
        return x + linear(m, p["blocks.down"], dtype), None

    if remat in (True, "full"):
        block = jax.checkpoint(block, policy=jax.checkpoint_policies.nothing_saveable)
    elif remat == "dots":           # keep the matmul outputs, recompute norms/RoPE/SiLU/attention
        block = jax.checkpoint(block, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    x = jnp.take(params["tok_emb"], idx, axis=0).astype(jnp.float32)
    x, _ = jax.lax.scan(block, x, {k: v for k, v in params.items() if k.startswith("blocks.")})
    x = rms_norm(x, params["norm_f"], eps)
    head = params["tok_emb"] if cfg.tie_embeddings else params["lm_head"]
    logits = linear(x, head, dtype).astype(jnp.float32)
    lse = jax.nn.logsumexp(logits, axis=-1)
    tgt = jnp.take_along_axis(logits, targets[..., None].astype(jnp.int32), axis=-1)[..., 0]
    ce = jnp.mean(lse - tgt)
    loss = ce + cfg.z_loss * jnp.mean(lse * lse) if cfg.z_loss > 0 else ce
    return loss, ce


# ----------------------------------------------------------------------------- optimizer

def newton_schulz(g, steps: int = 5, eps: float = 1e-7):
    """Batched quintic Newton-Schulz over the last two axes, in bfloat16 (as sapia.muon)."""
    a, b, c = _NS_COEFFS
    x = g.astype(jnp.bfloat16)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = jnp.swapaxes(x, -1, -2)
    norm = jnp.sqrt(jnp.sum(x.astype(jnp.float32) ** 2, axis=(-2, -1), keepdims=True))
    x = (x / (norm + eps).astype(jnp.bfloat16)).astype(jnp.bfloat16)
    for _ in range(steps):
        s = x @ jnp.swapaxes(x, -1, -2)
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = jnp.swapaxes(x, -1, -2)
    return x.astype(g.dtype)


def muon_split(key: str, cfg: ModelConfig) -> Optional[list[int]]:
    if key == "blocks.qkv":
        hd = cfg.head_dim
        return [cfg.n_head * hd, cfg.n_kv_head * hd, cfg.n_kv_head * hd]
    return None


def orthogonalised(key: str, update, cfg: ModelConfig, ns_steps: int):
    """Muon's update for a stacked (L, rows, cols) matrix, per projection for the fused q/k/v."""
    split = muon_split(key, cfg)
    parts = jnp.split(update, np.cumsum(split)[:-1].tolist(), axis=-2) if split else [update]
    out = [newton_schulz(u, ns_steps) * (0.2 * math.sqrt(max(u.shape[-2], u.shape[-1]))) for u in parts]
    return jnp.concatenate(out, axis=-2) if split else out[0]


def global_norm(tree) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in jax.tree_util.tree_leaves(tree)))


def apply_updates(params: dict, grads: dict, state: dict, lr, cfg: ModelConfig, optimizer: str,
                  weight_decay: float, betas: tuple[float, float], eps: float, momentum: float,
                  grad_clip: float, ns_steps: int = 5):
    """Gradient clipping + one Muon/AdamW step. Returns (params, state, grad_norm)."""
    gnorm = global_norm(grads)
    if grad_clip > 0:
        coef = jnp.minimum(1.0, grad_clip / (gnorm + 1e-6))
        grads = {k: g * coef for k, g in grads.items()}
    b1, b2 = betas
    step = state["step"] + 1
    t = step.astype(jnp.float32)
    bias1, bias2 = 1.0 - b1 ** t, 1.0 - b2 ** t
    new_p, mu, m, v = {}, dict(state["mu"]), dict(state["m"]), dict(state["v"])
    for k, p in params.items():
        g = grads[k].astype(jnp.float32)
        wd = weight_decay if decays(k) else 0.0
        if is_muon(k, optimizer):
            buf = momentum * state["mu"][k] + g
            mu[k] = buf
            upd = g + momentum * buf                                   # Nesterov
            if wd:
                p = p * (1.0 - lr * wd)
            new_p[k] = p - lr * orthogonalised(k, upd, cfg, ns_steps)
        else:
            m[k] = b1 * state["m"][k] + (1.0 - b1) * g
            v[k] = b2 * state["v"][k] + (1.0 - b2) * g * g
            denom = jnp.sqrt(v[k] / bias2) + eps
            if wd:
                p = p * (1.0 - lr * wd)
            new_p[k] = p - (lr / bias1) * m[k] / denom
    return new_p, {"mu": mu, "m": m, "v": v, "step": step}, gnorm


def model_config_dict(cfg: ModelConfig) -> dict:
    return asdict(cfg)
