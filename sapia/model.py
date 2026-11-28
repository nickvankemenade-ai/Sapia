"""Sapia: decoder-only transformer.

Pre-norm RMSNorm, RoPE, grouped-query attention with QK-norm, SwiGLU, tied embeddings,
no biases, z-loss on the output logits, and a KV cache for generation.

Runs on PyTorch >= 2.3. On >= 2.5 grouped-query attention uses SDPA's native ``enable_gqa``;
older versions repeat the K/V heads instead (same result, a little more memory).
"""
from __future__ import annotations

import inspect
import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

def _sdpa_supports_gqa() -> bool:
    """SDPA is a builtin without an inspectable signature, so probe it once with tiny tensors."""
    q, kv = torch.zeros(1, 2, 1, 2), torch.zeros(1, 1, 1, 2)
    try:
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except TypeError:
        return False
    return True


_SDPA_HAS_GQA = _sdpa_supports_gqa()


class RMSNorm(nn.Module):
    """Root-mean-square norm with a learned gain and no bias, computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).type_as(x)


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape (seq_len, head_dim // 2)."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE needs an even head dimension")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate feature pairs (x[i], x[i + D/2]) of ``x`` (..., T, D); cos/sin are (T, D/2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class KVCache:
    """Preallocated keys/values for every layer: (n_layer, B, n_kv_head, max_len, head_dim) each.

    ``pos`` is the number of positions already filled. ``GPT.forward`` advances it after the
    last layer has written its keys and values.
    """

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        self.max_len = max_len or cfg.block_size
        shape = (cfg.n_layer, batch_size, cfg.n_kv_head, self.max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store k/v (B, n_kv, T, hd) at the current position; return all keys/values so far."""
        t = k.size(2)
        end = self.pos + t
        if end > self.max_len:
            raise ValueError(f"KV cache full ({self.max_len} positions)")
        self.k[layer, :, :, self.pos:end] = k.to(self.k.dtype)
        self.v[layer, :, :, self.pos:end] = v.to(self.v.dtype)
        return self.k[layer, :, :, :end].to(k.dtype), self.v[layer, :, :, :end].to(v.dtype)

    def reset(self) -> None:
        self.pos = 0


class CausalSelfAttention(nn.Module):
    """Grouped-query self-attention: n_head query heads share n_kv_head key/value heads."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, (cfg.n_head + 2 * cfg.n_kv_head) * cfg.head_dim, bias=False)
        # Muon orthogonalises q, k and v separately (see sapia.muon).
        self.qkv.weight.muon_split = [cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim,
                                      cfg.n_kv_head * cfg.head_dim]
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.shape
        H, KV, hd = self.n_head, self.n_kv_head, self.head_dim
        q, k, v = self.qkv(x).split([H * hd, KV * hd, KV * hd], dim=-1)
        q = q.view(B, T, H, hd).transpose(1, 2)      # (B, H, T, hd)
        k = k.view(B, T, KV, hd).transpose(1, 2)     # (B, KV, T, hd)
        v = v.view(B, T, KV, hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        past = 0
        if cache is not None:
            past = cache.pos
            k, v = cache.update(self.layer_idx, k, v)
        S = k.size(2)

        # Masking: plain causal when queries and keys start together; no mask for a single new
        # token (it may see everything cached); an explicit mask for a chunk after a prefix.
        attn_mask, is_causal = None, False
        if past == 0:
            is_causal = T > 1
        elif T > 1:
            q_pos = torch.arange(past, past + T, device=x.device).unsqueeze(1)
            attn_mask = torch.arange(S, device=x.device).unsqueeze(0) <= q_pos   # (T, S)

        dropout_p = self.dropout if self.training else 0.0
        if KV != H and _SDPA_HAS_GQA:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                               is_causal=is_causal, enable_gqa=True)
        else:
            if KV != H:
                k = k.repeat_interleave(H // KV, dim=1)
                v = v.repeat_interleave(H // KV, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                               is_causal=is_causal)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(y))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        x = x + self.mlp(self.norm2(x))
        return x


class ModelOutput(NamedTuple):
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None      # ce + z_loss term; what is backpropagated
    ce: Optional[torch.Tensor] = None        # mean cross-entropy (comparable across runs)
    z: Optional[torch.Tensor] = None         # weighted z-loss term


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init for the layers that write into the residual stream.
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=resid_std)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Trainable parameters; a tied embedding/unembedding matrix is counted once."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def new_cache(self, batch_size: int, max_len: Optional[int] = None,
                  dtype: Optional[torch.dtype] = None) -> KVCache:
        ref = self.tok_emb.weight
        return KVCache(self.cfg, batch_size, max_len, device=ref.device, dtype=dtype or ref.dtype)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, *,
                cache: Optional[KVCache] = None, all_logits: bool = False) -> ModelOutput:
        """Run the model.

        With ``targets`` the logits cover every position and the loss is cross-entropy plus the
        z-loss (targets equal to -1 are ignored). Without targets only the last position's
        logits are computed unless ``all_logits=True``. With ``cache`` the tokens in ``idx``
        continue the cached sequence and the cache is advanced.
        """
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        if start + T > self.cfg.block_size:
            raise ValueError(f"sequence length {start + T} exceeds block_size {self.cfg.block_size}")
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        x = self.emb_dropout(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.pos += T
        x = self.norm_f(x)

        if targets is None:
            return ModelOutput(self.lm_head(x if all_logits else x[:, -1:, :]))

        logits = self.lm_head(x)
        flat = logits.view(-1, logits.size(-1)).float()
        tgt = targets.reshape(-1)
        ce = F.cross_entropy(flat, tgt, ignore_index=-1)
        loss, z = ce, None
        if self.cfg.z_loss > 0:
            valid = tgt != -1
            lse = torch.logsumexp(flat, dim=-1)
            z = self.cfg.z_loss * (lse.pow(2) * valid).sum() / valid.sum().clamp(min=1)
            loss = ce + z
        return ModelOutput(logits, loss, ce.detach(), None if z is None else z.detach())

    def param_groups_for(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters into named groups.

        ``hidden``: the 2D weights inside the blocks (Muon, or AdamW with weight decay).
        ``head``: an untied output projection (AdamW with weight decay).
        ``no_decay``: the embedding (no weight decay, as in OLMo 2 and SmolLM3) and all 1D parameters.
        """
        emb = self.tok_emb.weight
        hidden = [p for p in self.blocks.parameters() if p.requires_grad and p.dim() == 2]
        hidden_ids = {id(p) for p in hidden}
        head = [] if self.lm_head.weight is emb else [self.lm_head.weight]
        head_ids = {id(p) for p in head}
        no_decay = [p for p in self.parameters()
                    if p.requires_grad and id(p) not in hidden_ids and id(p) not in head_ids]
        return {"hidden": hidden, "head": head, "no_decay": no_decay}

    def configure_optimizer(self, learning_rate: float, weight_decay: float, betas: tuple[float, float],
                            eps: float, device_type: str, optimizer: str = "adamw",
                            momentum: float = 0.95, ns_steps: int = 5) -> torch.optim.Optimizer:
        """Build the optimizer.

        ``muon``: Muon for the hidden matrices, AdamW for the rest (one optimizer, see sapia.muon).
        ``adamw``: AdamW everywhere (fused on CUDA). Both apply weight decay to the hidden matrices
        and an untied head only, never to the embedding or 1D parameters. Groups are named
        ``muon`` / ``adamw_decay`` / ``adamw_no_decay``.
        """
        g = self.param_groups_for()
        if optimizer == "muon":
            from .muon import MuonAdamW

            groups = [{"name": "muon", "params": g["hidden"], "use_muon": True, "weight_decay": weight_decay},
                      {"name": "adamw_decay", "params": g["head"], "weight_decay": weight_decay},
                      {"name": "adamw_no_decay", "params": g["no_decay"], "weight_decay": 0.0}]
            groups = [x for x in groups if x["params"]]
            return MuonAdamW(groups, lr=learning_rate, momentum=momentum, ns_steps=ns_steps, betas=betas, eps=eps)
        if optimizer != "adamw":
            raise ValueError(f"unknown optimizer {optimizer!r}")
        groups = [{"name": "adamw_decay", "params": g["hidden"] + g["head"], "weight_decay": weight_decay},
                  {"name": "adamw_no_decay", "params": g["no_decay"], "weight_decay": 0.0}]
        fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if fused else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps, **extra)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 stop_token: Optional[int] = None, generator: Optional[torch.Generator] = None,
                 use_cache: bool = True) -> torch.Tensor:
        """Autoregressive sampling. ``temperature=0`` is greedy.

        With ``use_cache`` each new token costs one position, not the whole context. When the
        context fills up, the cache is rebuilt from the most recent 3/4 of ``block_size``.
        """
        block = self.cfg.block_size
        cache = self.new_cache(idx.size(0)) if use_cache else None
        pending = idx                        # tokens not yet fed to the model
        for _ in range(max_new_tokens):
            if cache is None:
                ctx = idx[:, -block:]
            else:
                if cache.pos + pending.size(1) > block:
                    cache.reset()
                    pending = idx[:, -(3 * block // 4):]
                ctx = pending
            logits = self(ctx, cache=cache).logits[:, -1, :].float()
            nxt = self._sample(logits, temperature, top_k, top_p, generator)
            idx = torch.cat((idx, nxt), dim=1)
            pending = nxt
            if stop_token is not None and bool((nxt == stop_token).all()):
                break
        return idx

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int], top_p: Optional[float],
                generator: Optional[torch.Generator]) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            # Drop tokens once the cumulative mass *before* them already exceeds top_p.
            remove = (probs.cumsum(dim=-1) - probs) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)
