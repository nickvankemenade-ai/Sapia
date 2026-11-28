"""Muon for the hidden weight matrices, AdamW for everything else, in one optimizer.

Muon (Jordan et al., 2024) replaces each 2D weight's momentum update with its nearest
semi-orthogonal matrix, computed with a few Newton-Schulz iterations. Sapia follows the
recipe from "Muon is Scalable for LLM Training" (Moonshot AI, 2025), which made it hold up at
larger scale:

* decoupled weight decay inside Muon, and
* the update rescaled to AdamW's typical RMS: ``0.2 * sqrt(max(rows, cols))``, so Muon and
  AdamW share one learning rate and one weight decay.

Param groups carry ``use_muon``. Muon groups may give a parameter a ``muon_split`` attribute
(sizes along dim 0) so a fused q/k/v projection is orthogonalised per projection.
Everything is a plain ``torch.optim.Optimizer``: ``state_dict``/``load_state_dict``,
``GradScaler`` and gradient clipping work as with AdamW.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch

# Quintic Newton-Schulz coefficients from the reference Muon implementation.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise a 2D matrix: push all its singular values towards ~1.

    Runs in bfloat16 for speed (the iteration is robust to it). Returns the input's dtype.
    """
    assert g.ndim == 2, "Newton-Schulz expects a matrix"
    a, b, c = _NS_COEFFS
    x = g.to(torch.bfloat16)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer with Muon param groups (``use_muon=True``) and AdamW param groups."""

    def __init__(self, param_groups: Iterable[dict], lr: float = 1e-3, weight_decay: float = 0.0,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, betas=betas, eps=eps, use_muon=False)
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(f"Muon needs 2D parameters, got shape {tuple(p.shape)}")

    @staticmethod
    def muon_scale(rows: int, cols: int) -> float:
        """Match AdamW's update RMS (~0.2) for an orthogonal update of this shape."""
        return 0.2 * math.sqrt(max(rows, cols))

    def _orthogonalised(self, p: torch.Tensor, update: torch.Tensor, steps: int) -> torch.Tensor:
        split: Optional[list[int]] = getattr(p, "muon_split", None)
        parts = update.split(split, dim=0) if split else (update,)
        out = [newton_schulz(u, steps) * self.muon_scale(*u.shape) for u in parts]
        return torch.cat(out, dim=0) if split else out[0]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"]:
                mom = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    update = g.add(buf, alpha=mom) if group["nesterov"] else buf
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(self._orthogonalised(p, update, group["ns_steps"]), alpha=-lr)
            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    state["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias1 = 1 - beta1 ** t
                    bias2 = 1 - beta2 ** t
                    denom = (state["exp_avg_sq"] / bias2).sqrt_().add_(eps)
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / bias1)
        return loss
