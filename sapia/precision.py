"""Mixed-precision selection.

Default is bf16 autocast with no GradScaler: on an A100 it is as fast as fp16 and it
cannot overflow the way fp16 does. fp16 (autocast + ``torch.cuda.amp.GradScaler``) and
fp32 stay available through the same ``--precision`` option.

Fallbacks, always logged by the caller through ``PrecisionPlan.reason``:
  * bf16 requested on a CUDA device where ``torch.cuda.is_bf16_supported()`` is False,
    or whose compute capability is below 8.0 (pre-Ampere: torch 2.3 reports "supported"
    there because it can emulate bf16, but it is much slower than fp16)
    -> fp16 + GradScaler.
  * any mixed-precision mode on CPU -> fp32 (CPU autocast is slow and GradScaler is
    CUDA-only, so plain fp32 is the sensible choice there).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch

from .config import normalize_precision


@dataclass(frozen=True)
class PrecisionPlan:
    requested: str      # what the user/config asked for
    mode: str           # what will actually run: bf16 | fp16 | fp32
    use_scaler: bool    # True only for fp16 on CUDA
    fell_back: bool     # mode differs from the request
    reason: str         # human-readable explanation, logged at startup

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[self.mode]

    def autocast(self, device_type: str):
        """Context manager for forward passes."""
        if self.mode == "fp32" or device_type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def make_scaler(self) -> "torch.cuda.amp.GradScaler":
        # torch.cuda.amp.GradScaler is the torch 2.3 spelling (torch.amp.GradScaler("cuda")
        # only became the preferred form in 2.4).
        return torch.cuda.amp.GradScaler(enabled=self.use_scaler)


def cuda_bf16_supported() -> bool:
    """``torch.cuda.is_bf16_supported()`` for the current device, False on any error."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def resolve_precision(requested: str, device_type: str, *,
                      bf16_supported: Optional[bool] = None,
                      capability_major: Optional[int] = None) -> PrecisionPlan:
    """Decide the precision mode that will actually run.

    ``bf16_supported`` and ``capability_major`` are injectable for tests; when None they
    are queried from the current CUDA device.
    """
    requested = normalize_precision(requested)

    if device_type != "cuda":
        if requested == "fp32":
            return PrecisionPlan(requested, "fp32", False, False, "fp32 on CPU")
        return PrecisionPlan(
            requested, "fp32", False, True,
            f"{requested} requested but the device is CPU: using fp32 "
            "(CPU autocast is slow and GradScaler is CUDA-only)",
        )

    if requested == "fp32":
        return PrecisionPlan(requested, "fp32", False, False, "fp32 (no autocast)")

    if requested == "fp16":
        return PrecisionPlan(requested, "fp16", True, False, "fp16 autocast + GradScaler")

    # requested == "bf16"
    if bf16_supported is None:
        bf16_supported = cuda_bf16_supported()
    if not bf16_supported:
        return PrecisionPlan(
            requested, "fp16", True, True,
            "bf16 requested but torch.cuda.is_bf16_supported() is False on this GPU: "
            "falling back to fp16 autocast + GradScaler",
        )
    if capability_major is None:
        try:
            capability_major = torch.cuda.get_device_capability()[0]
        except Exception:
            capability_major = None
    if capability_major is not None and capability_major < 8:
        # torch 2.3's is_bf16_supported() returns True whenever a bf16 tensor can be
        # created, which includes pre-Ampere GPUs (T4, RTX 20xx, ...) that have no native
        # bf16 support and run it far slower than fp16. Treat those as unsupported too.
        return PrecisionPlan(
            requested, "fp16", True, True,
            f"bf16 requested but this GPU (compute capability {capability_major}.x, pre-Ampere) "
            "has no native bf16 support: falling back to fp16 autocast + GradScaler",
        )
    return PrecisionPlan(requested, "bf16", False, False, "bf16 autocast, no GradScaler")
