"""Mixed-precision selection: bf16 by default, with automatic fallbacks."""
from __future__ import annotations

import contextlib

import pytest
import torch

from sapia import precision as precision_mod
from sapia.config import ConfigError, TrainConfig, build_config, get_preset, parse_train_args
from sapia.precision import resolve_precision


def test_bf16_is_the_default_everywhere():
    assert TrainConfig().precision == "bf16"
    assert get_preset("sapia").precision == "bf16"
    assert get_preset("tiny").precision == "bf16"
    assert build_config().precision == "bf16"
    assert parse_train_args([]).precision == "bf16"
    assert parse_train_args(["--preset", "tiny"]).precision == "bf16"


def test_precision_flag_selects_fp16_and_fp32():
    assert parse_train_args(["--precision", "fp16"]).precision == "fp16"
    assert parse_train_args(["--precision", "fp32"]).precision == "fp32"
    assert parse_train_args(["--precision", "float16"]).precision == "fp16"   # alias
    assert parse_train_args(["--precision", "BF16"]).precision == "bf16"
    with pytest.raises(ConfigError):
        build_config(overrides={"precision": "fp8"})


def test_bf16_on_supported_gpu_uses_no_scaler():
    plan = resolve_precision("bf16", "cuda", bf16_supported=True, capability_major=8)
    assert (plan.mode, plan.use_scaler, plan.fell_back) == ("bf16", False, False)
    assert plan.dtype is torch.bfloat16
    assert not plan.make_scaler().is_enabled()


def test_bf16_falls_back_to_fp16_with_scaler_when_unsupported():
    plan = resolve_precision("bf16", "cuda", bf16_supported=False, capability_major=6)
    assert (plan.mode, plan.use_scaler, plan.fell_back) == ("fp16", True, True)
    assert plan.requested == "bf16" and plan.dtype is torch.float16
    assert "falling back to fp16" in plan.reason and "GradScaler" in plan.reason


def test_bf16_fallback_queries_torch_when_not_injected(monkeypatch):
    """The real code path: torch.cuda.is_bf16_supported() is consulted and False triggers the fallback."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: calls.append(1) or False)
    plan = resolve_precision("bf16", "cuda")
    assert calls, "torch.cuda.is_bf16_supported() must be called"
    assert plan.mode == "fp16" and plan.use_scaler and plan.fell_back

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    plan = resolve_precision("bf16", "cuda")
    assert plan.mode == "bf16" and not plan.use_scaler and not plan.fell_back


def test_bf16_support_check_swallows_errors(monkeypatch):
    def boom():
        raise RuntimeError("no CUDA runtime")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", boom)
    assert precision_mod.cuda_bf16_supported() is False
    assert resolve_precision("bf16", "cuda").mode == "fp16"


def test_pre_ampere_gpu_falls_back_even_if_torch_reports_support():
    # torch 2.3 says "supported" whenever a bf16 tensor can be created (emulation on e.g. a T4).
    plan = resolve_precision("bf16", "cuda", bf16_supported=True, capability_major=7)
    assert (plan.mode, plan.use_scaler, plan.fell_back) == ("fp16", True, True)
    assert "pre-Ampere" in plan.reason


@pytest.mark.parametrize("requested", ["bf16", "fp16"])
def test_cpu_uses_fp32_and_reports_the_fallback(requested):
    plan = resolve_precision(requested, "cpu")
    assert (plan.mode, plan.use_scaler, plan.fell_back) == ("fp32", False, True)
    assert "CPU" in plan.reason
    assert isinstance(plan.autocast("cpu"), contextlib.nullcontext)
    assert not plan.make_scaler().is_enabled()


def test_cpu_fp32_is_not_a_fallback():
    plan = resolve_precision("fp32", "cpu")
    assert (plan.mode, plan.fell_back) == ("fp32", False)


def test_explicit_fp16_and_fp32_on_cuda():
    fp16 = resolve_precision("fp16", "cuda", bf16_supported=True, capability_major=8)
    assert (fp16.mode, fp16.use_scaler, fp16.fell_back) == ("fp16", True, False)
    fp32 = resolve_precision("fp32", "cuda")
    assert (fp32.mode, fp32.use_scaler, fp32.fell_back) == ("fp32", False, False)
    assert fp32.dtype is None
    assert isinstance(fp32.autocast("cuda"), contextlib.nullcontext)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_autocast_dtypes_on_real_gpu():
    lin = torch.nn.Linear(8, 8).cuda()
    x = torch.randn(2, 8, device="cuda")
    plan = resolve_precision("bf16", "cuda")
    expected = torch.bfloat16 if plan.mode == "bf16" else torch.float16
    with plan.autocast("cuda"):
        assert lin(x).dtype == expected
    fp16 = resolve_precision("fp16", "cuda")
    with fp16.autocast("cuda"):
        assert lin(x).dtype == torch.float16
    assert fp16.make_scaler().is_enabled()
