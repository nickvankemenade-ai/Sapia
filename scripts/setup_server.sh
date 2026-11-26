#!/usr/bin/env bash
# One-time setup on a fresh GPU server (tested target: Vultr GH200, Ubuntu, ARM64 + H100).
#   bash scripts/setup_server.sh
# Creates ~/sapia-venv, installs PyTorch with CUDA and Sapia, checks the GPU and runs the tests.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== system"
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "nvidia-smi failed: no GPU driver?"; exit 1; }

if ! command -v tmux >/dev/null || ! python3 -m venv --help >/dev/null 2>&1; then
  echo "== installing tmux and python venv support"
  sudo apt-get update -y && sudo apt-get install -y tmux python3-venv python3-dev
fi

VENV="$HOME/sapia-venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel

echo "== installing PyTorch (CUDA build; works on x86_64 and ARM64)"
pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
echo "== installing Sapia"
pip install -e ".[dev]" wandb

echo "== checking the GPU from PyTorch"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "| bf16", torch.cuda.is_bf16_supported())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
import time; t = time.time()
for _ in range(50): y = x @ x
torch.cuda.synchronize()
print(f"matmul: {50 * 2 * 4096**3 / (time.time() - t) / 1e12:.0f} TFLOP/s bf16")
PY

echo "== running the tests"
python -m pytest -q -p no:warnings

echo
echo "Setup done. Next: export your tokens (see docs/RUNBOOK.md), then: bash scripts/launch.sh"
