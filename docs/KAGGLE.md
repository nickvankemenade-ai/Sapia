# Training Sapia on Kaggle's free TPUs

Sapia trains on a TPU v5e-8 with `python -m sapia.train --tpu` (PyTorch/XLA, all 8 chips,
fully sharded, every step compiled and fused by XLA, Pallas flash attention). Kaggle sessions
stop after ~9 hours, so training runs as a **chain of sessions**: each one saves a checkpoint
just before its time is up, and the next one continues from it, exactly where it stopped.

Status: the TPU code path is written and its logic is tested on CPU; **it has not run on a real
TPU yet**. Do the 15-minute test session (step 3) before the first long one.

## 1. Data (done once, CPU notebooks)

Three CPU notebooks (or three versions of one) run `python -m sapia.prepare_mix --total_tokens
16e9 --only ...`, each producing a `sapia-mix/` folder in its output. Before training, turn each
finished output into a dataset: open the version → **Output** → **New Dataset**, named for example
`sapia-data-1`, `sapia-data-2`, `sapia-data-3`.

## 2. The training notebook (one notebook, one new version per session)

Create a notebook `sapia-train`:
- **Accelerator: TPU v5e-8**, **Internet: On**, **Persistence: Files only**.
- **Add Input**: the three `sapia-data-*` datasets. From session 2 on, also add **the previous
  version of `sapia-train` itself** (its output holds the checkpoint).
- Optional secrets: `HF_TOKEN`, `SAPIA_NTFY_TOPIC` (phone alerts), `WANDB_API_KEY`.

Cell 1: setup (installs PyTorch/XLA only if this image lacks it)

```python
import os, subprocess, sys
try:
    from kaggle_secrets import UserSecretsClient
    for name in ("SAPIA_NTFY_TOPIC", "WANDB_API_KEY", "HF_TOKEN"):
        try: os.environ[name] = UserSecretsClient().get_secret(name)
        except Exception: pass
except Exception:
    pass
try:
    import torch_xla  # noqa
    print("torch_xla", torch_xla.__version__)
except ImportError:
    !pip install -q 'torch~=2.6.0' 'torch_xla[tpu]~=2.6.0' -f https://storage.googleapis.com/libtpu-releases/index.html
!git clone -q -b design/architecture https://github.com/nickvankemenade-ai/Sapia.git /kaggle/working/Sapia || (cd /kaggle/working/Sapia && git pull -q)
%cd /kaggle/working/Sapia
!pip install -q tokenizers
```

Cell 2: join the data and find the previous checkpoint

```python
!python -m sapia.kaggle merge --out /tmp/sapia-mix
RESUME = !python -m sapia.kaggle latest
RESUME = RESUME[-1].strip() if RESUME and RESUME[-1].strip() else "auto"
print("resume from:", RESUME)
```

Cell 3: train until just before the session limit

```python
!python -m sapia.train --tpu --config configs/sapia.yaml \
    --data_dir /tmp/sapia-mix --out_dir /kaggle/working/run \
    --max_tokens 16e9 --resume {RESUME} \
    --max_hours 8.3 --keep_last 1 --save_best false --ckpt_minutes 45
```

Then **Save Version → Save & Run All (Commit)**. It runs in the background for up to 9 hours
(TPU sessions are shorter than CPU ones); `--max_hours 8.3` makes it save and stop with a
margin. The checkpoint (~9 GB) lands in this version's output under `run/checkpoints/`.

**Next session**: edit the notebook, **Add Input → Your Work → sapia-train**, pick the version
that just finished, and Save & Run All again. Cell 2 finds the newest checkpoint among the inputs
automatically. Repeat until the run reports `complete`; then `run/final.pt` holds the bf16 weights.

## 3. First time: a 15-minute test session

Run cell 3 once with `--max_hours 0.25` instead of 8.3 (interactive session is fine). Check the log for:
- `Sapia TPU training | ... 8 x TPU (SPMD, fully sharded)`: all 8 chips in use.
- `attention: Pallas flash attention kernel` and no `WARNING: flash attention failed`.
- `step ... | ... tok/s | mfu ...%`: the speed. Tokens/hour = tok/s x 3600; 16B tokens needs
  16e9 / (tok/s x 3600) hours in total.
- If it fails with a TPU out-of-memory ("RESOURCE_EXHAUSTED"), add `--tpu_tokens_per_chip 1024`.
- The first steps are slow while XLA compiles (minutes); that is normal and happens again once
  at the switch to the anneal phase.

## Notes

- Output limit: one checkpoint (~9 GB) + logs fits Kaggle's 20 GB; hence `--keep_last 1
  --save_best false`.
- The merged data folder lives in `/tmp`, outside the saved output; the data stays in the
  read-only input datasets.
- Checkpoints are ordinary PyTorch files: they load on CUDA or CPU (`python -m sapia.sample`).
- Changing `--max_tokens` between sessions is safe until the last 10% (the WSD decay).
