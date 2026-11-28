# Training Sapia on Kaggle's free TPUs

Sapia trains on a TPU v5e-8 in one of two ways, both using all 8 chips, fully sharded, every
step compiled and fused by XLA, with Pallas flash attention:

| flag | framework | notes |
|---|---|---|
| `--tpu-jax` | JAX | one compiled program per optimizer step; usually the faster one on TPU |
| `--tpu` | PyTorch/XLA | needs `torch_xla` installed |

(`--gpu` and `--cpu` exist too.) **Checkpoints are the same in every mode**: a session can stop
under `--tpu` and the next one continue under `--tpu-jax`, or the other way round, with the
same weights, optimizer state, data position and schedule.

Kaggle sessions stop after ~9 hours, so training runs as a **chain of sessions**: each one saves
a checkpoint just before its time is up, and the next one continues from it, exactly where it
stopped. Checkpoints are also saved every 45 minutes, on a stop signal and (JAX mode, if the
weights are still good) after a crash; each one is checksummed, and resuming falls back to an
older one if the newest is damaged.

Status: both TPU paths are tested on CPU (including resuming across modes and sharding over
several simulated devices); **neither has run on a real TPU yet**. Do the benchmark and test
session (step 3) before the first long one.

## 1. Data (done once, CPU notebooks)

Three CPU notebooks (or three versions of one) run `python -m sapia.prepare_mix --total_tokens
16e9 --only ...`, each producing a `sapia-mix/` folder in its output. Before training, turn each
finished output into a dataset: open the version → **Output** → **New Dataset**, named for example
`sapia-data-1`, `sapia-data-2`, `sapia-data-3`.

Use this cell, so that a hang restarts by itself. Prep writes its log to a file (never straight
into Kaggle's log pipe, which can stall); the cell copies new lines into the notebook log and, if
the file stops growing for 20 minutes (prep prints a line every 30 s), kills prep and starts it
again, resuming from what is on disk. Prep's own watchdog (exit code 75 after 15 minutes without
reading documents) does the same from the inside.

```python
import os, signal, subprocess, sys, time
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN set")
except Exception as e:
    print("no HF_TOKEN secret attached to this notebook:", e)
!git clone -q -b design/architecture https://github.com/nickvankemenade-ai/Sapia.git /tmp/Sapia
%cd /tmp/Sapia
!pip install -q tokenizers
ONLY = "reddit,finemath,gutenberg,fineweb_edu_top,ultrachat,stackexchange_top,reddit_top,wikipedia_anneal,finemath_top,dclm_anneal"
CMD = [sys.executable, "-u", "-m", "sapia.prepare_mix", "--total_tokens", "16e9", "--out_dir",
       "/kaggle/working/sapia-mix", "--num_workers", "4", "--only", ONLY]
LOG, QUIET_MINUTES = "/tmp/prep.log", 20
for attempt in range(1, 21):
    pos = os.path.getsize(LOG) if os.path.exists(LOG) else 0
    with open(LOG, "ab") as f:
        p = subprocess.Popen(CMD, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    changed = time.time()
    while p.poll() is None:
        time.sleep(15)
        size = os.path.getsize(LOG)
        if size != pos:
            with open(LOG, "rb") as f:
                f.seek(pos)
                print(f.read(size - pos).decode(errors="replace"), end="", flush=True)
            pos, changed = size, time.time()
        elif time.time() - changed > QUIET_MINUTES * 60:
            print(f"=== no output for {QUIET_MINUTES} min: killing attempt {attempt}", flush=True)
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    print(f"=== attempt {attempt}: exit code {p.returncode}", flush=True)
    if p.returncode == 0:
        break
```

The repo is cloned into `/tmp`, so the notebook's output (and the dataset made from it) holds only
`sapia-mix/`. Every data dataset should look the same: `sapia-mix/mix.json` plus
`sapia-mix/<phase>/<source>/`.

The log must not say "sending unauthenticated requests"; if it does, the `HF_TOKEN` secret is
not attached to this notebook (Add-ons → Secrets → tick it for this notebook).

## 2. The training notebook (the same notebook every session)

**Once:** create a notebook `sapia-train` (that exact name: the code looks up its own earlier
output under `nickvankemenade/sapia-train`).
- **Accelerator: TPU v5e-8**, **Internet: On**.
- **Add Input**: the three data datasets (they stay attached).
- **Add-ons -> Secrets**, each ticked for this notebook:
  - `KAGGLE_USERNAME` and `KAGGLE_KEY`: a Kaggle API key (kaggle.com -> Settings -> API -> Create
    New Token; the downloaded kaggle.json holds both). With it each session downloads the
    previous session's checkpoint by itself. (A newer-style token works as `KAGGLE_API_TOKEN`.)
  - `SAPIA_NTFY_TOPIC` (optional): phone updates via ntfy, a silent loss/MFU message every 15
    minutes plus alerts for start/end, crashes and every 1B tokens.

The only cell:

```python
!git clone -q -b design/architecture https://github.com/nickvankemenade-ai/Sapia.git /tmp/Sapia
!cd /tmp/Sapia && python -m sapia.kaggle train --hours 8
```

**Every session:** open the notebook, **Save Version -> Save & Run All (Commit)**. That's all.

What `python -m sapia.kaggle train` does: reads the secrets, installs what the TPU image lacks,
joins the attached data, downloads this notebook's last output, resumes from the newest
checkpoint it can find (that output, or any attached input), and trains with `--tpu-jax` and
Sapia's fixed settings (16B tokens, checkpoint every 45 minutes, newest checkpoint only) for
`--hours` hours of training. Setup and the final save add ~15 minutes, so 8 hours fits Kaggle's
9-hour limit. The output gets `run/checkpoints/` (~9 GB), `run/train.log` and
`run/metrics.jsonl` (both with the whole run's history). When the run reports `complete`,
`run/final.pt` holds the bf16 weights.

It refuses to start from step 0 when earlier progress might exist: when the Kaggle API fails,
when there are no API secrets and no attached checkpoint (first session ever without the API:
add `--fresh`), or when the last session started from step N but no checkpoint at or after N
can be found (it failed before saving: attach the newest version that has `run/checkpoints`).

Any training option can be added to the line, e.g. `--tpu_sharding dp --tpu_remat dots`, or
`--backend tpu` for PyTorch/XLA instead of JAX.

## 3. Optional: find the fastest setup (one ~30-minute interactive session)

```python
!git clone -q -b design/architecture https://github.com/nickvankemenade-ai/Sapia.git /tmp/Sapia
!cd /tmp/Sapia && python -m sapia.kaggle merge --out /tmp/sapia-mix && python -m sapia.bench_tpu --config configs/sapia.yaml --data_dir /tmp/sapia-mix
```

It tries five `--tpu-jax` setups, each in its own process: how the weights and gradients are
split over the 8 chips (`--tpu_sharding dp | zero2 | fsdp`) and whether activations are
recomputed to save memory (`--tpu_remat none | dots`). Each one compiles (a few minutes) and
trains 10 timed steps, and it prints tokens/s and MFU, or "out of TPU memory". The last lines
name the fastest, for example:

```
fastest: dp-dots: 140,000 tok/s, MFU 48.0% -> 16B tokens in ~32 TPU hours
use: --tpu-jax --tpu_sharding dp --tpu_remat dots
```

Add the two `--tpu_...` flags to the training line. The defaults (`zero2`, no remat) are used
otherwise.

What to look for in a session's log:
- `Sapia JAX training | 8 x TPU v5 lite ...`: all 8 chips in use.
- `data mix (main): fineweb_edu 45.0%, dclm 25.0%, ...`.
- `attention: Pallas flash attention kernel` and no `WARNING: flash attention failed`.
- `step ... | loss ... | ... tok/s | mfu ...%`: the speed.
- `checkpoint saved: ... verified`.
- The first steps are slow while XLA compiles (minutes); that happens again once at the switch to
  the anneal phase (longer sequences).
- Out of TPU memory ("RESOURCE_EXHAUSTED"): add `--tpu_remat dots`, then `--tpu_remat full`,
  then `--tpu_sharding fsdp`.
- If libtpu refuses to start because of an unknown flag, add `--tpu_xla_flags false`.

## Notes

- Output limit: one checkpoint (~9 GB) + logs fits Kaggle's 20 GB; hence `--keep_last 1
  --save_best false`.
- The merged data folder lives in `/tmp`, outside the saved output; the data stays in the
  read-only input datasets.
- Checkpoints are ordinary PyTorch files, whichever mode wrote them: they load on CUDA or CPU
  (`python -m sapia.sample`) and continue under `--tpu`, `--tpu-jax`, `--gpu` or `--cpu`.
  Switching mode between sessions only means adding `--backend tpu` (or removing it).
- Each checkpoint has a `.sha256` file next to it. If the newest checkpoint is damaged (for
  example a broken copy between sessions), resuming says so and uses the one before it.
- Changing `--max_tokens` between sessions is safe until the last 10% (the WSD decay).
