# Running Sapia on a GPU server (Vultr GH200)

Everything runs on the server inside `tmux`, so you can close your laptop. Every stage resumes
after an interruption: rerunning the same command continues where it stopped.

## 0. Before you start (on your phone / laptop)

- **Phone alerts**: install the **ntfy** app and subscribe to a random topic name, e.g.
  `sapia-7f3k9q2x`. Anyone who knows the name can read the messages, so keep it random.
- **Charts (optional)**: a free Weights & Biases account; copy your API key from wandb.ai/authorize.
- **Hugging Face (recommended)**: a free read token from huggingface.co/settings/tokens. Without
  it, downloads are rate-limited and data preparation is slower.

## 1. Start the server and log in

Create the GH200 instance in the Vultr console (Ubuntu image), then from your laptop:

```powershell
ssh root@<server-ip>
```

## 2. Get the code and set up (once, ~10 minutes)

```bash
git clone -b design/architecture https://github.com/nickvankemenade-ai/Sapia.git ~/Sapia
cd ~/Sapia
bash scripts/setup_server.sh
```

This installs PyTorch (CUDA, ARM64) and Sapia into `~/sapia-venv`, checks the GPU and runs the tests.
(If the repository is private, use a deploy key instead; see the conversation notes.)

## 3. Set your tokens and launch

```bash
export SAPIA_NTFY_TOPIC=sapia-7f3k9q2x       # your topic
export WANDB_API_KEY=...                      # optional
export HF_TOKEN=...                           # recommended
bash scripts/launch.sh
```

`launch.sh` starts `scripts/run_all.sh` in a tmux session called `sapia`:

1. **Data preparation** (`python -m sapia.prepare_mix`, ~7.5B tokens into `~/sapia-data/sapia-mix`).
   Several hours of downloading and tokenizing; uses the CPU cores only.
2. **Training** (`python -m sapia.train --config configs/sapia.yaml`), checkpoints every 30 minutes
   in `~/sapia-runs/sapia/checkpoints`, automatic restart from the last checkpoint after a crash.

You get an ntfy message when each stage starts, every 1B tokens, at the switch to the final
(anneal) phase, on any crash, and at the end.

**Detach and go to bed**: press `Ctrl+B`, then `D`. Close the laptop.

## 4. Checking in

| What | How |
|---|---|
| Phone | ntfy messages; W&B charts at wandb.ai (project `sapia`) |
| Live output | `ssh root@<server-ip>` then `tmux attach -t sapia` (leave with `Ctrl+B`, `D`; never `Ctrl+C`) |
| Is it running? | `tmux ls` |
| Logs | `~/sapia-runs/sapia/train.log`, `~/sapia-runs/sapia/prep.log` |
| GPU use | `nvidia-smi` |

## 5. Setting the token budget after the first hour

Training starts with `max_tokens = 7.5e9`. After ~30 minutes of training, read the speed from the
log (`tok/s`). Usable tokens ~= tok/s x hours left x 3600. To change it, stop and relaunch:

```bash
tmux attach -t sapia      # then Ctrl+C once: it saves a checkpoint and stops
export SAPIA_TOKENS=6.5e9  # your new total
bash scripts/launch.sh    # data prep is skipped (already done), training resumes
```

The learning rate is constant until the final 10% of `max_tokens` (warmup-stable-decay), so
changing `max_tokens` any time before that is safe. Data was prepared for 7.5B tokens; a smaller
total just leaves some unused, a larger one makes the loader repeat data.

## 6. Getting the model home

At the end, `~/sapia-runs/sapia/final.pt` holds the bf16 weights (~2.1 GB). On your laptop:

```powershell
scp root@<server-ip>:~/sapia-runs/sapia/final.pt C:\models\sapia\
```

Try it: `python -m sapia.sample --ckpt C:\models\sapia\final.pt --prompt "The history of"`.
Weights from any checkpoint: `python -m sapia.export --ckpt ~/sapia-runs/sapia --out final.pt`.

**Before destroying the instance**: copy `final.pt`, the newest `checkpoints/ckpt_*.pt` (to
continue training later) and the small files (`config.json`, `training_metrics.json`, logs).
Destroying the instance deletes its disk.
