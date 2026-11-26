#!/usr/bin/env bash
# Start (or reattach to) the whole run in a tmux session called "sapia".
#   bash scripts/launch.sh        # then detach with Ctrl+B, D; reattach with: tmux attach -t sapia
set -euo pipefail
cd "$(dirname "$0")/.."
if tmux has-session -t sapia 2>/dev/null; then
  echo "A 'sapia' session is already running; attaching (detach with Ctrl+B, then D)."
  exec tmux attach -t sapia
fi
# Pass the settings through to the tmux session.
ENV_ARGS=()
for v in SAPIA_DATA SAPIA_RUN SAPIA_TOKENS SAPIA_NTFY_TOPIC SAPIA_TRAIN_ARGS WANDB_API_KEY HF_TOKEN; do
  if [ -n "${!v:-}" ]; then ENV_ARGS+=(-e "$v=${!v}"); fi
done
tmux new-session -d -s sapia "${ENV_ARGS[@]}" "bash scripts/run_all.sh; echo; echo 'run_all.sh exited; press Enter to close'; read"
echo "Started in tmux session 'sapia'. Attaching now; detach with Ctrl+B, then D."
exec tmux attach -t sapia
