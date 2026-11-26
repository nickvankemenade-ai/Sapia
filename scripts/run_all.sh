#!/usr/bin/env bash
# Prepare the data mix, then train, resuming automatically after a crash.
# Normally started inside tmux by scripts/launch.sh. Safe to rerun at any time: every stage resumes.
#
# Settings (environment variables, all optional):
#   SAPIA_DATA        data directory           (default ~/sapia-data/sapia-mix)
#   SAPIA_RUN         run / checkpoint folder  (default ~/sapia-runs/sapia)
#   SAPIA_TOKENS      total training tokens    (default 15e9)
#   SAPIA_NTFY_TOPIC  ntfy topic for phone alerts
#   SAPIA_TRAIN_ARGS  extra flags for sapia.train, e.g. "--max_total_hours 95"
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$HOME/sapia-venv/bin/activate"

DATA="${SAPIA_DATA:-$HOME/sapia-data/sapia-mix}"
RUN="${SAPIA_RUN:-$HOME/sapia-runs/sapia}"
TOKENS="${SAPIA_TOKENS:-15e9}"
mkdir -p "$(dirname "$DATA")" "$RUN"

notify() {
  if [ -n "${SAPIA_NTFY_TOPIC:-}" ]; then
    curl -s -m 10 -H "Title: Sapia" -d "$1" "https://ntfy.sh/$SAPIA_NTFY_TOPIC" >/dev/null || true
  fi
  echo "[run_all] $1"
}

notify "Data preparation starting ($TOKENS tokens into $DATA)."
for attempt in 1 2 3 4 5; do
  if python -m sapia.prepare_mix --total_tokens "$TOKENS" --out_dir "$DATA" 2>&1 | tee -a "$RUN/prep.log"; then
    break
  fi
  notify "Data preparation failed (attempt $attempt/5); retrying in 60 s (it resumes)."
  sleep 60
  if [ "$attempt" = 5 ]; then notify "Data preparation failed 5 times; stopping. See $RUN/prep.log"; exit 1; fi
done
notify "Data preparation done. Training starts."

for attempt in $(seq 1 20); do
  # shellcheck disable=SC2086
  python -m sapia.train --config configs/sapia.yaml --data_dir "$DATA" --out_dir "$RUN" \
    --max_tokens "$TOKENS" ${SAPIA_TRAIN_ARGS:-}
  code=$?
  case $code in
    0) notify "Training finished (exit 0). Final weights: $RUN/final.pt"; exit 0 ;;
    2) notify "Training stopped with a configuration/data error (exit 2); not retrying. See $RUN/train.log"; exit 2 ;;
    3) notify "Training stopped on a time limit or signal (exit 3); not retrying."; exit 3 ;;
    *) notify "Training crashed (exit $code), restart $attempt/20 in 60 s from the last checkpoint."; sleep 60 ;;
  esac
done
notify "Training crashed 20 times; giving up. See $RUN/train.log"
exit 1
