#!/usr/bin/env bash
# This session's work on the GPU, in order of value; edit it before each session.
# `remote.sh start` runs it in tmux. A failed step is logged and the next one still runs.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
mkdir -p runs/logs

# The depth model needs transformers; a plain sync would also swap torch for the CUDA 13 build.
uv pip install -q transformers==5.17.0

# DINOv2 B with the depth grid, trained on all 716 photos: one set of weights instead of the five
# fold models whose average scored 4.80 on the leaderboard. Calibration comes from that config's
# out-of-fold run of 19.09, which is why that run folder must stay where it is.
run_full() {
  local config=$1 oof=$2 name
  name=$(basename "$config" .json)
  local log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}_full.log"
  echo "$(date +%H:%M:%S) start  $name --full  (log: $log)"
  if uv run python -m loadvision.cli.commands.finetune "$config" --device cuda --full \
      --calibrate-from "runs/finetune/$oof/oof.csv" >"$log" 2>&1; then
    echo "$(date +%H:%M:%S) done   $name: $(grep -h 'submission' "$log" | tail -1)"
  else
    echo "$(date +%H:%M:%S) FAILED $name, see $log"
  fi
}

# About 7 minutes.
run_full configs/finetune/dinov2_b14_depth.json run_20260919_183257_877c1e

echo "$(date +%H:%M:%S) plan finished"
