#!/usr/bin/env bash
# This session's work on the GPU, in order of value; edit it before each session.
# `remote.sh start` runs it in tmux. A failed step is logged and the next one still runs.
#
# Every run keeps its weights. A rented machine loses its disk when it stops, so a run whose
# weights stayed on it has to be trained again: that already cost us one session. Each head is
# trained twice - five fold models for an honest number, then one model on all 716 photos for
# the submission - so whichever head wins, nothing has to be run a second time.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
mkdir -p runs/logs

# The shipped fine-tuned DINOv2 B, out of fold 5.72: every head is compared with it photo for
# photo, so the paired difference and its error come out of the run itself.
REFERENCE=runs/reference/oof.csv

finetune() { uv run python -m loadvision.cli.commands.finetune "$@"; }

newest_run() { ls -td runs/finetune/*/ 2>/dev/null | head -1; }

# Five folds for the number, then all 716 photos for the submission, calibrated from the folds.
run_head() {
  local config=$1 name log folds
  name=$(basename "$config" .json)

  log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}_folds.log"
  echo "$(date +%H:%M:%S) start  $name, five folds  (log: $log)"
  if ! finetune "$config" --device cuda --reference "$REFERENCE" >"$log" 2>&1; then
    echo "$(date +%H:%M:%S) FAILED $name folds, see $log"
    return
  fi
  folds=$(newest_run)
  echo "$(date +%H:%M:%S) done   $name folds -> $folds"
  grep -hE 'mae|against|paired' "$log" | tail -6

  log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}_full.log"
  echo "$(date +%H:%M:%S) start  $name, all 716 photos  (log: $log)"
  if finetune "$config" --device cuda --full --calibrate-from "${folds}oof.csv" \
      >"$log" 2>&1; then
    echo "$(date +%H:%M:%S) done   $name full -> $(newest_run)"
  else
    echo "$(date +%H:%M:%S) FAILED $name full, see $log"
  fi
}

# The head that counts the share of the floor over the 1036 patch tokens instead of guessing it
# from the class token: the one change of this session that alters what the head can see.
run_head configs/finetune/dinov2_b14_occupancy.json

# A distribution over the grid the expert marked on, instead of one number.
run_head configs/finetune/dinov2_b14_bins.json

# Huber instead of L1: the label itself wanders by up to ten points, so errors under five
# deserve a gentler gradient.
run_head configs/finetune/dinov2_b14_huber.json

# The control: if a hidden layer helps, the linear head was short of capacity.
run_head configs/finetune/dinov2_b14_mlp.json

echo
echo "$(date +%H:%M:%S) plan finished. Weights on this machine:"
du -sh runs/finetune/*/ 2>/dev/null | tail -20
echo "Fetch them before stopping the machine: scripts/gpu/remote.sh fetch-weights"
