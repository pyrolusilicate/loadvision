#!/usr/bin/env bash
# Second session of 21.09: the head that counts the floor, now with the depth of every patch
# beside it, and the first real augmentation this project has had.
#
# Every run keeps its weights, folds first and then all 716 photos, so a winner needs no
# second training. `remote.sh start2` runs this in tmux.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
mkdir -p runs/logs

REFERENCE=runs/reference/oof.csv

# The depth model runs inside these configs.
uv pip install -q transformers==5.17.0

finetune() { uv run python -m loadvision.cli.commands.finetune "$@"; }
newest_run() { ls -td runs/finetune/*/ 2>/dev/null | head -1; }

run_head() {
  local config=$1 name log folds extra=${2:-}
  name=$(basename "$config" .json)

  log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}_folds.log"
  echo "$(date +%H:%M:%S) start  $name, five folds  (log: $log)"
  # shellcheck disable=SC2086
  if ! finetune "$config" --device cuda --reference "$REFERENCE" $extra >"$log" 2>&1; then
    echo "$(date +%H:%M:%S) FAILED $name folds, see $log"
    return
  fi
  folds=$(newest_run)
  echo "$(date +%H:%M:%S) done   $name folds -> $folds"
  grep -hE 'photos 716|reference' "$log" | tail -3

  log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}_full.log"
  echo "$(date +%H:%M:%S) start  $name, all 716 photos  (log: $log)"
  # shellcheck disable=SC2086
  if finetune "$config" --device cuda --full --calibrate-from "${folds}oof.csv" $extra \
      >"$log" 2>&1; then
    echo "$(date +%H:%M:%S) done   $name full -> $(newest_run)"
  else
    echo "$(date +%H:%M:%S) FAILED $name full, see $log"
  fi
}

# The beast: occupancy over the patches, each patch carrying its own depth.
run_head configs/finetune/dinov2_b14_occupancy_depth.json

# The same with the light, the grain and a blanked rectangle changed on every epoch. Until
# today the only augmentation this project had was the mirror.
run_head configs/finetune/dinov2_b14_occupancy_depth_aug.json

# Augmentation alone, on the shipped linear head, so its effect is measured by itself.
run_head configs/finetune/dinov2_b14_aug.json

# The 307 test photos, carrying the loads our best blend gave them, join the training half of
# every fold. They are 43% more photos than the dataset has, and they cannot flatter the
# measurement: no held-out photo is among them.
run_head configs/finetune/dinov2_b14_occupancy.json "--pseudo runs/reference/pseudo.csv"
run_head configs/finetune/dinov2_b14_aug.json "--pseudo runs/reference/pseudo.csv"

echo
echo "$(date +%H:%M:%S) plan2 finished."
du -sh runs/finetune 2>/dev/null
echo "Fetch before stopping: scripts/gpu/remote.sh fetch-weights"
