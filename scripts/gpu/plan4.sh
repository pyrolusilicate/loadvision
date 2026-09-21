#!/usr/bin/env bash
# Last run of 21.09: the recipe that scored 4.80 on the platform - DINOv2 with the depth grid -
# with the two heads that were not tried on it. Everything else this session measured as noise,
# so this is the last question left before the package is fixed.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
mkdir -p runs/logs

REFERENCE=runs/reference/oof.csv
uv pip install -q transformers==5.17.0
finetune() { uv run python -m loadvision.cli.commands.finetune "$@"; }
newest_run() { ls -td runs/finetune/*/ 2>/dev/null | head -1; }

run_head() {
  local config=$1 name log folds
  name=$(basename "$config" .json)
  log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}_folds.log"
  echo "$(date +%H:%M:%S) start  $name, five folds"
  if ! finetune "$config" --device cuda --reference "$REFERENCE" >"$log" 2>&1; then
    echo "$(date +%H:%M:%S) FAILED $name, see $log"
    return
  fi
  folds=$(newest_run)
  echo "$(date +%H:%M:%S) done   $name -> $folds"
  grep -hE 'photos 716|reference' "$log" | tail -2
}

run_head configs/finetune/dinov2_b14_depth_mlp.json
run_head configs/finetune/dinov2_b14_depth_bins.json

echo
echo "$(date +%H:%M:%S) plan4 finished."
