#!/usr/bin/env bash
# Fine-tunes configs one after another on the GPU, each logged to its own file. Start it in
# tmux: a dropped SSH connection must not stop a run. A failed config does not stop the rest.
# Usage: scripts/gpu/queue.sh configs/finetune/a.json configs/finetune/b.json ...
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
# Keeps the CUDA 12 torch that setup.sh installed; a sync would bring back the CUDA 13 one.
export UV_NO_SYNC=1

REFERENCE=${REFERENCE:-runs/reference/oof.csv}
FOLDS=${FOLDS:-configs/splits/folds.json}
mkdir -p runs/logs

for config in "$@"; do
  name=$(basename "$config" .json)
  log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}.log"
  extra=()
  if [ -f "$REFERENCE" ]; then
    extra=(--reference "$REFERENCE")
  fi
  echo "$(date +%H:%M:%S) start  $name  (log: $log)"
  if uv run python -m loadvision.cli.commands.finetune "$config" --device cuda --folds "$FOLDS" "${extra[@]}" >"$log" 2>&1; then
    echo "$(date +%H:%M:%S) done   $name: $(grep -h 'MAE before calibration' "$log" | tail -1)"
  else
    echo "$(date +%H:%M:%S) FAILED $name, see $log"
  fi
done
