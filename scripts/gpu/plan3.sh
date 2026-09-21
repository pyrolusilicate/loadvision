#!/usr/bin/env bash
# Third session of 21.09: strengthening the construction that the platform twice preferred.
#
# Two independent pairs say the fold ensemble beats a single model on all photos: the linear
# head scored 5.10 alone against 4.80 over five folds, and the occupancy head 5.24 against
# 4.88. Neither difference clears the leaderboard's own noise by itself, but both point the
# same way, so the construction is worth pushing rather than replacing. Two ways to push it,
# neither tried before:
#
#   ten folds  - every model trains on 645 photos instead of 573
#   seed bags  - every model trains on all 716, differing only in the seed
#
# Five members either way, because the answer has to arrive inside a second or two and one
# model costs 250 ms: five are 1.25 s, eight are 2 s, ten are 2.5 s. The service carries
# exactly what was submitted, so the budget binds the submission too. Of the ten fold models
# the first five are shipped; all ten are used for the out-of-fold number, which stays honest.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
mkdir -p runs/logs

REFERENCE=runs/reference/oof.csv
FOLDS10=configs/splits/folds10.json

finetune() { uv run python -m loadvision.cli.commands.finetune "$@"; }
newest_run() { ls -td runs/finetune/*/ 2>/dev/null | head -1; }

uv pip install -q transformers==5.17.0

# Ten folds of one recipe: the out-of-fold number is honest, and the ten models answer the
# test together.
run_ten() {
  local config=$1 name log folds
  name=$(basename "$config" .json)
  log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}_f10.log"
  echo "$(date +%H:%M:%S) start  $name, ten folds  (log: $log)"
  if ! finetune "$config" --device cuda --folds "$FOLDS10" --reference "$REFERENCE" \
      >"$log" 2>&1; then
    echo "$(date +%H:%M:%S) FAILED $name ten folds, see $log"
    return
  fi
  folds=$(newest_run)
  echo "$(date +%H:%M:%S) done   $name ten folds -> $folds"
  grep -hE 'photos 716|reference' "$log" | tail -3
}

# Several models of one recipe on all 716 photos, differing only in their seed. Each is a
# submission of its own; `blend --submit` averages them afterwards.
run_bag() {
  local config=$1 seeds=$2 oof=$3 name log seed
  name=$(basename "$config" .json)
  for seed in $(seq 0 $((seeds - 1))); do
    log="runs/logs/$(date +%Y%m%d_%H%M%S)_${name}_s${seed}.log"
    echo "$(date +%H:%M:%S) start  $name, all 716 photos, seed $seed"
    if finetune "$config" --device cuda --full --seed "$seed" \
        --calibrate-from "$oof" >"$log" 2>&1; then
      echo "$(date +%H:%M:%S) done   seed $seed -> $(newest_run)"
    else
      echo "$(date +%H:%M:%S) FAILED $name seed $seed, see $log"
    fi
  done
}

# The shipped recipe, on ten folds. It carried depth when it scored 4.80; three separate ways
# of feeding depth have now measured as nothing or worse, the last of them on 21.09 at
# +0.22 +- 0.34, so it is dropped here and the head is the only thing left changing.
run_ten configs/finetune/dinov2_b14.json
TEN_PLAIN=$(newest_run)

# The recipe behind 4.88, on ten folds.
run_ten configs/finetune/dinov2_b14_occupancy.json
TEN_OCC=$(newest_run)

# The same two recipes as bags of five models trained on every photo: more photos each than
# a fold model gets, the same five forward passes at inference.
run_bag configs/finetune/dinov2_b14.json 5 "${TEN_PLAIN}oof.csv"
run_bag configs/finetune/dinov2_b14_occupancy.json 5 "${TEN_OCC}oof.csv"

echo
echo "$(date +%H:%M:%S) plan3 finished."
du -sh runs/finetune 2>/dev/null
df -h / | tail -1
echo "Fetch before stopping: scripts/gpu/remote.sh fetch-weights"
