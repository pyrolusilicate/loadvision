#!/usr/bin/env bash
# Drives the GPU machine from this computer, one step per call. Everything a step prints also
# lands in runs/gpu/<time>_<step>.txt, so the session can be read back later.
# Usage: scripts/gpu/remote.sh upload|setup|quick|start|status|fetch|fetch-weights
set -euo pipefail
cd "$(dirname "$0")/../.."

KEY=${GPU_KEY:-$HOME/.ssh/loadvision-team-25}
# The address changes on every start of the machine, so it lives outside git: in GPU_HOST or in
# runs/gpu/host, one line such as yc-user@<ip> taken from the bot's /status.
HOST=${GPU_HOST:-$(cat runs/gpu/host 2>/dev/null || true)}
if [ -z "$HOST" ]; then
  echo "no machine address: put yc-user@<ip from /status> into runs/gpu/host" >&2
  exit 2
fi
# The model every experiment is compared with: the shipped fine-tuned DINOv2 B, 5.72.
REFERENCE=${REFERENCE:-runs/finetune/run_20260918_210429_19744c/oof.csv}
SSH=(ssh -i "$KEY" -o ConnectTimeout=20)

step=${1:?usage: remote.sh upload|setup|quick|start|status|fetch|fetch-weights}
mkdir -p runs/gpu
log="runs/gpu/$(date +%Y%m%d_%H%M%S)_${step}.txt"
run() { "$@" 2>&1 | tee -a "$log"; }

case "$step" in
  upload)
    run rsync -az --exclude .venv --exclude runs --exclude __pycache__ \
      --exclude 'Инструкция*' -e "ssh -i $KEY" ./ "$HOST:loadvision/"
    run "${SSH[@]}" "$HOST" mkdir -p loadvision/runs/reference
    run rsync -az -e "ssh -i $KEY" "$REFERENCE" "$HOST:loadvision/runs/reference/oof.csv"
    # The light files of earlier runs: a --full run calibrates from its own out-of-fold run,
    # and that run may live only on this computer.
    run rsync -az -e "ssh -i $KEY" --include='*/' --include='oof.csv' --include='settings.json' \
      --exclude='*' runs/finetune/ "$HOST:loadvision/runs/finetune/"
    ;;
  setup)
    run "${SSH[@]}" "$HOST" \
      'sudo apt-get update -qq && sudo apt-get install -y -qq tmux && cd loadvision && bash scripts/gpu/setup.sh'
    ;;
  quick)
    # One fold, one epoch on the real GPU: proves the whole path and measures an epoch.
    run "${SSH[@]}" "$HOST" \
      'cd loadvision && export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 && uv pip install -q transformers==5.17.0 && uv run python -m loadvision.cli.commands.finetune configs/finetune/checks/quick.json --device cuda --fold 0 --runs-dir runs/checks'
    ;;
  start)
    run "${SSH[@]}" "$HOST" \
      'cd loadvision && mkdir -p runs/logs && tmux new -d -s gpu "bash scripts/gpu/plan.sh 2>&1 | tee runs/logs/queue.txt" && echo "plan started in tmux session gpu"'
    ;;
  status)
    # The plan's own record, then only the run in progress: old logs of finished or aborted
    # runs stay on the machine but would only confuse the picture.
    run "${SSH[@]}" "$HOST" '
      cd loadvision
      echo "=== план ==="
      cat runs/logs/queue.txt 2>/dev/null || echo "план ещё не запускался"
      current=$(ls -t runs/logs/*.log 2>/dev/null | head -1)
      if [ -n "$current" ]; then
        echo "=== идёт сейчас: $(basename "$current" .log) ==="
        grep -E "fold [0-9]: training|epoch |photos 716|this run|test .* photos" "$current" | tail -3
      fi
      echo "=== GPU: загрузка, память ==="
      nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
      tmux has-session -t gpu 2>/dev/null && echo "план работает" || echo "план завершён - можно fetch и /vm_stop"
    '
    ;;
  fetch)
    # Everything but the weights: a ViT-L run keeps about 6 GB of fold models. They stay on
    # the machine's disk; `fetch-weights` brings them over when a package needs them.
    run rsync -az --exclude '*.safetensors' -e "ssh -i $KEY" "$HOST:loadvision/runs/finetune/" runs/finetune/
    run rsync -az -e "ssh -i $KEY" "$HOST:loadvision/runs/logs/" runs/gpu/logs/
    ;;
  fetch-weights)
    run rsync -az -e "ssh -i $KEY" "$HOST:loadvision/runs/finetune/" runs/finetune/
    ;;
  *)
    echo "unknown step: $step" >&2
    exit 2
    ;;
esac
echo "log: $log"
