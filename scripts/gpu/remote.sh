#!/usr/bin/env bash
# Drives the GPU machine from this computer, one step per call. Everything a step prints also
# lands in runs/gpu/<time>_<step>.txt, so the session can be read back later.
# Usage: scripts/gpu/remote.sh upload|setup|quick|start|status|fetch|fetch-weights
set -euo pipefail
cd "$(dirname "$0")/../.."

KEY=${GPU_KEY:-$HOME/.ssh/loadvision-team-25}
# Rented boxes answer on their own port: vast.ai hands out one per machine, the organizers' VM
# keeps 22. Set GPU_PORT, or put "user@ip PORT" in runs/gpu/host.
# The address changes on every start of the machine, so it lives outside git: in GPU_HOST or in
# runs/gpu/host, one line such as yc-user@<ip> taken from the bot's /status.
HOST=${GPU_HOST:-$(cat runs/gpu/host 2>/dev/null || true)}
PORT=${GPU_PORT:-$(echo "$HOST" | awk '{print $2}')}
HOST=$(echo "$HOST" | awk '{print $1}')
PORT=${PORT:-22}
if [ -z "$HOST" ]; then
  echo "no machine address: put user@<ip> [port] into runs/gpu/host" >&2
  exit 2
fi
# The model every experiment is compared with: the shipped fine-tuned DINOv2 B, 5.72.
REFERENCE=${REFERENCE:-runs/finetune/run_20260918_210429_19744c/oof.csv}
SSH=(ssh -i "$KEY" -p "$PORT" -o ConnectTimeout=20)
SHELL_SSH="ssh -i $KEY -p $PORT"

step=${1:?usage: remote.sh upload|setup|quick|start|start2|start3|status|fetch|fetch-weights}
mkdir -p runs/gpu
log="runs/gpu/$(date +%Y%m%d_%H%M%S)_${step}.txt"
run() { "$@" 2>&1 | tee -a "$log"; }

case "$step" in
  upload)
    run rsync -az --exclude .venv --exclude runs --exclude __pycache__ \
      --exclude 'Инструкция*' -e "$SHELL_SSH" ./ "$HOST:loadvision/"
    run "${SSH[@]}" "$HOST" mkdir -p loadvision/runs/reference
    run rsync -az -e "$SHELL_SSH" "$REFERENCE" "$HOST:loadvision/runs/reference/oof.csv"
    # The light files of earlier runs: a --full run calibrates from its own out-of-fold run,
    # and that run may live only on this computer.
    run rsync -az -e "$SHELL_SSH" --include='*/' --include='oof.csv' --include='settings.json' \
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
  start | start2 | start3 | start4)
    # start2 runs the second plan of a session in its own tmux window and its own queue file.
    plan=plan.sh; session=gpu; queue=queue.txt
    case "$step" in
      start2) plan=plan2.sh; session=gpu2; queue=queue2.txt ;;
      start3) plan=plan3.sh; session=gpu3; queue=queue3.txt ;;
      start4) plan=plan4.sh; session=gpu4; queue=queue4.txt ;;
    esac
    run "${SSH[@]}" "$HOST" \
      "cd loadvision && mkdir -p runs/logs && tmux new -d -s $session \"bash scripts/gpu/$plan 2>&1 | tee runs/logs/$queue\" && echo \"$plan started in tmux session $session\""
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
    run rsync -az --exclude '*.safetensors' -e "$SHELL_SSH" "$HOST:loadvision/runs/finetune/" runs/finetune/
    run rsync -az -e "$SHELL_SSH" "$HOST:loadvision/runs/logs/" runs/gpu/logs/
    echo "WEIGHTS ARE STILL ON THE MACHINE. Run fetch-weights before stopping it." | tee -a "$log"
    ;;
  fetch-weights)
    run rsync -az -e "$SHELL_SSH" "$HOST:loadvision/runs/finetune/" runs/finetune/
    run rsync -az -e "$SHELL_SSH" "$HOST:loadvision/runs/logs/" runs/gpu/logs/
    # The machine's disk goes when the machine goes, so say plainly what did and did not land.
    missing=0
    while IFS= read -r dir; do
      if [ -f "$dir/settings.json" ] && [ ! -f "$dir/encoder.safetensors" ] \
         && [ ! -d "$dir/folds" ]; then
        echo "NO WEIGHTS: $dir" | tee -a "$log"
        missing=$((missing + 1))
      fi
    done < <(find runs/finetune -mindepth 1 -maxdepth 1 -type d)
    run du -sh runs/finetune
    if [ "$missing" -gt 0 ]; then
      echo "$missing run(s) came without weights - do not stop the machine yet." | tee -a "$log"
      exit 1
    fi
    echo "every run has its weights; the machine can be stopped" | tee -a "$log"
    ;;
  *)
    echo "unknown step: $step" >&2
    exit 2
    ;;
esac
echo "log: $log"
