#!/usr/bin/env bash
# Prepares the rented GPU machine once: uv, Python 3.12, the project and a torch build the
# machine's driver can run, then proves the GPU trains. Safe to run again. Run on the machine.
set -euo pipefail
cd "$(dirname "$0")/../.."

# The lock pins torch 2.14 built for CUDA 13, which needs driver 580+. The machine has driver
# 535 (CUDA 12.2), where that build installs fine but never sees the GPU. The organizers
# verified this build on the same machine.
TORCH_SPEC=${TORCH_SPEC:-"torch==2.5.1 torchvision==0.20.1"}
TORCH_INDEX=${TORCH_INDEX:-"https://download.pytorch.org/whl/cu121"}

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv sync
# shellcheck disable=SC2086
uv pip install --reinstall $TORCH_SPEC --index-url "$TORCH_INDEX"

# From here on `uv run` must not re-sync to the lock, or it would put the CUDA 13 build back.
export UV_NO_SYNC=1

uv run python - <<'PY'
import timm
import torch

assert torch.cuda.is_available(), "torch does not see the GPU: check nvidia-smi and the torch build"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| CUDA", torch.version.cuda)
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).cuda()
batch = torch.randn(4, 3, 224, 224, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(batch).float().mean()
loss.backward()
print("forward and backward in bf16 on the GPU: ok")
PY

uv run pytest -q tests/ml/test_finetune.py
uv run python -m loadvision.cli.commands.finetune --help >/dev/null
echo "setup done: run scripts/gpu/queue.sh with fine-tuning configs"
