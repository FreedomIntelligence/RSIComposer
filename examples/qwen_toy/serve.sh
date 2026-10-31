#!/bin/zsh
# Reuse an installed pixi environment without installing or changing it.
set -eu
script_dir=${0:A:h}
export PYTHONDONTWRITEBYTECODE=1
export QWEN_DEVICE=${QWEN_DEVICE:-cpu}
export QWEN_THREADS=${QWEN_THREADS:-4}
exec pixi run --as-is \
  -m "${QWEN_PIXI_MANIFEST:-/home/jovyan/projects/vlm/SFT/Baseline/pixi.toml}" \
  -e "${QWEN_PIXI_ENV:-train}" \
  python -u "$script_dir/serve.py" "$@"
