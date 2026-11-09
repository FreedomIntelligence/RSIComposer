#!/bin/zsh
# Use a prepared model environment; QWEN_PYTHON may select its interpreter.
set -eu
script_dir=${0:A:h}
export PYTHONDONTWRITEBYTECODE=1
export QWEN_DEVICE=${QWEN_DEVICE:-cpu}
export QWEN_THREADS=${QWEN_THREADS:-4}
exec "${QWEN_PYTHON:-python}" -u "$script_dir/serve.py" "$@"
