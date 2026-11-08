#!/bin/zsh
# Submit via a thin kjob script that execs this shared file.
#SBATCH --job-name=qwen-toy-trainer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -eu
cd /home/jovyan/projects/RSISys
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${QWEN_THREADS:-4}
run_dir=${1:-runs/qwen-toy}
if (( $# > 0 )); then shift; fi
# Match the scheduler's exclusive device allocation on this cluster.
case ${NVIDIA_VISIBLE_DEVICES:-} in
  GPU-*)
    if [[ "$NVIDIA_VISIBLE_DEVICES" == *,* ]]; then
      print -u2 'Expected a single GPU UUID from the scheduler'
      exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
    ;;
  *) print -u2 'Missing an exclusive GPU UUID allocation'; exit 1 ;;
esac
exec timeout --signal=TERM --kill-after=20s 7180s \
  pixi run --as-is -m /home/jovyan/projects/vlm/SFT/Baseline/pixi.toml -e train \
  python -u experiments/qwen_toy/worker.py \
  --model "${QWEN_MODEL_PATH:-/home/jovyan/models/Qwen3.5-9B}" \
  --served-name "${QWEN_SERVED_NAME:-Qwen3.5-9B}" --run-dir "$run_dir" \
  --port "${QWEN_WORKER_PORT:-8766}" --device cuda --threads "${QWEN_THREADS:-4}" \
  --rounds "${QWEN_ROUNDS:-1}" --steps "${QWEN_STEPS:-24}" "$@"
