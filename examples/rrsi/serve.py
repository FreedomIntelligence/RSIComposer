"""Launch owned, local vLLM replicas; Ctrl+C closes their process groups."""

import argparse
import json
import os
import sys
from pathlib import Path

from rsisys.labs.math_parallel import run_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port", type=int, default=18400)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        parser.error("GPUs must be distinct nonnegative integers")
    directory = Path(args.log_dir)
    directory.mkdir(parents=True, exist_ok=False)
    commands = []
    environments = []
    for index, gpu in enumerate(gpus):
        commands.append(
            [
                args.python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(Path(args.model).resolve(strict=True)),
                "--served-model-name",
                "rsisys-fixed",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port + index),
                "--max-model-len",
                "16384",
                "--gpu-memory-utilization",
                "0.80",
                "--max-num-seqs",
                str(args.max_num_seqs),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--no-enable-log-requests",
                "--reasoning-parser",
                "qwen3",
                "--enable-prefix-caching",
            ]
        )
        if args.enforce_eager:
            commands[-1].append("--enforce-eager")
        environments.append(
            dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                OMP_NUM_THREADS="1",
            )
        )
    (directory / "launch.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "gpus": gpus,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "enforce_eager": args.enforce_eager,
                "endpoints": [
                    f"http://127.0.0.1:{args.port + i}/v1" for i in range(len(gpus))
                ],
            },
            indent=2,
        )
    )
    run_workers(commands, environments, directory, 7 * 24 * 3600)


if __name__ == "__main__":
    main()
