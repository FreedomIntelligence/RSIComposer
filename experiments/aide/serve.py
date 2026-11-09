"""Serve owned fixed-model replicas with native AIDE tool-review support."""
import argparse
import json
import os
from pathlib import Path
import sys

from rsisys.labs.math_parallel import run_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--port', type=int, default=18540)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--enforce-eager', action='store_true')
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    if not gpus or len(gpus) != len(set(gpus)) or any(g < 0 for g in gpus):
        parser.error('GPUs must be distinct nonnegative indices')
    root = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=False)
    commands, envs = [], []
    for i, gpu in enumerate(gpus):
        commands.append([args.python, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', str(Path(args.model).resolve(strict=True)), '--served-model-name', 'aide-fixed',
            '--host', '127.0.0.1', '--port', str(args.port + i), '--max-model-len', '24576',
            '--gpu-memory-utilization', '0.70', '--max-num-seqs', '128',
            '--max-num-batched-tokens', '32768', '--no-enable-log-requests',
            '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_xml',
            '--enable-auto-tool-choice', '--enable-prefix-caching'])
        if args.enforce_eager:
            commands[-1].append('--enforce-eager')
        envs.append(dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                         TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1'))
    (root / 'launch.json').write_text(json.dumps({'model': args.model, 'gpus': gpus,
        'ports': [args.port + i for i in range(len(gpus))], 'tool_call_parser': 'qwen3_xml',
        'enable_auto_tool_choice': True}, indent=2))
    run_workers(commands, envs, root, 7 * 24 * 3600)


if __name__ == '__main__':
    main()
