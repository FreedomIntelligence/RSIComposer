"""Optional offline data-parallel vLLM inference and torchrun LoRA training.

Each inference worker owns one GPU and only receives prompts. Worker processes
isolate the serving dependency environment from the framework and trainer.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

from rsisys.runtime.execution.workers import run_workers, worker_environment

from .engine import LoRAEngine


def shard_requests(rows, count, workers):
    return [[{'index': i, 'sample': s, 'prompt': row['prompt']}
             for i, row in enumerate(rows) for s in range(count)
             if (i * count + s) % workers == rank] for rank in range(workers)]


def gather_results(shards, rows, count):
    result = [[None] * count for _ in rows]
    for shard in shards:
        for item in shard:
            i, s = item['index'], item['sample']
            if (type(i) is not int or type(s) is not int or not 0 <= i < len(rows)
                    or not 0 <= s < count or result[i][s] is not None
                    or not isinstance(item['completion'], str)):
                raise ValueError('Invalid or duplicate inference worker output')
            result[i][s] = item['completion']
    if any(value is None for group in result for value in group):
        raise ValueError('Missing inference worker output')
    return result


class ParallelLoRAEngine(LoRAEngine):
    name = 'vllm-dp-ddp-lora-v1'

    def _env(self):
        # Propagate the actual imported framework snapshot, not an installed copy.
        return worker_environment(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')

    def generate(self, ref, rows, count, round_number, sample, *, adapter=None):
        gpu_ids = self.config.get('gpu_ids', [0])
        shards = shard_requests(rows, count, len(gpu_ids))
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='rsisys-inference-') as temporary:
            directory = Path(temporary)
            commands, environments, outputs = [], [], []
            for rank, (gpu, shard) in enumerate(zip(gpu_ids, shards, strict=True)):
                if not shard:
                    continue
                request = directory / f'request-{rank}.json'
                output = directory / f'output-{rank}.json'
                request.write_text(json.dumps({'config': self.config, 'model': str(Path(ref).resolve(strict=True)),
                    'adapter': str(adapter) if adapter is not None else None, 'requests': shard, 'round': round_number, 'sample': sample, 'rank': rank}))
                commands.append([self.config.get('inference_python', sys.executable), '-m',
                    'rsisys.backends.training.lora.worker', 'generate', str(request), str(output)])
                environments.append(dict(self._env(), CUDA_VISIBLE_DEVICES=str(gpu)))
                outputs.append(output)
            run_workers(commands, environments, directory, self.config.get('worker_timeout', 3600))
            result = gather_results([json.loads(path.read_text()) for path in outputs], rows, count)
        print(json.dumps({'stage': 'generate' if sample else 'evaluate', 'round': round_number,
            'backend': self.name, 'workers': len(outputs), 'completed': len(rows) * count,
            'elapsed_seconds': time.monotonic() - started}), file=sys.stderr, flush=True)
        return result

    def train(self, ref, rows, output, round_number):
        gpu_ids = self.config.get('gpu_ids', [0])
        with tempfile.TemporaryDirectory(prefix='rsisys-train-') as temporary:
            directory = Path(temporary)
            request = directory / 'request.json'
            request.write_text(json.dumps({'config': self.config, 'model': str(Path(ref).resolve(strict=True)),
                'rows': rows, 'output': str(output.resolve()), 'round': round_number}))
            command = [self.config.get('training_python', sys.executable), '-m', 'torch.distributed.run',
                '--standalone', '--nnodes=1', f'--nproc-per-node={len(gpu_ids)}',
                '-m', 'rsisys.backends.training.lora.worker', 'train', str(request)]
            run_workers([command], [dict(self._env(), CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_ids)))],
                directory, self.config.get('worker_timeout', 3600))
