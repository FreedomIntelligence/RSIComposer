"""Optional offline data-parallel vLLM inference and torchrun LoRA training.

Each inference worker owns one GPU and only receives prompts. Worker processes
isolate the serving dependency environment from the framework and trainer.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

from .math_engines import TransformersMathEngine


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


def run_workers(commands, environments, directory, timeout):
    """Fail fast, terminate whole process groups, and retain useful worker errors."""
    processes, streams = [], []
    started = time.monotonic()
    handlers = {}
    def cancelled(signum, frame):
        raise InterruptedError(f'GPU execution interrupted by signal {signum}')
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, cancelled)
    try:
        for rank, (command, env) in enumerate(zip(commands, environments, strict=True)):
            stream = (directory / f'worker-{rank}.log').open('w')
            streams.append(stream)
            env = dict(env, PATH=str(Path(command[0]).absolute().parent) + os.pathsep + env.get('PATH', ''))
            processes.append(subprocess.Popen(command, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            statuses = [p.poll() for p in processes]
            failed = next((i for i, status in enumerate(statuses) if status not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError(f'GPU worker {failed} failed ({statuses[failed]}):\n' +
                    (directory / f'worker-{failed}.log').read_text()[-12000:])
            if all(status == 0 for status in statuses):
                break
            if time.monotonic() - started > timeout:
                raise TimeoutError(f'GPU workers exceeded {timeout}s; last log:\n' +
                    (directory / 'worker-0.log').read_text()[-4000:])
            time.sleep(0.2)
        for rank in range(len(processes)):
            for line in (directory / f'worker-{rank}.log').read_text().splitlines():
                if line.startswith('RSISYS_WORKER '):
                    print(line, file=sys.stderr, flush=True)
    finally:
        # A launcher can exit before its CUDA descendants; signal its group too.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # Reaping the launcher does not prove that its descendants exited.
            # Always escalate for the owned group, including successful workers.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for stream in streams:
            stream.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


class ParallelMathEngine(TransformersMathEngine):
    name = 'vllm-dp-ddp-lora-v1'

    def _env(self):
        # Propagate the actual imported framework snapshot, not an installed copy.
        root = str(Path(__file__).resolve().parents[2])
        return dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''),
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')

    def _config(self):
        config = dict(self.config)
        if config.get('few_shot_data'):
            config['few_shot_data'] = str(self._path(config['few_shot_data']))
        return config

    def generate(self, ref, rows, count, round_number, sample):
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
                request.write_text(json.dumps({'config': self._config(), 'model': str(self._path(ref)),
                    'requests': shard, 'round': round_number, 'sample': sample, 'rank': rank}))
                commands.append([self.config.get('inference_python', sys.executable), '-m',
                    'rsisys.labs.math_worker', 'generate', str(request), str(output)])
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
            request.write_text(json.dumps({'config': self._config(), 'model': str(self._path(ref)),
                'rows': rows, 'output': str(output.resolve()), 'round': round_number}))
            command = [self.config.get('training_python', sys.executable), '-m', 'torch.distributed.run',
                '--standalone', '--nnodes=1', f'--nproc-per-node={len(gpu_ids)}',
                '-m', 'rsisys.labs.math_worker', 'train', str(request)]
            run_workers([command], [dict(self._env(), CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_ids)))],
                directory, self.config.get('worker_timeout', 3600))
