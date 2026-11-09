"""Raw-text engine shared by weight-evolving Labs; no domain or selection policy.

Optional CUDA dependencies execute in isolated serving/training interpreters.
Every request, worker log and response remains under the owning Lab artifact.
"""
import json
import os
import re
import sys

from rsisys.runtime.execution.workers import run_workers, worker_environment


def validate_engine(config):
    subdir = config.get('output_subdir')
    if 'output_subdir' in config and (not isinstance(subdir, str) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', subdir)):
        raise ValueError('output_subdir must be a simple run-local directory name')
    for key in ('inference_python', 'training_python', 'data_root', 'inference_cuda_home', 'training_cuda_home'):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f'{key} must be a nonempty executable')
    ids = config.get('gpu_ids', [0])
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('gpu_ids must be distinct nonnegative integers')
    for key, default in [('worker_timeout', 7200), ('max_length', 4096), ('max_tokens', 1024),
                         ('lora_rank', 32), ('outer_epochs', 2), ('batch_size', 1)]:
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(f'{key} must be a positive integer')
    import math
    for key, default in [('outer_lr', 3e-4), ('rl_lr', 5e-7)]:
        v = config.get(key, default)
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'{key} must be finite and positive')
    seed = config.get('seed', 73)
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    for key in ('gradient_checkpointing', 'enforce_eager'):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f'{key} must be boolean')
    temperature = config.get('temperature', 0.7)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if 'stop' in config and (not isinstance(config['stop'], list) or
            any(not isinstance(s, str) or not s for s in config['stop'])):
        raise ValueError('stop must be a list of nonempty strings')


class WeightEngine:
    name, simulated = 'raw-vllm-torch-lora-v1', False
    worker_module = 'rsisys.backends.training.worker'
    operation_workers = {'generate': 'rsisys.backends.training.worker'}

    def __init__(self, config):
        validate_engine(config)
        self.config = dict(config)

    def _env(self):
        return worker_environment(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')

    def _gpu_ids(self):
        ids = self.config.get('gpu_ids', [0])
        inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
        if inherited is not None and any(str(gpu) not in inherited.split(',') for gpu in ids):
            raise ValueError('gpu_ids are physical indices and must stay within inherited CUDA_VISIBLE_DEVICES')
        return ids

    def _run(self, mode, model, items, directory, *, sample=False, distributed=False, **extra):
        directory.mkdir(parents=True, exist_ok=False)
        gpu_ids = self._gpu_ids()
        commands, envs, outputs = [], [], []
        shards = [items] if distributed else [items[rank::len(gpu_ids)] for rank in range(len(gpu_ids))]
        for rank, shard in enumerate(shards):
            if not shard:
                continue
            request, output = directory / f'request-{rank}.json', directory / f'response-{rank}.json'
            request.write_text(json.dumps(dict(config=self.config, mode=mode, model=model,
                items=shard, sample=sample, rank=rank, output=str(output), **extra)))
            interpreter = self.config.get('inference_python' if mode == 'generate' else 'training_python', sys.executable)
            command = [interpreter]
            if distributed:
                command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc-per-node={len(gpu_ids)}']
            command += ['-m', self.operation_workers.get(mode, self.worker_module), str(request)]
            commands.append(command)
            envs.append(dict(self._env(), CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_ids)) if distributed else str(gpu_ids[rank])))
            outputs.append(output)
            cuda_home = self.config.get('inference_cuda_home' if mode == 'generate' else 'training_cuda_home')
            if cuda_home:
                envs[-1].update(CUDA_HOME=cuda_home,
                    PATH=cuda_home + '/bin' + os.pathsep + envs[-1].get('PATH', ''),
                    LD_LIBRARY_PATH=cuda_home + '/lib64' + os.pathsep + envs[-1].get('LD_LIBRARY_PATH', ''))
        run_workers(commands, envs, directory, self.config.get('worker_timeout', 7200))
        if distributed:
            return json.loads(outputs[0].read_text())
        result = [item for output in outputs for item in json.loads(output.read_text())]
        if len(result) != len(items) or {r['index'] for r in result} != {r['index'] for r in items}:
            raise ValueError('Missing, duplicate or unexpected worker items')
        return sorted(result, key=lambda item: item['index'])

    def generate(self, model, prompts, directory, *, sample, max_tokens=None, seed_offset=0,
                 temperature=None, top_p=None):
        items = [{'index': i, 'prompt': text} for i, text in enumerate(prompts)]
        options = {k: v for k, v in {'temperature': temperature, 'top_p': top_p}.items() if v is not None}
        return self._run('generate', model, items, directory, sample=sample,
                         max_tokens=max_tokens or self.config.get('max_tokens', 1024), seed_offset=seed_offset, **options)

    def train(self, model, rows, directory, checkpoint, *, algorithm='sft', seed_offset=0, reference_model=None):
        return self._run('train', model, rows, directory, distributed=True,
                         checkpoint=str(checkpoint), algorithm=algorithm, seed_offset=seed_offset,
                         reference_model=reference_model)
