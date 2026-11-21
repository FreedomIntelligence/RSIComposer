"""Raw-text engine shared by weight-evolving Labs; no domain or selection policy.

Optional CUDA dependencies execute in isolated serving/training interpreters.
Every request, worker log and response remains under the owning Lab artifact.
"""
import json
import os
import re
import sys

from rsicomposer.runtime.execution.workers import worker_environment
from rsicomposer.runtime.execution.resources import ResourceRequest
from rsicomposer.runtime.execution.executor import execution_capabilities


def validate_engine(config):
    if set(config) & {'gpu_ids', 'worker_timeout', 'outer_gpu_ids'}:
        raise ValueError('Device placement and worker timeout belong in host resources')
    subdir = config.get('output_subdir')
    if 'output_subdir' in config and (not isinstance(subdir, str) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', subdir)):
        raise ValueError('output_subdir must be a simple run-local directory name')
    for key in ('inference_python', 'training_python', 'data_root', 'inference_cuda_home', 'training_cuda_home'):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f'{key} must be a nonempty executable')
    for key, default in [('max_length', 4096), ('max_tokens', 1024),
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
    worker_module = 'rsicomposer.backends.training.worker'
    operation_workers = {'generate': 'rsicomposer.backends.training.worker'}

    def __init__(self, config):
        validate_engine(config)
        self.config = dict(config)

    def _env(self):
        return worker_environment(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')

    def bind_execution(self, context):
        self.resources, self.executor = execution_capabilities(context)
        self.resources.declare(self.resource_roles())

    def resource_roles(self):
        count = 0 if self.simulated or self.config.get('device') == 'cpu' else 1
        return {'rollout': ResourceRequest(gpus=count), 'trainer': ResourceRequest(gpus=count)}

    def _run(self, mode, model, items, directory, *, sample=False, distributed=False, **extra):
        with self.resources.acquire('rollout' if mode in ('generate', 'adapt') else 'trainer') as lease:
            directory.mkdir(parents=True, exist_ok=False)
            devices = lease.devices or (None,)
            commands, envs, outputs = [], [], []
            shards = [items] if distributed else [items[rank::len(devices)] for rank in range(len(devices))]
            for rank, shard in enumerate(shards):
                if not shard:
                    continue
                request, output = directory / f'request-{rank}.json', directory / f'response-{rank}.json'
                request.write_text(json.dumps(dict(config=self.config, mode=mode, model=model,
                    items=shard, sample=sample, rank=rank, output=str(output), **extra)))
                interpreter = self.config.get('inference_python' if mode == 'generate' else 'training_python', sys.executable)
                command = [interpreter]
                if distributed:
                    command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc-per-node={len(devices)}']
                command += ['-m', self.operation_workers.get(mode, self.worker_module), str(request)]
                commands.append(command)
                envs.append(dict(self._env(), CUDA_VISIBLE_DEVICES=','.join(str(i) for i in lease.devices) if distributed else (str(devices[rank]) if devices[rank] is not None else '')))
                outputs.append(output)
                cuda_home = self.config.get('inference_cuda_home' if mode == 'generate' else 'training_cuda_home')
                if cuda_home:
                    envs[-1].update(CUDA_HOME=cuda_home,
                        PATH=cuda_home + '/bin' + os.pathsep + envs[-1].get('PATH', ''),
                        LD_LIBRARY_PATH=cuda_home + '/lib64' + os.pathsep + envs[-1].get('LD_LIBRARY_PATH', ''))
            self.executor.run_workers(commands, envs, directory, lease)
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
