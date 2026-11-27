"""Host-supervised LoRA execution; the parent never loads CUDA weights."""
import json
from pathlib import Path
import sys
import tempfile

from .engine import LoRAEngine
from rsicomposer.runtime.execution.executor import WorkerSpec
from rsicomposer.runtime.execution.workers import worker_environment


class HostedLoRAEngine(LoRAEngine):
    train_worker = 'rsicomposer.backends.training.lora.hosted'
    wide_ddp_worker = 'rsicomposer.backends.training.lora.wide_ddp'

    def resource_roles(self):
        from rsicomposer.runtime.execution.resources import ResourceRequest
        roles = super().resource_roles()
        world = self.config.get('training_world_size', 1)
        if type(world) is not int or world not in (1, 4) or (world == 4 and self.config.get('device') != 'cuda'):
            raise ValueError('training_world_size must be 1 or be 4 on CUDA')
        if world == 4:
            roles['trainer'] = ResourceRequest(gpus=world)
        return roles

    def _call(self, mode, ref, rows, round_number, **options):
        graph = mode == 'generate' and self.config.get('rollout_backend') == 'vllm_graph'
        world = self.config.get('training_world_size', 1) if mode == 'train' else 1
        ddp = world == 4
        worker = (self.wide_ddp_worker if ddp else self.train_worker) if mode == 'train' else (
            'rsicomposer.backends.training.lora.vllm_rollout' if graph else 'rsicomposer.backends.training.lora.hosted')
        if graph or ddp:
            manager = getattr(self.resources, 'owner', self.resources)
            if manager.config.get('executor', 'local') != 'local':
                raise ValueError(('vllm_graph' if graph else 'wide DDP') + ' requires the local Host executor')
        with self.resources.acquire('rollout' if mode == 'generate' else 'trainer') as lease:
            if graph and len(lease.devices) != 1:
                raise ValueError('vllm_graph requires exactly one GPU in the rollout lease')
            if ddp and len(lease.devices) != world:
                raise ValueError('wide DDP requires one Host lease covering exactly four GPU devices')
            with tempfile.TemporaryDirectory(prefix='rsicomposer-transformers-',
                    dir=getattr(getattr(self.executor, 'owner', self.executor), 'root', None)) as temporary:
                directory = Path(temporary)
                request, output = directory / 'request.json', directory / 'output.json'
                request.write_text(json.dumps(dict(mode=mode, config=self.config, ref=str(ref), rows=rows,
                    round_number=round_number, options=options, output=str(output))))
                python = self.config.get('inference_python' if mode == 'generate' else 'training_python', sys.executable)
                log_path = directory / 'worker.log'
                command = [python, '-m', worker, str(request)]
                if ddp:
                    command = [python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                               '--nproc-per-node=4', '--module', worker, str(request)]
                with log_path.open('w') as log:
                    code = self.executor.run(WorkerSpec(command, worker_environment(), stdout=log), lease)
                if code:
                    raise RuntimeError(f'{self.name} worker failed: ' + log_path.read_text()[-6000:])
                if graph:
                    for line in log_path.read_text().splitlines():
                        if line.startswith('RSICOMPOSER_WORKER '):
                            print(line, file=sys.stderr, flush=True)
                return json.loads(output.read_text())

    def generate(self, ref, rows, count, round_number, sample, *, adapter=None, temperature=.7):
        return self._call('generate', ref, rows, round_number, count=count, sample=sample,
                          adapter=str(adapter) if adapter is not None else None, temperature=temperature)

    def train(self, ref, rows, output, round_number, *, steps=None):
        return self._call('train', ref, rows, round_number, output=str(output), steps=steps)


def run_request(engine_type):
    request = json.loads(Path(sys.argv[1]).read_text())
    engine = engine_type(request['config'])
    options = request['options']
    if request['mode'] == 'train':
        options['output'] = Path(options['output'])
    result = getattr(engine, request['mode'])(request['ref'], request['rows'],
        round_number=request['round_number'], **options)
    Path(request['output']).write_text(json.dumps(result))


if __name__ == '__main__':
    run_request(LoRAEngine)
