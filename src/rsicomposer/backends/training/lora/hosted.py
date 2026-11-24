"""Host-supervised transformers execution; the parent never loads CUDA weights."""
import json
from pathlib import Path
import sys
import tempfile

from .engine import LoRAEngine
from rsicomposer.runtime.execution.executor import WorkerSpec
from rsicomposer.runtime.execution.workers import worker_environment


class HostedLoRAEngine(LoRAEngine):
    train_worker = 'rsicomposer.backends.training.lora.hosted'

    def _call(self, mode, ref, rows, round_number, **options):
        with self.resources.acquire('rollout' if mode == 'generate' else 'trainer') as lease:
            with tempfile.TemporaryDirectory(prefix='rsicomposer-transformers-',
                    dir=getattr(getattr(self.executor, 'owner', self.executor), 'root', None)) as temporary:
                directory = Path(temporary)
                request, output = directory / 'request.json', directory / 'output.json'
                request.write_text(json.dumps(dict(mode=mode, config=self.config, ref=str(ref), rows=rows,
                    round_number=round_number, options=options, output=str(output))))
                python = self.config.get('inference_python' if mode == 'generate' else 'training_python', sys.executable)
                with (directory / 'worker.log').open('w') as log:
                    code = self.executor.run(WorkerSpec([python, '-m',
                        self.train_worker if mode == 'train' else 'rsicomposer.backends.training.lora.hosted',
                        str(request)], worker_environment(), stdout=log), lease)
                if code:
                    raise RuntimeError(f'{self.name} worker failed: ' + (directory / 'worker.log').read_text()[-6000:])
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
