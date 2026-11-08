"""Per-run artifact ownership and receipts for synchronous training Labs."""
import json
from hashlib import sha256
from pathlib import Path

from ..loops.contracts.coordination import OperationResult
from ..loops.contracts.decision import OperationSpec
from ..loops.contracts.lifecycle import ExecutionStatus


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class WeightArtifacts:
    simulated = False
    fields = {}
    engine_methods = ('generate', 'train')

    def __init__(self, context, config, *, engine=None):
        from .weight_engine import WeightEngine
        self.config = dict(config)
        self.engine = WeightEngine(config) if engine is None else engine
        if (not isinstance(getattr(self.engine, 'name', None), str) or not self.engine.name.strip()
                or type(getattr(self.engine, 'simulated', None)) is not bool
                or any(not callable(getattr(self.engine, name, None)) for name in self.engine_methods)):
            raise TypeError('Weight engine must declare name/version, simulated bool, and required operations')
        self.simulated = self.engine.simulated
        self.root = Path(context.data_dir).resolve() / self.method
        self.root.mkdir(parents=True, exist_ok=False)
        self.input_root = Path(config.get('data_root', Path.cwd())).resolve()
        self.hashes, self.inputs, self.models, self.seen = {}, {}, {}, set()
        self.artifact_models = {}

    def describe(self):
        return tuple(OperationSpec(name, f'{self.method} {name}; synchronous terminal receipt', {
            'type': 'object', 'properties': {'round': {'type': 'integer', 'minimum': 0},
                'model': {'type': 'string'}, **{key: {'type': kind} for key, kind in fields.items()}},
            'required': ['round', 'model', *fields], 'additionalProperties': False})
            for name, fields in self.fields.items())

    def path(self, ref):
        path = Path(ref)
        return (path if path.is_absolute() else self.input_root / path).resolve(strict=True)

    def read(self, ref, *, owned=False):
        path = self.path(ref)
        digest = file_hash(path)
        if owned:
            if self.hashes.get(str(path)) != digest:
                raise ValueError('Unowned or mutated stage artifact')
        else:
            prior = self.inputs.setdefault(str(path), digest)
            if prior != digest:
                raise ValueError('Input dataset changed during the experiment')
        return json.loads(path.read_text())

    def write(self, stage, round_number, rows):
        path = self.root / f'round-{round_number}' / f'{stage}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(rows, stream, ensure_ascii=False, allow_nan=False)
        digest = file_hash(path)
        self.hashes[str(path)] = digest
        if hasattr(self, '_active_model'):
            self.artifact_models[str(path)] = self._active_model
        return {'dataset': str(path), 'count': len(rows), 'sha256': digest}

    def preceding(self, ref, stage, round_number):
        expected = self.root / f'round-{round_number}' / f'{stage}.json'
        if self.path(ref) != expected:
            raise ValueError('Stage artifact has the wrong round or producer')
        if (str(expected) in self.artifact_models and
                self.artifact_models[str(expected)] != self._active_model):
            raise ValueError('Stage artifact was produced with a different parent model')
        return self.read(ref, owned=True)

    def execute(self, call):
        specs = {s.name: s for s in self.describe()}
        if call.name not in specs:
            raise ValueError('Unknown weight-evolution operation')
        props = specs[call.name].input_schema['properties']
        args = call.arguments
        if set(args) != set(props):
            raise ValueError('Missing or unexpected operation argument')
        for key, spec in props.items():
            value = args[key]
            if spec['type'] == 'integer' and (type(value) is not int or value < (0 if key == 'round' else 1)):
                raise ValueError(f'{key} must be a valid integer')
            if spec['type'] == 'string' and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{key} must be nonempty text')
        key = (args['round'], call.name)
        if key in self.seen:
            raise ValueError('Replay rejected; resume requires an explicit recovery adapter')
        self.seen.add(key)
        try:
            self._active_model = str(self.path(args['model']))
            if args['round'] == 0 and call.name == 'evaluate':
                self.initial_model = str(self.path(args['model']))
            elif str(self.path(args['model'])) not in {self.initial_model, *self.models}:
                raise ValueError('Model was not produced by this run')
            output = getattr(self, '_' + call.name)(args)
            result = OperationResult(f'{self.method}.{call.name} completed', (), self.simulated,
                data={'request': args, **output})
        except Exception as exc:
            result = OperationResult(f'{self.method}.{call.name} failed', (), self.simulated,
                ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}', data={'request': args})
        # A small independent receipt is usable even if the host journal is interrupted.
        path = self.root / f'round-{args["round"]}' / f'{call.name}-receipt.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump({'status': result.status.value, 'error': result.error, 'data': result.data}, stream, indent=2)
        return result

    def train_checkpoint(self, args, rows, *, algorithm='sft'):
        checkpoint = self.root / f'round-{args["round"]}' / 'model'
        metrics = self.engine.train(str(self.path(args['model'])), rows,
            self.root / f'round-{args["round"]}' / 'training-workers', checkpoint,
            algorithm=algorithm, seed_offset=args['round'], **(
                {'reference_model': self.initial_model} if algorithm == 'grpo' else {}))
        weights = sorted(checkpoint.glob('*.safetensors'))
        if not weights or not metrics.get('weights_changed') or metrics.get('steps', 0) < 1:
            raise ValueError('Training did not publish updated weights')
        manifest = {'parent': str(self.path(args['model'])), 'initial_model': self.initial_model,
            'dataset_sha256': file_hash(self.path(args['dataset'])), 'algorithm': algorithm,
            'weights': {p.name: file_hash(p) for p in weights}, 'metrics': metrics,
            'engine': self.engine.name, 'recipe': self.config,
            'verifier': getattr(getattr(self, 'verifier', None), 'name', None),
            'scorer': getattr(self, 'scorer_name', None)}
        (checkpoint / 'provenance.json').write_text(json.dumps(manifest, indent=2))
        self.models[str(checkpoint)] = manifest
        return {'model': str(checkpoint), 'training_performed': not self.simulated, 'training': metrics,
                'provenance_sha256': file_hash(checkpoint / 'provenance.json')}
