"""Per-run artifact ownership and receipts for synchronous training Labs."""
import json
import math
from copy import deepcopy
from pathlib import Path

from rsicomposer.api import WorkerLostError
from rsicomposer.api import DEV, Invariant, ExecutionStatus, JobRef, OperationResult, OperationSpec, file_hash


class WeightTrainingLab:
    simulated = False
    default_temperature = .7
    fields = {}
    engine_methods = ('generate', 'train')

    def __init__(self, context, config, *, engine=None):
        from rsicomposer.api import WeightEngine
        self.config = dict(config)
        self.engine = WeightEngine(config) if engine is None else engine
        if (not isinstance(getattr(self.engine, 'name', None), str) or not self.engine.name.strip()
                or type(getattr(self.engine, 'simulated', None)) is not bool
                or any(not callable(getattr(self.engine, name, None)) for name in self.engine_methods)):
            raise TypeError('Weight engine must declare name/version, simulated bool, and required operations')
        bind = getattr(self.engine, 'bind_execution', None)
        if callable(bind):
            bind(context)
        self.simulated = self.engine.simulated
        self.root = Path(context.data_dir).resolve() / config.get('output_subdir', self.method)
        # A resumed run rebuilds the Lab over its own existing output directory.
        self.root.mkdir(parents=True, exist_ok=bool(getattr(context, 'restoring', False)))
        self.input_root = Path(config.get('data_root', Path.cwd())).resolve()
        self.hashes, self.inputs, self.models, self.seen = {}, {}, {}, set()
        self.artifact_models = {}

    # Durable state: everything execute() changes. Subclasses add their own fields
    # through _extra_state/_restore_extra. _active_model is not captured: execute()
    # sets it before any use. No heavyweight objects live in this process; models
    # are published checkpoint directories written by worker subprocesses.
    def state(self):
        return {'hashes': dict(self.hashes), 'inputs': dict(self.inputs),
                'models': {path: deepcopy(manifest) for path, manifest in self.models.items()},
                'seen': set(self.seen), 'artifact_models': dict(self.artifact_models),
                'initial_model': getattr(self, 'initial_model', None), **self._extra_state()}

    def restore(self, state):
        base = {'hashes', 'inputs', 'models', 'seen', 'artifact_models', 'initial_model'}
        if set(state) != base | set(self._extra_state()):
            raise ValueError(f'Unexpected {self.method} Lab state')
        for label, files in (('artifact', state['hashes']), ('input', state['inputs'])):
            for ref, digest in files.items():
                path = Path(ref)
                if not path.is_file() or file_hash(path) != digest:
                    raise ValueError(f'{self.method} {label} file changed while the run was stopped: {ref}')
        for ref, manifest in state['models'].items():
            for name, digest in manifest['weights'].items():
                weight = Path(ref) / name
                if not weight.is_file() or file_hash(weight) != digest:
                    raise ValueError(f'{self.method} checkpoint changed while the run was stopped: {weight}')
        self.hashes, self.inputs = dict(state['hashes']), dict(state['inputs'])
        self.models = {path: deepcopy(manifest) for path, manifest in state['models'].items()}
        self.seen = {tuple(key) for key in state['seen']}
        self.artifact_models = dict(state['artifact_models'])
        if state['initial_model'] is None:
            self.__dict__.pop('initial_model', None)
        else:
            self.initial_model = state['initial_model']
        self._restore_extra(state)

    def _extra_state(self):
        return {}

    def _restore_extra(self, state):
        return None

    def resource_roles(self):
        roles = getattr(self.engine, 'resource_roles', None)
        return roles() if callable(roles) else {}

    def profiles(self):
        return ('round-weight-training/v1',)

    def split_roles(self):
        return {'dev': DEV}

    def invariants(self):
        return (Invariant.baseline_before('generate'),)

    def describe(self):
        specs = []
        for name, fields in self.fields.items():
            props = {'round': {'type': 'integer', 'minimum': 0},
                     ('parent' if name in ('train', 'fit') else 'model'): {'type': 'string'},
                     **{key: {'type': kind} for key, kind in fields.items()}}
            if 'temperature' in props:
                props['temperature']['default'] = self.config.get('temperature', self.default_temperature)
            required = list(props)
            if name == 'evaluate':
                props['baseline'] = {'type': 'boolean'}
            specs.append(OperationSpec(name, f'{self.method} {name}', {
                'type': 'object', 'properties': props, 'required': required, 'additionalProperties': False}))
        return tuple(specs)

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
        if not set(specs[call.name].input_schema["required"]) <= set(args) <= set(props):
            raise ValueError('Missing or unexpected operation argument')
        for key, spec in props.items():
            if key not in args:
                continue
            value = args[key]
            if spec['type'] == 'integer' and (type(value) is not int or value < (0 if key == 'round' else 1)):
                raise ValueError(f'{key} must be a valid integer')
            if spec['type'] == 'string' and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{key} must be nonempty text')
            if spec['type'] == 'number' and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be finite and positive')
            if spec['type'] == 'boolean' and type(value) is not bool:
                raise ValueError(f'{key} must be boolean')
        if call.name == 'generate' and args['condition'] != 'none':
            raise ValueError(f'{self.method} does not use label-conditioned generation')
        if call.name == 'evaluate' and args['split'] != 'dev':
            raise ValueError('Use the configured development evaluation cohort')
        key = (args['round'], call.name)
        if key in self.seen:
            raise ValueError('Replay rejected; resume requires an explicit recovery adapter')
        self.seen.add(key)
        try:
            model = args['parent'] if call.name in ('train', 'fit') else args['model']
            self._active_model = str(self.path(model))
            if args['round'] == 0 and call.name == 'evaluate':
                self.initial_model = str(self.path(model))
            elif self._active_model not in {self.initial_model, *self.models}:
                raise ValueError('Model was not produced by this run')
            if 'reference' in args and str(self.path(args['reference'])) not in {self.initial_model, *self.models}:
                raise ValueError('Reference was not the initial model or a checkpoint produced by this run')
            output = self._execute_operation(call)
            if isinstance(output, JobRef):
                return output
            result = OperationResult(f'{self.method}.{call.name} completed', (), self.simulated,
                data={'request': args, **output})
        except WorkerLostError:
            raise
        except Exception as exc:
            result = OperationResult(f'{self.method}.{call.name} failed', (), self.simulated,
                ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}', data={'request': args})
        # A small independent receipt is usable even if the host journal is interrupted.
        path = self.root / f'round-{args["round"]}' / f'{call.name}-receipt.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump({'status': result.status.value, 'error': result.error, 'data': result.data}, stream, indent=2)
        return result

    def _execute_operation(self, call):
        return getattr(self, '_' + call.name)(call.arguments)

    def train_checkpoint(self, args, rows, *, algorithm='sft'):
        checkpoint = self.root / f'round-{args["round"]}' / 'model'
        metrics = self.engine.train(str(self.path(args['parent'])), rows,
            self.root / f'round-{args["round"]}' / 'training-workers', checkpoint,
            algorithm=algorithm, seed_offset=args['round'], **(
                {'reference_model': str(self.path(args['reference']))} if algorithm == 'grpo' else {}))
        weights = sorted(checkpoint.glob('*.safetensors'))
        if not weights or not metrics.get('weights_changed') or metrics.get('steps', 0) < 1:
            raise ValueError('Training did not publish updated weights')
        manifest = {'parent': str(self.path(args['parent'])),
            'reference': str(self.path(args['reference'])) if algorithm == 'grpo' else None, 'initial_model': self.initial_model,
            'dataset_sha256': file_hash(self.path(args['dataset'])), 'algorithm': algorithm,
            'weights': {p.name: file_hash(p) for p in weights}, 'metrics': metrics,
            'engine': self.engine.name, 'recipe': self.config,
            'verifier': getattr(getattr(self, 'verifier', None), 'name', None),
            'scorer': getattr(self, 'scorer_name', None)}
        (checkpoint / 'provenance.json').write_text(json.dumps(manifest, indent=2))
        self.models[str(checkpoint)] = manifest
        return {'model': str(checkpoint), 'training_performed': not self.simulated, 'training': metrics,
                'provenance_sha256': file_hash(checkpoint / 'provenance.json')}
