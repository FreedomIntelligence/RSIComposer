"""Synchronous numeric-math Lab assembled from an engine and answer verifier.

The controller owns phase ordering and model selection. The Lab owns execution,
artifact provenance and held-out data checks. Components can be supplied by a
registered Lab factory without adding top-level registration object types.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

from rsisys.algorithms.rest_em.engine import (
    FixtureRestEMEngine,
    TransformersRestEMEngine,
)
from rsisys.api import WorkerLostError
from rsisys.api import (
    ExecutionStatus, DEV, Invariant, MetricSpec,
    OperationResult,
    OperationSpec,
    build_verifier,
    file_hash,
    parse_reference,
    validate_verifier,
)


def validate_config(config):
    allowed = {'mode', 'device', 'epochs', 'learning_rate', 'max_new_tokens', 'max_length',
               'seed', 'model_kind', 'engine', 'data_root', 'output_subdir',
               'generation_batch_size', 'train_batch_size', 'gradient_accumulation_steps',
               'prompt_format', 'few_shot_data', 'answer_format', 'verifier',
               'inference_python', 'training_python', 'gpu_memory_utilization',
               'max_num_seqs', 'max_num_batched_tokens', 'train_gradient_checkpointing'}
    if set(config) - allowed or config.get('engine', config.get('mode', 'fixture')) not in ('fixture', 'transformers', 'vllm'):
        raise ValueError('Unknown math Lab configuration or engine')
    if config.get('engine', config.get('mode')) == 'vllm' and config.get('device', 'cuda') != 'cuda':
        raise ValueError('vllm requires CUDA')
    for key in ('inference_python', 'training_python'):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f'{key} must be a nonempty executable path')
    memory = config.get('gpu_memory_utilization', 0.85)
    if type(memory) not in (int, float) or not math.isfinite(memory) or not 0 < memory < 1:
        raise ValueError('gpu_memory_utilization must be between zero and one')
    if type(config.get('train_gradient_checkpointing', False)) is not bool:
        raise ValueError('train_gradient_checkpointing must be boolean')
    if 'mode' in config and 'engine' in config and config['mode'] != config['engine']:
        raise ValueError('mode and engine must agree when both are supplied')
    if 'data_root' in config and (not isinstance(config['data_root'], str) or not config['data_root'].strip()):
        raise ValueError('data_root must be a nonempty path')
    subdir = config.get('output_subdir', 'math-lab')
    if not isinstance(subdir, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', subdir):
        raise ValueError('output_subdir must be a simple run-local directory name')
    if config.get('device', 'cuda') not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if config.get('model_kind', 'causal_lm') not in ('causal_lm', 'qwen3_5'):
        raise ValueError('model_kind must be causal_lm or qwen3_5')
    if config.get('prompt_format', 'chat') not in ('chat', 'completion'):
        raise ValueError('prompt_format must be chat or completion')
    if config.get('answer_format', 'numeric') not in ('numeric', 'latex'):
        raise ValueError('answer_format must be numeric or latex')
    validate_verifier(config.get('verifier', 'strict_numeric'))
    if 'few_shot_data' in config:
        if config.get('prompt_format', 'chat') != 'completion':
            raise ValueError('few_shot_data requires completion prompt format')
        if not isinstance(config['few_shot_data'], str) or not config['few_shot_data'].strip():
            raise ValueError('few_shot_data must be a nonempty path')
    for key, default in (('epochs', 1), ('max_new_tokens', 512), ('max_length', 2048),
                         ('generation_batch_size', 1), ('train_batch_size', 1),
                         ('gradient_accumulation_steps', 1), ('max_num_seqs', 128),
                         ('max_num_batched_tokens', 16384)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config.get('seed', 73)) is not int or config.get('seed', 73) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rate = config.get('learning_rate', 2e-4)
    if type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('learning_rate must be finite and positive')


class RestEMLab:
    def __init__(self, context, config, *, engine=None, verifier=None, input_root=None):
        validate_config(config)
        self.config = dict(config)
        self.input_root = Path(input_root or config.get('data_root', Path.cwd())).resolve()
        selected = config.get('engine', config.get('mode', 'fixture'))
        from rsisys.algorithms.rest_em.engine import ParallelRestEMEngine
        self.engine = engine if engine is not None else (
            FixtureRestEMEngine() if selected == 'fixture' else
            (ParallelRestEMEngine if selected == 'vllm' else TransformersRestEMEngine)(config, input_root=self.input_root))
        bind = getattr(self.engine, 'bind_execution', None)
        if callable(bind):
            bind(context)
        self.verifier = verifier if verifier is not None else build_verifier(config.get('verifier', 'strict_numeric'))
        for component, methods in ((self.engine, ('generate', 'train')), (self.verifier, ('parse', 'is_correct'))):
            if not isinstance(getattr(component, 'name', None), str) or not component.name.strip():
                raise TypeError('Math components must declare a nonempty name/version')
            if any(not callable(getattr(component, name, None)) for name in methods):
                raise TypeError('Invalid math component interface')
        if type(getattr(self.engine, 'simulated', None)) is not bool:
            raise TypeError('Math engine must declare simulated as a bool')
        self.simulated = self.engine.simulated
        self.root = Path(context.data_dir) / config.get('output_subdir', 'math-lab')
        # A resumed run rebuilds the Lab over its own existing output directory.
        self.root.mkdir(parents=True, exist_ok=bool(getattr(context, 'restoring', False)))
        self.initial = None
        self.seen = set()
        self.heldout_ids = self.heldout_prompts = None
        self.artifact_hashes = {}
        self.eval_ref = None

    def state(self):
        return {'initial': self.initial, 'seen': sorted(self.seen),
                'heldout_ids': None if self.heldout_ids is None else sorted(self.heldout_ids),
                'heldout_prompts': None if self.heldout_prompts is None else sorted(self.heldout_prompts),
                'artifact_hashes': dict(self.artifact_hashes),
                'eval_ref': None if self.eval_ref is None else list(self.eval_ref)}

    def restore(self, state):
        self.initial = state['initial']
        self.seen = {tuple(key) for key in state['seen']}
        self.heldout_ids = None if state['heldout_ids'] is None else set(state['heldout_ids'])
        self.heldout_prompts = None if state['heldout_prompts'] is None else set(state['heldout_prompts'])
        self.artifact_hashes = dict(state['artifact_hashes'])
        self.eval_ref = None if state['eval_ref'] is None else tuple(state['eval_ref'])
        for ref, digest in self.artifact_hashes.items():
            if file_hash(Path(ref)) != digest:
                raise ValueError(f'Published ReST^EM artifact changed while the run was stopped: {ref}')

    def resource_roles(self):
        roles = getattr(self.engine, 'resource_roles', None)
        return roles() if callable(roles) else {}

    def profiles(self):
        return ('round-weight-training/v1',)

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', self.verifier.name)

    def split_roles(self):
        return {'dev': DEV}

    def invariants(self):
        return (Invariant.baseline_before('generate'),)

    def final_evaluations(self):
        return ()

    def describe(self):
        fields = {
            'generate': {'model': 'string', 'dataset': 'string', 'samples_per_prompt': 'integer',
                         'temperature': 'number', 'condition': 'string'},
            'filter': {'samples': 'string', 'max_accepted_per_prompt': 'integer'},
            'train': {'parent': 'string', 'dataset': 'string'},
            'evaluate': {'model': 'string', 'dataset': 'string', 'split': 'string'},
            'select': {'model': 'string'},
        }
        return tuple(OperationSpec(name, f'ReST^EM {name}; returns a terminal receipt', {
            'type': 'object', 'properties': {'round': {'type': 'integer', 'minimum': 0},
                **{k: {'type': v} for k, v in props.items()},
                **({'steps': {'type': 'integer', 'minimum': 1}} if name == 'train' else {}),
                **({'baseline': {'type': 'boolean'}} if name == 'evaluate' else {})},
            'required': ['round', *props], 'additionalProperties': False,
        }) for name, props in fields.items())

    def _path(self, ref):
        path = Path(ref)
        return (path if path.is_absolute() else self.input_root / path).resolve(strict=True)

    def _read(self, ref):
        return [json.loads(line) for line in self._path(ref).read_text().splitlines() if line.strip()]

    def _problems(self, ref):
        rows = self._read(ref)
        if not rows or any(not isinstance(row, dict) or any(
                not isinstance(row.get(k), str) or not row[k].strip() for k in ('id', 'prompt', 'answer'))
                or parse_reference(self.verifier, row['answer']) is None for row in rows):
            raise ValueError('Expected nonempty JSONL: unique id, prompt and answer parseable by the selected verifier')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate problem IDs')
        return rows

    def _write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        ref, digest = str(path.resolve()), file_hash(path)
        self.artifact_hashes[ref] = digest
        return {'dataset': ref, 'count': len(rows), 'sha256': digest}

    def _owned_data(self, ref, round_number, filename):
        expected = str((self.root / f'round-{round_number}' / filename).resolve())
        if ref != expected or ref not in self.artifact_hashes:
            raise ValueError('Expected the artifact produced by the preceding stage of this round')
        if file_hash(Path(ref)) != self.artifact_hashes[ref]:
            raise ValueError('Produced dataset changed after publication')
        return self._read(ref)

    def execute(self, call):
        specs = {s.name: s for s in self.describe()}
        if call.name not in specs:
            raise ValueError('Unknown math operation')
        args = call.arguments
        props = specs[call.name].input_schema['properties']
        if not set(specs[call.name].input_schema["required"]) <= set(args) <= set(props):
            raise ValueError('Unexpected or missing math operation arguments')
        for key, spec in props.items():
            if key not in args:
                continue
            value = args[key]
            if spec['type'] == 'string' and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{key} must be nonempty text')
            if spec['type'] == 'integer' and (type(value) is not int or value < (0 if key == 'round' else 1)):
                raise ValueError(f'{key} is out of range')
        if 'temperature' in args and (type(args['temperature']) not in (int, float)
                or not math.isfinite(args['temperature']) or args['temperature'] < 0):
            raise ValueError('temperature must be finite and nonnegative')
        if call.name == 'generate' and args['condition'] not in ('none', 'answer'):
            raise ValueError('condition must be none or answer')
        key = (args['round'], call.name)
        if key in self.seen:
            raise ValueError('Repeated stage rejected: each (round, operation) runs once per run')
        self.seen.add(key)
        try:
            output = getattr(self, '_' + call.name)(args)
            return OperationResult(f'{call.name} completed', (), self.simulated,
                data={'request': args, 'training_performed': call.name == 'train' and not self.simulated, **output})
        except WorkerLostError:
            raise
        except Exception as exc:
            return OperationResult(f'{call.name} failed', (), self.simulated,
                ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}', data={'request': args})

    def _generate(self, args):
        rows = self._problems(args['dataset'])
        if self.heldout_ids is None:
            raise ValueError('Evaluate the held-out baseline before generation')
        if (self.heldout_ids & {r['id'] for r in rows}
                or self.heldout_prompts & {r['prompt'].strip() for r in rows}):
            raise ValueError('Training/evaluation problem overlap')
        output = []
        completions = self._completions(args['model'], rows, args['samples_per_prompt'], args['round'],
                                        args['temperature'] > 0, temperature=args['temperature'])
        if args['condition'] == 'answer':
            misses = [i for i, (row, answers) in enumerate(zip(rows, completions, strict=True))
                      if not any(self.verifier.is_correct(answer, row['answer']) for answer in answers)]
            hinted = [{**rows[i], 'prompt': rows[i]['prompt'] + '\nKnown correct answer: ' + rows[i]['answer']}
                      for i in misses]
            if hinted:
                repaired = self._completions(args['model'], hinted, args['samples_per_prompt'], args['round'],
                                            args['temperature'] > 0, temperature=args['temperature'])
                for i, answers in zip(misses, repaired, strict=True):
                    completions[i].extend(answers)
        for row, answers in zip(rows, completions, strict=True):
            for answer in answers:
                output.append({'id': row['id'], 'prompt': row['prompt'], 'completion': answer,
                               'answer': row['answer']})
        return self._write(self.root / f"round-{args['round']}" / 'generated.jsonl', output)

    def _filter(self, args):
        rows = self._owned_data(args['samples'], args['round'], 'generated.jsonl')
        counts, kept = Counter(), []
        for row in rows:
            if self.verifier.is_correct(row['completion'], row['answer']) and counts[row['id']] < args['max_accepted_per_prompt']:
                # Gold answers are verifier input only; training sees generated completion.
                kept.append({k: row[k] for k in ('id', 'prompt', 'completion')})
                counts[row['id']] += 1
        result = self._write(self.root / f"round-{args['round']}" / 'accepted.jsonl', kept)
        return {**result, 'verifier': self.verifier.name, 'source_sha256': file_hash(self._path(args['samples']))}

    def _train(self, args):
        rows = self._owned_data(args['dataset'], args['round'], 'accepted.jsonl')
        if not rows:
            raise ValueError('Cannot train on an empty accepted dataset')
        output = self.root / f"round-{args['round']}" / 'model'
        output.mkdir(parents=True, exist_ok=False)
        self.engine.train(args['parent'], rows, output, args['round'], steps=args.get('steps'))
        manifest = {'parent': args['parent'], 'dataset': args['dataset'],
                    'dataset_sha256': file_hash(self._path(args['dataset'])),
                    'round': args['round'], 'simulated': self.simulated,
                    'training_performed': not self.simulated, 'recipe': self.config,
                    'engine': self.engine.name, 'verifier': self.verifier.name}
        (output / 'rest-em.json').write_text(json.dumps(manifest, indent=2))
        return {'model': str(output.resolve()), 'manifest': manifest}

    def _evaluate(self, args):
        if args['split'] != 'dev':
            raise ValueError('ReST EM evaluates its configured development dataset')
        rows = self._problems(args['dataset'])
        if args['round'] == 0:
            self.initial = args['model']
            self.heldout_ids = {r['id'] for r in rows}
            self.heldout_prompts = {r['prompt'].strip() for r in rows}
            self.eval_ref = (str(self._path(args['dataset'])), file_hash(self._path(args['dataset'])))
        elif self.eval_ref != (str(self._path(args['dataset'])), file_hash(self._path(args['dataset']))):
            raise ValueError('Evaluation must use the unchanged held-out baseline dataset')
        completions = self._completions(args['model'], rows, 1, args['round'], False)
        results = [{'id': r['id'], 'completion': answers[0],
                    'correct': self.verifier.is_correct(answers[0], r['answer'])}
                   for r, answers in zip(rows, completions, strict=True)]
        artifact = self._write(self.root / f"round-{args['round']}" / 'evaluation.jsonl', results)
        return {'quality_metrics': {'accuracy': sum(r['correct'] for r in results) / len(results),
                                    'eval_samples': len(results)}, 'evidence': artifact,
                'eval_sha256': file_hash(self._path(args['dataset']))}

    def _select(self, args):
        model = args['model']
        if model != self.initial:
            manifest = Path(model) / 'rest-em.json'
            if not manifest.is_file() or not Path(model).resolve().is_relative_to(self.root.resolve()):
                raise ValueError('Selection must reference a model published by this Lab')
        self._write(self.root / 'selection.jsonl', [{'model': model}])
        return {'model': model}

    def _completions(self, ref, rows, count, round_number, sample, *, temperature=0.):
        return self.engine.generate(ref, rows, count, round_number, sample, temperature=temperature)
