"""Synchronous numeric-math Lab assembled from an engine and answer verifier.

The controller owns phase ordering and model selection. The Lab owns execution,
artifact provenance and held-out data checks. Components can be supplied by a
registered Lab factory without adding top-level registration object types.
"""
from collections import Counter
from hashlib import sha256
import json
import math
import re
from pathlib import Path

from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.labs.math.engines import FixtureMathEngine, TransformersMathEngine
from rsisys.labs.math.verifiers import StrictNumericVerifier, MathVerifyVerifier


def validate_config(config):
    allowed = {'mode', 'device', 'epochs', 'learning_rate', 'max_new_tokens', 'max_length',
               'seed', 'model_kind', 'engine', 'data_root', 'output_subdir',
               'generation_batch_size', 'train_batch_size', 'gradient_accumulation_steps',
               'prompt_format', 'few_shot_data', 'answer_format', 'verifier',
               'gpu_ids', 'inference_python', 'training_python', 'gpu_memory_utilization',
               'max_num_seqs', 'max_num_batched_tokens', 'worker_timeout', 'train_gradient_checkpointing'}
    if set(config) - allowed or config.get('engine', config.get('mode', 'fixture')) not in ('fixture', 'transformers', 'vllm'):
        raise ValueError('Unknown math Lab configuration or engine')
    if 'gpu_ids' in config:
        ids = config['gpu_ids']
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('gpu_ids must be distinct nonnegative GPU indices')
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
    if config.get('verifier', 'numeric') not in ('numeric', 'math_verify'):
        raise ValueError('verifier must be numeric or math_verify')
    if 'few_shot_data' in config:
        if config.get('prompt_format', 'chat') != 'completion':
            raise ValueError('few_shot_data requires completion prompt format')
        if not isinstance(config['few_shot_data'], str) or not config['few_shot_data'].strip():
            raise ValueError('few_shot_data must be a nonempty path')
    for key, default in (('epochs', 1), ('max_new_tokens', 512), ('max_length', 2048),
                         ('generation_batch_size', 1), ('train_batch_size', 1),
                         ('gradient_accumulation_steps', 1), ('max_num_seqs', 128),
                         ('max_num_batched_tokens', 16384), ('worker_timeout', 3600)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config.get('seed', 73)) is not int or config.get('seed', 73) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rate = config.get('learning_rate', 2e-4)
    if type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('learning_rate must be finite and positive')


class MathLab:
    def __init__(self, context, config, *, engine=None, verifier=None, input_root=None):
        validate_config(config)
        self.config = dict(config)
        self.input_root = Path(input_root or config.get('data_root', Path.cwd())).resolve()
        selected = config.get('engine', config.get('mode', 'fixture'))
        from rsisys.labs.math.parallel import ParallelMathEngine
        self.engine = engine if engine is not None else (
            FixtureMathEngine() if selected == 'fixture' else
            (ParallelMathEngine if selected == 'vllm' else TransformersMathEngine)(config, input_root=self.input_root))
        self.verifier = verifier if verifier is not None else (
            MathVerifyVerifier() if config.get('verifier') == 'math_verify' else StrictNumericVerifier())
        for component, methods in ((self.engine, ('generate', 'train')), (self.verifier, ('parse', 'is_correct'))):
            if not isinstance(getattr(component, 'name', None), str) or not component.name.strip():
                raise TypeError('Math components must declare a nonempty name/version')
            if any(not callable(getattr(component, name, None)) for name in methods):
                raise TypeError('Invalid math component interface')
        if type(getattr(self.engine, 'simulated', None)) is not bool:
            raise TypeError('Math engine must declare simulated as a bool')
        self.simulated = self.engine.simulated
        self.root = Path(context.data_dir) / config.get('output_subdir', 'math-lab')
        self.root.mkdir(parents=True, exist_ok=False)
        self.initial = None
        self.seen = set()
        self.heldout_ids = self.heldout_prompts = None
        self.artifact_hashes = {}
        self.eval_ref = None

    def describe(self):
        fields = {
            'generate': {'model': 'string', 'dataset': 'string', 'samples_per_prompt': 'integer'},
            'filter': {'samples': 'string', 'max_accepted_per_prompt': 'integer'},
            'train': {'initial_model': 'string', 'dataset': 'string'},
            'evaluate': {'model': 'string', 'dataset': 'string'},
        }
        return tuple(OperationSpec(name, f'ReST^EM {name}; returns a terminal receipt', {
            'type': 'object', 'properties': {'round': {'type': 'integer', 'minimum': 0},
                **{k: {'type': v} for k, v in props.items()}},
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
                or self.verifier.parse(row['answer']) is None for row in rows):
            raise ValueError('Expected nonempty JSONL: unique id, prompt and answer parseable by the selected verifier')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate problem IDs')
        return rows

    def _write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        ref, digest = str(path.resolve()), sha256(path.read_bytes()).hexdigest()
        self.artifact_hashes[ref] = digest
        return {'dataset': ref, 'count': len(rows), 'sha256': digest}

    def _owned_data(self, ref, round_number, filename):
        expected = str((self.root / f'round-{round_number}' / filename).resolve())
        if ref != expected or ref not in self.artifact_hashes:
            raise ValueError('Expected the artifact produced by the preceding stage of this round')
        if sha256(Path(ref).read_bytes()).hexdigest() != self.artifact_hashes[ref]:
            raise ValueError('Produced dataset changed after publication')
        return self._read(ref)

    def execute(self, call):
        specs = {s.name: s for s in self.describe()}
        if call.name not in specs:
            raise ValueError('Unknown math operation')
        args = call.arguments
        props = specs[call.name].input_schema['properties']
        if set(args) != set(props):
            raise ValueError('Unexpected or missing math operation arguments')
        for key, spec in props.items():
            value = args[key]
            if spec['type'] == 'string' and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{key} must be nonempty text')
            if spec['type'] == 'integer' and (type(value) is not int or value < (0 if key == 'round' else 1)):
                raise ValueError(f'{key} is out of range')
        key = (args['round'], call.name)
        if key in self.seen:
            raise ValueError('Repeated stage rejected; this example does not support replay/resume')
        self.seen.add(key)
        try:
            output = getattr(self, '_' + call.name)(args)
            return OperationResult(f'{call.name} completed', (), self.simulated,
                data={'request': args, 'training_performed': call.name == 'train' and not self.simulated, **output})
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
        completions = self._completions(args['model'], rows, args['samples_per_prompt'], args['round'], True)
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
        return {**result, 'verifier': self.verifier.name, 'source_sha256': sha256(self._path(args['samples']).read_bytes()).hexdigest()}

    def _train(self, args):
        rows = self._owned_data(args['dataset'], args['round'], 'accepted.jsonl')
        if not rows:
            raise ValueError('Cannot train on an empty accepted dataset')
        output = self.root / f"round-{args['round']}" / 'model'
        output.mkdir(parents=True, exist_ok=False)
        self.engine.train(args['initial_model'], rows, output, args['round'])
        manifest = {'initial_model': args['initial_model'], 'dataset': args['dataset'],
                    'dataset_sha256': sha256(self._path(args['dataset']).read_bytes()).hexdigest(),
                    'round': args['round'], 'simulated': self.simulated,
                    'training_performed': not self.simulated, 'recipe': self.config,
                    'engine': self.engine.name, 'verifier': self.verifier.name}
        (output / 'rest-em.json').write_text(json.dumps(manifest, indent=2))
        return {'model': str(output.resolve()), 'manifest': manifest}

    def _evaluate(self, args):
        rows = self._problems(args['dataset'])
        if args['round'] == 0:
            self.initial = args['model']
            self.heldout_ids = {r['id'] for r in rows}
            self.heldout_prompts = {r['prompt'].strip() for r in rows}
            self.eval_ref = (str(self._path(args['dataset'])), sha256(self._path(args['dataset']).read_bytes()).hexdigest())
        elif self.eval_ref != (str(self._path(args['dataset'])), sha256(self._path(args['dataset']).read_bytes()).hexdigest()):
            raise ValueError('Evaluation must use the unchanged held-out baseline dataset')
        completions = self._completions(args['model'], rows, 1, args['round'], False)
        results = [{'id': r['id'], 'completion': answers[0],
                    'correct': self.verifier.is_correct(answers[0], r['answer'])}
                   for r, answers in zip(rows, completions, strict=True)]
        artifact = self._write(self.root / f"round-{args['round']}" / 'evaluation.jsonl', results)
        return {'quality_metrics': {'accuracy': sum(r['correct'] for r in results) / len(results),
                                    'eval_samples': len(results)}, 'evidence': artifact,
                'eval_sha256': sha256(self._path(args['dataset']).read_bytes()).hexdigest()}

    def _completions(self, ref, rows, count, round_number, sample):
        return self.engine.generate(ref, rows, count, round_number, sample)
