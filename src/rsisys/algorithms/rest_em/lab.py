"""ReST^EM's Lab: a MathLab whose candidates are model checkpoints trained from its own samples.

Data, verifiers and scoring are MathLab's. The dataset manifest has a `train`
split (problems to sample from) and a `dev` split (measured after every round)
and selects each split's answer verifier; MathLab rejects problems that repeat
across splits. The engine generates and trains; evaluate() reuses MathLab's
solve/score/trace, and filtering reuses its verify().

The Controller owns phase ordering and model selection; it names a split in
`dataset` (or the split file's path). The Lab owns execution and artifact
provenance: generated and accepted samples are files under the run directory
that later stages must reference exactly as published.
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
    ExecutionStatus, Invariant,
    MathAttempt, MathLab, MetricSpec, ModelRef,
    OperationResult,
    OperationSpec,
    file_hash,
)


def validate_config(config):
    allowed = {'mode', 'device', 'epochs', 'learning_rate', 'max_new_tokens', 'max_length',
               'seed', 'model_kind', 'engine', 'output_subdir', 'manifest', 'grading_workers',
               'generation_batch_size', 'train_batch_size', 'gradient_accumulation_steps',
               'prompt_format', 'few_shot_data', 'answer_format',
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
    if not isinstance(config.get('manifest'), str) or not config['manifest'].strip():
        raise ValueError('manifest must be the path of a math dataset manifest with train and dev splits')
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
    if 'few_shot_data' in config:
        if config.get('prompt_format', 'chat') != 'completion':
            raise ValueError('few_shot_data requires completion prompt format')
        if not isinstance(config['few_shot_data'], str) or not config['few_shot_data'].strip():
            raise ValueError('few_shot_data must be a nonempty path')
    for key, default in (('epochs', 1), ('max_new_tokens', 512), ('max_length', 2048),
                         ('generation_batch_size', 1), ('train_batch_size', 1),
                         ('gradient_accumulation_steps', 1), ('max_num_seqs', 128),
                         ('max_num_batched_tokens', 16384), ('grading_workers', 1)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config.get('seed', 73)) is not int or config.get('seed', 73) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rate = config.get('learning_rate', 2e-4)
    if type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('learning_rate must be finite and positive')


class RestEMLab(MathLab):
    source_budgets = False  # Generation is set by this Lab, not by a benchmark.

    def __init__(self, context, config, *, engine=None, input_root=None):
        validate_config(config)
        self.config = dict(config)
        # Few-shot demonstrations are the engine's prompt context; relative paths start at the working directory.
        self.input_root = Path(input_root or Path.cwd()).resolve()
        selected = config.get('engine', config.get('mode', 'fixture'))
        from rsisys.algorithms.rest_em.engine import ParallelRestEMEngine
        self.engine = engine if engine is not None else (
            FixtureRestEMEngine() if selected == 'fixture' else
            (ParallelRestEMEngine if selected == 'vllm' else TransformersRestEMEngine)(config, input_root=self.input_root))
        bind = getattr(self.engine, 'bind_execution', None)
        if callable(bind):
            bind(context)
        if not isinstance(getattr(self.engine, 'name', None), str) or not self.engine.name.strip():
            raise TypeError('Math components must declare a nonempty name/version')
        if any(not callable(getattr(self.engine, name, None)) for name in ('generate', 'train')):
            raise TypeError('Invalid math component interface')
        if type(getattr(self.engine, 'simulated', None)) is not bool:
            raise TypeError('Math engine must declare simulated as a bool')
        self.simulated = self.engine.simulated
        MathLab.__init__(self, context, {
            'manifest': config['manifest'], 'split_roles': {'train': 'train', 'dev': 'dev'},
            'grading_workers': config.get('grading_workers', 1)})
        self.root = Path(context.data_dir) / config.get('output_subdir', 'math-lab')
        # A resumed run rebuilds the Lab over its own existing output directory.
        self.root.mkdir(parents=True, exist_ok=bool(getattr(context, 'restoring', False)))
        self.initial = None
        self.seen = set()
        self.artifact_hashes = {}

    def bind_actor(self, context, config):
        # The measured models are checkpoints the engine loads; the binding records the engine.
        return ModelRef('local-weights', self.engine.name, 'checkpoint')

    def state(self):
        return {'binding': self.binding(), 'initial': self.initial, 'seen': sorted(self.seen),
                'artifact_hashes': dict(self.artifact_hashes)}

    def restore(self, state):
        if state['binding'] != self.binding():
            raise ValueError('ReST^EM input binding changed since the checkpoint')
        self.initial = state['initial']
        self.seen = {tuple(key) for key in state['seen']}
        self.artifact_hashes = dict(state['artifact_hashes'])
        for ref, digest in self.artifact_hashes.items():
            if file_hash(Path(ref)) != digest:
                raise ValueError(f'Published ReST^EM artifact changed while the run was stopped: {ref}')

    def resource_roles(self):
        roles = getattr(self.engine, 'resource_roles', None)
        return roles() if callable(roles) else {}

    def profiles(self):
        return ('round-weight-training/v1',)

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', self.datasets['dev'][0].verifier.name)

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

    def evaluate_schema(self):
        return next(spec for spec in self.describe() if spec.name == 'evaluate').input_schema

    def _read(self, path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

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
            if call.name == 'evaluate':
                return self._evaluate(call, args)
            output = getattr(self, '_' + call.name)(args)
            return OperationResult(f'{call.name} completed', (), self.simulated,
                data={'request': args, 'training_performed': call.name == 'train' and not self.simulated, **output})
        except WorkerLostError:
            raise
        except Exception as exc:
            return OperationResult(f'{call.name} failed', (), self.simulated,
                ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}', data={'request': args})

    def _rows(self, problems):
        # The engine reads prompts; fixture engines also read the reference answer.
        return [{'id': problem.id, 'prompt': problem.prompt, 'answer': problem.answer} for problem in problems]

    def _checks(self, problems, completions):
        """One AnswerCheck per completion, graded by each problem's verifier."""
        return self.verify(problems, [MathAttempt(completion) for completion in completions])

    def _generate(self, args):
        if self.initial is None:
            raise ValueError('Evaluate the held-out baseline before generation')
        problems = self.datasets[self.split_named(args['dataset'], 'train')]
        rows = self._rows(problems)
        sample, temperature = args['temperature'] > 0, args['temperature']
        completions = self._completions(args['model'], rows, args['samples_per_prompt'], args['round'],
                                        sample, temperature=temperature)
        if args['condition'] == 'answer':
            flat = [(problem, answer) for problem, answers in zip(problems, completions, strict=True)
                    for answer in answers]
            checks = iter(self._checks([problem for problem, _ in flat], [answer for _, answer in flat]))
            misses = [i for i, answers in enumerate(completions)
                      if not any(next(checks).correct for _ in answers)]
            hinted = [{**rows[i], 'prompt': rows[i]['prompt'] + '\nKnown correct answer: ' + rows[i]['answer']}
                      for i in misses]
            if hinted:
                repaired = self._completions(args['model'], hinted, args['samples_per_prompt'], args['round'],
                                            sample, temperature=temperature)
                for i, answers in zip(misses, repaired, strict=True):
                    completions[i].extend(answers)
        output = [{'id': problem.id, 'prompt': problem.prompt, 'completion': answer}
                  for problem, answers in zip(problems, completions, strict=True) for answer in answers]
        return self._write(self.root / f"round-{args['round']}" / 'generated.jsonl', output)

    def _filter(self, args):
        rows = self._owned_data(args['samples'], args['round'], 'generated.jsonl')
        by_id = {problem.id: problem for problem in self.datasets['train']}
        problems = [by_id[row['id']] for row in rows]
        checks = self._checks(problems, [row['completion'] for row in rows])
        counts, kept = Counter(), []
        for row, check in zip(rows, checks, strict=True):
            if check.correct and counts[row['id']] < args['max_accepted_per_prompt']:
                # Reference answers are verifier input only; training sees the generated completion.
                kept.append(row)
                counts[row['id']] += 1
        result = self._write(self.root / f"round-{args['round']}" / 'accepted.jsonl', kept)
        return {**result, 'verifier': problems[0].verifier.name if problems else None,
                'source_sha256': file_hash(Path(args['samples']))}

    def _train(self, args):
        rows = self._owned_data(args['dataset'], args['round'], 'accepted.jsonl')
        if not rows:
            raise ValueError('Cannot train on an empty accepted dataset')
        output = self.root / f"round-{args['round']}" / 'model'
        output.mkdir(parents=True, exist_ok=False)
        self.engine.train(args['parent'], rows, output, args['round'], steps=args.get('steps'))
        manifest = {'parent': args['parent'], 'dataset': args['dataset'],
                    'dataset_sha256': file_hash(Path(args['dataset'])),
                    'round': args['round'], 'simulated': self.simulated,
                    'training_performed': not self.simulated, 'recipe': self.config,
                    'engine': self.engine.name, 'verifier': self.datasets['train'][0].verifier.name}
        (output / 'rest-em.json').write_text(json.dumps(manifest, indent=2))
        return {'model': str(output.resolve()), 'manifest': manifest}

    def _evaluate(self, call, args):
        if self.split_named(args['dataset'], 'dev') != args['split']:
            raise ValueError('ReST EM evaluates its configured development dataset')
        if args['round'] == 0:
            self.initial = args['model']
        return self.evaluate(call.call_id, args)

    def problems(self, arguments):
        return self.datasets[arguments['split']]

    def solve(self, problems, arguments):
        completions = self._completions(arguments['model'], self._rows(problems), 1, arguments['round'], False)
        return [MathAttempt(answers[0], simulated=self.simulated) for answers in completions]

    def result_data(self, records, ref, arguments):
        return {'request': arguments, 'model': arguments['model'], 'training_performed': False}

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
