"""STaR's Lab: a MathLab whose candidates are checkpoints fitted on their own correct rationales.

The dataset manifest has `train` (problems sampled from), `dev` (measured every
round) and sealed `test` splits, each selecting its answer verifier. Sampling and
fitting run in GPU workers (or the fixture engine); evaluation reuses MathLab's
solve/score/trace, and filtering reuses each problem's verifier.
"""

from collections import Counter
import json
import math
from pathlib import Path
import sys
from time import sleep

from rsisys.api import (
    Invariant, MathAttempt, MathLab, OperationResult, MetricSpec, ModelRef, OperationCall, OperationSpec, atomic_write, file_hash, json_bytes, report_training_log, ResourceRequest, WorkerSpec, execution_capabilities, worker_environment,
)

from .protocol import MAX_LENGTH, keep_reasoning, render_prompt, write_records
from .records import GLOBAL_BATCH, training_example


def atomic_json(path, value):
    atomic_write(path, json_bytes(value))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def filter_overlength(source, destination, base_model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    kept, dropped = [], 0
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        try:
            training_example(tokenizer, render_prompt(tokenizer, row["prompt"]),
                             row["completion"], max_tokens=MAX_LENGTH)
        except ValueError as exc:
            if str(exc) != f"Prompt plus completion exceeds {MAX_LENGTH} tokens":
                raise
            dropped += 1
        else:
            kept.append(row)
    if len(kept) < GLOBAL_BATCH:
        raise RuntimeError(f"STaR has only {len(kept)} eligible rationales after length filtering")
    write_records(destination, kept)
    return dropped


class StarLab(MathLab):
    source_budgets = False  # Generation is set by this Lab, not by a benchmark.

    def __init__(self, context, config, *, engine=None):
        if 'max_seconds' in config:
            raise ValueError('Configure host budget.seconds instead of STaR max_seconds')
        self.resources, self.executor = execution_capabilities(context)
        self.root = Path(context.data_dir)
        from .engine import FixtureStarEngine
        self.engine = engine if engine is not None else (FixtureStarEngine() if config.get('engine') == 'fixture' else None)
        self.simulated = self.engine is not None and self.engine.simulated
        MathLab.__init__(self, context, {
            'manifest': config['manifest'], 'split_roles': {'train': 'train', 'dev': 'dev', 'test': 'sealed'}})
        self.resources.declare(self.resource_roles())
        self.config = dict(config)
        self.selected = None
        self.default_steps = config.get('train_steps', 40)
        self.seed = config['seed']
        self.warmup_steps = config.get('warmup_steps', 100)
        self.base = Path(config['base_model']).resolve(strict=True)
        self.models, self.datasets_by_round, self.dataset_hashes = {'base': self.base}, {}, {}
        self.dev_file = self.root / 'development.jsonl'
        # A resumed run rebuilds the Lab over its own workspace; the cohorts and
        # development file it wrote before must match the configured data.
        restoring = bool(getattr(context, 'restoring', False))
        cohorts = {split: [row.id for row in rows] for split, rows in self.datasets.items()}
        cohort_file = self.root / 'cohorts.json'
        if restoring and cohort_file.exists() and read_json(cohort_file) != cohorts:
            raise ValueError('STaR data cohorts differ from the resumed run')
        atomic_json(cohort_file, cohorts)
        development = [{"id": row.id, "prompt": row.prompt, "answer": "\\boxed{" + row.expected + "}"}
                       for row in self.datasets["dev"]]
        if restoring and self.dev_file.exists():
            written = [json.loads(line) for line in self.dev_file.read_text(encoding='utf-8').splitlines()]
            if written != development:
                raise ValueError('STaR development file differs from the resumed run')
        else:
            write_records(self.dev_file, development)

    def bind_actor(self, context, config):
        # The measured models are checkpoints the workers load; the binding records the engine.
        return ModelRef('local-weights', getattr(self.engine, 'name', 'star-workers/v1'), 'checkpoint')

    def state(self):
        # Workers write checkpoint directories; the host persists the run budget.
        return {'binding': self.binding(), 'models': {name: str(path) for name, path in self.models.items()},
                'datasets': {number: str(path) for number, path in self.datasets_by_round.items()},
                'dataset_hashes': dict(self.dataset_hashes), 'selected': self.selected}

    def restore(self, state):
        if set(state) != {'binding', 'models', 'datasets', 'dataset_hashes', 'selected'}:
            raise ValueError('Unexpected STaR Lab state')
        if state['binding'] != self.binding():
            raise ValueError('STaR input binding changed since the checkpoint')
        if state['models'].get('base') != str(self.base):
            raise ValueError('STaR base model differs from the resumed run')
        if set(state['datasets']) != set(state['dataset_hashes']):
            raise ValueError('STaR dataset hashes do not cover every collected round')
        for number, ref in state['datasets'].items():
            path = Path(ref)
            if not path.is_file() or file_hash(path) != state['dataset_hashes'][number]:
                raise ValueError(f'STaR round {number} rationales changed while the run was stopped: {ref}')
        for name, ref in state['models'].items():
            if name != 'base' and not (Path(ref) / 'model.safetensors.index.json').is_file():
                raise ValueError(f'STaR checkpoint {name} is missing: {ref}')
        self.models = {name: Path(ref) for name, ref in state['models'].items()}
        self.datasets_by_round = {number: Path(ref) for number, ref in state['datasets'].items()}
        self.dataset_hashes = dict(state['dataset_hashes'])
        self.selected = state['selected']

    def profiles(self):
        return ('round-weight-training/v1',)

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', self.datasets['dev'][0].verifier.name)

    def invariants(self):
        return (Invariant.once_per('round', operation='generate'),
                Invariant.once_per('round', operation='train'))

    def final_evaluations(self):
        if self.selected is None:
            return ()
        return tuple(OperationCall('evaluate', {'round': 0, 'split': 'test', 'model': model,
                     'dataset': 'test', 'baseline': baseline})
                     for model, baseline in (('base', True), (self.selected, False)))

    def describe(self):
        fields = {
            'generate': {'model': 'string', 'dataset': 'string', 'samples_per_prompt': 'integer',
                         'temperature': 'number', 'condition': 'string'},
            'filter': {'samples': 'string', 'max_accepted_per_prompt': 'integer'},
            'train': {'parent': 'string', 'dataset': 'string'},
            'evaluate': {'model': 'string', 'split': 'string', 'dataset': 'string'},
            'select': {'model': 'string'},
        }
        return tuple(OperationSpec(name, f'STaR {name}; explicit artifact lineage', {
            'type': 'object', 'properties': {'round': {'type': 'integer', 'minimum': 0},
                **{key: {'type': kind} for key, kind in props.items()},
                **({'steps': {'type': 'integer', 'minimum': 1}} if name == 'train' else {}),
                **({'baseline': {'type': 'boolean'}} if name == 'evaluate' else {})},
            'required': ['round', *props], 'additionalProperties': False}) for name, props in fields.items())

    def execute(self, call):
        specs = {spec.name: spec.input_schema for spec in self.describe()}
        if call.name not in specs:
            raise ValueError('Unknown STaR operation')
        schema, request = specs[call.name], call.arguments
        if not set(schema['required']) <= set(request) <= set(schema['properties']):
            raise ValueError('Missing or unexpected STaR operation arguments')
        for key, value in request.items():
            kind = schema['properties'][key]['type']
            if kind == 'integer' and (type(value) is not int or value < (0 if key == 'round' else 1)):
                raise ValueError(f'{key} must be a valid integer')
            if kind == 'string' and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{key} must be nonempty text')
            if kind == 'number' and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f'{key} must be finite and nonnegative')
            if kind == 'boolean' and type(value) is not bool:
                raise ValueError(f'{key} must be bool')
        self._call_id = call.call_id
        if call.name == 'evaluate':
            return self._evaluate(call.call_id, request)
        summary, data = getattr(self, '_' + call.name)(request)
        return OperationResult(summary, (), self.simulated, data={'request': request, **data})

    def _model(self, ref):
        if ref in self.models:
            return ref
        path = Path(ref).resolve()
        matches = [key for key, value in self.models.items() if path == value]
        if len(matches) != 1:
            raise ValueError('Model must be the configured base or a committed checkpoint')
        return matches[0]

    def _items(self, problems):
        # Sampling workers read the question; the fixture engine and hinted prompts read the reference answer.
        return [{'id': row.id, 'question': row.prompt, 'answer': row.answer} for row in problems]

    def resource_roles(self):
        count = 0 if self.engine is not None else 4
        return {'rollout': ResourceRequest(gpus=count), 'trainer': ResourceRequest(gpus=count)}

    def workers(self, commands, lease):
        handles = []
        try:
            for command, env, log in commands:
                with log.open('x') as stream:
                    handles.append(self.executor.launch(WorkerSpec(command, env,
                        cwd=self.root, stdout=stream, devices=tuple(int(v) for v in
                            env.get('CUDA_VISIBLE_DEVICES', '').split(',') if v)), lease))
            while any(handle.poll() is None for handle in handles):
                for _, _, log in commands:
                    if log.name == 'worker.log':
                        report_training_log(log)
                if any(handle.poll() not in (None, 0) for handle in handles):
                    raise RuntimeError('STaR worker failed; inspect operation logs')
                sleep(.05)
            for _, _, log in commands:
                if log.name == 'worker.log':
                    report_training_log(log)
            if any(handle.returncode for handle in handles):
                raise RuntimeError('STaR worker failed; inspect operation logs')
        finally:
            for handle in handles:
                handle.terminate()

    def _sample(self, items, model, directory, *, count=1, temperature=0., hinted=False):
        directory.mkdir()
        if self.engine is not None:
            return self.engine.generate(items, self.models[model], count=count,
                                        temperature=temperature, hinted=hinted)
        with self.resources.acquire('rollout') as lease:
            expanded = [(item, sample) for item in items for sample in range(count)]
            commands, receipts = [], []
            for rank, device in enumerate(lease.devices):
                shard = [(index, item) for index, (item, _) in enumerate(expanded)
                         if index % len(lease.devices) == rank]
                if not shard:
                    continue
                prefix = directory / f'gpu-{rank}'
                input_path, output = prefix.with_suffix('.input.json'), prefix.with_suffix('.result.json')
                payload = [{'id': str(index), 'question': item['question'],
                            **({'hint': item['answer']} if hinted else {})} for index, item in shard]
                env = worker_environment(CUDA_VISIBLE_DEVICES=str(device))
                commands.append(([sys.executable, '-B', '-m', 'rsisys.algorithms.star.sample',
                                  str(self.models[model]), str(input_path), str(output),
                                  '--temperature', str(temperature)], env, prefix.with_suffix('.log')))
                atomic_json(input_path, payload)
                receipts.append((output, [row['id'] for row in payload]))
            self.workers(commands, lease)
            generated = {}
            for path, ids in receipts:
                responses = read_json(path)
                if [row['id'] for row in responses] != ids:
                    raise RuntimeError(f'Unexpected STaR generation identities in {path}')
                generated.update((row['id'], row['text']) for row in responses)
            return {item['id']: [generated[str(index * count + sample)] for sample in range(count)]
                    for index, item in enumerate(items)}

    def _evaluate(self, call_id, request):
        split = request['split']
        self._model(request['model'])
        if split not in ('dev', 'test') or self.split_named(request['dataset'], 'dev' if split == 'dev' else 'sealed') != split:
            raise ValueError('Evaluation must use a fixed held-out split')
        return self.evaluate(call_id, request)

    def problems(self, arguments):
        return self.datasets[arguments['split']]

    def solve(self, problems, arguments):
        directory = self.root / f"evaluation-{self._call_id}"
        outputs = self._sample(self._items(problems), self._model(arguments['model']), directory)
        return [MathAttempt(outputs[row.id][0], simulated=self.simulated) for row in problems]

    def result_data(self, records, ref, arguments):
        return {'protocol': f'GSM8K STaR {self._call_id}', 'request': arguments,
                'model': self._model(arguments['model'])}

    def _generate(self, request):
        number, model = request['round'], self._model(request['model'])
        if request['condition'] not in ('none', 'answer'):
            raise ValueError('condition must be none or answer')
        rationalize = request['condition'] == 'answer'
        problems = self.datasets[self.split_named(request['dataset'], 'train')]
        items = self._items(problems)
        directory = self.root / f'round-{number}'
        directory.mkdir()
        options = {'count': request['samples_per_prompt'], 'temperature': request['temperature']}
        outputs = self._sample(items, model, directory / 'plain', **options)
        accepted, misses = [], []
        for problem, item in zip(problems, items, strict=True):
            kept = [row for text in outputs[item['id']] if (row := keep_reasoning(problem, text)) is not None]
            if not kept:
                misses.append((problem, item))
            accepted.extend({**row, 'origin': 'plain'} for row in kept)
        plain_count = len(accepted)
        if rationalize and misses:
            hinted = self._sample([item for _, item in misses], model, directory / 'rationalized',
                                  hinted=True, **options)
            for problem, item in misses:
                for text in hinted[item['id']]:
                    row = keep_reasoning(problem, text)
                    if row is not None:
                        accepted.append({**row, 'origin': 'rationalized'})
        path = directory / 'generated.jsonl'
        write_records(path, accepted)
        metrics = {'generated_plain': len(items) * options['count'], 'accepted_plain': plain_count,
                   'generated_rationalized': len(misses) * options['count'] if rationalize else 0,
                   'accepted_rationalized': len(accepted) - plain_count, 'accepted_total': len(accepted)}
        atomic_json(directory / 'collection.json', {'model': model, **metrics})
        self.datasets_by_round[number] = path
        self.dataset_hashes[number] = file_hash(path)
        return f'Round {number}: accepted {len(accepted)} rationales', {
            'dataset': str(path), 'count': len(accepted), 'collection': metrics}

    def _owned_dataset(self, number, ref):
        dataset = self.datasets_by_round[number]
        if ref != str(dataset) or file_hash(dataset) != self.dataset_hashes[number]:
            raise ValueError('Dataset differs from the committed round artifact')
        return dataset

    def _filter(self, request):
        number = request['round']
        source = self._owned_dataset(number, request['samples'])
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        kept, counts = [], Counter()
        for row in rows:
            if counts[row['id']] < request['max_accepted_per_prompt']:
                kept.append(row)
                counts[row['id']] += 1
        destination = source.parent / 'accepted.jsonl'
        write_records(destination, kept)
        self.datasets_by_round[number] = destination
        self.dataset_hashes[number] = file_hash(destination)
        return f'Filtered {len(kept)} accepted rationales', {'dataset': str(destination), 'count': len(kept)}

    def _train(self, request):
        number, steps = request['round'], request.get('steps', self.default_steps)
        dataset = self._owned_dataset(number, request['dataset'])
        parent = self.models[self._model(request['parent'])]
        directory = dataset.parent / "fit"
        directory.mkdir()
        filtered = directory / "supervised.jsonl"
        if self.engine is not None:
            checkpoint = directory / 'checkpoint'
            metrics = self.engine.train(parent, dataset, checkpoint, steps=steps)
            model = f'round-{number}'
            self.models[model] = checkpoint
            return f'Fixture round {number}: simulated training', {
                'model': model, 'checkpoint': str(checkpoint), 'training_metrics': metrics,
                'training_performed': False}
        with self.resources.acquire('trainer') as lease:
            dropped = filter_overlength(dataset, filtered, parent)
            command = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node", str(len(lease.devices)), "--module", "rsisys.algorithms.star.fit",
                       "--base-model", str(parent), "--train-jsonl", str(filtered),
                       "--validation-jsonl", str(self.dev_file), "--output-dir", str(directory),
                       "--seed", str(self.seed + number), "--max-steps", str(steps),
                       "--dropped-overlength", str(dropped), "--warmup-steps", str(self.warmup_steps)]
            self.workers([(command, worker_environment(CUDA_VISIBLE_DEVICES=','.join(map(str, lease.devices))), directory / "worker.log")], lease)
            receipt = read_json(directory / "training_result.json")
            checkpoint = directory / "checkpoint"
            if (receipt["optimizer_steps"] != steps or receipt["text_tensors_changed"] < 1
                    or receipt["base_model"] != str(parent)
                    or receipt["checkpoint"] != str(checkpoint)
                    or not (checkpoint / "model.safetensors.index.json").is_file()):
                raise RuntimeError("STaR did not produce the expected updated checkpoint")
            model = f"round-{number}"
            self.models[model] = checkpoint
            metrics = {key: receipt[key] for key in (
                "optimizer_steps", "text_tensors_changed", "completion_loss_first",
                "completion_loss_last", "train_rows", "dropped_overlength")}
            return f"第 {number} 轮完成 {steps} 次更新", {
                "model": model, "checkpoint": str(checkpoint), "training_metrics": metrics}

    def _select(self, request):
        self.selected = self._model(request['model'])
        atomic_json(self.root / 'selection.json', {'model': self.selected})
        return 'Recorded selected model for host final evaluation', {'model': request['model']}
