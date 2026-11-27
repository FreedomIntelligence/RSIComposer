"""Shared receipt and split handling for host-run task benchmarks."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path

from rsicomposer.loops.contracts.benchmark import Benchmark, canonical_json, choose_primary_metric
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.data import DatasetBinding
from rsicomposer.loops.contracts.decision import OperationSpec
from rsicomposer.loops.contracts.domain_agent import allowed_trust
from rsicomposer.loops.contracts.execution import LabInfrastructureError, LabToolError
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.profiles import MetricSpec, SPLIT_ROLES


def digest(value):
    return sha256(canonical_json(value).encode()).hexdigest()


def checked_file(path, expected):
    contents = Path(path).read_bytes()
    if sha256(contents).hexdigest() != expected:
        raise ValueError(f'Benchmark file hash mismatch: {path}')
    return contents


def installed_callable(path):
    module, separator, name = path.partition(':')
    if not separator or not module or not name:
        raise ValueError('Runner must be an installed module:callable')
    fn = getattr(import_module(module), name)
    if not callable(fn):
        raise ValueError(f'Runner is not callable: {path}')
    return fn


class TaskEvaluationLab:
    """A domain supplies public inputs, a runtime, and authoritative metrics."""

    protocol_version = 'task-evaluation/v1'
    # name -> (primary metric, fixed scorer name, required benchmark options)
    adapters = {}

    @classmethod
    def validate_config(cls, config):
        if set(config) - {'benchmarks', 'primary_metric', 'split_roles', 'runner', 'runner_revision', 'trust'}:
            raise ValueError(f'Unknown {cls.domain} Lab settings')
        if 'host-code' not in allowed_trust(config):
            raise ValueError('The installed runner uses host code; set trust = ["host-code"] in the Lab config')
        for key in ('runner', 'runner_revision'):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f'{cls.domain} requires {key}')
        if ':' not in config['runner']:
            raise ValueError('Runner must be an installed module:callable')
        entries = config.get('benchmarks')
        if not isinstance(entries, dict) or not entries or set(entries) - cls.adapters.keys():
            raise ValueError(f'Select one or more benchmarks from {sorted(cls.adapters)}')
        for name, entry in entries.items():
            required = {'manifest', 'sha256'} | cls.adapters[name][2]
            if (not isinstance(entry, dict) or set(entry) != required or any(
                    not isinstance(value, str) or not value.strip() for value in entry.values())):
                raise ValueError(f'{name} benchmark requires {sorted(required)}')
        roles = config.get('split_roles')
        if (not isinstance(roles, dict) or not roles or any(
                not isinstance(role, str) or role not in SPLIT_ROLES for role in roles.values())):
            raise ValueError('Assign explicit split roles with split_roles')
        choose_primary_metric(config.get('primary_metric'), [Benchmark(n, cls.domain) for n in entries],
                              aggregates={})

    def __init__(self, context, config):
        self.validate_config(config)
        config = deepcopy(config)
        self.context = context
        self.runner = installed_callable(config['runner'])
        self.roles = dict(config['split_roles'])
        self.entries, self.datasets, self.benchmarks = {}, {}, {}
        bindings = []
        for name, entry in sorted(config['benchmarks'].items()):
            path = Path(entry['manifest']).resolve()
            content = json.loads(checked_file(path, entry['sha256']))
            if content.get('schema_version') != 1 or content.get('benchmark') != name:
                raise ValueError(f'Invalid {name} task manifest: {path}')
            splits = content['splits']
            if not isinstance(splits, dict) or not splits or set(splits) - self.roles.keys():
                raise ValueError(f'{name} requires nonempty configured splits')
            protocol = {'scorer': self.adapters[name][1], 'runtime': config['runner'],
                        'runtime_revision': config['runner_revision'], 'lab': self.protocol_version,
                        **{k: entry[k] for k in self.adapters[name][2]}}
            benchmark = self.benchmarks[name] = Benchmark(name, self.domain, protocol, {
                'manifest': str(path), 'sha256': entry['sha256']})
            self.entries[name] = entry
            seen_ids, seen_inputs = set(), set()
            for split, tasks in sorted(splits.items()):
                if not isinstance(tasks, list) or not tasks:
                    raise ValueError(f'{name}/{split} must list at least one task')
                for task in tasks:
                    public = task['input']
                    if (not isinstance(task['id'], str) or not task['id'].strip()
                            or not isinstance(public, dict) or not public
                            or not isinstance(task['reference'], dict)):
                        raise ValueError(f'Invalid task in {name}/{split}')
                    public_digest = digest(public)
                    if task['id'] in seen_ids or public_digest in seen_inputs:
                        raise ValueError(f'Duplicate task or task input in {name}: {task["id"]}')
                    seen_ids.add(task['id'])
                    seen_inputs.add(public_digest)
                    self.datasets.setdefault(split, []).append((name, task))
                bindings.append(DatasetBinding(
                    name=f'{name}/{split}', split=split, role=self.roles[split], source=str(path),
                    schema_version=self.protocol_version, content_digest=digest(tasks),
                    row_ids=tuple(task['id'] for task in tasks),
                    provenance={'benchmark': name, 'manifest_sha256': entry['sha256'],
                                'protocol_id': benchmark.protocol_id, 'source': content.get('source', {})}))
        if set(self.roles) != set(self.datasets):
            raise ValueError('Each split role must name an installed split')
        self.bindings = tuple(bindings)
        _, self.primary = choose_primary_metric(config.get('primary_metric'), self.benchmarks.values(), aggregates={})

    def split_roles(self):
        return {name: SPLIT_ROLES[role] for name, role in self.roles.items()}

    def data_bindings(self):
        return self.bindings

    def primary_metric(self):
        benchmark = self.benchmarks[self.primary]
        return MetricSpec(f'{self.primary}/{self.adapters[self.primary][0]}', 'maximize',
                          f'{self.adapters[self.primary][1]}/{benchmark.protocol_id}')

    def evaluation_operations(self):
        """`evaluate` is this Lab's measurement; see describe()."""
        return ('evaluate',)

    def state(self):
        return {'protocol': self.protocol_version, 'primary': self.primary,
                'bindings': [binding.identity for binding in self.bindings]}

    def restore(self, state):
        if state != self.state():
            raise ValueError(f'{self.domain} Lab bindings changed since the checkpoint')

    def describe(self):
        return (
            OperationSpec('inspect', f'Describe the {self.domain} benchmark splits.',
                          {'type': 'object', 'properties': {}, 'additionalProperties': False}, read_only=True),
            OperationSpec('evaluate', 'Evaluate an instruction with the configured agent runtime.',
                          {'type': 'object', 'properties': {
                              'split': {'type': 'string', 'enum': list(self.datasets)},
                              'instruction': {'type': 'string', 'minLength': 1},
                              'task_ids': {'type': 'array', 'items': {'type': 'string'},
                                           'minItems': 1, 'uniqueItems': True}},
                           'required': ['split', 'instruction'], 'additionalProperties': False}),
        )

    def execute(self, call):
        if call.name == 'inspect' and not call.arguments:
            return OperationResult('Benchmark environment', (), False, data={
                'benchmarks': {name: benchmark.identity for name, benchmark in self.benchmarks.items()},
                'splits': {split: [{'id': f'{name}/{task["id"]}', 'benchmark': name} for name, task in rows]
                           for split, rows in self.datasets.items() if self.split_roles()[split].inputs_visible},
            }, call_id=call.call_id)
        if call.name == 'evaluate' and {'split', 'instruction'} <= call.arguments.keys() <= {
                'split', 'instruction', 'task_ids'}:
            return self.evaluate(call.call_id, **call.arguments)
        raise LabToolError('Use the documented Lab operation arguments')

    def evaluate(self, call_id, split, instruction, task_ids=None):
        if not isinstance(split, str) or split not in self.datasets:
            raise LabToolError('Unknown evaluation split')
        if not isinstance(instruction, str) or not instruction.strip():
            raise LabToolError('An instruction is required')
        sealed = self.split_roles()[split].after_finish_only
        rows = self.datasets[split]
        if task_ids is not None:
            if sealed:
                raise LabToolError('The sealed evaluation must use the entire split')
            if (not isinstance(task_ids, list) or not task_ids
                    or any(not isinstance(ident, str) for ident in task_ids) or len(set(task_ids)) != len(task_ids)):
                raise LabToolError('task_ids must be distinct benchmark-qualified IDs')
            by_id = {f'{name}/{task["id"]}': (name, task) for name, task in rows}
            if set(task_ids) - by_id.keys():
                raise LabToolError('Unknown benchmark-qualified task ID')
            rows = [by_id[ident] for ident in sorted(task_ids)]
        records, totals = [], {}
        for name, task in rows:
            try:
                metrics, evidence = self.measure(name, task, instruction)
                if self.adapters[name][0] not in metrics or any(
                        type(value) not in (float, int) or not 0 <= value <= 1 for value in metrics.values()):
                    raise ValueError('A bounded official grade is required')
            except Exception as error:
                message = f'{name} trial or grader failed ({type(error).__name__})'
                if not sealed:
                    message += f' for {task["id"]}: {error}'
                raise LabInfrastructureError(OperationResult(
                    'Benchmark measurement incomplete', (), False, status=ExecutionStatus.FAILED,
                    error=message, call_id=call_id)) from error
            totals.setdefault(name, []).append(metrics)
            records.append({'benchmark': name, 'id': task['id'], 'metrics': metrics, 'evidence': evidence})
        quality = {}
        for name, grades in totals.items():
            keys = grades[0].keys()
            if any(grade.keys() != keys for grade in grades):
                raise ValueError(f'Inconsistent per-task metrics from {name}')
            quality.update({f'{name}/{key}': sum(grade[key] for grade in grades) / len(grades) for key in keys})
        ref = self.context.artifacts.put(canonical_json({'split': split, 'instruction': instruction,
            'binding': self.state(), 'records': records}).encode(),
            kind='sealed-evaluation' if sealed else f'{self.domain}-evaluation')
        return OperationResult('Benchmark evaluation completed', (ref,), False, data={
            'split': split, 'quality_metrics': quality, 'evidence_refs': [asdict(ref)],
            'execution_metrics': {'num_samples': len(records)},
            'benchmarks': {name: self.benchmarks[name].protocol_id for name in totals},
        }, call_id=call_id)
