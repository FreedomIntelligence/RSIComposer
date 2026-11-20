"""Reusable coding task environment, with no RSI search or adoption policy.

Methods submit pinned artifacts to a host-installed agent adapter. This Lab owns
tasks, sandbox trials, grading, evidence, budgets and resumable measurement.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import time

from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.data import DatasetBinding
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.profiles import MetricSpec, SPLIT_ROLES
from rsisys.loops.contracts.execution import LabToolError
from .agents import RUNTIMES
from .artifacts import digest, snapshot
from rsisys.loops.contracts.benchmark import choose_primary_metric
from rsisys.loops.contracts.domain_agent import (
    DomainAgentRejected, DomainAgent, allowed_trust, read_domain_agent_data, require_trust)
from .benchmarks import load_task_manifests, validate_benchmark_config
from .harbor_runner import atomic_json
from .trial_runner import HarborTrialRunner, infrastructure_failure


OPTIONS = {'experiment_dir', 'benchmarks', 'primary_metric', 'python', 'endpoints', 'model', 'workers',
           'seed', 'split_roles', 'trust'}
PROTOCOL_VERSION = 'coding-environment/v9'
# Reported only as named; there is no implicit cross-benchmark total.
AGGREGATES = {'benchmark-mean': 'harbor-official-reward/benchmark-mean/v1'}


def validate_config(config):
    if set(config) - OPTIONS:
        raise ValueError('Unknown Coding Lab options: ' + str(set(config) - OPTIONS))
    for key in ('experiment_dir', 'python', 'model'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('Coding Lab requires ' + key)
    benchmarks = validate_benchmark_config(config)
    allowed = allowed_trust(config)
    for benchmark in benchmarks:
        runtime = RUNTIMES[benchmark.protocol['runtime']['name']](benchmark.protocol['runtime']['options'])
        require_trust(runtime.accepts, allowed, where=f'Benchmark {benchmark.name!r} runtime {runtime.name!r}')
    choose_primary_metric(config.get('primary_metric'), benchmarks, aggregates=AGGREGATES)
    endpoints = config.get('endpoints')
    if not isinstance(endpoints, list) or not endpoints or any(
            not isinstance(e, str) or not e.startswith(('http://', 'https://')) for e in endpoints):
        raise ValueError('Coding Lab requires explicit inference endpoints')
    if 'workers' in config and (type(config['workers']) is not int or config['workers'] < 1):
        raise ValueError('Positive integer required: workers')
    if 'seed' in config and (type(config['seed']) is not int or not 0 <= config['seed'] < 2**32):
        raise ValueError('seed must be a nonnegative 32-bit integer')


class CodingLab:
    """Task-scene Lab reusable by prompt, code and agent-configuration methods.

    An installed adapter controls how the immutable artifact is passed to the
    agent. The Lab never proposes, mutates, ranks, selects or adopts candidates.
    Split roles are declared per instance and enforced by the host.
    """
    validate_config = staticmethod(validate_config)

    def __init__(self, context, config):
        validate_config(config)
        self.config = deepcopy(config)
        self.root = Path(config['experiment_dir']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        execution_context = context if context is not None else SimpleNamespace(data_dir=self.root)
        self.manifest, self.datasets, self.benchmarks = load_task_manifests(self.config)
        self.protocols = {benchmark.name: benchmark for benchmark in self.benchmarks}
        self.validate_tasks()
        roles = self.config.get('split_roles', {})
        if (not isinstance(roles, dict) or set(roles) - set(self.datasets)
                or any(not isinstance(role, str) or role not in SPLIT_ROLES for role in roles.values())):
            raise ValueError('split_roles must map installed splits to named profile roles')
        # One runtime per benchmark: the agent a benchmark's tasks run under is part of its protocol.
        self.runtimes = {b.name: RUNTIMES[b.protocol['runtime']['name']](b.protocol['runtime']['options'])
                         for b in self.benchmarks}
        self.agent = {name: runtime.to_json() for name, runtime in self.runtimes.items()}
        self._runner = HarborTrialRunner(execution_context, self.config, self.root / 'agents')
        package = Path(__file__).parent
        self.protocol = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                         for name in ('agent.py', 'agents.py', 'artifacts.py', 'benchmarks.py', 'lab.py',
                                      'harbor_runner.py', 'history.py', 'trial_runner.py')}
        self.binding = {'protocol_version': PROTOCOL_VERSION, 'manifest': self.manifest,
                        'config': self.config, 'agent': self.agent, 'source': self.protocol}
        self.fingerprint = digest(self.binding)
        binding = self.root / 'binding.json'
        self.evaluations = self.root / 'evaluations'
        self.artifacts = self.root / 'artifacts'
        self.measurements = self.root / 'measurements'
        with self._lock():
            if binding.exists() and json.loads(binding.read_text()) != self.binding:
                raise ValueError('Experiment binding changed; use a new experiment_dir')
            atomic_json(binding, self.binding)
            (self.root / 'agents').mkdir(exist_ok=True)
            for name, spec in self.agent.items():
                atomic_json(self.root / 'agents' / (name + '.json'), spec)
            self.evaluations.mkdir(exist_ok=True)
            self.artifacts.mkdir(exist_ok=True)
            self.measurements.mkdir(exist_ok=True)

    def validate_tasks(self):
        if not isinstance(self.datasets, dict) or not self.datasets:
            raise ValueError('Nonempty named task splits required')
        seen, path_splits = set(), {}
        for split, rows in self.datasets.items():
            if not isinstance(split, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', split):
                raise ValueError('Invalid task split name')
            if not isinstance(rows, list) or not rows:
                raise ValueError('Nonempty task lists required')
            for row in rows:
                if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                        for key in ('benchmark', 'id', 'path')) or not isinstance(row.get('files'), dict):
                    raise ValueError('Invalid task manifest row: expected benchmark, id, path and file hashes')
                key = (row['benchmark'], row['id'])
                if key in seen:
                    raise ValueError('Task overlap or duplicate: ' + str(key))
                seen.add(key)
                path = Path(row['path'])
                if path.is_symlink():
                    raise ValueError('Task directories must not be symlinks: ' + str(path))
                if not path.is_absolute() or not (path / 'task.toml').is_file() or not (path / 'instruction.md').is_file():
                    raise ValueError('Missing real benchmark task: ' + str(path))
                if path_splits.setdefault(path.resolve(), split) != split:
                    raise ValueError('Task directory overlaps splits: ' + str(path))
                if not {'task.toml', 'instruction.md'} <= row['files'].keys():
                    raise ValueError('Task manifest must hash task.toml and instruction.md')
                declared = set(row['files'])
                for relative in declared:
                    entry = Path(relative)
                    if not relative or entry.is_absolute() or '..' in entry.parts or entry.as_posix() != relative:
                        raise ValueError('Task manifest contains an invalid file path: ' + repr(relative))
                actual = set()
                for entry in sorted(path.rglob('*')):
                    relative = entry.relative_to(path).as_posix()
                    if entry.is_symlink():
                        raise ValueError('Task directories must not contain symlinks: ' + relative)
                    if entry.is_file():
                        actual.add(relative)
                    elif not entry.is_dir():
                        raise ValueError('Unsupported task entry: ' + relative)
                if actual != declared:
                    missing = sorted(actual - declared)
                    extra = sorted(declared - actual)
                    details = []
                    if missing:
                        details.append('unlisted task files: ' + ', '.join(missing))
                    if extra:
                        details.append('missing task files: ' + ', '.join(extra))
                    raise ValueError('Task manifest must hash exactly all task files; ' + '; '.join(details))
                for relative, sha in row['files'].items():
                    entry = path / relative
                    if hashlib.sha256(entry.read_bytes()).hexdigest() != sha:
                        raise ValueError('Frozen task changed: ' + str(entry))

    @contextmanager
    def _lock(self):
        with (self.root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def state(self):
        # Evaluations, artifacts and measurements are resumable files in experiment_dir.
        return {'fingerprint': self.fingerprint}

    def restore(self, state):
        if state['fingerprint'] != self.fingerprint:
            raise ValueError('Coding task environment binding changed since the checkpoint')

    def profiles(self):
        return ('coding-task/v1',)

    def primary_metric(self):
        kind, value = choose_primary_metric(self.config.get('primary_metric'), self.benchmarks, aggregates=AGGREGATES)
        if kind == 'benchmark':
            return MetricSpec(self.metric_name(value), 'maximize',
                              f'harbor-official-reward/task-mean/{self.protocols[value].protocol_id}')
        return MetricSpec('benchmark_reward', 'maximize', AGGREGATES[value])

    @staticmethod
    def metric_name(benchmark):
        return f'{benchmark}/task_reward'

    def split_roles(self):
        return {name: SPLIT_ROLES[self._split_role_name(name)] for name in self.datasets}

    def _split_role_name(self, split):
        return self.config.get('split_roles', {}).get(
            split, 'sealed' if split in ('test', 'heldout', 'sealed') else 'dev')

    def data_bindings(self):
        """Publish stable references for every configured benchmark split.

        The binding describes the frozen cohort and its access role. Task file
        formats and benchmark-specific records remain owned by CodingLab.
        """
        bindings = []
        for split, rows in sorted(self.datasets.items()):
            role = self._split_role_name(split)
            benchmarks = sorted({row['benchmark'] for row in rows})
            provenance = {'benchmarks': benchmarks, 'task_count': len(rows)}
            source = str(self.root / 'binding.json')
            provenance['manifests'] = {name: self.manifest['benchmarks'][name]['manifest'] for name in benchmarks}
            provenance['protocols'] = {name: self.protocols[name].protocol_id for name in benchmarks}
            cohort = [{
                'id': row['id'],
                'benchmark': row['benchmark'],
                'files': dict(sorted(row['files'].items())),
            } for row in rows]
            bindings.append(DatasetBinding(
                name='coding/' + split,
                split=split,
                role=role,
                source=source,
                schema_version='coding-task-manifest/v1',
                content_digest=digest({'split': split, 'tasks': cohort}),
                provenance=provenance,
            ))
        return tuple(bindings)

    def describe(self):
        artifact = {'type': 'object', 'properties': {
            'path': {'type': 'string', 'minLength': 1},
            'sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}},
            'required': ['path', 'sha256'], 'additionalProperties': False}
        return (
            OperationSpec('inspect', 'Describe the installed coding task environment and measured evaluations.',
                          {'type': 'object', 'properties': {}, 'additionalProperties': False}, read_only=True),
            OperationSpec('evaluate', 'Measure an immutable artifact using the installed agent, frozen task '
                          'split and fixed execution budgets. No search or candidate selection is performed.',
                          {'type': 'object', 'properties': {
                              'artifact': artifact, 'split': {'type': 'string', 'enum': list(self.datasets)},
                              'trials': {'type': 'integer', 'minimum': 1, 'maximum': 16},
                              'task_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1,
                                           'uniqueItems': True}},
                           'required': ['artifact', 'split'], 'additionalProperties': False}),
            OperationSpec('read_trace', 'Read a stored agent trace page by evaluation and trial index.',
                          {'type': 'object', 'properties': {
                              'evaluation': {'type': 'string'}, 'split': {'type': 'string'}, 'index': {'type': 'integer', 'minimum': 0},
                              'offset': {'type': 'integer', 'minimum': 0}},
                           'required': ['evaluation', 'split', 'index', 'offset'], 'additionalProperties': False},
                          read_only=True),
        )

    def execute(self, call):
        args = dict(call.arguments)
        if call.name == 'inspect' and not args:
            return OperationResult('Coding task environment', (), False, data={
                'protocol': PROTOCOL_VERSION, 'splits': {
                    key: [{'id': r['id'], 'benchmark': r['benchmark']} for r in rows]
                    for key, rows in self.datasets.items() if self.split_roles()[key].inputs_visible},
                'agent': self.agent, 'protocol_fingerprint': self.fingerprint,
                'evaluations': [p.parent.name for p in sorted(self.evaluations.glob('*/summary.json'))],
            })
        if call.name == 'evaluate' and {'artifact', 'split'} <= args.keys() <= {'artifact', 'split', 'trials', 'task_ids'}:
            return OperationResult('Coding evaluation completed', (), False, data=self.evaluate(**args))
        if call.name == 'read_trace' and set(args) == {'evaluation', 'split', 'index', 'offset'}:
            return OperationResult('Coding trace page', (), False, data=self.read_trace(**args))
        raise LabToolError('Use exactly the documented Coding Lab operation arguments')

    def _request(self, artifact, split, trials, task_ids):
        if split not in self.datasets:
            raise LabToolError('Unknown task split')
        if type(trials) is not int or not 1 <= trials <= 16:
            raise LabToolError('trials must be between 1 and 16')
        rows = self.datasets[split]
        if task_ids is not None:
            if (not isinstance(task_ids, list) or not task_ids
                    or any(not isinstance(t, str) for t in task_ids)
                    or len(task_ids) != len(set(task_ids))):
                raise LabToolError('task_ids must be distinct task identifiers')
            lookup = defaultdict_rows(rows)
            if any(len(lookup.get(t, [])) != 1 for t in task_ids):
                raise LabToolError('Unknown or ambiguous task ID in this split')
            wanted = set(task_ids)
            rows = [r for r in rows if r['id'] in wanted]
        pinned = snapshot(artifact, self.artifacts)
        self.check_domain_agent(pinned, rows)
        request = {'artifact_sha256': pinned.sha256, 'split': split, 'trials': trials,
                   'tasks': rows, 'protocol_fingerprint': self.fingerprint,
                   'protocols': {name: self.protocols[name].protocol_id for name in sorted({r['benchmark'] for r in rows})}}
        return pinned, request

    def check_domain_agent(self, pinned, rows):
        """Every benchmark in the request must accept this domain agent's shape; reject before any trial runs."""
        for name in sorted({row['benchmark'] for row in rows}):
            spec = self.runtimes[name].accepts
            agent = DomainAgent.from_path(spec.kind, pinned.path)
            try:
                if spec.kind == 'agent-tree':
                    if not Path(pinned.path).is_dir():
                        raise DomainAgentRejected('An agent-tree domain agent must be a directory')
                else:
                    spec.check(read_domain_agent_data(agent))
            except DomainAgentRejected as error:
                raise LabToolError(f'Benchmark {name!r} runtime {self.runtimes[name].name!r} rejects the '
                                   f'domain agent: {error}') from error

    def evaluate(self, artifact, split, trials=1, task_ids=None):
        """Public method-neutral measurement entry point; safe to reattach/repeat.

        Identical inputs reuse receipts. Increase trials or use a new frozen seed
        binding for additional independent samples, rather than silently rerunning.
        """
        with self._lock():
            self.validate_tasks()
            pinned, request = self._request(artifact, split, trials, task_ids)
            key = 'evaluation-' + digest(request)
            root = self.evaluations / key
            cached = root / 'summary.json'
            if cached.exists():
                result = json.loads(cached.read_text())
                if result['request'] != request:
                    raise ValueError('Cached coding evaluation identity changed')
                return result
            root.mkdir(exist_ok=True)
            atomic_json(root / 'request.json', request)
            items = [(row, trial) for row in request['tasks'] for trial in range(trials)]
            started = time.monotonic()
            def work(item):
                row, trial = item
                # Seed and endpoint follow the task's own identity, never its position in the merged
                # split, so a benchmark runs the same way whatever else the Lab mounts.
                slot = int(digest({'benchmark': row['benchmark'], 'task': row['id']})[:12], 16)
                seed = (self.config.get('seed', 73) + slot % 100000 * 10000 + trial * 1000) % 2**32
                protocol = self.protocols[row['benchmark']]
                identity = {'artifact_sha256': pinned.sha256, 'task': row, 'seed': seed,
                            'protocol_fingerprint': self.fingerprint, 'protocol_id': protocol.protocol_id}
                root = self.measurements / digest(identity)
                root.mkdir(exist_ok=True)
                atomic_json(root / 'identity.json', identity)
                receipt = root / 'task-000.json'
                endpoint = self.config['endpoints'][(slot + trial) % len(self.config['endpoints'])]
                def completed(data):
                    if data.get('task') != row['path']:
                        raise infrastructure_failure('Cached receipt task identity mismatch: ' + str(receipt))
                    if type(data.get('score')) not in (int, float) or not 0 <= data['score'] <= 1:
                        raise infrastructure_failure('Invalid bounded score in receipt: ' + str(receipt))
                    data.update(task_id=row['id'], benchmark=row['benchmark'], endpoint=endpoint,
                                trial=trial, seed=seed, artifact_sha256=pinned.sha256,
                                domain_agent_sha256=pinned.sha256, protocol_id=protocol.protocol_id,
                                runtime={'name': self.runtimes[row['benchmark']].name,
                                         'version': self.runtimes[row['benchmark']].version})
                    atomic_json(receipt, data)
                    return data
                if receipt.exists():
                    data = json.loads(receipt.read_text())
                    if data.get('measured') is True:
                        self._runner.reconcile(root)
                        return completed(data)
                for _ in range(2):
                    code, trial_name = self._runner.run(root, row, pinned, endpoint, seed, protocol.protocol, row['benchmark'])
                    if code == 0 and receipt.exists():
                        data = json.loads(receipt.read_text())
                        if data.get('measured') is True:
                            return completed(data)
                    if receipt.exists():
                        receipt.replace(root / (trial_name + '-incomplete.json'))
                raise infrastructure_failure('Coding task has no valid grade after retries: ' + row['id'] +
                                             '; inspect ' + str(root))
            with ThreadPoolExecutor(max_workers=self.config.get('workers', 14)) as pool:
                measured = [future.result() for future in [pool.submit(copy_context().run, work, item) for item in items]]
            pinned.verify()
            return self._summarize(key, root, request, measured, time.monotonic() - started)

    def _summarize(self, key, root, request, rows, seconds):
        by_benchmark = {}
        for family in sorted({r['benchmark'] for r in rows}):
            subset = [r for r in rows if r['benchmark'] == family]
            by_benchmark[family] = {'score': sum(r['score'] for r in subset) / len(subset),
                                   'samples': len(subset), 'solved': sum(r['score'] == 1 for r in subset),
                                   'protocol_id': self.protocols[family].protocol_id}
        usages = [(r.get('agent_result') or {}) for r in rows]
        cost = {key: sum(u.get('n_' + key, 0) or 0 for u in usages)
                for key in ('input_tokens', 'output_tokens', 'cache_tokens')}
        cost['total_tokens'] = cost['input_tokens'] + cost['output_tokens']
        # Each benchmark's reward is reported on its own; scores are measurements, not an objective
        # imposed on the search method, and no cross-benchmark total is formed here.
        metrics = {self.metric_name(name): value['score'] for name, value in by_benchmark.items()}
        kind, declared = choose_primary_metric(self.config.get('primary_metric'), self.benchmarks, aggregates=AGGREGATES)
        if kind == 'aggregate':
            # Declared in the Lab's config: the unweighted mean of the mounted benchmarks' rewards.
            names = sorted(self.protocols)
            if set(by_benchmark) == set(names):
                metrics['benchmark_reward'] = sum(by_benchmark[n]['score'] for n in names) / len(names)
        data = {'evaluation_id': key, 'request': request, 'split': request['split'],
                'artifact_sha256': request['artifact_sha256'],
                'quality_metrics': metrics, 'by_benchmark': by_benchmark, 'samples': len(rows), 'cost': cost,
                'timing': {'elapsed_seconds': seconds}, 'tasks': rows,
                'dataset_sha256': digest(request['tasks']), 'protocol_fingerprint': self.fingerprint}
        atomic_json(root / 'summary.json', data)
        return data

    def read_trace(self, evaluation, index, offset, split=None):
        if (not isinstance(evaluation, str) or not re.fullmatch(r'evaluation-[a-f0-9]{64}', evaluation)
                or type(index) is not int or index < 0 or type(offset) is not int or offset < 0):
            raise LabToolError('Invalid evaluation or trace page')
        summary = self.evaluations / evaluation / 'summary.json'
        if not summary.exists():
            raise LabToolError('Unknown completed evaluation')
        data = json.loads(summary.read_text())
        if split is not None and data['split'] != split:
            raise LabToolError('Trace does not belong to the requested split')
        rows = data['tasks']
        if index >= len(rows):
            raise LabToolError('Invalid trace index')
        official = Path(rows[index]['official_result'])
        agent = official.parent / 'agent'
        paths = [agent / 'coding-trace.jsonl', agent / 'trajectory.json']
        trace = next((p for p in paths if p.is_file()), None)
        content = trace.read_text() if trace else 'No agent trajectory was produced'
        offset = min(offset, len(content))
        end = min(len(content), offset + 6000)
        return {'text': content[offset:end], 'eof': end == len(content),
                'next_offset': None if end == len(content) else end}

    def resource_roles(self):
        return self._runner.resource_roles()

    def runtime_context(self):
        return json.dumps({'task_splits': {k: len(v) for k, v in self.datasets.items()},
                           'completed_evaluations': len(list(self.evaluations.glob('*/summary.json')))})

    def close(self):
        # Worker ownership remains journaled for reattachment if a controller stops.
        pass


def defaultdict_rows(rows):
    result = {}
    for row in rows:
        result.setdefault(row['id'], []).append(row)
    return result
