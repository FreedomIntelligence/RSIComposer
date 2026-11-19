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
import os
from pathlib import Path
import re
import signal
import subprocess
from types import SimpleNamespace
import time
from uuid import uuid4

from rsisys.loops.contracts.coordination import OperationResult, ExecutionStatus
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.profiles import DEV, SEALED, MetricSpec, SPLIT_ROLES
from rsisys.loops.contracts.execution import LabInfrastructureError, LabToolError
from .agents import HarborCodingAgent
from .artifacts import digest, snapshot
from .harbor_runner import atomic_json


OPTIONS = {'experiment_dir', 'manifest', 'python', 'endpoints', 'model', 'workers',
           'max_steps', 'max_tokens', 'total_tokens', 'agent_timeout', 'verifier_timeout',
           'seed', 'thinking', 'split_roles'}


def _infrastructure_failure(message):
    return LabInfrastructureError(OperationResult('Coding environment infrastructure failure', (), False,
        status=ExecutionStatus.FAILED, error=message))


def validate_config(config):
    if set(config) - OPTIONS:
        raise ValueError('Unknown Coding Lab options: ' + str(set(config) - OPTIONS))
    for key in ('experiment_dir', 'manifest', 'python', 'model'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('Coding Lab requires ' + key)
    endpoints = config.get('endpoints')
    if not isinstance(endpoints, list) or not endpoints or any(
            not isinstance(e, str) or not e.startswith(('http://', 'https://')) for e in endpoints):
        raise ValueError('Coding Lab requires explicit inference endpoints')
    for key in ('workers', 'max_steps', 'max_tokens', 'total_tokens', 'agent_timeout', 'verifier_timeout'):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError('Positive integer required: ' + key)
    if 'thinking' in config and type(config['thinking']) is not bool:
        raise ValueError('thinking must be boolean')
    if 'seed' in config and (type(config['seed']) is not int or not 0 <= config['seed'] < 2**32):
        raise ValueError('seed must be a nonnegative 32-bit integer')


class CodingLab:
    """Task-scene Lab reusable by prompt, code and agent-configuration methods.

    An installed adapter controls how the immutable artifact is passed to the
    agent. The Lab never proposes, mutates, ranks, selects or adopts candidates.
    Split roles are declared per instance and enforced by the host.
    """
    validate_config = staticmethod(validate_config)

    def __init__(self, context, config, *, agent=None):
        validate_config(config)
        self.config = deepcopy(config)
        self.root = Path(config['experiment_dir']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._execution_context = context if context is not None else SimpleNamespace(data_dir=self.root)
        self.manifest = json.loads(Path(config['manifest']).read_text())
        self.datasets = self.manifest.get('splits', {
            k: self.manifest[k] for k in ('search', 'heldout') if k in self.manifest})
        self.validate_tasks()
        roles = self.config.get('split_roles', {})
        if (not isinstance(roles, dict) or set(roles) - set(self.datasets)
                or any(not isinstance(role, str) or role not in SPLIT_ROLES for role in roles.values())):
            raise ValueError('split_roles must map installed splits to named profile roles')
        self.agent = (agent or HarborCodingAgent()).to_json()
        package = Path(__file__).parent
        self.protocol = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                         for name in ('agent.py', 'agents.py', 'artifacts.py', 'lab.py',
                                      'harbor_runner.py', 'history.py')}
        self.binding = {'protocol_version': 'coding-environment/v4', 'manifest': self.manifest,
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
            atomic_json(self.root / 'agent.json', self.agent)
            self.evaluations.mkdir(exist_ok=True)
            self.artifacts.mkdir(exist_ok=True)
            self.measurements.mkdir(exist_ok=True)

    def validate_tasks(self):
        if not isinstance(self.datasets, dict) or not self.datasets:
            raise ValueError('Nonempty named task splits required')
        seen = set()
        for split, rows in self.datasets.items():
            if not isinstance(split, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', split):
                raise ValueError('Invalid task split name')
            if not isinstance(rows, list) or not rows:
                raise ValueError('Nonempty task lists required')
            for row in rows:
                key = (row['benchmark'], row['id'])
                if key in seen:
                    raise ValueError('Task overlap or duplicate: ' + str(key))
                seen.add(key)
                path = Path(row['path'])
                if not path.is_absolute() or not (path / 'task.toml').is_file() or not (path / 'instruction.md').is_file():
                    raise ValueError('Missing real benchmark task: ' + str(path))
                if not {'task.toml', 'instruction.md'} <= row['files'].keys():
                    raise ValueError('Task manifest must hash task.toml and instruction.md')
                for relative, sha in row['files'].items():
                    entry = path / relative
                    if not entry.resolve().is_relative_to(path.resolve()):
                        raise ValueError('Task hash entry leaves its task directory')
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
        return MetricSpec('task_reward', 'maximize', 'harbor-official-reward/task-mean/v1')

    def split_roles(self):
        return {name: SPLIT_ROLES[self.config['split_roles'][name]]
                if name in self.config.get('split_roles', {}) else
                (SEALED if name in ('test', 'heldout', 'sealed') else DEV)
                for name in self.datasets}

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
                'protocol': 'coding-environment/v4', 'splits': {
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
        request = {'artifact_sha256': pinned.sha256, 'split': split, 'trials': trials,
                   'tasks': rows, 'protocol_fingerprint': self.fingerprint}
        return pinned, request

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
                original_index = self.datasets[split].index(row)
                seed = (self.config.get('seed', 73) + original_index * 10000 + trial * 1000) % 2**32
                identity = {'artifact_sha256': pinned.sha256, 'task': row, 'seed': seed,
                            'protocol_fingerprint': self.fingerprint}
                measurement_root = self.measurements / digest(identity)
                measurement_root.mkdir(exist_ok=True)
                root, index = measurement_root, 0
                atomic_json(root / 'identity.json', identity)
                receipt = root / 'task-000.json'
                endpoint = self.config['endpoints'][(original_index + trial) % len(self.config['endpoints'])]
                def completed(data):
                    if data.get('task') != row['path']:
                        raise _infrastructure_failure('Cached receipt task identity mismatch: ' + str(receipt))
                    if type(data.get('score')) not in (int, float) or not 0 <= data['score'] <= 1:
                        raise _infrastructure_failure('Invalid bounded score in receipt: ' + str(receipt))
                    data.update(task_id=row['id'], benchmark=row['benchmark'], endpoint=endpoint,
                                trial=trial, seed=seed, artifact_sha256=pinned.sha256)
                    atomic_json(receipt, data)
                    return data
                if receipt.exists():
                    data = json.loads(receipt.read_text())
                    if data.get('measured') is True:
                        inflight = root / f'task-{index:03d}-inflight.json'
                        if inflight.exists():
                            record = json.loads(inflight.read_text())
                            self._run_process(record['command'], root, index, record['trial_name'])
                        return completed(data)
                for _ in range(2):
                    trial_name = f"{index:03d}-{re.sub('[^a-zA-Z0-9_-]', '-', row['id'])[:65]}-{uuid4().hex[:8]}"
                    cmd = [self.config['python'], '-m', 'rsisys.labs.scenes.coding.harbor_runner',
                           '--task', row['path'], '--artifact', pinned.path,
                           '--agent-spec', str(self.root / 'agent.json'), '--endpoint', endpoint,
                           '--output', str(root / 'trials'), '--trial-name', trial_name,
                           '--receipt', str(receipt), '--model', self.config['model']]
                    for option, name, default in [('max-steps', 'max_steps', 48), ('max-tokens', 'max_tokens', 4096),
                        ('total-tokens', 'total_tokens', 98304), ('agent-timeout', 'agent_timeout', 1800),
                        ('verifier-timeout', 'verifier_timeout', 1800)]:
                        cmd.extend(['--' + option, str(self.config.get(name, default))])
                    # Stable across subsets and repetitions; a task has one seed stream.
                    cmd.extend(['--seed', str(seed)])
                    if self.config.get('thinking', False):
                        cmd.append('--thinking')
                    code = self._run_process(cmd, root, index, trial_name)
                    if code == 0 and receipt.exists():
                        data = json.loads(receipt.read_text())
                        if data.get('measured') is True:
                            return completed(data)
                    if receipt.exists():
                        receipt.replace(root / (trial_name + '-incomplete.json'))
                raise _infrastructure_failure('Coding task has no valid grade after retries: ' + row['id'] +
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
                                   'samples': len(subset), 'solved': sum(r['score'] == 1 for r in subset)}
        usages = [(r.get('agent_result') or {}) for r in rows]
        cost = {key: sum(u.get('n_' + key, 0) or 0 for u in usages)
                for key in ('input_tokens', 'output_tokens', 'cache_tokens')}
        cost['total_tokens'] = cost['input_tokens'] + cost['output_tokens']
        # Scores are measurements, not an objective imposed on the search method.
        score = sum(r['score'] for r in rows) / len(rows)
        data = {'evaluation_id': key, 'request': request, 'split': request['split'],
                'artifact_sha256': request['artifact_sha256'], 'score': score,
                'quality_metrics': {'task_reward': score}, 'by_benchmark': by_benchmark, 'samples': len(rows), 'cost': cost,
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
        from rsisys.runtime.execution.resources import ResourceRequest
        return {'evaluator': ResourceRequest(timeout=7200)}

    def _run_process(self, cmd, root, index, trial_name):
        """Reattach a live submitted worker before considering a retry.

        A submission is journaled before spawn. PID discovery covers a parent
        crash between spawn and PID persistence. A live Docker project without
        a live worker is uncertain and requires explicit host reconciliation.
        """
        record_path = root / f'task-{index:03d}-inflight.json'
        child = None
        if record_path.exists():
            record = json.loads(record_path.read_text())
            trial_name = record['trial_name']
            cmd = record['command']
        else:
            record = {'trial_name': trial_name, 'command': cmd, 'submitted_at': time.time()}
            atomic_json(record_path, record)
            with (root / (trial_name + '.log')).open('wb') as log:
                from rsisys.runtime.execution.executor import execution_capabilities, WorkerSpec
                from rsisys.runtime.execution.workers import worker_environment
                resources, executor = execution_capabilities(self._execution_context)
                resources.declare(self.resource_roles())
                with resources.acquire('evaluator') as lease:
                    child = executor.launch(WorkerSpec(cmd, worker_environment(), stdout=log,
                        detached=True, cancel_signal=signal.SIGINT), lease)
            if callable(getattr(child, 'reference', None)):
                record['worker'] = child.reference()
            else:
                record['pid'] = child.worker_pid
            atomic_json(record_path, record)

        if 'worker' not in record and child is None:
            configured = getattr(self._execution_context, 'resources', None)
            owner = getattr(configured, 'owner', configured)
            if getattr(owner, 'config', {}).get('executor', 'local') == 'ray':
                from rsisys.runtime.execution.executor import execution_capabilities
                _, executor = execution_capabilities(self._execution_context)
                reference = executor.find_job(cmd)
                if reference is None:
                    raise _infrastructure_failure('Remote submission ownership is uncertain; reconcile ' + str(record_path))
                record['worker'] = reference
                atomic_json(record_path, record)

        if 'worker' in record:
            from rsisys.runtime.execution.executor import execution_capabilities
            _, executor = execution_capabilities(self._execution_context)
            child = child or executor.reattach(record['worker'])
            try:
                code = child.wait(timeout=7200)
            except (TimeoutError, subprocess.TimeoutExpired):
                child.terminate()
                raise _infrastructure_failure('Owned remote task exceeded host wall time; reconcile ' + str(record_path))
            result = child.run_cleanup(['docker', 'ps', '-aq', '--filter',
                'name=^/' + re.escape(trial_name.lower()) + '__'])
            if result['code'] != 0 or result['stdout'].strip():
                raise _infrastructure_failure('Uncertain live/orphaned remote task container; inspect ' + str(record_path))
            record_path.replace(root / (trial_name + '-submission.json'))
            return code

        def live_pid():
            # A saved PID is a hint, not proof of ownership: match the exact
            # unique trial name and receipt root in the actual command line.
            candidates = ([Path('/proc') / str(record['pid'])] if 'pid' in record else [])
            if not candidates:
                candidates = [p for p in Path('/proc').iterdir() if p.name.isdigit()]
            for p in candidates:
                try:
                    args = (p / 'cmdline').read_bytes().decode(errors='replace').split('\0')
                    if trial_name in args and str(root / f'task-{index:03d}.json') in args:
                        return int(p.name)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
            return None

        started = time.monotonic()
        while (child.poll() is None if child is not None else live_pid() is not None):
            if time.monotonic() - started > 7200:
                pid = child.pid if child is not None else live_pid()
                if pid is not None:
                    # SIGINT lets asyncio/Harbor run cancellation cleanup. Do
                    # not start another task until cleanup can be reconciled.
                    try:
                        os.kill(pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                raise _infrastructure_failure('Owned task exceeded host wall time; reconcile ' + str(record_path))
            time.sleep(2)
        result = subprocess.run(['docker', 'ps', '-aq', '--filter',
                                 'name=^/' + re.escape(trial_name.lower()) + '__'],
                                capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or result.stdout.strip():
            raise _infrastructure_failure('Uncertain live/orphaned task container; inspect ' + str(record_path))
        record_path.replace(root / (trial_name + '-submission.json'))
        return child.returncode if child is not None else 0

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
