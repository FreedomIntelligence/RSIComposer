"""Meta-agent instruction-search policy over the reusable coding environment."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import time

from rsicomposer.api import (DomainAgentRejected, DomainAgentSpec, CodingLab, CodingArtifact, DEV, SEALED, Invariant, MetricSpec,
                        OperationCall, OperationResult, OperationSpec, LabToolError, atomic_write, json_bytes,
                        emit_observation, report_version, report_version_evaluation, report_version_selection)

BASELINE = 'Inspect the project, implement a correct minimal fix, and run relevant tests before finishing.'
POLICY_OPTIONS = {'min_candidates', 'max_candidates'}
POLICY_VERSION = 'coding-meta-agent/v5'
# The domain agent this method's RSI policy registers: the shape every mounted benchmark's runtime must also accept.
REGISTERED = DomainAgentSpec('instruction', {
    'instruction': {'type': 'string', 'min_length': 20, 'max_length': 12000, 'required': True},
    'hypothesis': {'type': 'string', 'required': True}}, 'data')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, data):
    atomic_write(Path(path), json_bytes(data))


def validate_config(config):
    CodingLab.validate_config({k: v for k, v in config.items() if k not in POLICY_OPTIONS})
    roles = config.get('split_roles', {})
    expected_roles = {'search': 'dev', 'heldout': 'sealed'}
    if not isinstance(roles, dict) or any(
            split not in expected_roles or role != expected_roles[split]
            for split, role in roles.items()):
        raise ValueError('coding_meta_agent requires search=dev and heldout=sealed')
    for key in POLICY_OPTIONS:
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError('Positive integer required: ' + key)
    if config.get('min_candidates', 3) > config.get('max_candidates', 6):
        raise ValueError('min_candidates exceeds max_candidates')


class CodingSearchLab:
    """Method-owned proposal/selection rules; tasks and grading stay in CodingLab."""
    validate_config = staticmethod(validate_config)

    def __init__(self, context, config):
        validate_config(config)
        self.config = deepcopy(config)
        self.artifacts = context.artifacts
        self.selected = None
        self.root = Path(config['experiment_dir']).resolve()
        prior = self.root / 'binding.json'
        if prior.exists() and json.loads(prior.read_text()).get('policy') != POLICY_VERSION:
            raise ValueError('Experiment binding changed; use a new experiment_dir')
        env_config = {k: v for k, v in config.items() if k not in POLICY_OPTIONS}
        env_config['experiment_dir'] = str(self.root / 'environment')
        self.environment = CodingLab(context, env_config)
        self.manifest = self.environment.datasets
        if set(self.manifest) != {'search', 'heldout'}:
            raise ValueError('coding_meta_agent requires search and heldout splits')
        self.fingerprint = self.environment.fingerprint
        # Assembly-time check: each mounted benchmark's runtime must accept the domain agents this method registers.
        for name, runtime in self.environment.runtimes.items():
            if runtime.accepts.kind != REGISTERED.kind or set(REGISTERED.fields) - set(runtime.accepts.fields):
                raise ValueError(f'Benchmark {name!r} runtime {runtime.name!r} does not accept the instruction '
                                 'domain agents this method registers')
        self.candidates = self.root / 'candidates'
        self.evaluations = self.root / 'evaluations'
        with self._lock():
            binding = {'policy': POLICY_VERSION, 'config': self.config,
                       'environment': self.fingerprint,
                       'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            path = self.root / 'binding.json'
            if path.exists() and json.loads(path.read_text()) != binding:
                raise ValueError('Experiment binding changed; use a new experiment_dir')
            atomic_json(path, binding)
            self.candidates.mkdir(exist_ok=True)
            self.evaluations.mkdir(exist_ok=True)
            baseline = self.candidates / 'baseline.json'
            if not baseline.exists():
                atomic_json(baseline, {'id': 'baseline', 'instruction': BASELINE,
                            'hypothesis': 'Initial minimal coding instruction', 'sha256': digest(BASELINE)})

    def state(self):
        # Candidates, evaluations and task receipts persist under experiment_dir.
        # Selection is restored from the host checkpoint, not inferred from files.
        return {'environment': self.environment.state(), 'selected': self.selected}

    def restore(self, state):
        self.environment.restore(state['environment'])
        if state['selected'] is not None:
            self._candidate(state['selected'])
        self.selected = state['selected']

    def _validate_manifest(self):
        self.environment.validate_tasks()

    @contextmanager
    def _lock(self):
        with (self.root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def _candidate(self, ident):
        if not isinstance(ident, str) or ident != Path(ident).name:
            raise LabToolError('Invalid candidate ID')
        path = self.candidates / (ident + '.json')
        if not path.is_file():
            raise LabToolError('Unknown candidate ID')
        data = json.loads(path.read_text())
        if data['id'] != ident or data['sha256'] != digest(data['instruction']):
            raise RuntimeError('Immutable candidate changed')
        return data, path

    def resource_roles(self):
        return self.environment.resource_roles()

    def profiles(self):
        return ('artifact-search/v1',)

    def primary_metric(self):
        # The environment's declaration: one benchmark, or the aggregate the config names.
        return self.environment.primary_metric()

    def split_roles(self):
        return {'search': DEV, 'heldout': SEALED}

    def data_bindings(self):
        return self.environment.data_bindings()

    def invariants(self):
        return (Invariant.baseline_before('select'), Invariant.measured_selection(),
                Invariant.min_candidates(self.config.get('min_candidates', 3)))

    def final_evaluations(self):
        return tuple(OperationCall('evaluate', {'candidate_id': ident, 'split': 'heldout', 'indices': None})
            for ident in dict.fromkeys(('baseline', self.selected))) if self.selected is not None else ()

    def describe(self):
        text = {'type': 'string', 'minLength': 1}
        def spec(name, description, properties, required):
            return OperationSpec(name, description, {'type': 'object', 'properties': properties,
                'required': required, 'additionalProperties': False})
        return (
            spec('inspect', 'Read persisted experiment state and an immutable instruction. '
                 'Start with candidate_id=baseline.', {'candidate_id': text}, ['candidate_id']),
            spec('register', 'Register a candidate object containing a general coding instruction and hypothesis. '
                 'The worker supplies JSON shell actions and equal budgets; improve diagnosis, editing or testing. '
                 'No task IDs, cached solutions, grader access or task-specific patches. No measurement is performed.',
                 {'candidate': {'type': 'object', 'properties': {'instruction': text, 'hypothesis': text},
                                'required': ['instruction', 'hypothesis']}}, ['candidate']),
            spec('evaluate', 'Measure a registered candidate on a named split, optionally selecting task indices. '
                 'The host reserves heldout until after finish. Infrastructure failures remain incomplete.',
                 {'candidate_id': text, 'split': text, 'indices': {'type': ['array', 'null']}},
                 ['candidate_id', 'split', 'indices']),
            spec('read_trace', 'Read a search agent trajectory page.',
                 {'candidate_id': text, 'task_index': {'type': 'integer', 'minimum': 0},
                  'offset': {'type': 'integer', 'minimum': 0}}, ['candidate_id', 'task_index', 'offset']),
            spec('select', 'Record a measured selection, then finish. Host invariants require baseline and '
                 'the configured minimum measured candidates; the host performs final evaluation after finish.',
                 {'candidate_id': text}, ['candidate_id']),
        )

    @staticmethod
    def _aggregate(data):
        keys = ('candidate', 'split', 'score', 'by_benchmark', 'samples', 'cost',
                'harness_sha256', 'dataset_sha256', 'protocol_fingerprint', 'candidate_id',
                'quality_metrics', 'valid', 'scored', 'scores', 'trace_refs', 'feedback', 'baseline')
        return {key: data[key] for key in keys if key in data}

    def _feedback(self, data):
        # Full official records already live in measurement summaries.
        # Keep the final receipt small enough to remain native in a bounded
        # decision context, so the policy can recognize its completion facts.
        if 'tasks' in data:
            result = self._aggregate(data)
            result['tasks'] = [dict(index=index, task_id=row.get('task_id'),
                benchmark=row.get('benchmark'), score=row.get('score'),
                agent_status=((row.get('agent_result') or {}).get('metadata') or {}).get('status'),
                output_tokens=(row.get('agent_result') or {}).get('n_output_tokens'))
                for index, row in enumerate(data['tasks'])]
            return result
        return data

    def _result(self, message, data):
        return OperationResult(message, (), False, data=self._feedback(data))

    def execute(self, call):
        with self._lock():
            return self._execute(call)

    def _execute(self, call):
        args = dict(call.arguments)
        specs = {spec.name: spec for spec in self.describe()}
        if call.name not in specs:
            raise LabToolError('Unknown coding search operation')
        schema = specs[call.name].input_schema
        if set(args) != set(schema['required']):
            raise LabToolError('Use exactly the documented operation arguments')
        if call.name == 'register':
            candidate = args['candidate']
            try:
                REGISTERED.check(candidate)
            except DomainAgentRejected as error:
                raise LabToolError('Candidate requires instruction (20–12000 characters) and hypothesis: ' + str(error)) from error
            instruction, hypothesis = candidate['instruction'], candidate['hypothesis']
            # Meta-agent cannot smuggle benchmark task names or copied questions.
            for rows in (self.manifest['search'], self.manifest['heldout']):
                for row in rows:
                    if row['id'] in instruction:
                        raise LabToolError('Candidate must be task independent')
            if instruction == BASELINE:
                raise LabToolError('A proposal must differ from the baseline')
            ident = 'candidate-' + digest(instruction)[:12]
            path = self.candidates / (ident + '.json')
            if not path.exists():
                if len(list(self.candidates.glob('candidate-*.json'))) >= self.config.get('max_candidates', 6):
                    raise LabToolError('Candidate budget exhausted; select a measured candidate')
                atomic_json(path, {'id': ident, 'instruction': instruction, 'hypothesis': hypothesis,
                                   'sha256': digest(instruction)})
            report_version(ident, label=ident)
            emit_observation('coding_meta_agent.candidate', call.call_id, subject=ident,
                             components={'instruction': instruction})
            return self._result('Candidate registered, not evaluated', {'candidate_id': ident})
        candidate, path = self._candidate(args['candidate_id'])
        if call.name == 'inspect':
            return self._result('Persisted coding experiment', {'candidate': candidate,
                'state': json.loads(self.runtime_context()),
                'budgets': {**{key: self.config[key] for key in POLICY_OPTIONS if key in self.config},
                            'protocols': {name: benchmark.protocol
                                          for name, benchmark in self.environment.protocols.items()}},
                'protocol_fingerprint': self.fingerprint})
        if call.name == 'read_trace':
            index, offset = args['task_index'], args['offset']
            if type(index) is not int or not 0 <= index < len(self.manifest['search']) or type(offset) is not int or offset < 0:
                raise LabToolError('Invalid trace index or offset')
            result = self._cached(candidate['id'], 'search')
            if result is None:
                raise LabToolError('Evaluate this candidate first')
            official = Path(result['tasks'][index]['official_result'])
            trace = official.parent / 'agent' / 'coding-trace.jsonl'
            content = trace.read_text() if trace.exists() else 'No agent trajectory was produced'
            end = min(len(content), offset + 6000)
            return self._result('Search trace page', {'text': content[offset:end],
                'eof': end == len(content), 'next_offset': None if end == len(content) else end})
        if call.name == 'evaluate':
            data = self._measure(candidate, path, args['split'], args['indices'])
            ref = self.artifacts.put(json.dumps(data, allow_nan=False).encode(), kind='coding-evaluation')
            data = dict(data, candidate_id=candidate['id'], valid=True, scored=True,
                baseline=candidate['id'] == 'baseline',
                scores=[row['score'] for row in data['tasks']],
                trace_refs=[asdict(ref)] * len(data['tasks']),
                feedback=[f"{row['task_id']}: reward {row['score']}" for row in data['tasks']])
            if candidate['id'] == 'baseline':
                report_version('baseline', label='Baseline', round_index=0, parent_ids=())
            report_version_evaluation(candidate['id'], call.call_id)
            return self._result('Coding candidate evaluated', data)
        self.selected = candidate['id']
        atomic_json(self.root / 'selection.json', {'candidate_id': self.selected,
                    'selected_at': time.time(), 'protocol_fingerprint': self.fingerprint})
        report_version_selection(self.selected, step=len(list(self.candidates.glob('candidate-*.json'))),
                                 event_id=call.call_id, action='kept' if self.selected == 'baseline' else 'selected')
        return self._result('Selection recorded; finish to request host final evaluation', {
            'candidate_id': self.selected, 'selected_candidate': self.selected, 'selection_recorded': True})

    def _cached(self, ident, split, indices=None):
        part = split if indices is None else split + '-indices-' + digest(indices)
        p = self.evaluations / ident / part / 'summary.json'
        return json.loads(p.read_text()) if p.exists() else None

    def _measure(self, candidate, candidate_path, split, indices):
        self._validate_manifest()
        rows = self.manifest.get(split)
        if rows is None:
            raise LabToolError('Unknown coding split')
        if indices is not None and (not isinstance(indices, list) or not indices
                or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
                or len(set(indices)) != len(indices)):
            raise LabToolError('indices must be distinct valid task offsets')
        selected = rows if indices is None else [rows[i] for i in indices]
        missing = self.environment.primary_metric_benchmarks() - {row['benchmark'] for row in selected}
        if missing:
            raise LabToolError('Evaluation must include the benchmarks required by the primary metric; missing: '
                               + ', '.join(sorted(missing)))
        cached = self._cached(candidate['id'], split, indices)
        if cached is not None:
            return cached
        measurement = self.environment.evaluate(CodingArtifact.from_path(candidate_path), split,
            task_ids=None if indices is None else [row['id'] for row in selected])
        data = dict(measurement, candidate=candidate['id'], harness_sha256=candidate['sha256'])
        # The search objective is the primary metric the Lab config declares; every benchmark's own
        # reward stays in quality_metrics beside it.
        data['score'] = measurement['quality_metrics'][self.primary_metric().name]
        part = split if indices is None else split + '-indices-' + digest(indices)
        atomic_json(self.evaluations / candidate['id'] / part / 'summary.json', data)
        return data

    def runtime_context(self):
        states = {}
        for p in sorted(self.candidates.glob('*.json')):
            d = self._cached(p.stem, 'search')
            states[p.stem] = {'search_score': d['score'] if d else None}
        return json.dumps({'candidates': states, 'selected': self.selected,
                           'min_candidates': self.config.get('min_candidates', 3),
                           'max_candidates': self.config.get('max_candidates', 6),
                           'sealed_evaluation': 'host_after_finish'})

    def close(self):
        self.environment.close()
