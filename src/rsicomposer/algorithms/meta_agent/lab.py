"""Bounded harness research Lab for the model-driven meta-agent method.

Share the fixed-model measurement engine with RRSI, without its proposal,
critic, edit-budget or selection algorithm. The meta-agent owns those choices.
"""
from dataclasses import asdict, replace
import json
import math
import subprocess
from pathlib import Path

from rsicomposer.algorithms.shared.harness_lab import FixedModelHarnessLab, validate_config as validate_measurement
from rsicomposer.api import (
    ArtifactRef,
    DEV, SEALED, Invariant, ExecutionStatus,
    HarnessExecutionError,
    LabToolError,
    OperationCall,
    OperationResult,
    OperationSpec,
    emit_observation,
    file_hash,
    report_version,
    report_version_evaluation,
    report_version_selection,
    sandbox_step,
    validate_source,
)


# Fields of a harness trial record shown to the research model.
_TRACE_FIELDS = ('question', 'completion', 'correct', 'error')
_OPTIONS = {'initial_harness', 'max_candidates', 'trials', 'final_trials', 'output_subdir'}


def validate_config(config):
    validate_measurement({k: v for k, v in config.items() if k not in _OPTIONS})
    if not isinstance(config.get('initial_harness'), str) or not config['initial_harness'].strip():
        raise ValueError('meta_agent math Lab requires initial_harness')
    for key, default, maximum in (('max_candidates', 8, 1000), ('trials', 2, 16), ('final_trials', 2, 16)):
        value = config.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('Invalid meta_agent ' + key)
    output = config.get('output_subdir', 'harness-math')
    if not isinstance(output, str) or Path(output).is_absolute() or '..' in Path(output).parts:
        raise ValueError('Invalid output_subdir')


class HarnessResearchLab:
    """Immutable harness candidates; the host owns split access and final evaluation."""

    def __init__(self, context, config, *, measurement_factory=None, sandbox_executor=None):
        validate_config(config)
        source = Path(config['initial_harness']).read_text()
        validate_source(source)
        self.config = dict(config)
        self.measurement = (measurement_factory or FixedModelHarnessLab)(context, {k: v for k, v in config.items() if k not in _OPTIONS})
        # A resumed run rebuilds the Lab over its own existing candidate directory.
        self.root = Path(context.data_dir) / config.get('output_subdir', 'harness-math') / 'candidates'
        self.root.mkdir(parents=True, exist_ok=bool(getattr(context, 'restoring', False)))
        path = self.root / 'baseline.py'
        if not path.exists():
            path.write_text(source)
        self.candidates = {'baseline': (path, file_hash(path))}
        self.measured = set()
        self.selected = None
        self.search_traces = {}
        self.rejections = 0
        self.executor = sandbox_executor or sandbox_step

    def state(self):
        files = lambda rows: {key: [str(path), digest] for key, (path, digest) in rows.items()}
        return {'candidates': files(self.candidates), 'measured': sorted(self.measured),
                'selected': self.selected,
                'search_traces': {key: asdict(ref) for key, ref in self.search_traces.items()}, 'rejections': self.rejections,
                'measurement': self.measurement.state()}

    def restore(self, state):
        rows = lambda saved: {key: (Path(path), digest) for key, (path, digest) in saved.items()}
        self.candidates = rows(state['candidates'])
        self.search_traces = {key: ArtifactRef(**ref) for key, ref in state['search_traces'].items()}
        for ref in self.search_traces.values():
            self.measurement.artifacts.read(ref)
        for path, digest in self.candidates.values():
            if file_hash(path) != digest:
                raise ValueError(f'Immutable meta_agent file changed while the run was stopped: {path}')
        self.measured, self.selected = set(state['measured']), state['selected']
        self.rejections = state['rejections']
        self.measurement.restore(state['measurement'])

    def profiles(self):
        return ('artifact-search/v1',)

    def primary_metric(self):
        return self.measurement.primary_metric()

    def split_roles(self):
        return {'search': DEV, 'heldout': SEALED}

    def invariants(self):
        return (Invariant.baseline_before('select'), Invariant.measured_selection())

    def final_evaluations(self):
        return tuple(OperationCall('evaluate', {'candidate_id': ident, 'split': 'heldout', 'indices': None})
            for ident in dict.fromkeys(('baseline', self.selected))) if self.selected is not None else ()

    def describe(self):
        def spec(name, description, properties, required):
            return OperationSpec(name, description, {'type': 'object', 'properties': properties,
                'required': required, 'additionalProperties': False})
        return (
            spec('inspect', 'Read immutable source; the initial candidate_id is baseline.',
                 {'candidate_id': {'type': 'string'}}, ['candidate_id']),
            spec('register', 'Register a candidate object with complete Python source and hypothesis. '
                 'Define step(problem, history); return messages to call the fixed model or answer to finish. '
                 'The driver calls the model. No filesystem/network/model/grader access from Python. '
                 'Use general mechanisms and boxed answers, without question literals or cached answers.',
                 {'candidate': {'type': 'object', 'properties': {'source': {'type': 'string'},
                    'hypothesis': {'type': 'string'}}, 'required': ['source', 'hypothesis']}}, ['candidate']),
            spec('evaluate', 'Measure a registered candidate on a named split, optionally selecting indices. '
                 'The host reserves heldout for after Controller finish.',
                 {'candidate_id': {'type': 'string'}, 'split': {'type': 'string'},
                  'indices': {'type': ['array', 'null']}}, ['candidate_id', 'split', 'indices']),
            spec('read_trace', 'Read a stored search trace excerpt; follow next_offset until eof.',
                 {'evaluation': {'type': 'string'}, 'index': {'type': 'integer', 'minimum': 0},
                  'offset': {'type': 'integer', 'minimum': 0}}, ['evaluation', 'index', 'offset']),
            spec('select', 'Record the selected measured candidate; then finish. '
                 'The host performs sealed final evaluation after finish.',
                 {'candidate_id': {'type': 'string'}}, ['candidate_id']),
        )

    def _candidate(self, ident):
        if ident not in self.candidates:
            raise LabToolError('Unknown candidate ID')
        path, expected = self.candidates[ident]
        if file_hash(path) != expected:
            raise ValueError('Immutable candidate changed')
        return path

    def _measure(self, ident, split, trials, indices):
        path = self._candidate(ident)
        registered = self.measurement.execute(OperationCall('register', {'candidate': {'harness': str(path)}}))
        result = self.measurement.execute(OperationCall('evaluate', {
            'candidate_id': registered.data['candidate_id'], 'indices': indices,
            'split': split, 'trials': trials}))
        score = result.data.get('score')
        if (result.status != ExecutionStatus.COMPLETED or result.simulated
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise RuntimeError('A completed, real measurement with a valid score is required')
        return result

    def execute(self, call):
        args = dict(call.arguments)
        specs = {spec.name: spec for spec in self.describe()}
        if call.name not in specs:
            raise LabToolError('Unknown meta_agent math operation')
        schema = specs[call.name].input_schema
        if set(schema['required']) - set(args) or set(args) - set(schema['properties']):
            raise LabToolError('Use the documented required arguments: ' + ', '.join(schema['required']))
        for name, value in args.items():
            expected = schema['properties'][name]['type']
            if ((expected == 'string' and (not isinstance(value, str) or not value.strip()))
                    or (expected == 'integer' and (type(value) is not int or value < 0))
                    or (expected == 'object' and not isinstance(value, dict))
                    or (expected == ['array', 'null'] and value is not None and not isinstance(value, list))):
                raise LabToolError('Invalid argument: ' + name)
        if call.name == 'read_trace':
            return self._read_trace(args)
        if call.name == 'register':
            if len(self.candidates) - 1 >= self.config.get('max_candidates', 8):
                raise LabToolError('Candidate budget exhausted; select an existing candidate')
            candidate = args['candidate']
            if set(candidate) != {'source', 'hypothesis'}:
                raise LabToolError('Candidate requires source and hypothesis')
            source, hypothesis = candidate['source'], candidate['hypothesis']
            if not isinstance(source, str) or not source.strip():
                raise LabToolError('Complete Python source is required')
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                raise LabToolError('A hypothesis is required')
            try:
                validate_source(source)
            except (ValueError, SyntaxError) as exc:
                raise LabToolError('Invalid candidate source: ' + str(exc)) from exc
            for rows in self.measurement.datasets.values():
                if any(len(row.prompt) > 20 and row.prompt in source for row in rows):
                    raise LabToolError('Question literals cannot be embedded in a harness')
            ident = f'candidate-{len(self.candidates)}'
            path = self.root / (ident + '.py')
            path.write_text(source)
            try:
                for history in ([], [{'content': r'Checked derivation. \boxed{17}', 'tokens': 5,
                                     'prompt_tokens': 2, 'completion_tokens': 3,
                                     'endpoint': 'local', 'finish_reason': 'stop'}]):
                    self.executor(path, 'Solve a new mathematical problem using general reasoning.', history)
            except (HarnessExecutionError, subprocess.TimeoutExpired) as exc:
                self.rejections += 1
                path.replace(self.root / f'rejected-{self.rejections}.py')
                raise LabToolError('Harness interface check failed: ' + str(exc)) from exc
            self.candidates[ident] = (path, file_hash(path))
            (self.root / (ident + '.json')).write_text(json.dumps({'hypothesis': hypothesis, 'sha256': file_hash(path)}))
            report_version(ident, label=ident)
            emit_observation('meta_agent.candidate', call.call_id, subject=ident, components={'source': source})
            return OperationResult('Candidate registered; no measurement or adoption yet', (), False,
                                   data={'candidate_id': ident, 'sha256': file_hash(path)})
        ident = args['candidate_id']
        path = self._candidate(ident)
        if call.name == 'inspect':
            return OperationResult('Immutable candidate source', (), False, data={
                'candidate_id': ident, 'source': path.read_text(), 'max_calls': self.config.get('max_calls', 3),
                'search_trials': self.config.get('trials', 2), 'max_candidates': self.config.get('max_candidates', 8)})
        if call.name == 'evaluate':
            split = args['split']
            trials = self.config.get('final_trials', 2) if split == 'heldout' else self.config.get('trials', 2)
            result = self._measure(ident, split, trials, args['indices'])
            data = dict(result.data, request=args, candidate_id=ident, baseline=ident == 'baseline')
            if result.status == ExecutionStatus.COMPLETED:
                if ident == 'baseline':
                    report_version(ident, label='Baseline', round_index=0, parent_ids=())
                report_version_evaluation(ident, call.call_id)
            if split == 'heldout':
                return replace(result, data=data)
            self.measured.add(ident)
            trace_ref = ArtifactRef(**data['traces'])
            evaluation = trace_ref.artifact_id
            self.search_traces[evaluation] = trace_ref
            # Aggregate feedback stays complete; large trial tables stay in the archive.
            data['evaluation'] = evaluation
            row = json.loads(self.measurement.artifacts.read(trace_ref))['records'][0]
            data['example_excerpt'] = {k: (row[k][:512] if isinstance(row[k], str) else row[k])
                                       for k in _TRACE_FIELDS}
            data['example_is_excerpt'] = True
            return replace(result, data=data)
        self.selected = ident
        report_version_selection(ident, step=len(self.candidates) - 1, event_id=call.call_id,
                                 action='kept' if ident == 'baseline' else 'selected')
        return OperationResult('Selection recorded; finish to request host final evaluation', (), False,
            data={'selection_recorded': True, 'selected_candidate': ident, 'candidate_id': ident})

    def _read_trace(self, args):
        evaluation, index, offset = args['evaluation'], args['index'], args['offset']
        if evaluation not in self.search_traces:
            raise LabToolError('Unknown search evaluation')
        if type(index) is not int or index < 0 or type(offset) is not int or offset < 0:
            raise LabToolError('Trace index and offset must be nonnegative integers')
        ref = self.search_traces[evaluation]
        expected = ref.digest
        records = json.loads(self.measurement.artifacts.read(ref))['records']
        if index >= len(records):
            raise LabToolError('Trace index is outside this evaluation')
        row = records[index]
        text = json.dumps({k: row[k] for k in _TRACE_FIELDS}, ensure_ascii=False)
        if offset > len(text):
            raise LabToolError('Trace offset is outside this record')
        end = min(len(text), offset + 2000)
        return OperationResult('Search trace page; follow next_offset until eof', (), False, data={
            'evaluation': evaluation, 'index': index, 'offset': offset, 'text': text[offset:end],
            'eof': end == len(text), 'next_offset': None if end == len(text) else end,
            'trace_sha256': expected})

    def runtime_context(self):
        return json.dumps({'candidates': list(self.candidates), 'search_measured': sorted(self.measured),
                           'selected': self.selected, 'sealed_evaluation': 'host_after_finish'})

    def close(self):
        close = getattr(self.measurement, 'close', None)
        if close is not None:
            close()
