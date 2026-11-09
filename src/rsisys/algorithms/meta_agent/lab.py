"""Bounded harness research Lab for the model-driven meta-agent method.

Share the fixed-model measurement engine with RRSI, without its proposal,
critic, edit-budget or selection algorithm. The meta-agent owns those choices.
"""
from dataclasses import replace
import json
import math
import subprocess
from pathlib import Path

from rsisys.algorithms.shared.harness_lab import FixedModelHarnessLab, validate_config as validate_measurement
from rsisys.harnesses.python_step import HarnessExecutionError, sandbox_step, validate_source
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.storage.files import file_hash


_OPTIONS = {'initial_harness', 'max_candidates', 'trials', 'final_trials'}


def validate_config(config):
    validate_measurement({k: v for k, v in config.items() if k not in _OPTIONS})
    if not isinstance(config.get('initial_harness'), str) or not config['initial_harness'].strip():
        raise ValueError('meta_agent math Lab requires initial_harness')
    for key, default, maximum in (('max_candidates', 8, 1000), ('trials', 2, 16), ('final_trials', 2, 16)):
        value = config.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('Invalid meta_agent ' + key)


class HarnessResearchLab:
    """Immutable candidate IDs; heldout is opened only after selection is sealed."""

    def __init__(self, context, config, *, measurement_factory=None, sandbox_executor=None):
        validate_config(config)
        source = Path(config['initial_harness']).read_text()
        validate_source(source)
        self.config = dict(config)
        self.measurement = (measurement_factory or FixedModelHarnessLab)(context, {k: v for k, v in config.items() if k not in _OPTIONS})
        self.root = self.measurement.root / 'candidates'
        self.root.mkdir()
        path = self.root / 'baseline.py'
        path.write_text(source)
        self.candidates = {'baseline': (path, file_hash(path))}
        self.measured = set()
        self.selected = None
        self.final_result = None
        self.final_measurements = {}
        self.search_traces = {}
        self.rejections = 0
        self.executor = sandbox_executor or sandbox_step

    def describe(self):
        def spec(name, description, properties, required):
            return OperationSpec(name, description, {'type': 'object', 'properties': properties,
                'required': required, 'additionalProperties': False})
        return (
            spec('inspect', 'Read immutable candidate source. Initial candidate ID is baseline.',
                 {'candidate': {'type': 'string'}}, ['candidate']),
            spec('propose', 'Register a complete Python step(problem, history) harness. No weights change. '
                 'Return messages to call the fixed model or answer to finish; no filesystem/network access. '
                 'History contains content and token counts. At most the installed max_calls model calls per problem. '
                 'The driver calls the model; never call a model or grader from Python. Define all helpers yourself. '
                 'Use general mechanisms and boxed math answers; no question literals or cached answers.',
                 {'source': {'type': 'string'}, 'hypothesis': {'type': 'string', 'minLength': 1}}, ['source', 'hypothesis']),
            spec('evaluate', 'Measure one candidate on frozen search data; feedback includes a bounded trace sample.',
                 {'candidate': {'type': 'string'}}, ['candidate']),
            spec('read_trace', 'Read a search-trace excerpt. index identifies a trial; offset counts characters. '
                 'Follow next_offset until eof. Heldout traces are not available during search.',
                 {'evaluation': {'type': 'string'}, 'index': {'type': 'integer', 'minimum': 0},
                  'offset': {'type': 'integer', 'minimum': 0}}, ['evaluation', 'index', 'offset']),
            spec('finalize', 'Seal selection using search evidence. Evaluate baseline and selected candidate on '
                 'heldout with equal budgets. Selection cannot change and no further search is permitted. '
                 'Returns measurements; improvement is not guaranteed.',
                 {'candidate': {'type': 'string'}}, ['candidate']),
        )

    def _candidate(self, ident):
        if ident not in self.candidates:
            raise LabToolError('Unknown candidate ID')
        path, expected = self.candidates[ident]
        if file_hash(path) != expected:
            raise ValueError('Immutable candidate changed')
        return path

    def _measure(self, ident, split, trials):
        path = self._candidate(ident)
        result = self.measurement.execute(OperationCall('evaluate', {
            'harness': str(path), 'dataset': self.config[split + '_data'],
            'split': split, 'trials': trials, 'round': len(self.measured)}))
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
                    or (expected == 'integer' and (type(value) is not int or value < 0))):
                raise LabToolError('Invalid argument: ' + name)
        if self.selected is not None and call.name != 'finalize':
            raise LabToolError('Selection is sealed; further search is closed')
        if call.name == 'read_trace':
            return self._read_trace(args)
        if call.name == 'propose':
            if len(self.candidates) - 1 >= self.config.get('max_candidates', 8):
                raise LabToolError('Candidate budget exhausted; select an existing candidate')
            source, hypothesis = args['source'], args['hypothesis']
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                raise LabToolError('A hypothesis is required')
            try:
                validate_source(source)
            except (ValueError, SyntaxError) as exc:
                raise LabToolError('Invalid candidate source: ' + str(exc)) from exc
            for rows in self.measurement.datasets.values():
                if any(len(row['prompt']) > 20 and row['prompt'] in source for row in rows):
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
            return OperationResult('Candidate registered; no measurement or adoption yet', (), False,
                                   data={'candidate': ident, 'sha256': file_hash(path)})
        ident = args['candidate']
        path = self._candidate(ident)
        if call.name == 'inspect':
            return OperationResult('Immutable candidate source', (), False, data={
                'candidate': ident, 'source': path.read_text(), 'max_calls': self.config.get('max_calls', 3),
                'search_trials': self.config.get('trials', 2), 'max_candidates': self.config.get('max_candidates', 8)})
        if call.name == 'evaluate':
            result = self._measure(ident, 'search', self.config.get('trials', 2))
            self.measured.add(ident)
            data = dict(result.data, candidate=ident)
            trace_path = Path(data['traces'])
            evaluation = trace_path.stem
            self.search_traces[evaluation] = (trace_path, file_hash(trace_path))
            # Aggregate feedback stays complete; large trial tables stay in the archive.
            data.pop('rewards', None)
            data['evaluation'] = evaluation
            row = json.loads(trace_path.read_text().splitlines()[0])
            data['example_excerpt'] = {k: (row[k][:512] if isinstance(row[k], str) else row[k])
                                       for k in ('prompt', 'completion', 'correct', 'error')}
            data['example_is_excerpt'] = True
            return replace(result, data=data)
        if self.selected is not None and ident != self.selected:
            raise LabToolError('Final candidate cannot change after heldout access')
        if not {'baseline', ident} <= self.measured:
            raise LabToolError('Measure baseline and selected candidate on search data before finalization')
        self.selected = ident
        if self.final_result is None:
            trials = self.config.get('final_trials', 2)
            for candidate in dict.fromkeys(('baseline', ident)):
                if candidate not in self.final_measurements:
                    self.final_measurements[candidate] = self._measure(candidate, 'heldout', trials)
            baseline = self.final_measurements['baseline']
            selected = self.final_measurements[ident]
            data = {'selected_candidate': ident, 'baseline': baseline.data, 'selected': selected.data,
                    'score_delta': selected.data['score'] - baseline.data['score'],
                    'training_performed': False, 'selection_sealed': True,
                    'self_reflection_implemented': False}
            (self.root.parent / 'final-selection.json').write_text(json.dumps(data, indent=2))
            self.final_result = OperationResult('Sealed selection measured on heldout', (), False, data=data)
        return self.final_result

    def _read_trace(self, args):
        evaluation, index, offset = args['evaluation'], args['index'], args['offset']
        if evaluation not in self.search_traces:
            raise LabToolError('Unknown search evaluation')
        if type(index) is not int or index < 0 or type(offset) is not int or offset < 0:
            raise LabToolError('Trace index and offset must be nonnegative integers')
        path, expected = self.search_traces[evaluation]
        if file_hash(path) != expected:
            raise ValueError('Immutable search trace changed')
        lines = path.read_text().splitlines()
        if index >= len(lines):
            raise LabToolError('Trace index is outside this evaluation')
        row = json.loads(lines[index])
        text = json.dumps({k: row[k] for k in ('prompt', 'completion', 'correct', 'error')}, ensure_ascii=False)
        if offset > len(text):
            raise LabToolError('Trace offset is outside this record')
        end = min(len(text), offset + 2000)
        return OperationResult('Search trace page; follow next_offset until eof', (), False, data={
            'evaluation': evaluation, 'index': index, 'offset': offset, 'text': text[offset:end],
            'eof': end == len(text), 'next_offset': None if end == len(text) else end,
            'trace_sha256': expected})

    def runtime_context(self):
        return json.dumps({'candidates': list(self.candidates), 'search_measured': sorted(self.measured),
                           'selected': self.selected, 'final_evaluation_complete': self.final_result is not None})

    def close(self):
        close = getattr(self.measurement, 'close', None)
        if close is not None:
            close()
