"""Registered controller for the pinned, unmodified native AIDE Agent.

Agent.step's execution callback crosses the public Lab boundary. Search,
draft/debug/improve generation, feedback interpretation and final selection stay
inside upstream AIDE; the host never substitutes its own search policy.

Upstream (MIT): https://github.com/WecoAI/aideml/tree/ad73d227af481bea3de06f884f41b2993dd6453a
Loaded from a separate clean checkout; no upstream implementation is bundled.
"""
import math
import re
import sys
from copy import deepcopy
from pathlib import Path

from rsicomposer.api import (
    ArtifactRef, Decision, OperationCall, ResearchBridge, UpstreamAgentPolicy,
    checkout, file_hash as digest, valid_replay_point,
)

AIDE_REVISION = 'ad73d227af481bea3de06f884f41b2993dd6453a'


def validate_config(config):
    allowed = {'upstream_root', 'worker_python', 'task_file',
               'iterations', 'lab', 'seed', 'timeout', 'endpoint', 'model',
               'max_tokens', 'native_config', 'disable_thinking', 'output_subdir', 'execution_timeout'}
    if set(config) - allowed:
        raise ValueError('Unknown aide controller option')
    for key in ('upstream_root', 'task_file', 'endpoint', 'model'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('aide requires ' + key)
    for key, default in (('iterations', 3), ('timeout', 600), ('max_tokens', 8192), ('execution_timeout', 300)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError('Invalid aide ' + key)
    if type(config.get('seed', 73)) is not int:
        raise ValueError('Invalid aide seed')
    if not config['endpoint'].startswith(('http://', 'https://')):
        raise ValueError('aide endpoint must be an explicit OpenAI API URL')
    if 'worker_python' in config and (not isinstance(config['worker_python'], str) or not config['worker_python']):
        raise ValueError('Invalid aide worker_python')
    if type(config.get('disable_thinking', True)) is not bool:
        raise ValueError('Invalid aide disable_thinking')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', config.get('lab', 'aide')):
        raise ValueError('Invalid aide Lab alias')
    output = config.get('output_subdir', 'aide-research')
    if not isinstance(output, str) or not output or Path(output).is_absolute() or '..' in Path(output).parts:
        raise ValueError('Invalid aide output_subdir')
    native = config.get('native_config', {})
    if not isinstance(native, dict) or set(native) - {'num_drafts', 'debug_prob', 'max_debug_depth',
            'k_fold_validation', 'data_preview', 'temperature', 'metric_maximize'}:
        raise ValueError('Invalid aide native_config')
    for key in ('num_drafts', 'max_debug_depth', 'k_fold_validation'):
        if key in native and (type(native[key]) is not int or native[key] < 1):
            raise ValueError('Invalid native AIDE ' + key)
    for key, high in (('debug_prob', 1), ('temperature', 2)):
        if key in native and (type(native[key]) not in (int, float) or not math.isfinite(native[key])
                              or not 0 <= native[key] <= high):
            raise ValueError('Invalid native AIDE ' + key)
    if 'data_preview' in native and type(native['data_preview']) is not bool:
        raise ValueError('Invalid native AIDE data_preview')
    if native.get('metric_maximize') is not None and type(native['metric_maximize']) is not bool:
        raise ValueError('Invalid native AIDE metric_maximize')


class AIDEController(UpstreamAgentPolicy):
    """Method hooks for native AIDE preview, callbacks and sealed selection."""
    validate_config = staticmethod(validate_config)

    def __init__(self, context, config, *, bridge_factory=ResearchBridge):
        validate_config(config)
        checkout(config['upstream_root'], AIDE_REVISION)
        self.artifacts = context.artifacts
        self.lab = config.get('lab', 'aide')
        self.phase, self.preview_files, self.selection = 'preview', None, None
        self.native_report = {}
        super().__init__(context, config,
            root=Path(context.data_dir) / config.get('output_subdir', 'aide-research'),
            worker_module='rsicomposer.algorithms.aide.worker',
            worker_python=config.get('worker_python', sys.executable), researcher=config,
            timeout=config.get('timeout', 600), bridge_factory=bridge_factory,
            resume_validator=valid_resume)

    def method_state(self):
        return {name: deepcopy(getattr(self, name)) for name in
                ('phase', 'preview_files', 'selection', 'native_report')}

    def restore_method_state(self, state):
        expected = {'phase', 'preview_files', 'selection', 'native_report'}
        if set(state) != expected or state['phase'] not in ('preview', 'search', 'select'):
            raise ValueError('Unexpected AIDE method state')
        if state['phase'] != 'preview' and (not isinstance(state['preview_files'], dict)
                or set(state['preview_files']) != {'train.csv', 'validation.csv'}):
            raise ValueError('AIDE search requires Lab-provided preview artifacts')
        for name in expected:
            setattr(self, name, deepcopy(state[name]))

    def required_operations(self):
        return tuple(self.lab + '.' + name for name in ('preview', 'execute', 'select'))

    def required_profiles(self):
        return {self.lab: 'program-execution/v1'}

    def before_upstream(self, window):
        if self.phase == 'preview':
            if self.pending is None:
                return self.invoke(self.lab + '.preview', {}, reason='Native AIDE public preview')
            data = self.consume(window)
            files = data.get('files')
            if not isinstance(files, dict) or set(files) != {'train.csv', 'validation.csv'}:
                raise ValueError('AIDE preview must contain exactly train.csv and validation.csv')
            for raw in files.values():
                ref = ArtifactRef(**raw)
                if ref.kind != 'aide-public-input':
                    raise ValueError('Unexpected AIDE public-input artifact kind')
                self.artifacts.read(ref)
            self.preview_files, self.phase = deepcopy(files), 'search'
            self.start_worker({**self.config, 'root': str(self.root.resolve()),
                'upstream_root': str(Path(self.config['upstream_root']).resolve()),
                'task_file': str(Path(self.config['task_file']).resolve()),
                'preview_files': {name: self.artifacts.read(ArtifactRef(**ref)).decode('utf-8')
                                  for name, ref in files.items()}})
        elif self.phase == 'select':
            self.consume(window)
            self.stopped = 'Native AIDE search and selection completed; host sealed evaluation follows'
            self.close()
            return Decision('finish', reason=self.stopped)
        return None

    def _program(self, item):
        candidate = Path(item['candidate'])
        if candidate.is_absolute() or '..' in candidate.parts:
            raise ValueError('AIDE candidates must be run-relative paths')
        path = (self.root / candidate).resolve(strict=True)
        if not path.is_relative_to(self.root.resolve()) or digest(path) != item['sha256']:
            raise ValueError('Invalid AIDE candidate identity')
        return path.read_text()

    def event_operation(self, event):
        if not valid_resume(event.get('resume')) or event['resume']['snapshot'] != event['step']:
            raise ValueError('AIDE execute event lacks a valid resume point')
        return OperationCall(self.lab + '.execute', {'program': self._program(event),
            'split': 'validation', 'step': event['step']})

    def receipt_message(self, event, data):
        return {'kind': 'execution_result', 'execution_result': {
            'term_out': [data['term_out']], 'exec_time': data['exec_time'],
            'exc_type': data['exc_type'], 'exc_info': data.get('exc_info'),
            'exc_stack': data.get('exc_stack')}}

    def finished_event(self, event):
        self.native_report, self.phase = deepcopy(event['report']), 'select'
        evaluations = event['evaluations']
        if evaluations:
            if len(evaluations) != 2 or {row['label'] for row in evaluations} != {'initial', 'selected'}:
                raise ValueError('AIDE worker must record initial and selected programs')
            self.selection = {row['label']: self._program(row) for row in evaluations}
            return self.invoke(self.lab + '.select', self.selection, reason='Native AIDE selection')
        self.stopped = 'Native AIDE found no valid solution'
        return Decision('finish', reason=self.stopped)

    def report(self):
        return {'method': 'aide', 'status': 'failed' if self.failure else 'completed' if self.stopped and self.selection else
                ('no_valid_candidate' if self.stopped else 'running'),
                'upstream_revision': AIDE_REVISION, 'native': deepcopy(self.native_report),
                'selected': bool(self.selection), 'sealed_evaluation': 'host_after_finish', 'training_performed': False,
                'simulated': False,
                'scope': 'GSM8K LLM inference-program task adaptation; not original benchmark reproduction'}


def valid_resume(point):
    """Native AIDE step snapshot plus the common replay-log position."""
    return (isinstance(point, dict) and set(point) == {'snapshot', 'snapshot_sha256', 'replies'}
            and type(point['snapshot']) is int and point['snapshot'] >= 0
            and isinstance(point['snapshot_sha256'], str) and len(point['snapshot_sha256']) == 64
            and valid_replay_point(point['replies']))


def build_aide(context, config):
    return AIDEController(context, config)
