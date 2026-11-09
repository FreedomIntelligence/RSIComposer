"""Registered controller for the pinned, unmodified native AIDE Agent.

Agent.step's execution callback crosses the public Lab boundary. Search,
draft/debug/improve generation, feedback interpretation and final selection stay
inside upstream AIDE; the host never substitutes its own search policy.

Upstream (MIT): https://github.com/WecoAI/aideml/tree/ad73d227af481bea3de06f884f41b2993dd6453a
Loaded from a separate clean checkout; no upstream implementation is bundled.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys

from .research_bridge import ResearchBridge
from ..labs.harness_runtime import digest
from ..labs.pinned_upstream import checkout
from ..loops.contracts.decision import Decision, OperationCall
from ..loops.contracts.lifecycle import ExecutionStatus

AIDE_REVISION = 'ad73d227af481bea3de06f884f41b2993dd6453a'


def validate_config(config):
    allowed = {'upstream_root', 'worker_python', 'task_file', 'input_dir',
               'iterations', 'lab', 'seed', 'timeout', 'endpoint', 'model',
               'max_tokens', 'native_config', 'disable_thinking', 'output_subdir', 'execution_timeout'}
    if set(config) - allowed:
        raise ValueError('Unknown aide controller option')
    for key in ('upstream_root', 'task_file', 'input_dir', 'endpoint', 'model'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('aide requires ' + key)
    for key, default in (('iterations', 3), ('timeout', 600), ('max_tokens', 8192), ('execution_timeout', 300)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError('Invalid aide ' + key)
    if type(config.get('seed', 73)) is not int:
        raise ValueError('Invalid aide seed')
    if not config['endpoint'].startswith(('http://', 'https://')):
        raise ValueError('aide endpoint must be an explicit OpenAI-compatible URL')
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


class AIDEController:
    validate_config = staticmethod(validate_config)

    def __init__(self, context, config, *, bridge_factory=ResearchBridge):
        validate_config(config)
        checkout(config['upstream_root'], AIDE_REVISION)
        self.config = deepcopy(config)
        self.root = Path(context.data_dir) / config.get('output_subdir', 'aide-research')
        self.root.mkdir(parents=True, exist_ok=False)
        self.lab = config.get('lab', 'aide')
        self.bridge_factory = bridge_factory
        self.bridge = None
        self.pending = None
        self.phase = 'search'
        self.final_queue = []
        self.results = {}
        self.native_report = {}
        self.stopped = None
        self.failure = None

    def required_operations(self):
        return (self.lab + '.execute', self.lab + '.evaluate')

    def _start(self):
        cfg = {**self.config, 'root': str(self.root.resolve()),
               'upstream_root': str(Path(self.config['upstream_root']).resolve()),
               'task_file': str(Path(self.config['task_file']).resolve()),
               'input_dir': str(Path(self.config['input_dir']).resolve())}
        request = self.root / 'request.json'
        request.write_text(json.dumps(cfg, indent=2))
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                   PYTHONUNBUFFERED='1', OPENAI_BASE_URL=cfg['endpoint'], OPENAI_API_KEY='local-unused')
        self.bridge = self.bridge_factory([cfg.get('worker_python', sys.executable), '-m',
            'rsisys.controllers.aide_worker', str(request)], log=self.root / 'worker.log', env=env)

    def _invoke(self, name, args):
        path = Path(args['candidate']).resolve(strict=True)
        if not path.is_relative_to(self.root.resolve()) or digest(path) != args['sha256']:
            raise ValueError('Invalid AIDE candidate identity')
        self.pending = OperationCall(self.lab + '.' + name, args)
        return Decision('invoke', operation=deepcopy(self.pending),
                        reason='Native AIDE ' + name)

    def decide(self, window):
        if self.stopped is not None:
            return Decision('finish', reason=self.stopped)
        if self.bridge is None:
            self._start()
        if self.pending is not None:
            if window.last_operation != self.pending or window.feedback is None:
                raise RuntimeError('AIDE requires its matching terminal Lab receipt')
            feedback = window.feedback
            if feedback.data.get('request') != self.pending.arguments:
                raise ValueError('AIDE receipt request mismatch')
            if feedback.status != ExecutionStatus.COMPLETED or feedback.simulated:
                self.stopped = ('AIDE rejected a simulated Lab receipt' if feedback.simulated else
                                'AIDE infrastructure failed: ' + str(feedback.error))
                self.failure = self.stopped
                return Decision('finish', reason=self.stopped)
            if self.phase == 'search':
                self.bridge.send({'kind': 'execution_result', 'execution_result': feedback.data['execution_result']})
            else:
                self.results[self.pending.arguments['label']] = deepcopy(feedback.data)
            self.pending = None
        if self.phase == 'search':
            event = self.bridge.receive(self.config.get('timeout', 600))
            if event['kind'] == 'execute':
                path = Path(event['candidate']).resolve(strict=True)
                if not path.is_relative_to(self.root.resolve()) or digest(path) != event['sha256']:
                    raise ValueError('Invalid AIDE candidate identity')
                return self._invoke('execute', {k: event[k] for k in ('candidate', 'sha256', 'step')})
            if event['kind'] != 'finished':
                raise ValueError('Unexpected AIDE worker event')
            self.native_report = event['report']
            self.final_queue = deepcopy(event['evaluations'])
            self.phase = 'final'
        if self.final_queue:
            item = self.final_queue.pop(0)
            return self._invoke('evaluate', item)
        if self.results and not self.results.get('selected', {}).get('quality_metrics'):
            self.failure = 'Selected AIDE candidate failed final heldout execution'
        self.stopped = (self.failure or ('Native AIDE search and final heldout evaluation completed'
                        if self.results else 'Native AIDE found no valid solution'))
        return Decision('finish', reason=self.stopped)

    def report(self):
        return {'method': 'aide', 'status': 'failed' if self.failure else 'completed' if self.stopped and self.results else
                ('no_valid_candidate' if self.stopped else 'running'),
                'upstream_revision': AIDE_REVISION, 'native': deepcopy(self.native_report),
                'evaluations': deepcopy(self.results), 'training_performed': False,
                'simulated': False,
                'scope': 'GSM8K LLM inference-program task adaptation; not original benchmark reproduction'}

    def close(self):
        if self.bridge is not None:
            self.bridge.close()


def build_aide(context, config):
    return AIDEController(context, config)
