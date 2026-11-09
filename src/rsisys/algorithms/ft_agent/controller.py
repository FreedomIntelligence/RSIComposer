"""Registered FT-Agent reference adapter using pinned upstream policy code.
Upstream (MIT): https://github.com/microsoft/RD-Agent/tree/484776c211e4fbbeef03e0ec00d6bbee7362a4f4
Loaded from a separate clean checkout; no upstream implementation is bundled.
"""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

from rsisys.loops.contracts.decision import Decision, OperationCall
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.algorithms.ft_agent.upstream import FT_AGENT_REVISION
from rsisys.runtime.execution.research_bridge import ResearchBridge
from rsisys.runtime.execution.workers import worker_environment


def validate_config(config):
    allowed = {'upstream_root', 'python', 'iterations', 'lab', 'researcher', 'timeout', 'output_subdir'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Unknown ft_agent controller option')
    if not isinstance(config.get('upstream_root'), str) or not config['upstream_root'].strip():
        raise ValueError('ft_agent requires the pinned upstream_root')
    for key, default in (('iterations', 2), ('timeout', 3600)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(key + ' must be a positive integer')
    for key, default in (('lab', 'finetune'), ('output_subdir', 'ft-agent')):
        if not isinstance(config.get(key, default), str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', config.get(key, default)):
            raise ValueError('Invalid ' + key)
    researcher = config.get('researcher')
    if not isinstance(researcher, dict) or not researcher.get('endpoints') or not researcher.get('model'):
        raise ValueError('ft_agent requires researcher model and endpoints')
    if 'python' in config and (not isinstance(config['python'], str) or not config['python'].strip()):
        raise ValueError('python must be an interpreter path')


class FTAgentController:
    def __init__(self, context, config, *, bridge_factory=ResearchBridge):
        validate_config(config)
        self.config = deepcopy(config)
        self.root = Path(context.data_dir) / config.get('output_subdir', 'ft-agent')
        self.root.mkdir(parents=True, exist_ok=False)
        self.bridge_factory, self.bridge = bridge_factory, None
        self.pending, self.event, self.result = None, None, None
        self.stage = 'baseline'

    def required_operations(self):
        alias = self.config.get('lab', 'finetune')
        return tuple(alias + '.' + name for name in ('baseline', 'prepare', 'train', 'final'))

    def _invoke(self, name, arguments):
        self.pending = OperationCall(self.config.get('lab', 'finetune') + '.' + name, arguments)
        return Decision('invoke', operation=deepcopy(self.pending))

    def _consume(self, window):
        if window.last_operation != self.pending or window.feedback is None:
            raise RuntimeError('FT-Agent requires matching terminal Lab feedback')
        result = window.feedback
        if result.status != ExecutionStatus.COMPLETED or result.simulated:
            raise RuntimeError('FT-Agent requires a real terminal Lab receipt: ' + str(result.error))
        if result.data.get('request') != self.pending.arguments:
            raise ValueError('FT-Agent Lab receipt request mismatch')
        if self.stage in ('baseline', 'final') and not result.data.get('ok'):
            raise RuntimeError('FT-Agent ' + self.stage + ' failed: ' + str(result.data.get('error')))
        self.pending = None
        return deepcopy(result.data)

    def decide(self, window):
        if self.stage == 'baseline' and self.pending is None:
            return self._invoke('baseline', {})
        if self.stage == 'baseline':
            baseline = self._consume(window)
            config = {**self.config, **baseline['policy_context'], 'baseline': baseline['benchmark'],
                      'directory': str(self.root), 'iterations': self.config.get('iterations', 2)}
            request = self.root / 'worker-request.json'
            request.write_text(json.dumps(config, indent=2))
            env = worker_environment(
                       CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
            self.bridge = self.bridge_factory([self.config.get('python', sys.executable), '-m',
                'rsisys.algorithms.ft_agent.worker', str(request)], log=self.root / 'worker.log', env=env)
            self.stage = 'research'
        elif self.stage == 'research' and self.pending is not None:
            data = self._consume(window)
            self.bridge.send({'kind': 'feedback', 'stage': self.event['stage'],
                              'iteration': self.event['iteration'], 'data': data})
        elif self.stage == 'final':
            final = self._consume(window)
            self.result['final'] = final
            self.stage = 'finished'
            self.close()
        if self.stage == 'finished':
            return Decision('finish', reason='FT-Agent research budget completed')
        self.event = self.bridge.receive(self.config.get('timeout', 3600))
        if self.event['kind'] == 'execute':
            if self.event.get('stage') not in ('prepare', 'train'):
                raise ValueError('Unknown FT-Agent execution stage')
            return self._invoke(self.event['stage'], {'iteration': self.event['iteration'], 'files': self.event['files']})
        if self.event['kind'] != 'finished':
            raise ValueError('Unexpected FT-Agent worker event')
        self.result = self.event
        self.stage = 'final'
        return self._invoke('final', {'iteration': self.result['selected_iteration']})

    def report(self):
        return {'method': 'ft_agent', 'upstream_revision': FT_AGENT_REVISION,
                'profile': 'local-reference', 'status': 'completed' if self.stage == 'finished' else self.stage,
                'training_performed': bool((self.result or {}).get('final', {}).get('training_performed')),
                'simulated': False,
                'paper_gain_reproduced': False, 'result': deepcopy(self.result)}

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
