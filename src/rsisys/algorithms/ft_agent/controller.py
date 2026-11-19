"""Registered FT-Agent reference adapter using pinned upstream policy code.
Upstream (MIT): https://github.com/microsoft/RD-Agent/tree/484776c211e4fbbeef03e0ec00d6bbee7362a4f4
Loaded from a separate clean checkout; no upstream implementation is bundled.
"""
from copy import deepcopy
from pathlib import Path
import re
import sys

from rsisys.api import (Decision, OperationCall, ResearchBridge, UpstreamAgentPolicy, valid_replay_point)
from rsisys.algorithms.ft_agent.upstream import FT_AGENT_REVISION


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


class FTAgentController(UpstreamAgentPolicy):
    """Method hooks for the native FT loop's baseline and callback payloads."""
    def __init__(self, context, config, *, bridge_factory=ResearchBridge):
        validate_config(config)
        self.lab = config.get('lab', 'finetune')
        self.stage, self.result = 'baseline', None
        super().__init__(context, config,
            root=Path(context.data_dir) / config.get('output_subdir', 'ft-agent'),
            worker_module='rsisys.algorithms.ft_agent.worker',
            worker_python=config.get('python', sys.executable), researcher=config['researcher'],
            timeout=config.get('timeout', 3600), bridge_factory=bridge_factory)

    def method_state(self):
        return {'stage': self.stage, 'result': deepcopy(self.result)}

    def restore_method_state(self, state):
        if set(state) != {'stage', 'result'} or state['stage'] not in ('baseline', 'research', 'selection', 'finished'):
            raise ValueError('Unexpected FT-Agent method state')
        self.stage, self.result = state['stage'], deepcopy(state['result'])

    def required_operations(self):
        return tuple(self.lab + '.' + name for name in ('baseline', 'prepare', 'train', 'select'))

    def before_upstream(self, window):
        if self.stage == 'baseline':
            if self.pending is None:
                return self.invoke(self.lab + '.baseline', {})
            baseline = self.consume(window)
            if not baseline.get('ok'):
                raise RuntimeError('FT-Agent baseline failed: ' + str(baseline.get('error')))
            config = {**self.config, **baseline['policy_context'], 'baseline': baseline['benchmark'],
                'directory': str(self.root.resolve()), 'iterations': self.config.get('iterations', 2)}
            self.stage = 'research'
            self.start_worker(config)
        elif self.stage == 'selection':
            selection = self.consume(window)
            if not selection.get('ok'):
                raise RuntimeError('FT-Agent selection failed: ' + str(selection.get('error')))
            self.result['selection'], self.stage = selection, 'finished'
            self.close()
        if self.stage == 'finished':
            return Decision('finish', reason='FT-Agent research budget completed')
        return None

    def event_operation(self, event):
        if event.get('stage') not in ('prepare', 'train'):
            raise ValueError('Unknown FT-Agent execution stage')
        if not valid_replay_point(event.get('resume')):
            raise ValueError('FT-Agent execute event lacks a valid resume point')
        return OperationCall(self.lab + '.' + event['stage'],
            {'iteration': event['iteration'], 'files': event['files']})

    def receipt_message(self, event, data):
        return {'kind': 'feedback', 'stage': event['stage'], 'iteration': event['iteration'], 'data': data}

    def finished_event(self, event):
        self.result, self.stage = deepcopy(event), 'selection'
        return self.invoke(self.lab + '.select', {'iteration': event['selected_iteration']})

    def report(self):
        return {'method': 'ft_agent', 'upstream_revision': FT_AGENT_REVISION,
                'profile': 'local-reference', 'status': 'failed' if self.failure else 'completed' if self.stage == 'finished' else self.stage,
                'training_performed': bool((self.result or {}).get('selection', {}).get('training_performed')),
                'simulated': False,
                'paper_gain_reproduced': False, 'sealed_evaluation': 'host_after_finish', 'result': deepcopy(self.result)}
