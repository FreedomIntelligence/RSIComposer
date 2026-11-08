"""ReST^EM scheduling and receipt contract (Singh et al., arXiv:2312.06585).

Generate with the current model; filter binary successes; fit the INITIAL model
on this round's accepted data. Evaluation observes outcomes, not an acceptance
gate. Model/data references must identify immutable artifacts. No backend imports.
"""
from copy import deepcopy
import math
import re

from ..loops.contracts.decision import Decision, OperationCall
from ..loops.contracts.lifecycle import ExecutionStatus


def _ref(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError(f'{label} must be a nonempty reference of at most 1024 characters')
    return value


class RestEMController:
    """Synchronous terminal-receipt protocol; no hidden meta-agent or replay."""

    def __init__(self, *, initial_model, train_data, eval_data, lab='math',
                 iterations=2, samples_per_prompt=4, max_accepted_per_prompt=4):
        self.validate_config(dict(initial_model=initial_model, train_data=train_data,
            eval_data=eval_data, lab=lab, iterations=iterations,
            samples_per_prompt=samples_per_prompt,
            max_accepted_per_prompt=max_accepted_per_prompt))
        self.initial_model = self.current_model = initial_model
        self.train_data, self.eval_data, self.lab = train_data, eval_data, lab
        self.iterations, self.samples_per_prompt = iterations, samples_per_prompt
        self.max_accepted_per_prompt = max_accepted_per_prompt
        self.round = 0
        self.pending = None
        self.stage = 'evaluate'
        self.generated = self.accepted = self.candidate = None
        self.generated_count = 0
        self.history = []
        self.stopped = None

    @staticmethod
    def validate_config(config):
        allowed = {'initial_model', 'train_data', 'eval_data', 'lab', 'iterations',
                   'samples_per_prompt', 'max_accepted_per_prompt'}
        if set(config) - allowed:
            raise ValueError('Unknown rest_em option')
        for key in ('initial_model', 'train_data', 'eval_data'):
            _ref(config.get(key), key)
        if config['train_data'] == config['eval_data']:
            raise ValueError('ReST^EM requires separate training and held-out evaluation data')
        if not isinstance(config.get('lab', 'math'), str) or not re.fullmatch(
                r'[A-Za-z][A-Za-z0-9_-]*', config.get('lab', 'math')):
            raise ValueError('lab must be a registered Lab alias')
        for key, default in (('iterations', 2), ('samples_per_prompt', 4),
                             ('max_accepted_per_prompt', 4)):
            if type(config.get(key, default)) is not int or config.get(key, default) < 1:
                raise ValueError(f'{key} must be a positive integer')

    def _call(self):
        args = {'round': self.round}
        if self.stage == 'generate':
            args.update(model=self.current_model, dataset=self.train_data,
                        samples_per_prompt=self.samples_per_prompt)
        elif self.stage == 'filter':
            args.update(samples=self.generated, max_accepted_per_prompt=self.max_accepted_per_prompt)
        elif self.stage == 'train':
            args.update(initial_model=self.initial_model, dataset=self.accepted)
        else:
            args.update(model=self.candidate or self.current_model, dataset=self.eval_data)
        return OperationCall(f'{self.lab}.{self.stage}', args)

    def _consume(self, feedback):
        data, args = feedback.data, self.pending.arguments
        # Every successful receipt echoes the full small request. Large sample
        # tables remain outside the decision window, behind artifact references.
        if data.get('request') != args:
            raise ValueError('ReST^EM receipt request does not match the pending operation')
        if self.stage in ('generate', 'filter'):
            count = data.get('count')
            if type(count) is not int or count < 0:
                raise ValueError('Sample count must be a nonnegative integer')
            ref = _ref(data.get('dataset'), 'dataset')
            if self.stage == 'generate':
                self.generated, self.generated_count = ref, count
                self.stage = 'filter'
            else:
                if count > self.generated_count:
                    raise ValueError('Accepted count exceeds generated count')
                self.accepted = ref
                if count == 0:
                    self.stopped = f'No accepted samples in round {self.round}; no training performed for this round'
                self.stage = 'train'
        elif self.stage == 'train':
            candidate = _ref(data.get('model'), 'model')
            if candidate in {self.initial_model, self.current_model, *(r['model'] for r in self.history)}:
                raise ValueError('Training must publish a distinct immutable model reference')
            self.candidate = candidate
            self.stage = 'evaluate'
        else:
            metrics = data.get('quality_metrics')
            if not isinstance(metrics, dict) or not metrics or any(
                    not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v)
                    for k, v in metrics.items()):
                raise ValueError('Evaluation requires finite measured quality_metrics')
            self.history.append({'round': self.round, 'model': args['model'],
                                 'quality_metrics': deepcopy(metrics), 'simulated': feedback.simulated})
            if self.candidate is not None:
                self.current_model, self.candidate = self.candidate, None
            if self.round >= self.iterations:
                self.stopped = 'Configured ReST^EM rounds completed; quality change is recorded in evaluation receipts'
            else:
                self.round += 1
                self.stage = 'generate'

    def decide(self, window):
        if self.stopped is not None:
            return Decision('finish', reason=self.stopped)
        if self.pending is not None:
            if window.feedback is None or window.last_operation != self.pending:
                raise RuntimeError('ReST^EM requires the matching terminal receipt before advancing')
            if window.feedback.status != ExecutionStatus.COMPLETED:
                self.stopped = f'ReST^EM stopped after {self.pending.name}: {window.feedback.status.value}'
            else:
                self._consume(window.feedback)
            self.pending = None
        if self.stopped is not None:
            return Decision('finish', reason=self.stopped)
        self.pending = self._call()
        return Decision('invoke', operation=deepcopy(self.pending),
                        reason=f'ReST^EM round {self.round}: {self.stage}')
