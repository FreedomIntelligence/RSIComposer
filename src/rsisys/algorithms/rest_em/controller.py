"""ReST^EM scheduling and receipt contract (Singh et al., arXiv:2312.06585).

Generate with the current model; filter binary successes; fit the INITIAL model
on this round's accepted data. Evaluation observes outcomes, not an acceptance
gate. Model/data references must identify immutable artifacts. No backend imports.
"""
from copy import deepcopy
import math
import re

from rsisys.api import Decision, ExecutionStatus, OperationCall


def _ref(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError(f'{label} must be a nonempty reference of at most 1024 characters')
    return value


class RestEMController:
    """Synchronous terminal-receipt protocol; no hidden meta-agent or replay."""

    def __init__(self, *, initial_model, train_data, eval_data, lab='math',
                 iterations=2, samples_per_prompt=4, max_accepted_per_prompt=4, temperature=0.7):
        self.validate_config(dict(initial_model=initial_model, train_data=train_data,
            eval_data=eval_data, lab=lab, iterations=iterations,
            samples_per_prompt=samples_per_prompt,
            max_accepted_per_prompt=max_accepted_per_prompt, temperature=temperature))
        self.initial_model = self.current_model = initial_model
        self.train_data, self.eval_data, self.lab = train_data, eval_data, lab
        self.iterations, self.samples_per_prompt = iterations, samples_per_prompt
        self.max_accepted_per_prompt = max_accepted_per_prompt
        self.temperature = temperature
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
                   'samples_per_prompt', 'max_accepted_per_prompt', 'temperature'}
        if set(config) - allowed:
            raise ValueError('Unknown rest_em option')
        for key in ('initial_model', 'train_data', 'eval_data'):
            _ref(config.get(key), key)
        if config['train_data'] == config['eval_data']:
            raise ValueError('ReST^EM requires separate training and held-out evaluation data')
        if not isinstance(config.get('lab', 'math'), str) or not re.fullmatch(
                r'[A-Za-z][A-Za-z0-9_-]*', config.get('lab', 'math')):
            raise ValueError('lab must be a registered Lab alias')
        temperature = config.get('temperature', .7)
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        for key, default in (('iterations', 2), ('samples_per_prompt', 4),
                             ('max_accepted_per_prompt', 4)):
            if type(config.get(key, default)) is not int or config.get(key, default) < 1:
                raise ValueError(f'{key} must be a positive integer')

    def required_operations(self):
        return tuple(f'{self.lab}.{name}' for name in ('generate', 'filter', 'train', 'evaluate'))

    def required_profiles(self):
        return {self.lab: 'round-weight-training/v1'}

    def _call(self):
        args = {'round': self.round}
        if self.stage == 'generate':
            args.update(model=self.current_model, dataset=self.train_data,
                        samples_per_prompt=self.samples_per_prompt, temperature=self.temperature, condition='none')
        elif self.stage == 'filter':
            args.update(samples=self.generated, max_accepted_per_prompt=self.max_accepted_per_prompt)
        elif self.stage == 'train':
            args.update(parent=self.initial_model, dataset=self.accepted)
        else:
            args.update(model=self.candidate or self.current_model, dataset=self.eval_data, split='dev', baseline=self.round == 0)
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

    _STATE = ('current_model', 'round', 'pending', 'stage', 'generated', 'accepted', 'candidate',
              'generated_count', 'history', 'stopped')

    def state(self):
        return {name: deepcopy(getattr(self, name)) for name in self._STATE}

    def restore(self, state):
        if set(state) != set(self._STATE):
            raise ValueError('Unexpected ReST^EM controller state')
        for name in self._STATE:
            setattr(self, name, deepcopy(state[name]))

    def decide(self, window):
        feedback = window.feedbacks[0] if window.feedbacks else None
        if self.stopped is not None:
            return Decision('finish', reason=self.stopped)
        if self.pending is not None:
            if (feedback is None or feedback.call_id != self.pending.call_id):
                raise RuntimeError('ReST^EM requires the matching terminal receipt before advancing')
            if feedback.status != ExecutionStatus.COMPLETED:
                self.stopped = f'ReST^EM stopped after {self.pending.name}: {feedback.status.value}'
            else:
                self._consume(feedback)
            self.pending = None
        if self.stopped is not None:
            return Decision('finish', reason=self.stopped)
        self.pending = self._call()
        return Decision.invoke(deepcopy(self.pending), reason=f'ReST^EM round {self.round}: {self.stage}')
