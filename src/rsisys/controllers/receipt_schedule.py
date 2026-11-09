"""Receipt-driven staged controller machinery, independent of any RSI algorithm.

Subclasses define operation phases and method-specific option constraints.
"""
import math
import re
from copy import deepcopy

from rsisys.loops.contracts.decision import Decision, OperationCall
from rsisys.loops.contracts.lifecycle import ExecutionStatus


def _ref(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError(f'{label} must be a nonempty reference of at most 1024 characters')
    return value


class ReceiptScheduleController:
    method = None
    phases = ()
    option = None
    default_width = 1
    minimum_width = 1

    def __init__(self, *, initial_model, train_data, eval_data, lab=None,
                 iterations=2, **options):
        config = dict(initial_model=initial_model, train_data=train_data,
                      eval_data=eval_data, lab=lab or self.method, iterations=iterations,
                      **options)
        self.validate_config(config)
        self.initial_model = self.current_model = initial_model
        self.train_data, self.eval_data = train_data, eval_data
        self.lab, self.iterations = config['lab'], iterations
        self.width = config.get(self.option, self.default_width)
        self.round, self.phase = 0, 'evaluate'
        self.pending = self.artifact = self.stopped = None
        self.models, self.history = {initial_model}, []
        self.outcome, self.training_rounds, self.real_training_rounds = 'running', 0, 0

    @classmethod
    def validate_config(cls, config):
        allowed = {'initial_model', 'train_data', 'eval_data', 'lab', 'iterations', cls.option}
        if set(config) - allowed:
            raise ValueError(f'Unknown {cls.method} option')
        for key in ('initial_model', 'eval_data'):
            _ref(config.get(key), key)
        learning = config.get('train_data')
        datasets = learning if isinstance(learning, list) else [learning]
        if not datasets or (isinstance(learning, list) and len(learning) != config.get('iterations', 2)):
            raise ValueError('train_data list must provide exactly one dataset per iteration')
        for dataset in datasets:
            _ref(dataset, 'train_data')
        if config['eval_data'] in datasets:
            raise ValueError('Learning and independent evaluation data must differ')
        if not isinstance(config.get('lab', cls.method), str) or not re.fullmatch(
                r'[A-Za-z][A-Za-z0-9_-]*', config.get('lab', cls.method)):
            raise ValueError('Invalid Lab alias')
        for key, default in (('iterations', 2), (cls.option, cls.default_width)):
            if type(config.get(key, default)) is not int or config.get(key, default) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if config.get(cls.option, cls.default_width) < cls.minimum_width:
            raise ValueError(f'{cls.method} requires {cls.option} >= {cls.minimum_width}')

    def _call(self):
        args = {'round': self.round, 'model': self.current_model}
        if self.phase == 'evaluate':
            args['dataset'] = self.eval_data
        elif self.phase == self.phases[0]:
            dataset = self.train_data[self.round - 1] if isinstance(self.train_data, list) else self.train_data
            args.update(dataset=dataset, **{self.option: self.width})
        else:
            args['dataset'] = self.artifact
        return OperationCall(f'{self.lab}.{self.phase}', args)

    def required_operations(self):
        """Optional registration capability check, before any expensive Lab call."""
        return tuple(f'{self.lab}.{name}' for name in ('evaluate', *self.phases))

    def report(self):
        return {'method': self.method, 'status': self.outcome, 'reason': self.stopped,
                'rounds_completed': self.training_rounds, 'iterations': self.iterations,
                'initial_model': self.initial_model, 'current_model': self.current_model,
                'training_performed': self.real_training_rounds > 0,
                'real_training_rounds': self.real_training_rounds,
                'evaluations': deepcopy(self.history)}

    def _consume(self, feedback):
        data = feedback.data
        if data.get('request') != self.pending.arguments:
            raise ValueError('Weight-evolution receipt does not match pending request')
        if self.phase == 'evaluate':
            metrics = data.get('quality_metrics')
            if not isinstance(metrics, dict) or not metrics or any(
                    not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v)
                    for k, v in metrics.items()):
                raise ValueError('Evaluation must return finite measured metrics')
            self.history.append({'round': self.round, 'model': self.current_model,
                                 'metrics': deepcopy(metrics), 'simulated': feedback.simulated})
            if self.round == self.iterations:
                self.outcome = 'completed'
                self.stopped = f'{self.method} configured rounds completed; inspect independent evaluation receipts'
                return
            self.round = 1
            self.phase = self.phases[0]
        elif self.phase == self.phases[-1]:
            model = _ref(data.get('model'), 'model')
            if model in self.models:
                raise ValueError('Training must publish a distinct immutable checkpoint')
            if not feedback.simulated and data.get('training_performed') is not True:
                raise ValueError('Real training receipt must attest an actual optimizer update')
            self.models.add(model)
            self.current_model = model
            self.training_rounds += 1
            self.real_training_rounds += int(not feedback.simulated)
            if self.round == self.iterations:
                self.phase = 'evaluate'
            else:
                self.round += 1
                self.phase = self.phases[0]
        else:
            count = data.get('count')
            if type(count) is not int or count < 0:
                raise ValueError('Artifact count must be a nonnegative integer')
            artifact = _ref(data.get('dataset'), 'dataset')
            digest = data.get('sha256')
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('Published datasets require a SHA256 identity')
            self.artifact = artifact
            if count == 0:
                self.outcome = 'stopped'
                self.stopped = f'{self.method} stopped: empty {self.phase} artifact'
            else:
                self.phase = self.phases[self.phases.index(self.phase) + 1]

    def decide(self, window):
        if self.stopped:
            return Decision('finish', reason=self.stopped)
        if self.pending:
            if window.feedback is None or window.last_operation != self.pending:
                raise RuntimeError('Matching terminal feedback is required before advancing')
            if window.feedback.status != ExecutionStatus.COMPLETED:
                self.outcome = 'failed' if window.feedback.status == ExecutionStatus.FAILED else 'cancelled'
                self.stopped = f'{self.method} stopped after {self.pending.name}: {window.feedback.status.value}'
            else:
                self._consume(window.feedback)
            self.pending = None
        if self.stopped:
            return Decision('finish', reason=self.stopped)
        self.pending = self._call()
        return Decision('invoke', operation=deepcopy(self.pending),
                        reason=f'{self.method} round {self.round}: {self.phase}')

