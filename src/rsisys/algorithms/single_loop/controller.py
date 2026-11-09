"""Fixed iteration scheduling with feedback from a configured Lab operation."""
from copy import deepcopy

from rsisys.components import _json
from rsisys.loops.contracts.decision import Decision, OperationCall
from rsisys.loops.contracts.lifecycle import ExecutionStatus


class SingleLoopController:
    """Repeat one Lab iteration, observing its receipt before the next call.

    The Lab owns the configured experiment pipeline (for example generate/filter/train/evaluate).
    This controller is a scheduling template, not an implementation of ReST or RL.
    """
    def __init__(self, operation, *, iterations=1, arguments=None, stop_on_failure=True):
        self.validate_config(dict(operation=operation, iterations=iterations,
                                  arguments={} if arguments is None else arguments,
                                  stop_on_failure=stop_on_failure))
        self.operation, self.iterations = operation, iterations
        self.arguments = deepcopy(arguments or {})
        self.stop_on_failure = stop_on_failure
        self.submitted = 0

    @staticmethod
    def validate_config(config):
        if set(config) - {'operation', 'iterations', 'arguments', 'stop_on_failure'}:
            raise ValueError('Unknown single_loop option')
        if not isinstance(config.get('operation'), str) or not config['operation'].strip():
            raise ValueError('single_loop requires a Lab operation name')
        count = config.get('iterations', 1)
        if type(count) is not int or count < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(config.get('arguments', {}), dict):
            raise TypeError('arguments must be an object')
        _json(config.get('arguments', {}))
        if type(config.get('stop_on_failure', True)) is not bool:
            raise TypeError('stop_on_failure must be boolean')

    def decide(self, window):
        if self.submitted:
            if window.feedback is None or window.last_operation != OperationCall(self.operation, self.arguments):
                raise RuntimeError('single_loop requires the previous iteration receipt')
            if window.feedback.status != ExecutionStatus.COMPLETED and self.stop_on_failure:
                return Decision('finish', reason='Lab iteration failed; stopped by configured policy')
        if self.submitted >= self.iterations:
            return Decision('finish', reason='Configured iterations completed; see receipts for outcomes')
        self.submitted += 1
        return Decision('invoke', operation=OperationCall(self.operation, deepcopy(self.arguments)),
                        reason=f'Run configured Lab iteration {self.submitted}')
