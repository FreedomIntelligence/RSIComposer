"""Copy this directory outside RSISys; only the installed public API is needed.

CPU interface example, not an LLM training algorithm or evidence of RSI gains.
"""
from rsisys.api import Decision, LabRequest, LabFeedback, OperationSpec


class CounterLab:
    def __init__(self, context, config):
        self.value = config.get('initial', 0)

    def describe(self):
        return (OperationSpec('step', 'Increment the CPU example counter', {
            'type': 'object', 'properties': {}, 'additionalProperties': False}),)

    def execute(self, call):
        if call.name != 'step' or call.arguments:
            raise ValueError('Expected step with empty arguments')
        self.value += 1
        return LabFeedback('CPU example iteration completed', (), False, data={'value': self.value})


class TargetController:
    def __init__(self, context, config):
        self.target = config.get('target', 3)
        self.operation = config.get('operation', 'counter.step')

    def decide(self, window):
        if window.feedback is not None and window.feedback.data['value'] >= self.target:
            return Decision('finish', reason='Observed counter target in the Lab receipt')
        return Decision('invoke', operation=LabRequest(self.operation, {}))


def validate_lab(config):
    if set(config) - {'initial'} or type(config.get('initial', 0)) is not int:
        raise ValueError('counter accepts an integer initial value')


def validate_controller(config):
    if set(config) - {'target', 'operation'} or type(config.get('target', 3)) is not int:
        raise ValueError('target controller accepts an integer target and an operation name')
    if not isinstance(config.get('operation', 'counter.step'), str):
        raise ValueError('operation must be text')


def register(registry):
    registry.register_lab('counter', CounterLab, version='1', validate_config=validate_lab)
    registry.register_controller('target', TargetController, version='1', validate_config=validate_controller)
