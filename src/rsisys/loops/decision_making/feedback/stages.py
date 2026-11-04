"""Bind public product functions and isolate the old journal-shaped workflow."""
from copy import deepcopy
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.runtime.files import json_bytes
from .context_memory import feedback_context_event, _callable_identity
from .model_gateway import FeedbackModelGateway


def component_configuration(component):
    value = {'type': f'{type(component).__module__}.{type(component).__qualname__}'}
    if callable(component):
        value['callable'] = _callable_identity(component)
    configuration = getattr(component, 'configuration', None)
    if callable(configuration):
        settings = configuration()
        if not isinstance(settings, dict):
            raise TypeError('Feedback component configuration must be a JSON object')
        value['configuration'] = deepcopy(settings)
    json_bytes(value)
    return value


class CallbackFeedbackProcessor:
    """SDK/function integration. Call allowance is deployment configuration.

    The callback returns only its product. The adapter does not interpret keys
    such as status, context, proposals or charged_calls as host metadata.
    """
    def __init__(self, process, *, max_model_calls=0, recover=None, configuration=None):
        if not callable(process) or (recover is not None and not callable(recover)):
            raise TypeError('Feedback process and optional recovery must be callable')
        if type(max_model_calls) is not int or max_model_calls < 0:
            raise ValueError('Feedback call allowance must be a nonnegative integer')
        self.process, self.recover = process, recover
        self.max_model_calls = max_model_calls
        self._configuration = deepcopy(configuration or {})

    def configuration(self):
        return {'process': _callable_identity(self.process),
                'recover': _callable_identity(self.recover) if self.recover else None,
                'max_model_calls': self.max_model_calls, 'settings': deepcopy(self._configuration)}


class AccountedFeedbackModel:
    """Small capability passed to a product, without exposing the worker row."""
    def __init__(self, backend, model, limits, *, path=None):
        self.model = model
        self._max_output_tokens = limits['max_output_tokens']
        self._gateway = FeedbackModelGateway(backend, model,
            max_calls=limits['max_calls'], max_input_tokens=limits['max_input_tokens'], path=path)
        self.supports_tool_calling = getattr(backend, 'supports_tool_calling', False) is True
        self.supports_response_schema = getattr(backend, 'supports_response_schema', False) is True

    def generate(self, request):
        if request.max_output_tokens > self._max_output_tokens:
            raise ValueError('Feedback request exceeds host output limit')
        return self._gateway.generate(request)


class FeedbackStageBinding:
    """Only this compatibility adapter passes private rows to old processors."""
    def __init__(self, component, *, proposal=False):
        self.component = component
        self.legacy = (component is not None and hasattr(component, 'max_calls_per_job')
                       and callable(getattr(component, 'process', None)))
        self.method = ('propose' if proposal and callable(getattr(component, 'propose', None)) else 'process')
        if component is None:
            self.max_calls = 0
            return
        operation = getattr(component, self.method, None)
        if not callable(operation) and not callable(component):
            raise TypeError('Feedback component requires process(event, model) or propose(event, model)')
        self.max_calls = getattr(component, 'max_calls_per_job' if self.legacy else 'max_model_calls', 0)
        if type(self.max_calls) is not int or self.max_calls < 0:
            raise ValueError('Feedback components require a nonnegative call allowance')
        component_configuration(component)

    def configuration(self):
        if self.legacy and callable(getattr(self.component, 'configuration', None)):
            return deepcopy(self.component.configuration())
        return {'protocol': 'product-v1', 'component': component_configuration(self.component),
                'max_model_calls': self.max_calls}

    def process(self, job, backend, limits, *, path=None):
        if self.legacy:
            return self.component.process(deepcopy(job), backend, dict(limits))
        model = AccountedFeedbackModel(backend, ModelRef(**job['model']), limits, path=path)
        try:
            operation = getattr(self.component, self.method, None)
            output = (operation if callable(operation) else self.component)(feedback_context_event(job), model)
            json_bytes(output)
        except Exception as exc:
            return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                    'charged_calls': model._gateway.attempted, 'proposals': [],
                    'response': {'model_exchanges': model._gateway.snapshot()}}
        return {'status': 'completed', 'error': None, 'charged_calls': model._gateway.attempted,
                'proposals': [], 'output': deepcopy(output),
                'simulated': any(e.get('response', {}).get('simulated', False)
                                 for e in model._gateway.snapshot()),
                'response': {'model_exchanges': model._gateway.snapshot()}}

    def recover(self, job):
        recovery = getattr(self.component, 'recover', None)
        if not callable(recovery):
            return None
        if self.legacy:
            return recovery(deepcopy(job))
        output = recovery(feedback_context_event(job))
        # None means no committed product. Conservatively retain the stage's
        # entire reservation: a recovered business product cannot prove usage.
        if output is None:
            return None
        json_bytes(output)
        return {'status': 'completed', 'charged_calls': min(self.max_calls, job['reserved_calls']),
                'proposals': [], 'output': deepcopy(output), 'response': {'recovered': True}}
