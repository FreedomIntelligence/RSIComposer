"""Host binding for one product protocol: process(event, model)."""
from copy import deepcopy
import json

from rsisys.harnesses.decision.feedback.context_memory import (
    _callable_identity,
    feedback_context_event,
)
from rsisys.harnesses.decision.feedback.model_gateway import FeedbackModelGateway
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.runtime.storage.files import json_bytes


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
    """Adapt a product function; model allowance belongs to deployment settings."""
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
    """Explicit bounded model capability; accounting stays inside the host."""
    def __init__(self, backend, model, limits, *, path=None):
        self.model = model
        self.max_output_tokens = limits['max_output_tokens']
        self._gateway = FeedbackModelGateway(backend, model,
            max_calls=limits['max_calls'], max_input_tokens=limits['max_input_tokens'], path=path)
        self.supports_tool_calling = getattr(backend, 'supports_tool_calling', False) is True
        self.supports_response_schema = getattr(backend, 'supports_response_schema', False) is True

    def generate(self, request):
        if request.max_output_tokens > self.max_output_tokens:
            raise ValueError('Feedback request exceeds host output limit')
        return self._gateway.generate(request)

    @property
    def requests_reserved(self):
        return sum(entry['status'] != 'input_too_large' for entry in self._gateway.snapshot())


class FeedbackStageBinding:
    """Keep opaque product output separate from execution/accounting metadata."""
    actual_exchanges_read_path = '/feedback_response/model_exchanges'
    def __init__(self, component):
        self.component = component
        if component is None:
            self.max_calls = 0
            return
        if hasattr(component, 'max_calls_per_job'):
            raise TypeError('Feedback components must use process(event, model) and max_model_calls')
        operation = getattr(component, 'process', None)
        if not callable(operation) and not callable(component):
            raise TypeError('Feedback component requires process(event, model)')
        self.max_calls = getattr(component, 'max_model_calls', 0)
        if type(self.max_calls) is not int or self.max_calls < 0:
            raise ValueError('Feedback components require a nonnegative call allowance')
        component_configuration(component)

    def configuration(self):
        return {'protocol': 'product-v1', 'component': component_configuration(self.component),
                'max_model_calls': self.max_calls}

    def run(self, job, backend, limits, *, path=None):
        model = None
        try:
            event = feedback_context_event(job)
            model = AccountedFeedbackModel(backend, ModelRef(**job['model']), limits, path=path)
            operation = getattr(self.component, 'process', None)
            output = (operation if callable(operation) else self.component)(event, model)
            json_bytes(output)
        except Exception as exc:
            # Existing exchanges can mean an uncertain prior request. Without a
            # fresh gateway retain this stage's complete reserved allowance.
            return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                    'charged_calls': model._gateway.attempted if model else limits['max_calls'],
                    'proposals': [],
                    'response': {'model_exchanges': model._gateway.snapshot() if model else []}}
        exchanges = model._gateway.snapshot()
        return {'status': 'completed', 'error': None, 'charged_calls': model._gateway.attempted,
                'proposals': [], 'output': deepcopy(output),
                'simulated': any(e.get('response', {}).get('simulated', False) for e in exchanges),
                'response': {'model_exchanges': exchanges}}

    def recover_job(self, job, *, path=None):
        recovery = getattr(self.component, 'recover', None)
        if not callable(recovery):
            return None
        output = recovery(feedback_context_event(job))
        if output is None:
            return None
        json_bytes(output)
        exchanges = []
        if path is not None and path.exists():
            document = json.loads(path.read_bytes())
            if (not isinstance(document, dict) or set(document) != {'version', 'exchanges'}
                    or type(document['version']) is not int or document['version'] != 1
                    or not isinstance(document['exchanges'], list)):
                raise ValueError('Invalid durable feedback exchanges')
            exchanges = document['exchanges']
            if (len(exchanges) > min(self.max_calls, job['reserved_calls'])
                    or any(not isinstance(entry, dict) or entry.get('status') not in
                           ('reserved', 'failed', 'completed', 'input_too_large')
                           or not isinstance(entry.get('request'), dict)
                           or entry['request'].get('model') != job['model'] for entry in exchanges)):
                raise ValueError('Durable feedback exchanges differ from the recovered job')
        return {'status': 'completed', 'charged_calls': len(exchanges), 'proposals': [],
                'output': deepcopy(output),
                'simulated': any(entry.get('response', {}).get('simulated', False) for entry in exchanges),
                'response': {'recovered': True, 'model_exchanges': exchanges}}
