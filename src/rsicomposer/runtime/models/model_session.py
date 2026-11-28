"""Host model admission and accounting for explicit model routes."""
from dataclasses import asdict
import threading

from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.runtime.models.accounting import AccountedModelBackend
from rsicomposer.runtime.models.activation import validate_activation_route


def validate_model_route(context, model):
    """Check an ordinary candidate against the run's declared or checkpoint-prepared refs."""
    if not isinstance(model, ModelRef) or model.backend_id not in context.backends:
        raise ValueError('Controller model must use an available host backend')
    session = getattr(context, 'model_session', None)
    backend = context.backends[model.backend_id]
    checker = getattr(session if session is not None else backend, 'is_model_authorized', None)
    if callable(checker):
        if not checker(model):
            raise PermissionError('Controller model is outside the registered host routes')
    elif model not in context.roles.values() and model not in getattr(getattr(context, 'controller_host', None), 'prepared_models', {}):
        raise PermissionError('Controller model requires a declared route or prepared training result')


def prepared_model_target(context, model, training, *, prepared):
    """Resolve the admission owner before deployment and again after the readiness receipt."""
    if not isinstance(model, ModelRef) or model != training.model:
        raise ValueError('Prepared model must match the trained version')
    delegated = (getattr(context, '_delegated_models', None) or {}).get('backends', {}).get(model.backend_id)
    if delegated is not None and asdict(model) not in delegated.get('models', ()):
        raise PermissionError('Prepared model is outside the delegated model pins')
    session = context.model_session
    target = session if session is not None else context.backends.get(model.backend_id)
    # An external activator can still report deployment pending, but its ready
    # receipt alone cannot create a host route or expand a delegated pin.
    if prepared and target is None:
        raise ValueError('Prepared model has no available host backend')
    if not training.simulated:
        validate_activation_route(target, training.base, model, prepared=prepared)
    return target


_SESSION_BUDGET = object()


class HostModelSession:
    def __init__(self, backends, models, *, budget=_SESSION_BUDGET):
        self.backends, self.models = dict(backends), tuple(models)
        self.configured_models = self.models
        self._models_lock = threading.RLock()
        if budget is not _SESSION_BUDGET and budget is not None and not callable(getattr(budget, "reserve_model_call", None)):
            raise TypeError("Session budget must provide model admission")
        self._budget = budget
        if not self.models or any(not isinstance(b, AccountedModelBackend) for b in self.backends.values()):
            raise TypeError('Model sessions require host-accounted backends and explicit model bindings')
        if any(model.backend_id not in self.backends for model in self.models):
            raise ValueError('Unknown model backend in session')

    def generate(self, request, *, attribution=None):
        if not self.is_model_authorized(request.model):
            raise PermissionError('Model is not registered in this host session')
        backend = self.backends[request.model.backend_id]
        if self._budget is _SESSION_BUDGET:
            return backend.generate(request, attribution=attribution)
        return backend.generate(request, attribution=attribution, budget=self._budget)

    def count_tokens(self, request):
        if not self.is_model_authorized(request.model):
            raise PermissionError('Model is not registered in this host session')
        return self.backends[request.model.backend_id].count_tokens(request)

    def derive(self, budget):
        return HostModelSession(self.backends, self.models, budget=budget)

    def is_model_authorized(self, model):
        return (isinstance(model, ModelRef) and model in self.models and model.backend_id in self.backends
                and self.backends[model.backend_id].is_model_authorized(model))

    def validate_activation_route(self, base, model, *, prepared):
        backend = self.backends.get(model.backend_id)
        if backend is None:
            raise PermissionError('Prepared model cannot introduce an unregistered backend')
        validate_activation_route(backend, base, model, prepared=prepared)

    def authorize_prepared_model(self, model):
        """Extend admission at a host checkpoint after preparation was confirmed."""
        if not isinstance(model, ModelRef) or model.backend_id not in self.backends:
            raise PermissionError('Prepared model cannot introduce an unregistered backend')
        with self._models_lock:
            self.backends[model.backend_id].authorize_prepared_model(model)
            if model not in self.models:
                self.models += (model,)
