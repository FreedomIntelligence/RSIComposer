"""Host model admission and accounting for explicit model routes."""
import threading

from rsisys.loops.contracts.model_service import ModelRef
from rsisys.runtime.models.accounting import AccountedModelBackend


def validate_model_route(context, model):
    """Check an ordinary candidate against declared or checkpoint-prepared refs."""
    owner = getattr(context, '_evidence_owner', context)
    if not isinstance(model, ModelRef) or model.backend_id not in owner.backends:
        raise ValueError('Controller model must use an available host backend')
    session = getattr(owner, 'model_session', None)
    backend = owner.backends[model.backend_id]
    checker = getattr(session if session is not None else backend, 'is_model_authorized', None)
    if callable(checker):
        if not checker(model):
            raise PermissionError('Controller model is outside the registered host routes')
    elif model not in owner.roles.values() and model not in getattr(getattr(owner, 'controller_host', None), 'prepared_models', {}):
        raise PermissionError('Controller model requires a declared route or prepared training result')


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

    def authorize_prepared_model(self, model):
        """Extend admission at a host checkpoint after preparation was confirmed."""
        if not isinstance(model, ModelRef) or model.backend_id not in self.backends:
            raise PermissionError('Prepared model cannot introduce an unregistered backend')
        with self._models_lock:
            self.backends[model.backend_id].authorize_prepared_model(model)
            if model not in self.models:
                self.models += (model,)
