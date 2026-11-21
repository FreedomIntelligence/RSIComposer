"""Ephemeral parent model routes that debit one nested run's reserved budget."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from uuid import uuid4

from rsicomposer.runtime.execution.budget import HostBudget
from rsicomposer.runtime.models.accounting import AccountedModelBackend
from rsicomposer.runtime.models.model_session import HostModelSession
from rsicomposer.runtime.models.model_session import validate_model_route


class _AllocatedBudget:
    """Admission stays attached to every still-reserved ancestor allocation."""
    def __init__(self, parent, child, ident):
        self.parent, self.child, self.ident = parent, child, ident
        self.root, self.limits = child.root, child.limits

    def with_active(self, callback):
        def check(parent):
            with parent._transaction() as state:
                allocation = state['allocations'].get(self.ident)
                if (allocation is None or allocation['state'] != 'reserved'
                        or allocation['root'] != str(self.child.root.resolve())
                        or allocation['limits'] != self.child.limits):
                    raise PermissionError('Nested model budget allocation is no longer active')
                return callback()
        return _with_active(self.parent, check)

    def reserve_model_call(self):
        return self.with_active(self.child.reserve_model_call)


def _with_active(budget, callback):
    if isinstance(budget, _AllocatedBudget):
        return budget.with_active(lambda: callback(budget.child))
    if not isinstance(budget, HostBudget):
        raise PermissionError('Delegation requires a registered nested HostBudget allocation')
    return callback(budget)


def _registered_child(parent, ident, root, limits):
    def verify(budget):
        allocation = budget.allocation(ident)
        if (allocation['state'] != 'reserved' or allocation['root'] != root
                or allocation['limits'] != limits):
            raise PermissionError('Nested model allocation identity mismatch')
        child = HostBudget(root, limits, restoring=True)
        return _AllocatedBudget(parent, child, ident)
    return _with_active(parent, verify)


def delegate_session(session, raw):
    """Rebind a root provider session only to its verified descendant ledger."""
    if (not isinstance(raw, dict) or set(raw) != {'allocation_id', 'root', 'limits', 'call_id'}
            or not isinstance(raw['call_id'], str) or not raw['call_id']
            or raw['allocation_id'] != sha256(raw['call_id'].encode()).hexdigest()[:32]):
        raise PermissionError('Invalid nested model delegation identity')
    budget = _registered_child(session._budget, raw['allocation_id'], raw['root'], raw['limits'])
    return session.derive(budget), {'component': 'nested:' + raw['call_id'], 'decision_id': None,
                                  'call_id': raw['call_id']}


def model_route_key(config):
    """Transport identity before delegation; ephemeral gateways never enter it."""
    raw = {'endpoints': config.get('endpoints') or [config.get('endpoint')],
           'model': config.get('model'), 'native_schema_mode': config.get('native_schema_mode', 'exact'),
           'disable_thinking': config.get('disable_thinking', True),
           'credential_env': config.get('credential_env'), 'timeout': config.get('timeout', 180)}
    raw.update({key: config[key] for key in ('serving_backend', 'engine_version') if key in config})
    return sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()


class _DelegatedSession(HostModelSession):
    def __init__(self, backends, models, *, budget, counters):
        super().__init__(backends, models, budget=budget)
        self._counters = counters

    def count_tokens(self, request):
        if request.model not in self.models:
            raise PermissionError('Model is not registered in this delegated session')
        counter = self._counters.get(request.model.backend_id)
        if counter is None:
            raise NotImplementedError('This delegated provider has no native token counter')
        return _with_active(self._budget, lambda _: counter(request))

    def derive(self, budget):
        return _DelegatedSession(self.backends, self.models, budget=budget, counters=self._counters)


def _configured_routes(value):
    if isinstance(value, dict):
        if isinstance(value.get('model'), str) and ('endpoint' in value or 'endpoints' in value):
            yield value
        for item in value.values():
            yield from _configured_routes(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _configured_routes(item)


def prepare_nested_models(access, child_budget, call_id, host_config, version, labs):
    """Return private transport overrides, token-only environment and cleanup.

    The caller binds ``access`` when exposing this helper on AssemblyContext.
    Tokens live only in the returned environment and server capability table.
    Original provider addresses/model identities remain in saved configuration.
    """
    if not isinstance(call_id, str) or not call_id:
        raise ValueError('Nested model delegation requires its parent call_id')
    candidate_model = getattr(version, 'model', None)
    if candidate_model is not None:
        validate_model_route(access.context, candidate_model)
    component = access.component('nested:' + call_id)
    approved_routes = {model_route_key(config) for config in _configured_routes(host_config)}
    with access._lock:
        existing_components = tuple(access._components.values())
    for existing in existing_components:
        with existing._lock:
            approved_routes.update(existing._route_configs)
    attribution = {'component': component.name, 'decision_id': None, 'call_id': call_id}
    delegated, environment, capabilities = {'backends': {}, 'routes': {}}, {}, []
    allocation_id = sha256(call_id.encode()).hexdigest()[:32]
    allocation = {'allocation_id': allocation_id, 'root': str(child_budget.root.resolve()),
                  'limits': child_budget.limits, 'call_id': call_id}
    allocated_budget = _registered_child(access.context.budget, allocation_id,
                                         allocation['root'], allocation['limits'])
    remote_capabilities = []
    revoke_handoff = lambda: None

    def cleanup():
        revoke_handoff()
        for endpoint in capabilities:
            endpoint.revoke()
        from urllib.error import HTTPError
        failed, errors = [], []
        for endpoint, token in tuple(remote_capabilities):
            from rsicomposer.runtime.execution.remote_services import post_worker
            try:
                post_worker(endpoint + '/revoke', token, {})
            except HTTPError as error:
                if error.code != 401:
                    failed.append((endpoint, token))
                    errors.append(error)
                error.close()
            except Exception as error:
                failed.append((endpoint, token))
                errors.append(error)
        remote_capabilities[:] = failed
        if errors:
            raise ExceptionGroup('Nested model capability cleanup failed', errors)

    def ready():
        nonlocal revoke_handoff
        revoke_handoff = access._register_nested_environment(environment,
            lambda: allocated_budget.with_active(lambda: None))
        return delegated, environment, cleanup

    def issue(session):
        endpoint = access.server.issue(component.name, session, attribution_provider=lambda: dict(attribution))
        capabilities.append(endpoint)
        issued = endpoint.environment()
        variable = 'RSISYS_NESTED_MODEL_TOKEN_' + uuid4().hex.upper()
        environment[variable] = issued['OPENAI_API_KEY']
        return {'endpoint': issued['OPENAI_BASE_URL'], 'credential_env': variable,
                'models': [asdict(model) for model in session.models]}

    def redelegate(route):
        import os
        from rsicomposer.runtime.execution.remote_services import post_worker
        token = os.environ.get(route['credential_env'])
        if not token:
            raise PermissionError('Nested model delegation credential is unavailable')
        result = post_worker(route['endpoint'] + '/delegate', token, allocation)
        variable = 'RSISYS_NESTED_MODEL_TOKEN_' + uuid4().hex.upper()
        environment[variable] = result['token']
        remote_capabilities.append((result['endpoint'], result['token']))
        return {'endpoint': result['endpoint'], 'credential_env': variable,
                'models': list(route['models'])}

    try:
        inherited = getattr(access.context, '_delegated_models', None)
        if inherited is not None:
            if candidate_model is not None:
                route = inherited.get('backends', {}).get(candidate_model.backend_id)
                if route is None or asdict(candidate_model) not in route.get('models', ()):
                    raise PermissionError('Nested candidate model is outside its delegated model pins')
            for group in ('backends', 'routes'):
                for key, route in inherited.get(group, {}).items():
                    delegated[group][key] = redelegate(route)
            return ready()
        models = dict(access.context.roles)
        if candidate_model is not None:
            models['decision_maker'] = candidate_model
        session = getattr(access.context, 'model_session', None)
        for backend_id, transport in access.context.backends.items():
            raw = transport.transport if isinstance(transport, AccountedModelBackend) else transport
            binding = getattr(raw, 'binding', None)
            if binding is None or binding.kind == 'mock':
                continue
            selected = tuple(dict.fromkeys(model for model in models.values() if model.backend_id == backend_id))
            if not selected:
                continue
            account = (session.backends.get(backend_id) if session is not None else None)
            if account is None and isinstance(transport, AccountedModelBackend):
                account = transport
            native_binding = (replace(binding, kind='vllm') if binding.kind in ('chat_completions', 'http')
                              else binding)
            if account is None:
                # The application's basic Chat transport does not implement
                # native schemas/tokenization. Nested harnesses may require both;
                # use the host's full serving adapter for the same pinned binding.
                wire = (raw if getattr(raw, 'supports_response_schema', False) else
                        access.transport_factory(native_binding, models={model: model.model_id for model in selected}))
                account = component._account('nested-backend:' + backend_id, wire,
                                              max_input_tokens=None, models=selected)
                raw = wire
            if account.token_counter is not None or account.provider_token_counter or account.tokenizer is not None:
                counter = account.count_tokens
            elif callable(getattr(raw, 'count_tokens', None)):
                counter = raw.count_tokens
            elif binding.kind == 'openai_chat_completions':
                counter = None
            else:
                counter = access.transport_factory(native_binding,
                    models={model: model.model_id for model in selected}).count_tokens
            endpoint = _DelegatedSession({backend_id: account}, selected, budget=allocated_budget,
                                         counters={backend_id: counter} if counter is not None else {})
            delegated['backends'][backend_id] = issue(endpoint)

        configurations = [host_config, getattr(version, 'config', None), labs]
        seen = set()
        for config in _configured_routes(configurations):
            key = model_route_key(config)
            if key in seen:
                continue
            if key not in approved_routes:
                raise PermissionError('Nested candidate model route was not declared by its parent host')
            seen.add(key)
            view = component.endpoint(config, purpose='nested:' + key)
            original = view._endpoint
            original_session = original._session
            counters = {backend_id: counter for backend_id, account in original_session.backends.items()
                        if callable(counter := getattr(account.transport, 'count_tokens', None))}
            endpoint = _DelegatedSession(original_session.backends, original_session.models,
                                         budget=allocated_budget, counters=counters)
            delegated['routes'][key] = issue(endpoint)
        return ready()
    except BaseException:
        cleanup()
        raise
