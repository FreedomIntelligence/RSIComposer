"""Assembly-owned model routes; algorithm workers see only scoped loopback access."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from hashlib import sha256
import json
import re
import threading

from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.runtime.models.accounting import AccountedModelBackend
from rsisys.runtime.models.component_endpoint import ComponentModelServer
from rsisys.runtime.models.model_session import HostModelSession
from rsisys.runtime.models.delegation import model_route_key


class _BackendView:
    def __init__(self, component, model_backend):
        self.component, self.backend = component, model_backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @property
    def binding(self):
        transport = self.backend.transport if isinstance(self.backend, AccountedModelBackend) else self.backend
        return transport.binding

    def generate(self, request):
        return self.component.generate(request)


class _EndpointView:
    def __init__(self, component, endpoint):
        self._component, self._endpoint = component, endpoint

    def __getattr__(self, name):
        return getattr(self._endpoint, name)

    def environment(self):
        # Construction outside an operation is allowed for long-lived upstream
        # Controllers; their root token still requires an unambiguous active scope.
        if self._component._local.get() is None:
            return self._endpoint.environment()
        return self._component._environment(self._endpoint)


class ComponentModels:
    """A component's host model capabilities and explicitly configured model routes."""
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.max_concurrent = 1
        self._endpoints, self._accounts = {}, {}
        self._route_configs = {}
        self._lock, self._active = threading.RLock(), {}
        self._scope_endpoints = {}
        self._local = ContextVar('rsisys-component-model-scope', default=None)
        self.backends = {key: _BackendView(self, backend) for key, backend in owner.context.backends.items()}
        self.models = tuple(owner.context.roles.values())

    @contextmanager
    def scope(self, *, decision_id=None, call_id=None):
        scope = {'component': self.name, 'decision_id': decision_id, 'call_id': call_id}
        ident = object()
        token = self._local.set(scope)
        with self._lock:
            self._active[ident] = scope
        try:
            yield self
        finally:
            self._local.reset(token)
            with self._lock:
                self._active.pop(ident, None)
                endpoints = self._scope_endpoints.pop(ident, [])
            for endpoint in endpoints:
                endpoint.revoke()

    def attribution(self):
        if self.owner.closed:
            raise PermissionError('Host model access is closed')
        direct = self._local.get()
        if direct is not None:
            return dict(direct)
        with self._lock:
            scopes = list(self._active.values())
            if len(scopes) != 1:
                raise PermissionError('Component model request requires an unambiguous host scope')
            return dict(scopes[0])

    def direct_attribution(self):
        # Trusted in-process services (e.g. Memory processing) may run outside
        # a decision. Keep that absence explicit rather than inventing lineage.
        if self.owner.closed:
            raise PermissionError('Host model access is closed')
        if self._local.get() is not None:
            return dict(self._local.get())
        with self._lock:
            active = bool(self._active)
        return self.attribution() if active else {'component': self.name, 'decision_id': None, 'call_id': None}

    def _account(self, key, transport, *, max_calls=1000000, max_concurrent=1,
                 max_input_tokens=None, provider_token_counter=False, response_attempts=1, models=()):
        with self._lock:
            if key not in self._accounts:
                root = self.owner.root / sha256((self.name + ':' + key).encode()).hexdigest()
                # A route unused before interruption has no ledger to recover.
                resume = self.owner.context.restoring and (root / 'model_calls.json').exists()
                self._accounts[key] = AccountedModelBackend(transport, None, root,
                    max_calls=max_calls, max_concurrent=max_concurrent, max_input_tokens=max_input_tokens,
                    provider_token_counter=provider_token_counter, response_attempts=response_attempts,
                    resume=resume, allowed_models=models or None)
            self._accounts[key].host_budget = getattr(self.owner.context, 'budget', None)
            return self._accounts[key]

    def generate(self, request):
        attribution = self.direct_attribution()
        delegated = request.model.backend_id in (getattr(self.owner.context, '_delegated_models', None) or {}).get('backends', {})
        session = self.owner.context.model_session
        if session is not None:
            if delegated:
                if request.model not in session.models:
                    raise PermissionError('Model is not registered in this host session')
                return session.backends[request.model.backend_id].generate(request, attribution=attribution, budget=None)
            for backend in session.backends.values():
                backend.host_budget = getattr(self.owner.context, 'budget', None)
            return session.generate(request, attribution=attribution)
        backend = self.owner.context.backends[request.model.backend_id]
        if isinstance(backend, AccountedModelBackend):
            if delegated:
                return backend.generate(request, attribution=attribution, budget=None)
            backend.host_budget = getattr(self.owner.context, 'budget', None)
            return backend.generate(request, attribution=attribution)
        account = self._account('backend:' + request.model.backend_id, backend, max_concurrent=self.max_concurrent)
        if delegated:
            return account.generate(request, attribution=attribution, budget=None)
        return account.generate(request, attribution=attribution)

    def metered_backend(self, transport, *, name, **limits):
        component = self
        binding = getattr(transport, 'binding', None)
        overrides = getattr(self.owner.context, '_delegated_models', None) or {}
        delegated = binding is not None and any(binding.endpoint == route['endpoint']
            and binding.credential_env == route['credential_env'] for group in ('backends', 'routes')
            for route in overrides.get(group, {}).values())
        models = tuple(limits.get('models', ()))
        if binding is not None and callable(getattr(transport, 'bind', None)):
            from rsisys.runtime.models.model_session import validate_model_route
            for model in models:
                if delegated:
                    transport.bind(model, model.model_id)
                    continue
                validate_model_route(self.owner.context, model)
                session = getattr(self.owner.context, 'model_session', None)
                source = (session.backends.get(model.backend_id) if session is not None else
                          self.owner.context.backends.get(model.backend_id))
                source = source.transport if isinstance(source, AccountedModelBackend) else source
                source_binding = getattr(source, 'binding', None)
                if (source_binding is None or binding.backend_id != source_binding.backend_id
                        or binding.endpoint != source_binding.endpoint):
                    continue
                aliases = getattr(source, 'models', None)
                if isinstance(aliases, dict):
                    if model in aliases:
                        transport.bind(model, aliases[model])
                else:
                    transport.bind(model, model.model_id)
        key = 'custom:' + json.dumps({'name': name, 'models': [asdict(model) for model in models]}, sort_keys=True)
        account = self._account(key, transport, **limits)
        class Metered:
            def __getattr__(self, key):
                return getattr(account, key)

            def generate(self, request):
                if delegated:
                    return account.generate(request, attribution=component.direct_attribution(), budget=None)
                return account.generate(request, attribution=component.direct_attribution())
        return Metered()

    def endpoint(self, config, *, purpose='model', max_calls=1000000, max_concurrent=1,
                 max_input_tokens=None, provider_token_counter=False, response_attempts=1):
        """Bind provider configuration in the host; return an ephemeral worker capability."""
        endpoints = config.get('endpoints') or [config.get('endpoint')]
        model_name = config['model']
        identity = json.dumps({'purpose': purpose, 'endpoints': endpoints, 'model': model_name,
            'max_calls': max_calls, 'max_concurrent': max_concurrent, 'max_input_tokens': max_input_tokens,
            'provider_token_counter': provider_token_counter, 'response_attempts': response_attempts,
            'native_schema_mode': config.get('native_schema_mode', 'exact'),
            'disable_thinking': config.get('disable_thinking', True),
            'credential_env': config.get('credential_env')}, sort_keys=True)
        with self._lock:
            if identity in self._endpoints:
                return _EndpointView(self, self._endpoints[identity])
            if not endpoints or any(not isinstance(url, str) or not url.startswith(('http://', 'https://'))
                                    for url in endpoints):
                raise ValueError('Explicit model endpoints are required')
            if not isinstance(model_name, str) or not model_name:
                raise ValueError('Explicit model name is required')
            backend_id = 'route-' + sha256(identity.encode()).hexdigest()[:16]
            model = ModelRef(backend_id, model_name, 'configured-deployment')
            overrides = getattr(self.owner.context, '_delegated_models', None)
            delegated = None if overrides is None else overrides.get('routes', {}).get(model_route_key(config))
            if overrides is not None and delegated is None:
                raise PermissionError('Model route is outside the nested run delegation')
            wire_endpoints = [delegated['endpoint']] if delegated is not None else endpoints
            transports = [self.owner.transport_factory(BackendBinding(backend_id, 'chat_completions', url,
                credential_env=delegated['credential_env'] if delegated is not None else config.get('credential_env'),
                timeout_seconds=config.get('timeout', 180), json_mode=False),
                models={model: model_name}, native_schema_mode=config.get('native_schema_mode', 'exact'),
                generation_options=({'chat_template_kwargs': {'enable_thinking': not config.get('disable_thinking', True)}}
                    if config.get('disable_thinking', True) is not None else {}))
                for url in wire_endpoints]
            class RoundRobin:
                supports_response_schema = True
                supports_tool_calling = True
                def __init__(self):
                    self.lock, self.index = threading.Lock(), 0

                def generate(self, request):
                    with self.lock:
                        transport = transports[self.index % len(transports)]
                        self.index += 1
                    return transport.generate(request)

                def count_tokens(self, request):
                    return transports[0].count_tokens(request)
            backend = self._account('route:' + identity, RoundRobin(), max_calls=max_calls,
                max_concurrent=max_concurrent, max_input_tokens=max_input_tokens,
                provider_token_counter=provider_token_counter, response_attempts=response_attempts, models=[model])
            session = (HostModelSession({backend_id: backend}, [model], budget=None) if delegated is not None else
                       HostModelSession({backend_id: backend}, [model]))
            endpoint = self.owner.server.issue(self.name, session, attribution_provider=self.attribution)
            self._endpoints[identity] = endpoint
            self._route_configs[model_route_key(config)] = json.loads(json.dumps(config, allow_nan=False))
            return _EndpointView(self, endpoint)

    def background_endpoint(self, config, *, purpose='model', attribution=None, **limits):
        """A revocable model capability for one host-owned background activity."""
        attribution = self.direct_attribution() if attribution is None else dict(attribution)
        if (set(attribution) != {'component', 'decision_id', 'call_id'} or attribution['component'] != self.name
                or any(value is not None and (not isinstance(value, str) or not value)
                       for value in attribution.values())):
            raise ValueError('Invalid background model attribution')
        view = self.endpoint(config, purpose=purpose, **limits)
        def source():
            if self.owner.closed:
                raise PermissionError('Host model access is closed')
            return dict(attribution)
        return self.owner.server.issue(self.name, view._endpoint._session, attribution_provider=source)

    def _environment(self, endpoint):
        scope = self.attribution()
        with self._lock:
            matches = [ident for ident, active in self._active.items() if active == scope]
            if len(matches) != 1:
                raise PermissionError('Worker credentials require one exact active host scope')
            ident = matches[0]
            def attribution():
                with self._lock:
                    if ident not in self._active:
                        raise PermissionError('Worker operation scope has ended')
                    return dict(scope)
            bound = self.owner.server.issue(self.name, endpoint._session, attribution_provider=attribution)
            self._scope_endpoints.setdefault(ident, []).append(bound)
            return bound.environment()

    def pool(self, config, *, purpose='model', max_calls=1000000, max_concurrent=1):
        endpoint = self.endpoint(config, purpose=purpose, max_calls=max_calls, max_concurrent=max_concurrent)
        component = self
        class ScopedPool:
            def __init__(self):
                self.endpoint = endpoint

            def complete(self, *args, **kwargs):
                env = component._environment(endpoint)
                pool = component.owner.pool_factory([env['OPENAI_BASE_URL']], config['model'], api_key=env['OPENAI_API_KEY'],
                    max_tokens=config.get('max_tokens', 2048), temperature=config.get('temperature', .7),
                    timeout=config.get('timeout', 180))
                return pool.complete(*args, **kwargs)
        return ScopedPool()


class HostModelAccess:
    def __init__(self, context, *, pool_factory, transport_factory):
        self.context = context
        self.pool_factory, self.transport_factory = pool_factory, transport_factory
        self._temporary = None
        if context.data_dir is None:
            from tempfile import TemporaryDirectory
            from pathlib import Path
            self._temporary = TemporaryDirectory(prefix='rsisys-models-')
            self.root = Path(self._temporary.name)
        else:
            self.root = context.data_dir / 'model_accounts'
        self._server, self._components, self._nested_handoffs = None, {}, {}
        self.closed = False
        self._lock = threading.RLock()

    @property
    def server(self):
        with self._lock:
            if self._server is None:
                from rsisys.runtime.execution.remote_services import service_addresses
                bind_host, advertise_host = service_addresses(self.context)
                self._server = ComponentModelServer(bind_host=bind_host, advertise_host=advertise_host)
            return self._server

    def component(self, name):
        with self._lock:
            if name not in self._components:
                self._components[name] = ComponentModels(self, name)
            return self._components[name]

    def _register_nested_environment(self, environment, active):
        """Remember the exact output of successful local or upstream delegation."""
        if any(not re.fullmatch(r'RSISYS_NESTED_MODEL_TOKEN_[0-9A-F]{32}', name)
               or not isinstance(value, str) or not value for name, value in environment.items()):
            raise ValueError('Invalid host-created nested model environment')
        ident = object()
        with self._lock:
            if self.closed:
                raise RuntimeError('Host model access is closed')
            self._nested_handoffs[ident] = (dict(environment), active)
            # Upstream-issued tokens do not appear in our local model server.
            getattr(self.context, '_evidence_secrets', set()).update(environment.values())
        def revoke():
            with self._lock:
                self._nested_handoffs.pop(ident, None)
        return revoke

    def nested_environment(self, supplied):
        """Authorize only exact registered values while their allocation is live."""
        if not supplied:
            return {}
        with self._lock:
            handoffs = tuple(self._nested_handoffs.items())
        environment = {}
        for ident, (issued, active) in handoffs:
            try:
                active()
            except PermissionError:
                continue
            with self._lock:
                if not self.closed and ident in self._nested_handoffs:
                    environment.update({key: value for key, value in issued.items() if supplied.get(key) == value})
        return environment

    def close(self):
        with self._lock:
            self.closed = True
            self._nested_handoffs.clear()
        if self._server is not None:
            self._server.close()
        if self._temporary is not None:
            self._temporary.cleanup()
