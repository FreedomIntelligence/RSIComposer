"""Collect host-owned credentials. Only values of at least 8 characters match inside text."""
import os

from rsicomposer.runtime.evidence.view import _SECRET_KEY


_PLACEHOLDERS = {'test', 'dummy', 'placeholder', 'changeme', 'none', 'null'}


def _configured_secrets(value, environment):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'credential_env' and isinstance(item, str):
                yield environment.get(item)
            elif isinstance(key, str) and _SECRET_KEY.fullmatch(key) and isinstance(item, str):
                yield item
            else:
                yield from _configured_secrets(item, environment)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _configured_secrets(item, environment)


def host_secret_values(context):
    """Read explicit host routes and capabilities, never arbitrary object graphs.

    Keep discovered values for the lifetime of this host: revoked capabilities or
    removed credentials may still occur in previously recorded evidence.
    """
    environment = dict(os.environ)
    values = [value for key, value in environment.items() if _SECRET_KEY.fullmatch(key)]
    for config in (getattr(context, 'host_config', {}), getattr(context, '_delegated_models', {})):
        values.extend(_configured_secrets(config, environment))
    backends = list(context.backends.values())
    session = getattr(context, 'model_session', None)
    if session is not None:
        backends.extend(session.backends.values())
    access = getattr(context, '_model_access', None)
    if access is not None:
        with access._lock:
            components = tuple(access._components.values())
            server = access._server
        if server is not None:
            values.extend(server.secret_values)
        for component in components:
            with component._lock:
                configs = tuple(component._route_configs.values())
                backends.extend(component._accounts.values())
            for config in configs:
                values.extend(_configured_secrets(config, environment))
    for backend in backends:
        transport = getattr(backend, 'transport', backend)
        binding = getattr(transport, 'binding', None)
        variable = getattr(binding, 'credential_env', None)
        if variable:
            values.append(environment.get(variable))
        values.append(getattr(transport, 'api_key', None))
    services = getattr(context, '_worker_services', None)
    if services is not None:
        values.extend(services.secret_values)
    context._evidence_secrets.update(value for value in values
                                    if isinstance(value, str) and value.strip()
                                    and value.strip().casefold() not in _PLACEHOLDERS)
    return tuple(context._evidence_secrets)
