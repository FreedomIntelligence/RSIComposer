"""Discover installed method packages through their register(registry) entry."""
from importlib import import_module
from pkgutil import iter_modules


def register_builtins(registry):
    for info in sorted(iter_modules(__path__), key=lambda item: item.name):
        if not info.ispkg or info.name.startswith('_'):
            continue
        module = import_module(f'{__name__}.{info.name}')
        register = getattr(module, 'register', None)
        if not callable(register) or register(registry) is not None:
            raise TypeError(f'Method package {info.name} must define register(registry) returning None')
