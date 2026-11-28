"""Resolve explicitly declared memory lifecycle and host capabilities."""
from rsicomposer.runtime.capabilities import optional_method


def memory_lifecycle(memory):
    factory = optional_method(memory, 'run_lifecycle')
    if factory is None:
        return None
    lifecycle = factory()
    if any(optional_method(lifecycle, name) is None
           for name in ('prepare', 'validate_host', 'for_run', 'install')):
        raise TypeError('Memory run_lifecycle() requires prepare, validate_host, for_run and install')
    return lifecycle
