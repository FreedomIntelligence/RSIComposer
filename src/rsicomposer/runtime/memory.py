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


def memory_host(harness):
    factory = optional_method(harness, 'memory_host')
    host = factory() if factory is not None else None
    if any(optional_method(host, name) is None for name in ('history', 'install')):
        raise TypeError('Research memory requires a memory_host() port with history() and install(loop, memory)')
    return host
