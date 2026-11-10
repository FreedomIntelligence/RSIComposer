"""Optional reusable harness implementations; loop mechanics do not import these."""


def _python_step(context, config):
    from functools import partial
    from .python_step import sandbox_step
    from rsisys.runtime.orchestration.run_policy import positive
    if set(config) - {'timeout'}:
        raise ValueError('Unknown python_step harness option')
    timeout = config.get('timeout', 5)
    positive(timeout, 'timeout')
    return partial(sandbox_step, timeout=timeout)


def _agent_memory(context, config):
    from .decision.agent_memory import AgentMemory
    return AgentMemory(**config)


def register_components(registry):
    """Register reusable implementations; methods own binding and cleanup."""
    registry.register_component('harness', 'python_step', _python_step, version='1')
    registry.register_component('memory', 'agent_memory', _agent_memory, version='1')
