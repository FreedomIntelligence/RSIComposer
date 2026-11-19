"""Host readiness validation before a method updates a model-role binding."""
from copy import deepcopy

from rsisys.loops.contracts.model_service import ModelRef


def prepare_model_activation(result, activator, *, idempotency_key: str) -> ModelRef:
    """Confirm availability independently of changing a policy's role binding."""
    if activator is None:
        if result.simulated:
            return result.model  # Mock references do not represent deployed weights.
        raise RuntimeError('Real training results require a host model activation provider before adoption')
    ready = activator.activate(deepcopy(result), idempotency_key=idempotency_key)
    if not isinstance(ready, ModelRef) or ready != result.model:
        raise ValueError('Activated model must match the requested trained version')
    return ready
