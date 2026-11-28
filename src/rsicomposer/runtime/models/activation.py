"""Host readiness validation before a method updates a model-role binding."""
from copy import deepcopy

from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.runtime.capabilities import optional_method


def validate_activation_route(backend, base: ModelRef, model: ModelRef, *, prepared: bool) -> None:
    """Ask a serving backend how it addresses an exact trained revision.

    Before activation, reject routes that deployment cannot make addressable.
    After activation, reject a missing exact binding. This port is read-only:
    it neither deploys nor grants access. Local backends with no special routing
    constraints need only the activator's exact readiness receipt.
    """
    check = optional_method(backend, 'validate_activation_route')
    if check is not None and check(base, model, prepared=prepared) is not None:
        raise TypeError('validate_activation_route must return None or raise')


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
