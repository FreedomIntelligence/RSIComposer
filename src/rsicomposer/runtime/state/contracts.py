"""Shared recovery contracts and selection, independent of snapshot storage."""
from typing import Protocol

from rsicomposer.runtime.capabilities import optional_method


class RecoveryContract(Protocol):
    """Capture is observational; restore reattaches state without submitting work.

    Include dependent policy/backend/tool state, durable job IDs and receipts.
    The caller supplies the original factories again; never serialize credentials
    or callables, and reject state that cannot safely reattach to existing effects.
    """
    contract_id: str
    def capture(self, application) -> object: ...
    def restore(self, application, state: object) -> None: ...


class SnapshotContract(RecoveryContract, Protocol):
    """Portable capture additionally validates a quiescent, restorable boundary.

    validate_snapshot returns None or raises before capture. Supplying a durable
    contract without this port does not authorize a fallback portable serializer.
    """
    def validate_snapshot(self, application) -> None: ...


def validate_contract(contract, *, portable=False):
    methods = ('capture', 'restore', 'validate_snapshot') if portable else ('capture', 'restore')
    if (not isinstance(getattr(contract, 'contract_id', None), str) or not contract.contract_id.strip()
            or any(optional_method(contract, name) is None for name in methods)):
        raise TypeError('Recovery contract needs contract_id and callable ports: ' + ', '.join(methods))
    return contract


def select_contract(supplied=None, *, has_state_adapters=False, portable=False):
    """Explicit contract or assembled state ports.

    Portable snapshots never silently substitute a different contract for the
    active durable one; callers must provide explicit portable integration.
    """
    from rsicomposer.runtime.state.component_state import AssemblyRecovery
    if supplied is not None:
        contract = supplied
    elif has_state_adapters:
        contract = AssemblyRecovery()
    else:
        raise TypeError('Recovery needs an explicit contract or assembled state adapters')
    return validate_contract(contract, portable=portable)
