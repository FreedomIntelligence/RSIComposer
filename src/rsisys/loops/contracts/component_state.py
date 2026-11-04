"""Optional persistence ports, independent of module internals and execution."""
from typing import Protocol


class ComponentState(Protocol):
    """An integration owns the complete state of one assembled component.

    state_id versions the adapter and its interpretation. capture is read-only;
    restore validates and attaches existing state, never launches or resubmits
    work. Values cross the host's explicit JSON codec; no dynamic class loading.
    Include configuration, identities, receipts and dependent private state, or
    reject capture/restore when those cannot be represented safely.
    """
    state_id: str

    def capture(self) -> object: ...
    def restore(self, state: object) -> None: ...


class PortableComponentState(ComponentState, Protocol):
    """Opt-in portable snapshot boundary, checked before capture and after restore.

    Raise unless all effects are terminal and all required data is in capture or
    immutable resolvable references. No running tasks, credentials, private paths
    or assumptions about a previous workspace. No execution inside validation.
    """
    def validate_snapshot(self) -> None: ...
