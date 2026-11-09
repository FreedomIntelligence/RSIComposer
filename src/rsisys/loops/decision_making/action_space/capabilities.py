"""Explicit, instance-local capability composition; no automatic discovery or transport clients."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.decision_making.action_space import DecisionActionDefinition

Validator = Callable[[dict[str, object]], OperationCall]
Handler = Callable[[dict[str, object]], OperationResult | JobRef]


@dataclass(frozen=True)
class _SpecDefinition:
    spec: OperationSpec
    validate: Validator


@dataclass(frozen=True)
class Capability:
    """A validation definition and its explicitly supplied execution handler."""

    definition: DecisionActionDefinition
    handler: Handler

    def __post_init__(self):
        spec = getattr(self.definition, "spec", None)
        if not isinstance(spec, OperationSpec):
            raise TypeError("Capability definition requires an OperationSpec")
        if not isinstance(spec.name, str) or not spec.name.strip():
            raise ValueError("Capability names must be nonempty strings")
        if not callable(getattr(self.definition, "validate", None)):
            raise TypeError(f"Capability requires a validator: {spec.name}")
        if not callable(self.handler):
            raise TypeError(f"Capability requires a callable handler: {spec.name}")

    @classmethod
    def from_spec(cls, spec: OperationSpec, validator: Validator, handler: Handler) -> "Capability":
        return cls(_SpecDefinition(spec, validator), handler)


class CapabilityProvider(Protocol):
    """Describe already configured capabilities, including their handlers."""

    def describe(self) -> Iterable[Capability]: ...


class Capabilities:
    """Validate a finite collection before injecting definitions and bindings."""

    def __init__(self, capabilities: Iterable[Capability] = ()):
        items = tuple(capabilities)
        names: set[str] = set()
        for capability in items:
            if not isinstance(capability, Capability):
                raise TypeError("Expected Capability instances with definitions and handlers")
            capability.__post_init__()
            name = capability.definition.spec.name
            if name in names:
                raise ValueError(f"Capability names must be unique: {name}")
            names.add(name)
        self._items = items

    @classmethod
    def from_provider(cls, provider: object, *, method: Literal["describe", "discover"] = "describe",
                      explicit: Iterable[Capability] = ()) -> "Capabilities":
        """Call only the selected provider method once; propagate provider failures."""
        if method not in ("describe", "discover"):
            raise ValueError("Provider method must be describe or discover")
        describe = getattr(provider, method, None)
        if not callable(describe):
            raise TypeError(f"Capability provider requires callable {method}()")
        return cls((*describe(), *explicit))

    def __iter__(self) -> Iterator[Capability]:
        return iter(self._items)

    @property
    def definitions(self) -> tuple[DecisionActionDefinition, ...]:
        return tuple(capability.definition for capability in self._items)

    @property
    def bindings(self) -> dict[str, Handler]:
        return {capability.definition.spec.name: capability.handler for capability in self._items}
