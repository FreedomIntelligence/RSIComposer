from copy import deepcopy
from dataclasses import dataclass

from .artifacts import ArtifactRef
from .model_service import ModelRef, Role


@dataclass(frozen=True)
class SystemVersion:
    """Immutable identity for a composition; component formats belong to providers."""
    version: str
    components: dict[str, ArtifactRef]
    model: ModelRef | None = None
    parent_version: str | None = None

    def __post_init__(self):
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError('System version must be nonempty')
        if self.parent_version is not None and (not isinstance(self.parent_version, str)
                or not self.parent_version.strip() or self.parent_version == self.version):
            raise ValueError('Parent must identify a different system version')
        if not isinstance(self.components, dict) or any(
                not isinstance(k, str) or not k.strip() or not isinstance(v, ArtifactRef)
                for k, v in self.components.items()):
            raise TypeError('Components must map nonempty names to ArtifactRef')
        if self.model is not None and not isinstance(self.model, ModelRef):
            raise TypeError('Model must be ModelRef or None')
        object.__setattr__(self, 'components', deepcopy(self.components))


@dataclass(frozen=True)
class Candidate:
    """Proposed component changes; training base and experiment parent are distinct."""
    candidate_id: str
    base_version: str
    changes: dict[str, ArtifactRef]
    affected_roles: tuple[Role, ...]
    parent_system: str | None = None

    def __post_init__(self):
        if self.parent_system is not None and (not isinstance(self.parent_system, str)
                or not self.parent_system.strip()):
            raise ValueError('parent_system must be nonempty or None')
        object.__setattr__(self, 'changes', deepcopy(self.changes))
