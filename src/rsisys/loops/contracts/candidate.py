from copy import deepcopy
from dataclasses import dataclass, field

from .artifacts import ArtifactRef
from .model_service import ModelRef, Role


@dataclass(frozen=True)
class SystemVersion:
    """Immutable identity for a composition; component formats belong to providers."""
    version: str
    components: dict[str, ArtifactRef]
    model: ModelRef | None = None
    parent_version: str | None = None
    # Named targets distinguish meta/decision and Lab/actor models. `model`
    # remains the default binding for legacy manifests and readers.
    models: dict[str, ModelRef] = field(default_factory=dict)

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
        if not isinstance(self.models, dict) or any(not isinstance(k, str) or not k.strip()
                or not isinstance(v, ModelRef) for k, v in self.models.items()):
            raise TypeError('Models must map nonempty target names to ModelRef')
        object.__setattr__(self, 'models', deepcopy(self.models))

    def model_for(self, target):
        # A named manifest is explicit: do not silently bind an undeclared role
        # to the legacy single-model alias.
        return self.models.get(target) if self.models else self.model


@dataclass(frozen=True)
class Candidate:
    """Proposed component changes; training base and experiment parent are distinct."""
    candidate_id: str
    base_version: str
    changes: dict[str, ArtifactRef]
    affected_roles: tuple[Role, ...]
    parent_system: str | None = None
    model_bases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError('Candidate ID must be nonempty')
        if not isinstance(self.base_version, str) or not self.base_version.strip():
            raise ValueError('Candidate base version must be nonempty')
        if not isinstance(self.affected_roles, (list, tuple)) or any(
                not isinstance(v, str) or not v.strip() for v in self.affected_roles):
            raise ValueError('Candidate affected_roles must name targets')
        object.__setattr__(self, 'affected_roles', tuple(self.affected_roles))
        if self.parent_system is not None and (not isinstance(self.parent_system, str)
                or not self.parent_system.strip()):
            raise ValueError('parent_system must be nonempty or None')
        object.__setattr__(self, 'changes', deepcopy(self.changes))
        if not isinstance(self.model_bases, dict) or any(not isinstance(k, str) or not k.strip()
                or not isinstance(v, str) or not v.strip() for k, v in self.model_bases.items()):
            raise TypeError('Model bases must map target names to revisions')
        object.__setattr__(self, 'model_bases', deepcopy(self.model_bases))
