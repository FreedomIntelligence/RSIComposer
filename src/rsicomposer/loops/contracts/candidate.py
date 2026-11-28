from copy import deepcopy
from dataclasses import dataclass, field

from .artifacts import ArtifactRef
from .model_service import Role


@dataclass(frozen=True)
class Candidate:
    """Training artifacts and their intended roles, not permission to install changes."""
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
