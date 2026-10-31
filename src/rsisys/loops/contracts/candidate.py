from dataclasses import dataclass

from .artifacts import ArtifactRef
from .model_service import Role


@dataclass(frozen=True)
class Candidate:
    """Versioned proposed changes, not an accepted version or a system snapshot."""

    candidate_id: str
    base_version: str
    changes: dict[str, ArtifactRef]
    affected_roles: tuple[Role, ...]
