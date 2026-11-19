from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactRef:
    """Data-only reference; its consumer owns URI semantics and content access."""

    artifact_id: str
    kind: str
    uri: str
    version: str | None = None
    digest: str | None = None

    def __post_init__(self):
        for name in ('artifact_id', 'kind', 'uri'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('version', 'digest'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{name} must be a nonempty string or None')
