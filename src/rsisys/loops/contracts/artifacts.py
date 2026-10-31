from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    uri: str
    version: str
    digest: str
