from dataclasses import dataclass, field

from rsisys.loops.contracts.model_service import ModelRef


@dataclass(frozen=True)
class SnapshotManifest:
    """Local quiescent snapshot manifest; external weights and jobs are not captured.

    References must be immutable and secret-free. A remote model reference alone
    does not guarantee that its provider can restore the requested revision.
    """

    snapshot_id: str
    parent_snapshot_id: str | None
    role_bindings: dict[str, ModelRef]
    loop_state_refs: dict[str, str]
    learner_state_refs: dict[str, str]
    environment_state_refs: dict[str, str]
    experience_revision: str
    runtime_state_ref: str
    code_revision: str
    config_ref: str
    dependency_lock_ref: str
    component_refs: dict[str, str]
    random_state_refs: tuple[str, ...]
    resume_policy: dict[str, str]
    schema_version: int = 33
    kind: str = "component_quiescent"
    label: str = ""
    file_digests: dict[str, str] = field(default_factory=dict)
