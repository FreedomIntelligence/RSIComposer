"""Optional backend contracts; no discovery, transport or deployment implementation."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .artifacts import ArtifactRef
from .jobs import JobOutcome, JobRef
from .model_service import ModelRef
from .training import ResumeMode, TrainingOrigin, TrainingResult


def _names(value: tuple[str, ...], name: str) -> None:
    if (not isinstance(value, tuple)
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{name} must be a tuple of unique nonempty strings')


@dataclass(frozen=True)
class BackendCapabilities:
    """Explicit opt-in; empty fields advertise no support, not a wildcard.

    Checkpoint support means exporting a job checkpoint. Consuming an existing
    checkpoint is governed separately by resume_modes. Declarations do not
    replace backend validation of recipes, data, checkpoint content or updates.
    """

    methods: tuple[str, ...] = ()
    resume_modes: tuple[ResumeMode, ...] = (ResumeMode.RESTART,)
    update_fields: tuple[str, ...] = ()
    supports_checkpoint: bool = False
    supports_reattach: bool = False

    def __post_init__(self):
        _names(self.methods, 'methods')
        _names(self.update_fields, 'update_fields')
        if (not isinstance(self.resume_modes, tuple)
                or any(not isinstance(mode, ResumeMode) for mode in self.resume_modes)
                or len(set(self.resume_modes)) != len(self.resume_modes)):
            raise ValueError('resume_modes must be a tuple of unique ResumeMode values')
        if type(self.supports_checkpoint) is not bool or type(self.supports_reattach) is not bool:
            raise TypeError('Checkpoint and reattach support flags must be bools')


@runtime_checkable
class BackendCapabilityProvider(Protocol):
    """Optional on existing learners; absence preserves their existing API."""

    def capabilities(self) -> BackendCapabilities: ...


def _validate(capabilities: BackendCapabilities) -> None:
    if not isinstance(capabilities, BackendCapabilities):
        raise TypeError('Expected BackendCapabilities')
    capabilities.__post_init__()


def validate_training_capabilities(capabilities: BackendCapabilities, *, method: str,
                                   origin: TrainingOrigin | None = None) -> None:
    """Fail before submission when a declared method/resume mode is unsupported."""
    _validate(capabilities)
    if not isinstance(method, str) or not method.strip():
        raise ValueError('method must be a nonempty string')
    if origin is not None:
        if not isinstance(origin, TrainingOrigin):
            raise TypeError('origin must be TrainingOrigin or None')
        origin.__post_init__()
    if method not in capabilities.methods:
        raise NotImplementedError(f'Unsupported training method: {method}')
    mode = origin.resume_mode if origin is not None else ResumeMode.RESTART
    if mode not in capabilities.resume_modes:
        raise NotImplementedError(f'Unsupported resume mode: {mode}')


def validate_update_capabilities(capabilities: BackendCapabilities, fields: tuple[str, ...]) -> None:
    """Validate field names only; the backend still validates values and versions."""
    _validate(capabilities)
    _names(fields, 'fields')
    if not fields:
        raise ValueError('An update requires at least one field')
    unsupported = set(fields) - set(capabilities.update_fields)
    if unsupported:
        raise NotImplementedError(f'Unsupported update fields: {", ".join(sorted(unsupported))}')


def require_checkpoint(capabilities: BackendCapabilities) -> None:
    _validate(capabilities)
    if not capabilities.supports_checkpoint:
        raise NotImplementedError('Backend does not support checkpoint export')


def require_reattach(capabilities: BackendCapabilities) -> None:
    _validate(capabilities)
    if not capabilities.supports_reattach:
        raise NotImplementedError('Backend does not support reattach')


class CheckpointingTrainingBackend(Protocol):
    """Export an actual checkpoint, not a controller snapshot or guessed URI."""

    def checkpoint(self, job: JobRef, *, idempotency_key: str) -> ArtifactRef: ...


class ReattachableTrainingBackend(Protocol):
    """Observe the same durable job after reconnecting; never submit a new job."""

    def reattach(self, job: JobRef) -> JobOutcome[TrainingResult]: ...


@dataclass(frozen=True)
class DeploymentRef:
    deployment_id: str
    backend: str

    def __post_init__(self):
        for name in ('deployment_id', 'backend'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')


class DeploymentStatus(StrEnum):
    PREPARING = 'preparing'
    READY = 'ready'
    RELEASING = 'releasing'
    RELEASED = 'released'
    FAILED = 'failed'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class DeploymentState:
    """Readiness is an explicit serving observation, independent of job success.

    UNKNOWN never implies failure or readiness. A successful deployment job alone
    does not imply READY. READY confirms this exact model is available to infer.
    The optional job identifies external work, not the source training job.
    """

    deployment: DeploymentRef
    model: ModelRef
    status: DeploymentStatus
    job: JobRef | None = None
    error: str | None = None

    def __post_init__(self):
        if not isinstance(self.deployment, DeploymentRef):
            raise TypeError('deployment must be a DeploymentRef')
        self.deployment.__post_init__()
        if not isinstance(self.model, ModelRef) or any(
                not isinstance(value, str) or not value.strip()
                for value in (self.model.backend_id, self.model.model_id, self.model.revision)):
            raise ValueError('model must be a ModelRef with nonempty identity fields')
        if not isinstance(self.status, DeploymentStatus):
            raise TypeError('status must be a DeploymentStatus')
        if self.job is not None and (not isinstance(self.job, JobRef) or any(
                not isinstance(value, str) or not value.strip()
                for value in (self.job.job_id, self.job.backend))):
            raise ValueError('job must be a JobRef with nonempty identity fields or None')
        if self.error is not None and (not isinstance(self.error, str) or not self.error.strip()):
            raise ValueError('error must be a nonempty string or None')
        if self.status == DeploymentStatus.FAILED and self.error is None:
            raise ValueError('A failed deployment requires an error')
        if self.error is not None and self.status not in (DeploymentStatus.FAILED, DeploymentStatus.UNKNOWN):
            raise ValueError('Only failed or unknown deployments may carry an error')

    @property
    def ready(self) -> bool:
        return self.status == DeploymentStatus.READY


class DeploymentBackend(Protocol):
    """Serving lifecycle without changing role bindings or implementing a harness.

    Same key and model must prepare the same deployment; conflicting reuse fails.
    Query and reattach observe existing work and preserve deployment/model/job
    identities. Release is idempotent and may remain RELEASING until confirmed;
    it never means model deletion. Unsupported operations raise NotImplementedError.
    A ModelActivator adapter may return a model only after observing READY and
    checking exact model equality. These contracts do not perform activation.
    """

    def prepare(self, model: ModelRef, *, idempotency_key: str) -> DeploymentState: ...
    def query(self, deployment: DeploymentRef) -> DeploymentState: ...
    def release(self, deployment: DeploymentRef, *, idempotency_key: str) -> DeploymentState: ...


class ReattachableDeploymentBackend(DeploymentBackend, Protocol):
    """Optional reconnect to the same external deployment, never a new prepare."""

    def reattach(self, deployment: DeploymentRef) -> DeploymentState: ...
