"""Execution boundary. The mock never invokes a shell or creates a container."""

from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import uuid4

from rsisys.loops.contracts.jobs import JobStatus


@dataclass(frozen=True)
class SandboxSpec:
    image_ref: str = "mock"
    timeout_seconds: int = 60

    def __post_init__(self):
        import math
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("sandbox timeout must be positive")


@dataclass(frozen=True)
class SandboxRef:
    sandbox_id: str


@dataclass(frozen=True)
class ExecRequest:
    argv: tuple[str, ...]
    timeout_seconds: int = 30

    def __post_init__(self):
        if (not isinstance(self.argv, (list, tuple)) or not self.argv
                or not all(isinstance(arg, str) and '\0' not in arg for arg in self.argv)
                or not self.argv[0]):
            raise ValueError("argv must contain strings")
        object.__setattr__(self, 'argv', tuple(self.argv))
        import math
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("execution timeout must be positive")


@dataclass(frozen=True)
class ProcessRef:
    process_id: str
    sandbox: SandboxRef


@dataclass(frozen=True)
class ProcessState:
    status: JobStatus
    stdout: str
    stderr: str
    exit_code: int | None
    simulated: bool
    failure_origin: Literal['program', 'timeout', 'infrastructure'] | None = None


@dataclass(frozen=True)
class SandboxCapabilities:
    simulated: bool
    snapshot: bool = False
    fork: bool = False


class SandboxBackend(Protocol):
    def capabilities(self) -> SandboxCapabilities: ...

    def create(self, spec: SandboxSpec) -> SandboxRef: ...

    def exec(self, sandbox: SandboxRef, request: ExecRequest) -> ProcessRef: ...

    def poll(self, process: ProcessRef) -> ProcessState: ...

    def cancel(self, process: ProcessRef) -> None: ...

    def close(self, sandbox: SandboxRef) -> None: ...


class MockSandboxBackend:
    """Only models an immediate `echo`; unsupported commands fail explicitly.

    Timeout values are descriptive in this immediate mock; no isolation,
    real process execution, or provider snapshot guarantees are provided.
    """

    def __init__(self):
        self._active: set[SandboxRef] = set()
        self._processes: dict[ProcessRef, ProcessState] = {}
        self.execution_count = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(simulated=True)

    def create(self, spec: SandboxSpec) -> SandboxRef:
        if spec.image_ref != "mock":
            raise NotImplementedError("Only the mock sandbox image is available")
        ref = SandboxRef(str(uuid4()))
        self._active.add(ref)
        return ref

    def exec(self, sandbox: SandboxRef, request: ExecRequest) -> ProcessRef:
        if sandbox not in self._active:
            raise ValueError("Sandbox is closed or unknown")
        self.execution_count += 1
        process = ProcessRef(str(uuid4()), sandbox)
        if request.argv[0] == "echo":
            state = ProcessState(JobStatus.SUCCEEDED, " ".join(request.argv[1:]) + "\n", "", 0, True)
        else:
            state = ProcessState(JobStatus.FAILED, "", "Unsupported mock command", 127, True)
        self._processes[process] = state
        return process

    def poll(self, process: ProcessRef) -> ProcessState:
        return self._processes[process]

    def cancel(self, process: ProcessRef) -> None:
        state = self._processes[process]
        if state.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            self._processes[process] = ProcessState(JobStatus.CANCELLED, state.stdout, state.stderr, None, True)

    def close(self, sandbox: SandboxRef) -> None:
        self._active.discard(sandbox)
