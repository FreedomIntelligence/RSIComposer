from dataclasses import dataclass
from typing import Protocol

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.runtime.artifacts import ArtifactStore
from rsisys.runtime.sandbox import ExecRequest, SandboxBackend, SandboxRef, SandboxSpec

from ..action_space import Action


@dataclass(frozen=True)
class Transition:
    observation: str
    terminated: bool
    simulated: bool
    reward: float | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    error: str | None = None


class Environment(Protocol):
    def reset(self, directive: Directive) -> str: ...

    def step(self, action: Action) -> Transition: ...

    def close(self) -> None: ...


class SandboxEchoEnvironment:
    """Demo task semantics on a sandbox backend that completes exec immediately."""

    def __init__(self, sandbox: SandboxBackend, artifacts: ArtifactStore):
        self.sandbox, self.artifacts = sandbox, artifacts
        self._session: SandboxRef | None = None

    def reset(self, directive: Directive) -> str:
        self.close()
        inputs = [self.artifacts.get(ref).decode() for ref in directive.input_refs]
        self._session = self.sandbox.create(SandboxSpec())
        return f"Task: {directive.objective}; prepared inputs: {inputs}"

    def step(self, action: Action) -> Transition:
        if self._session is None:
            raise RuntimeError("Reset the environment before acting")
        if action.name != "echo":
            raise ValueError("Environment does not support this action")
        process = self.sandbox.exec(self._session, ExecRequest(("echo", action.text)))
        result = self.sandbox.poll(process)
        if result.status not in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            raise NotImplementedError("Demo environment requires immediate command completion")
        ref = self.artifacts.put((result.stdout + result.stderr).encode(), kind="execution_output")
        error = result.stderr or f"exit={result.exit_code}" if result.status == JobStatus.FAILED else None
        return Transition(result.stdout, True, result.simulated, artifacts=(ref,), error=error)

    def close(self) -> None:
        if self._session is not None:
            self.sandbox.close(self._session)
            self._session = None
