"""Composition root. Business flows stay inside loops; backends are injected."""

from dataclasses import dataclass
import json
from pathlib import Path

from rsisys.assembly import (
    AssemblyContext, AssemblyRunner, DecisionFactory, ExplorationFactory, LabExecutionFactory,
    assemble_runner, module_assembly,
)
from rsisys.runtime.evidence.retrieval import BasicExperienceSystem
from rsisys.runtime.evidence.storage import InMemoryExperienceStore
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.runtime.orchestration.run_policy import RunPolicy
from rsisys.runtime.storage.artifacts import InMemoryArtifactStore
from rsisys.runtime.models.model import MockModelBackend
from rsisys.runtime.execution.sandbox import MockSandboxBackend


@dataclass(frozen=True)
class DemoConfig:
    objective: str
    rounds: int
    backends: dict[str, BackendBinding]
    roles: dict[str, ModelRef]

    @classmethod
    def load(cls, path: Path) -> "DemoConfig":
        raw = json.loads(path.read_text())
        if raw["mode"] != "demo":
            raise NotImplementedError("Only the explicit demo mode is available.")
        if type(raw["rounds"]) is not int or raw["rounds"] < 1:
            raise ValueError("rounds must be a positive integer")
        backends = {key: BackendBinding(backend_id=key, **value) for key, value in raw.get("backends", {}).items()}
        roles = {key: ModelRef(**value) for key, value in raw.get("roles", {}).items()}
        if set(roles) - {'decision_maker', 'actor'}:
            raise ValueError('Unknown model role; use decision_maker or actor')
        for model in roles.values():
            if model.backend_id not in backends:
                raise ValueError(f"Unknown model backend: {model.backend_id}")
        return cls(raw["objective"], raw["rounds"], backends, roles)


def build_demo(config: DemoConfig, *, with_learning: bool = False,
               decision_factory: DecisionFactory | None = None,
               exploration_factory: ExplorationFactory | None = None,
               lab_factory: LabExecutionFactory | None = None,
               action_permissions: ActionPermissions | None = None, run_policy: RunPolicy | None = None,
               cleanup_timeout_seconds: float = 30, research=None, research_backend=None,
               feedback_adapter_factory=None, modules=None) -> tuple[AssemblyRunner, InMemoryExperienceStore]:
    modules = module_assembly(modules, decision_factory=decision_factory,
        exploration_factory=exploration_factory, lab_factory=lab_factory,
        research=research, research_backend=research_backend, feedback_adapter_factory=feedback_adapter_factory)
    if action_permissions is not None and not isinstance(action_permissions, ActionPermissions):
        raise TypeError('Expected ActionPermissions')
    if any(binding.kind != "mock" for binding in config.backends.values()):
        raise NotImplementedError("Only mock backends are assembled by the demo.")
    backends = {key: MockModelBackend() for key in config.backends}
    store = InMemoryExperienceStore()
    experience = BasicExperienceSystem(store)
    artifacts = InMemoryArtifactStore()
    sandbox = MockSandboxBackend()
    context = AssemblyContext(
        experience=experience, artifacts=artifacts, backends=backends, sandbox=sandbox,
        roles=dict(config.roles), with_learning=with_learning,
        action_permissions=action_permissions if action_permissions is not None else ActionPermissions(),
    )
    for backend in backends.values():
        if callable(getattr(backend, 'close', None)):
            context.add_cleanup(backend.close)
    if callable(getattr(sandbox, 'close_all', None)):
        context.add_owned_cleanup(sandbox.close_all)
    try:
        return assemble_runner(context, modules=modules, run_policy=run_policy,
                               cleanup_timeout_seconds=cleanup_timeout_seconds), store
    except BaseException as exc:
        try:
            context.close()
        except BaseException as cleanup_error:
            exc.add_note(f"Assembly cleanup also failed: {cleanup_error}")
        raise
