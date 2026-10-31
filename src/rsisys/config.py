"""Composition root. Business flows stay inside loops; backends are injected."""

from dataclasses import dataclass
import json
from pathlib import Path

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace
from rsisys.loops.decision_making.harness.tools import DemoDecisionTools
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.environment import SandboxEchoEnvironment
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.loops.exploration.harness.learner.backends.mock import MockActorLearner
from rsisys.loops.exploration.harness.evaluation.mock import MockEvaluator
from rsisys.loops.exploration.harness.policy import ModelActorPolicy
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.model import MockModelBackend
from rsisys.runtime.runner import Runner
from rsisys.runtime.sandbox import MockSandboxBackend


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
        backends = {key: BackendBinding(backend_id=key, **value) for key, value in raw["backends"].items()}
        roles = {key: ModelRef(**value) for key, value in raw["roles"].items()}
        if set(roles) != {"decision_maker", "actor"}:
            raise ValueError("Exactly decision_maker and actor role bindings are required")
        for model in roles.values():
            if model.backend_id not in backends:
                raise ValueError(f"Unknown model backend: {model.backend_id}")
        return cls(raw["objective"], raw["rounds"], backends, roles)


def build_demo(config: DemoConfig, *, with_learning: bool = False) -> tuple[Runner, InMemoryExperienceStore]:
    if any(binding.kind != "mock" for binding in config.backends.values()):
        raise NotImplementedError("Only mock backends are assembled by the demo.")
    backends = {key: MockModelBackend() for key in config.backends}
    store = InMemoryExperienceStore()
    experience = BasicExperienceSystem(store)
    artifacts = InMemoryArtifactStore()
    sandbox = MockSandboxBackend()
    decision_model, actor_model = config.roles["decision_maker"], config.roles["actor"]
    modes = ("execute", "configure_prompt") + (("sft", "rl", "evaluate") if with_learning else ())
    decision = DecisionMakingLoop(
        DefaultDecisionHarness(
            ModelDecisionPolicy(backends[decision_model.backend_id], decision_model),
            DemoDecisionTools(sandbox, artifacts), experience,
            learner=MockDecisionLearner(artifacts) if with_learning else None,
        ), DemoDecisionActionSpace(directive_modes=modes),
    )
    exploration = ExplorationLoop(
        DefaultExplorationHarness(
            ModelActorPolicy(backends[actor_model.backend_id], actor_model),
            SandboxEchoEnvironment(sandbox, artifacts), experience,
            learner=MockActorLearner(artifacts) if with_learning else None,
            evaluator=MockEvaluator(artifacts) if with_learning else None,
        ), EchoActionSpace(),
    )
    return Runner(decision, exploration), store
