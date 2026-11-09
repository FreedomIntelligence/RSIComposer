"""Explicit composition of the bundled decision and Actor harnesses.

These are optional implementation choices, loaded only when selected by an assembly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rsisys.harnesses.actor.action_space import CommandActionSpace, EchoActionSpace
from rsisys.harnesses.actor.commands import ExplorationCommands
from rsisys.harnesses.actor.engine import DefaultExplorationHarness
from rsisys.harnesses.actor.environment import SandboxEchoEnvironment
from rsisys.harnesses.actor.environment.local import LocalProcessEnvironment
from rsisys.harnesses.actor.evaluation.mock import MockEvaluator
from rsisys.harnesses.actor.learner.backends.mock import MockActorLearner
from rsisys.harnesses.actor.memory import ExplorationMemory
from rsisys.harnesses.actor.policy import ModelActorPolicy
from rsisys.harnesses.decision.action_space import (
    DemoDecisionActionSpace,
    LocalDecisionActionSpace,
)
from rsisys.harnesses.decision.engine import DefaultDecisionHarness
from rsisys.harnesses.decision.learner.backends.mock import MockDecisionLearner
from rsisys.harnesses.decision.memory import DecisionMemory
from rsisys.harnesses.decision.policy import ModelDecisionPolicy
from rsisys.harnesses.decision.research import ResearchAgenda
from rsisys.harnesses.decision.tools import DemoDecisionTools
from rsisys.harnesses.decision.tools.local import LocalDecisionTools
from rsisys.harnesses.decision.tools.queries import QueryTools
from rsisys.harnesses.decision.tools.research import ResearchTools
from rsisys.loops.contracts.backend import BackendCapabilityProvider
from rsisys.loops.contracts.model_service import ModelRef, ModelUpdatable
from rsisys.loops.contracts.queries import QueryProvider
from rsisys.loops.contracts.training import ResumeMode
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.loop import ExplorationLoop

if TYPE_CHECKING:
    from rsisys.assembly import AssemblyContext


def build_default_exploration(context: AssemblyContext) -> ExplorationLoop:
    if 'actor' not in context.roles:
        raise ValueError('Legacy exploration requires an actor role; supply lab_factory for direct Lab execution')
    if context.execution == "local_process":
        space = CommandActionSpace()
        environment = LocalProcessEnvironment(context.sandbox, context.artifacts)
    else:
        space = EchoActionSpace()
        environment = SandboxEchoEnvironment(context.sandbox, context.artifacts)
    context.add_owned_cleanup(environment.close)
    model = context.roles["actor"]
    memory_path = context.data_dir / "exploration_memory.json" if context.data_dir is not None else None
    return ExplorationLoop(
        DefaultExplorationHarness(
            ModelActorPolicy(context.backends[model.backend_id], model), environment, context.experience,
            learner=MockActorLearner(context.artifacts) if context.with_learning else None,
            evaluator=MockEvaluator(context.artifacts) if context.with_learning else None,
            memory=ExplorationMemory(memory_path), memory_limit=context.memory_limit,
        ), space,
    )


def _decision_directives(context: AssemblyContext, exploration: ExplorationLoop):
    """Infer only known built-in handlers; custom execution requires a declaration."""
    declared = context.exploration_directives
    if declared is not None and not isinstance(declared, DirectiveDefinitions):
        raise TypeError("exploration_directives must be a DirectiveDefinitions catalog")
    harness = exploration.harness
    if (type(harness) is not DefaultExplorationHarness
            or type(harness.commands) is not ExplorationCommands or harness.commands.loop is not harness):
        if declared is None:
            raise ValueError("Custom exploration harness requires decision_factory or explicit "
                             "context.exploration_directives")
        return declared, declared.modes()

    executable = callable(getattr(harness.policy, "act", None)) and all(
        callable(getattr(harness.environment, method, None)) for method in ("reset", "step", "close")
    )
    trainable = (harness.learner is not None and harness.training.backend is not None
                 and isinstance(getattr(harness.policy, "model", None), ModelRef))
    available = {
        "execute": executable,
        "configure_prompt": True,
        "sft": trainable,
        "rl": trainable and executable and isinstance(harness.policy, ModelUpdatable),
        "evaluate": harness.evaluation.backend is not None,
    }
    if isinstance(harness.training.backend, BackendCapabilityProvider):
        capabilities = harness.training.backend.capabilities()
        # Built-in SFT/local RL start fresh; an unrelated backend method or
        # resume-only backend does not make either built-in command executable.
        for mode in ('sft', 'rl'):
            available[mode] = (available[mode] and mode in capabilities.methods
                               and ResumeMode.RESTART in capabilities.resume_modes)
    handlers = harness.commands.handlers
    if declared is not None:
        for mode in declared.modes():
            if mode not in handlers:
                raise ValueError(f"Declared directive has no exploration handler: {mode}")
            if mode in harness.commands.builtin_modes and not available.get(mode, False):
                raise ValueError(f"Declared directive lacks required exploration capabilities: {mode}")
        return declared, declared.modes()

    # Keep the default catalog identity for built-in snapshot compatibility.
    # Replaced/new handlers are only advertised through an explicit declaration.
    modes = tuple(mode for mode in handlers
                  if mode in harness.commands.builtin_modes and available.get(mode, False))
    return DirectiveDefinitions(), modes


def build_default_decision(context: AssemblyContext, exploration: ExplorationLoop, *, policy=None) -> DecisionMakingLoop:
    """Reuse default tools/harness with an independently supplied public RSI policy.

    Custom policies receive DecisionInput through ContextDecisionPolicy. The
    host retains its optional model identity, permissions and execution loop. A policy
    making model calls must use the host's accounted backend integration.
    Whole-harness replacement remains the decision_factory boundary.
    """
    model = context.roles.get("decision_maker")
    if policy is None and model is None:
        raise ValueError('The meta-agent controller requires a decision_maker model; '
                         'supply a policy or decision_factory for a model-free controller')
    open_lab = context.lab_execution is not None
    directives, modes = (DirectiveDefinitions(()), ()) if open_lab else _decision_directives(context, exploration)
    query_tools = (QueryTools(exploration, artifacts=context.artifacts)
                   if not open_lab and isinstance(exploration, QueryProvider) and exploration.describe_queries() else None)
    memory_path = context.data_dir / "decision_memory.json" if context.data_dir is not None else None
    memory = DecisionMemory(memory_path)
    agenda = ResearchAgenda(context.data_dir / "decision_research.json" if context.data_dir is not None else None)
    research_tools = ResearchTools(agenda, memory, experience=context.experience)
    definitions = (*(query_tools.definitions if query_tools is not None else ()), *research_tools.definitions)
    bindings = {**(query_tools.bindings if query_tools is not None else {}), **research_tools.bindings}
    if open_lab:
        from rsisys.harnesses.decision.tools.lab import LabExecutionTools
        tools = LabExecutionTools(context.lab_execution, extra_definitions=definitions, extra_bindings=bindings)
        space = ModularDecisionActionSpace(tools.definitions, directive_modes=(), directives=directives,
                                           permissions=context.action_permissions)
    elif context.data_dir is None:
        tools = DemoDecisionTools(context.sandbox, context.artifacts, extra_bindings=bindings)
        space = DemoDecisionActionSpace(directive_modes=modes, directives=directives, extra_definitions=definitions,
                                        permissions=context.action_permissions)
    else:
        tools = LocalDecisionTools(context.artifacts, extra_bindings=bindings)
        space = LocalDecisionActionSpace(directive_modes=modes, directives=directives, extra_definitions=definitions,
                                         permissions=context.action_permissions)
    tools.query_tools = query_tools
    tools.research_tools = research_tools
    if policy is None:
        policy = ModelDecisionPolicy(context.backends[model.backend_id], model,
            allowed_actions=() if open_lab else tuple(spec.name for spec in exploration.action_space.describe()),
            directives=directives, project_context=context.project_context)
    else:
        from rsisys.harnesses.decision.public_policy import (
            ContextDecisionPolicy,
            DecisionInputAdapter,
        )
        policy = ContextDecisionPolicy(policy, input_adapter=DecisionInputAdapter(directives=directives))
    return DecisionMakingLoop(
        DefaultDecisionHarness(
            policy,
            tools, context.experience,
            learner=MockDecisionLearner(context.artifacts) if context.with_learning else None,
            experience_limit=context.experience_limit,
            memory=memory, memory_limit=context.memory_limit, research_agenda=agenda,
            model=model,
        ), space,
    )
