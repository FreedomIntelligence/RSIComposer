"""Host harness construction for explicitly supplied Controller policies."""
from dataclasses import dataclass
from rsisys.component_context import ControllerContext, controller_wiring
from rsisys.harnesses.decision.engine import ResearchControllerHarness
from rsisys.harnesses.decision.memory import DecisionMemory
from rsisys.harnesses.decision.research import ResearchAgenda
from rsisys.harnesses.decision.tools.research import ResearchTools
from rsisys.harnesses.decision.tools.lab import LabExecutionTools
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.loop import DecisionMakingLoop


@dataclass(frozen=True)
class DecisionHarnessParts:
    tools: object
    action_space: object
    memory: DecisionMemory
    agenda: ResearchAgenda


def build_decision_parts(context, execution):
    owner = controller_wiring(context)
    if execution is None or execution is not owner.lab_execution:
        raise TypeError('Controller harness requires an explicit Lab execution provider')
    memory = DecisionMemory(context.data_dir / 'decision_memory.json' if context.data_dir else None)
    agenda = ResearchAgenda(context.data_dir / 'decision_research.json' if context.data_dir else None)
    research_tools = ResearchTools(agenda, memory, experience=owner.experience)
    host = getattr(owner, '_host_capabilities', None)
    tools = LabExecutionTools(execution,
        extra_definitions=(*research_tools.definitions, *(host.definitions if host is not None else ())),
        extra_bindings={**research_tools.bindings, **(host.bindings if host is not None else {})})
    space = ModularDecisionActionSpace(tools.definitions, permissions=owner.action_permissions)
    tools.research_tools = research_tools
    return DecisionHarnessParts(tools, space, memory, agenda)


def build_default_decision(context, execution, *, policy=None):
    from rsisys.harnesses.decision.public_policy import ResearchPolicyAdapter
    parts = build_decision_parts(context, execution)
    owner = controller_wiring(context)
    harness = ResearchControllerHarness(local_tools=parts.tools, experience=owner.experience,
        experience_limit=owner.experience_limit, memory=parts.memory,
        memory_limit=owner.memory_limit, research_agenda=parts.agenda)
    model = context.roles.get('decision_maker')
    native = policy is None
    models = context.models if isinstance(context, ControllerContext) else owner._model_access.component('controller') if native else None
    if native:
        from rsisys.harnesses.decision.policy import ModelDecisionPolicy
        if model is None:
            raise ValueError('The selected meta-agent Controller requires a decision_maker model')
        policy = ResearchPolicyAdapter(ModelDecisionPolicy(
            models.backends[model.backend_id], model, project_context=owner.project_context), harness)
    return DecisionMakingLoop(policy, harness, parts.action_space,
        host_execution=owner.host_execution, lab_execution=execution, model=model,
        model_scope=models.scope if native else None)
