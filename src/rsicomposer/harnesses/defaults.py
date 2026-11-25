"""Host harness construction for explicitly supplied Controller policies."""
from dataclasses import dataclass
from rsicomposer.component_context import controller_model_access, decision_wiring
from rsicomposer.harnesses.decision.engine import ResearchControllerHarness
from rsicomposer.harnesses.decision.memory import DecisionMemory
from rsicomposer.harnesses.decision.research import ResearchAgenda
from rsicomposer.harnesses.decision.tools.research import ResearchTools
from rsicomposer.harnesses.decision.tools.lab import LabExecutionTools
from rsicomposer.loops.decision_making.action_space import ModularDecisionActionSpace
from rsicomposer.loops.decision_making.loop import DecisionMakingLoop


@dataclass(frozen=True)
class DecisionHarnessParts:
    tools: object
    action_space: object
    memory: DecisionMemory
    agenda: ResearchAgenda


def build_decision_parts(context, execution):
    return _build_decision_parts(context.data_dir, execution, decision_wiring(context))


def _build_decision_parts(data_dir, execution, ports):
    if execution is None or execution is not ports.lab_execution:
        raise TypeError('Controller harness requires an explicit Lab execution provider')
    memory = DecisionMemory(data_dir / 'decision_memory.json' if data_dir else None)
    agenda = ResearchAgenda(data_dir / 'decision_research.json' if data_dir else None)
    research_tools = ResearchTools(agenda, memory, experience=ports.experience)
    host = ports.host_capabilities
    tools = LabExecutionTools(execution,
        extra_definitions=(*research_tools.definitions, *(host.definitions if host is not None else ())),
        extra_bindings={**research_tools.bindings, **(host.bindings if host is not None else {})})
    space = ModularDecisionActionSpace(tools.definitions, permissions=ports.action_permissions)
    tools.research_tools = research_tools
    return DecisionHarnessParts(tools, space, memory, agenda)


def build_default_decision(context, execution, *, policy=None):
    from rsicomposer.harnesses.decision.public_policy import ResearchPolicyAdapter
    ports = decision_wiring(context)
    parts = _build_decision_parts(context.data_dir, execution, ports)
    harness = ResearchControllerHarness(local_tools=parts.tools, experience=ports.experience,
        experience_limit=ports.experience_limit, memory=parts.memory,
        memory_limit=ports.memory_limit, research_agenda=parts.agenda)
    model = context.roles.get('decision_maker')
    native = policy is None
    models = controller_model_access(context) if native else None
    if native:
        from rsicomposer.harnesses.decision.policy import ModelDecisionPolicy
        if model is None:
            raise ValueError('The selected meta-agent Controller requires a decision_maker model')
        policy = ResearchPolicyAdapter(ModelDecisionPolicy(
            models.backends[model.backend_id], model, project_context=ports.project_context), harness)
    return DecisionMakingLoop(policy, harness, parts.action_space,
        host_execution=ports.host_execution, lab_execution=execution, model=model,
        model_scope=models.scope if native else None)


def build_harness_decision(context, execution, *, policy, harness):
    """Run a Policy with a user-supplied ControllerHarness.

    The action space is the Lab operations plus the Harness's own tools. Lab
    calls go to the Lab; every other name goes to harness.execute_tool().
    """
    from rsicomposer.loops.decision_making.action_space.capabilities import Capability
    ports = decision_wiring(context)
    for port in ('window', 'observe', 'tools', 'execute_tool'):
        if not callable(getattr(harness, port, None)):
            raise TypeError(f'ControllerHarness requires {port}()')
    local = tuple(harness.tools())
    tools = LabExecutionTools(execution, extra_definitions=tuple(
        Capability.from_spec(spec, lambda arguments, name=spec.name: _local_call(name, arguments),
                             lambda arguments: None).definition for spec in local))
    space = ModularDecisionActionSpace(tools.definitions, permissions=ports.action_permissions)
    return DecisionMakingLoop(policy, harness, space, host_execution=ports.host_execution,
                              lab_execution=execution, model=context.roles.get('decision_maker'))


def _local_call(name, arguments):
    from rsicomposer.loops.contracts.decision import OperationCall
    return OperationCall(name, arguments)
