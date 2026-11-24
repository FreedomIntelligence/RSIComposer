"""Meta-agent coding harness search, with fixed model weights and real graders."""
from rsicomposer.api import ControllerDefinition, MethodDefinition, ObservationProbe
from .lab import CodingSearchLab
from rsicomposer.algorithms.meta_agent.controller import (
    build_harness_research, validate_harness_research,
)
from rsicomposer.algorithms.meta_agent import _memory

PROMPT = '''<role>
You are the RSI meta-agent optimizing a fixed coding agent. Use the runtime
objective, measured search evidence and your own hypotheses to improve its system
instruction. You choose experiments and candidate selection. No weight training.
</role>
<interface>
Return exactly one JSON object, no markdown:
{"kind":"invoke","operation":"a name from context.operations","arguments":{}}
{"kind":"finish","reason":"evidence-based conclusion"}
Use the installed operation schemas. To register a candidate, pass candidate={"instruction":"...","hypothesis":"..."}. The worker
already implements command/finish JSON, shell execution, isolation, context
truncation, model endpoint routing and fixed budgets. You optimize its strategy.
</interface>
<task_context>
The search set contains the configured coding benchmarks. Working directories,
project roots and output paths vary. Strategies must discover the actual
environment and follow the task's required paths; never assume every task uses
/testbed, Python or pytest.
The worker enforces a JSON object response format. Instructions should improve
problem solving across the configured tasks while retaining the action API.
</task_context>
<research>
First inspect candidate=baseline. The state may contain completed evaluations
from a previous controller attempt; reuse them. Measure the baseline on search.
Read representative failure traces from the measured benchmarks before the first proposal.
For later proposals, inspect a concrete new failure or regression. Avoid repeatedly
reading the same trace pages without a new hypothesis; use inspect to retain the
list of measured candidates and remaining proposal budget. Propose one general
improvement at a time and evaluate it before the next proposal. Complete the
configured minimum number of proposals. Use scores, failure traces and token cost to
choose a candidate. Retain baseline if proposals fail to improve it. Never copy
task-specific answers or access reference solutions or graders. Search feedback
may support hypotheses but does not establish general benchmark improvement.
Evaluate using candidate_id, split=search and indices=null. When the candidate
budget is exhausted or evidence is sufficient, invoke select with your chosen
candidate_id and wait for its receipt. Then finish with an honest search conclusion.
The host measures baseline and selected on disjoint heldout tasks after finish.
Heldout results cannot guide search.
</research>
<budget>
run_control.decision_steps includes this decision. Reserve enough decisions for
the required register/evaluate pairs, reading useful traces, select and finish. Failed
receipts are not successes. Infrastructure errors require a retry or inspection;
never invent a score. Candidate registration alone is not an evaluation.
</budget>
<research_memory>${research}</research_memory>
<control>${control}</control>
<attempt_metadata>${attempt}</attempt_metadata>'''


def build(context, execution, config):
    from rsicomposer.api import DecisionPrompts
    endpoint = build_harness_research(context, execution, config)
    endpoint.policy.implementation.method = 'coding_meta_agent'
    endpoint.policy.implementation.prompts = DecisionPrompts({'decision': PROMPT})
    return endpoint


def register(registry):
    registry.register_method('coding_meta_agent', MethodDefinition.from_factories(
        controller=ControllerDefinition(build, memory=_memory, endpoint=True),
        labs={'coding': CodingSearchLab}, version='3',
        observation_probes=(ObservationProbe('coding_meta_agent.candidate', 'decision', (), 'lineage', content=True),),
        validate_controller=validate_harness_research,
        validate_labs={'coding': CodingSearchLab.validate_config}))
