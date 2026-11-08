"""Mock asynchronous experiment demo: no threads, GPU work or real model training.

Run: PYTHONPATH=src python3.12 examples/components/parallel_experiments.py
The scripted researcher uses the same ordinary capability bindings as a model
policy. Only this mock backend supports changing ``remaining_polls`` live.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import EvaluationSpec, TrainSpec
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions, training_directive, builtin_directives
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.control import ExperimentControls
from rsisys.loops.exploration.harness import ConcurrentExplorationHarness
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.evaluation import EvaluationResult
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.jobs import MockJobQueue
from rsisys.runtime.runner import Runner

BASE = ModelRef('mock', 'actor', 'base')


class MockPolicy:
    def __init__(self):
        self.model = BASE

    def update_model(self, model):
        self.model = model


class MockExperimentBackend:
    def __init__(self, ident, *, pending_polls=0):
        self.ident = ident
        self.queue = MockJobQueue(ident, pending_polls=pending_polls)
        self.submissions = []
        self.polls = []
        self.cancellations = []
        self.updates = {}
        self.lose_update_response = False

    def submit(self, request, *, idempotency_key):
        self.submissions.append((deepcopy(request), idempotency_key))

        def result():
            if isinstance(request, EvaluationSpec):
                raise AssertionError('EvaluationExecution must translate its request')
            if hasattr(request, 'protocol_ref'):
                return EvaluationResult(request.candidate.candidate_id, request.protocol_ref,
                    {'quality': 0.9 if request.candidate.candidate_id == 'fast' else 0.7}, (), True)
            return TrainingResult(request.base, replace(request.base, revision=self.ident),
                                  Candidate(self.ident, request.base.revision, {}, ('actor',)), simulated=True)

        return self.queue.submit(request, result, idempotency_key=idempotency_key)

    def poll(self, job):
        self.polls.append(job)
        return self.queue.poll(job)

    def cancel(self, job):
        self.cancellations.append(job)
        self.queue.cancel(job)

    def validate_update(self, job, request):
        self.queue._get(job)
        if set(request.changes) != {'remaining_polls'}:
            raise NotImplementedError('Mock backend supports only remaining_polls; no training hyperparameter hot updates')
        value = request.changes['remaining_polls']
        if type(value) is not int or value < 0:
            raise ValueError('remaining_polls must be a nonnegative integer')

    def update(self, job, request):
        self.validate_update(job, request)
        key = (job, request.idempotency_key)
        if key in self.updates:
            if self.updates[key] != request:
                raise ValueError('Backend update key conflict')
            return
        self.queue._get(job).remaining_polls = request.changes['remaining_polls']
        self.updates[key] = deepcopy(request)
        if self.lose_update_response:
            self.lose_update_response = False
            raise OSError('Mock update response lost after application')


class ScriptedResearcher:
    """Explicitly scripted scheduling fixture, not evidence of LLM autonomy."""
    def __init__(self):
        self.issued = set()
        self.updated = False
        self.compared = False
        self.contexts = []

    def decide(self, context):
        self.contexts.append(deepcopy(context))
        for ident in ('slow', 'fast'):
            if ident not in self.issued:
                self.issued.add(ident)
                return Decision('explore', directive=Directive(ident, context.objective, 'same-base candidate',
                    mode='train', payload=TrainSpec(ident, 'mock', BASE), simulated=True))
        if not self.updated:
            self.updated = True
            return Decision('invoke', operation=OperationCall('experiment_update', {
                'directive_id': 'slow', 'expected_version': 0, 'idempotency_key': 'slow-update-1',
                'changes': {'remaining_polls': 4}}))
        if len(context.candidates) == 2:
            for candidate in context.candidates:
                ident = 'eval-' + candidate.candidate_id
                if ident not in self.issued and len(context.outstanding) < 2:
                    self.issued.add(ident)
                    return Decision('explore', directive=Directive(ident, context.objective, 'same-protocol evaluation',
                        mode='evaluate', payload=EvaluationSpec(candidate, 'mock-quality-v1'), simulated=True))
        if len(context.candidate_evaluations) == 2:
            if not self.compared:
                self.compared = True
                return Decision('invoke', operation=OperationCall('experiment_compare', {
                    'candidate_ids': ['slow', 'fast'], 'metric': 'quality', 'higher_is_better': True}))
            return Decision('finish', reason='Explicit comparison recorded')
        return Decision('invoke', operation=OperationCall('experiment_inspect', {
            'directive_id': context.outstanding[0].directive_id}))


def build_demo():
    experience = BasicExperienceSystem(InMemoryExperienceStore())
    children = {}

    def factory(directive):
        backend = MockExperimentBackend(directive.directive_id,
            pending_polls=8 if directive.directive_id == 'slow' else 0)
        child = DefaultExplorationHarness(MockPolicy(), object(), experience, learner=backend,
            evaluator=backend, command_handlers={'train': TrainingCommandHandler()})
        child.state.instructions = f'Isolated instruction for {directive.directive_id}'
        children[directive.directive_id] = child
        return child

    exploration = ExplorationLoop(ConcurrentExplorationHarness(factory, max_in_flight=2), EchoActionSpace())
    policy = ScriptedResearcher()
    harness = DefaultDecisionHarness(policy, BoundDecisionTools({}), experience, max_outstanding=2)
    catalog = DirectiveDefinitions((*builtin_directives(), training_directive()))
    decision = DecisionMakingLoop(harness, ModularDecisionActionSpace((), directives=catalog))
    controls = ExperimentControls(exploration, decision)
    decision.action_space = ModularDecisionActionSpace(controls.definitions, directives=catalog)
    harness.tools = BoundDecisionTools(controls.bindings)
    return Runner(decision, exploration, parallel=True), children, policy


def run_demo():
    runner, children, policy = build_demo()
    completion_order, max_in_flight = [], 0
    for _ in range(80):
        result = runner.step('Compare two mock candidates from the same base')
        max_in_flight = max(max_in_flight, len(runner.exploration.pending_all()))
        if result.kind == 'completed':
            completion_order.append(result.directive_id)
        if result.kind == 'finished':
            break
    else:
        raise RuntimeError('Mock experiment demo did not finish')
    ranking = runner.decision_making.compare_candidates(('slow', 'fast'), metric='quality')
    assert max_in_flight == 2 and completion_order.index('fast') < completion_order.index('slow')
    assert runner.exploration.inspect('slow').version == 1
    assert not runner.busy
    return {'simulated': True, 'policy': 'scripted fixture', 'max_in_flight': max_in_flight,
            'completion_order': completion_order, 'update_version': runner.exploration.inspect('slow').version,
            'ranking': [asdict(row) for row in ranking],
            'tools_used': [row.details['operation']['name'] for row in runner.decision_making.harness.memory.history()
                           if row.kind == 'operation']}


if __name__ == '__main__':
    print(json.dumps(run_demo(), indent=2))
