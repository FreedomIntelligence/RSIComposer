"""Decision-policy improvement capabilities inside the ordinary Lab lifecycle.

This module supplies transport/admission boundaries, not a dataset algorithm or
trainer. Installed backends own recipes and execution. Completed training returns
a candidate; it never updates the live decision policy. JobBook retains uncertain
work and stable submission keys. Durable recovery needs an explicit integration.
"""
from copy import deepcopy
from dataclasses import asdict, replace

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.improvement import (
    TrajectoryRef, DecisionDatasetBuildRequest, DecisionDatasetBuildResult)
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.metrics import validated_metrics
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinition, DirectiveDefinitions, DirectiveSpec, _arguments)
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.exploration.harness.commands import CommandHandler
from rsisys.runtime.jobs import JobBook


class _ImprovementCommand(CommandHandler):
    def __init__(self, backend):
        for method in ('submit', 'poll', 'cancel'):
            if not callable(getattr(backend, method, None)):
                raise TypeError('Improvement backend requires ' + method)
        self.jobs = JobBook(backend)

    def prepare(self, harness, state):
        if not isinstance(state.directive.payload, self.request_type):
            raise TypeError('Improvement directive requires ' + self.request_type.__name__)
        if state.directive.input_refs:
            raise ValueError('Improvement inputs must be explicit references in the typed request')
        state.directive.payload.__post_init__()
        self.validate_request(state.directive.payload)
        state.request = deepcopy(state.directive.payload)

    def validate_request(self, request):
        pass

    def advance(self, harness, state, action_space):
        if state.job is None:
            if state.cancel_requested:
                return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED,
                    'Cancelled before backend submission', state.directive.simulated)
            state.job = self.jobs.submit(state.request, idempotency_key=state.idempotency_key)
            state.phase = 'running'

        def checked(outcome):
            state.data['backend_outcome'] = asdict(outcome)
            if outcome.status == JobStatus.SUCCEEDED:
                try:
                    if not isinstance(outcome.result, self.result_type):
                        raise TypeError('Improvement backend returned the wrong result type')
                    outcome.result.__post_init__()
                    self.validate_result(state.request, outcome.result)
                except (ValueError, TypeError) as exc:
                    return replace(outcome, status=JobStatus.FAILED, result=None,
                        error=f'Improvement result rejected: {type(exc).__name__}: {exc}',
                        simulated=outcome.simulated or bool(getattr(outcome.result, 'simulated', False)))
            return outcome

        outcome = (self.jobs.cancel if state.cancel_requested else self.jobs.poll)(
            state.job, normalize=checked)
        state.job_progress = outcome.progress
        if not outcome.terminal:
            return None
        simulated = (state.directive.simulated or outcome.simulated
                     or bool(outcome.result and outcome.result.simulated))
        if outcome.status != JobStatus.SUCCEEDED:
            status = (ExecutionStatus.CANCELLED if outcome.status == JobStatus.CANCELLED
                      else ExecutionStatus.FAILED)
            return Feedback(state.directive.directive_id, status, 'Improvement task ' + status.value,
                simulated, jobs=(state.job,), error=outcome.error)
        return self.feedback(state, outcome.result, simulated)


class DecisionDatasetCommand(_ImprovementCommand):
    request_type, result_type = DecisionDatasetBuildRequest, DecisionDatasetBuildResult

    def validate_result(self, request, result):
        if result.trajectory_refs != request.trajectory_refs or result.recipe_ref != request.recipe_ref:
            raise ValueError('Dataset result differs from the requested sources or recipe')

    def feedback(self, state, result, simulated):
        return Feedback(state.directive.directive_id, ExecutionStatus.COMPLETED,
            f'Decision dataset constructed: {result.example_count} examples; not a trained policy',
            simulated, artifact_refs=(*result.data_refs, result.report_ref),
            evidence_refs=(result.report_ref.artifact_id,), jobs=(state.job,))


class DecisionPolicyTrainingCommand(_ImprovementCommand):
    request_type, result_type = DecisionTrainingRequest, TrainingResult

    def validate_request(self, request):
        if any(not isinstance(v, str) or not v.strip() for v in
               (request.base.backend_id, request.base.model_id, request.base.revision)):
            raise ValueError('Policy training requires a complete base model identity')
        if not request.data_refs:
            raise ValueError('Decision policy training requires constructed dataset artifacts')
        for ref in request.data_refs:
            ref.__post_init__()
            if not ref.version or not ref.digest:
                raise ValueError('Training datasets must be versioned and fingerprinted')

    def validate_result(self, request, result):
        if result.base != request.base:
            raise ValueError('Trained policy base differs from the requested model')
        if result.candidate.affected_roles != ('decision_maker',):
            raise ValueError('Decision policy training must target only decision_maker')
        validated_metrics(result.metrics, 'Decision training metrics')
        if type(result.simulated) is not bool:
            raise TypeError('Training simulated must be a bool')

    def feedback(self, state, result, simulated):
        return Feedback(state.directive.directive_id, ExecutionStatus.COMPLETED,
            'Decision policy training completed; candidate awaits evaluation and adoption',
            simulated, training_metrics=result.metrics,
            artifact_refs=tuple(result.candidate.changes.values()),
            candidates=(result.candidate,), jobs=(state.job,))


def improvement_handlers(*, dataset_builder=None, decision_learner=None):
    """Advertise only installed capabilities; no backend is enabled by default."""
    handlers = {}
    if dataset_builder is not None:
        handlers['build_decision_dataset'] = DecisionDatasetCommand(dataset_builder)
    if decision_learner is not None:
        handlers['train_decision_policy'] = DecisionPolicyTrainingCommand(decision_learner)
    if not handlers:
        raise ValueError('An improvement Lab needs an installed capability')
    return handlers


def improvement_directives(modes=('build_decision_dataset', 'train_decision_policy')):
    """Public parsers for normal meta-agent explore decisions; no I/O."""
    text = {'type': 'string', 'minLength': 1}
    trajectory = {'type': 'object', 'properties': {
        'archive_id': text, 'trajectory_id': text,
        'digest': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}},
        'required': ['archive_id', 'trajectory_id', 'digest'], 'additionalProperties': False}
    artifact = {'type': 'object', 'properties': {**{k: text for k in ('artifact_id', 'kind', 'uri')},
        'version': text, 'digest': text},
        'required': ['artifact_id', 'kind', 'uri', 'version', 'digest'], 'additionalProperties': False}
    model = {'type': 'object', 'properties': {k: text for k in ('backend_id', 'model_id', 'revision')},
        'required': ['backend_id', 'model_id', 'revision'], 'additionalProperties': False}

    def dataset(arguments, candidates):
        _arguments(arguments, ('trajectory_refs', 'recipe_ref'))
        return DecisionDatasetBuildRequest(tuple(TrajectoryRef(**r) for r in arguments['trajectory_refs']),
                                           arguments['recipe_ref'])

    def training(arguments, candidates):
        _arguments(arguments, ('base', 'data_refs', 'recipe_ref'), ('method',))
        return DecisionTrainingRequest(ModelRef(**arguments['base']), (), arguments['recipe_ref'],
            tuple(ArtifactRef(**r) for r in arguments['data_refs']), arguments.get('method', 'sft'))

    def definition(mode, description, properties, required, parser):
        return DirectiveDefinition(DirectiveSpec(mode, description,
            {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False},
            accepts_input_refs=False), parser)

    definitions = {
        'build_decision_dataset': definition('build_decision_dataset',
            'Build decision-policy training data in this Lab using pinned archived trajectories and an explicit recipe. '
            'Returns dataset/report artifacts; does not train or activate a policy.',
            {'trajectory_refs': {'type': 'array', 'minItems': 1, 'items': trajectory}, 'recipe_ref': text},
            ['trajectory_refs', 'recipe_ref'], dataset),
        'train_decision_policy': definition('train_decision_policy',
            'Train a decision-policy candidate from versioned dataset artifacts and an explicit base/recipe. '
            'Does not evaluate, activate or change the live meta-agent model.',
            {'base': model, 'data_refs': {'type': 'array', 'minItems': 1, 'items': artifact},
             'recipe_ref': text, 'method': text}, ['base', 'data_refs', 'recipe_ref'], training)}
    if len(set(modes)) != len(modes) or any(m not in definitions for m in modes):
        raise ValueError('Unknown or duplicate improvement capability')
    return DirectiveDefinitions(definitions[m] for m in modes)


def improvement_exploration_factory(*, dataset_builder=None, decision_learner=None):
    """Install Lab capabilities through the existing Application/Demo factory.

    One Lab occupies the existing exploration port. This is not a cross-Lab
    scheduler or a durable recovery contract. Host budgets still own admission.
    """
    def build(context):
        from rsisys.assembly import build_default_exploration
        from rsisys.loops.exploration.harness.commands import ExplorationCommands
        from rsisys.loops.exploration.harness.policy import ModelActorPolicy
        handlers = improvement_handlers(dataset_builder=dataset_builder, decision_learner=decision_learner)
        loop = build_default_exploration(context)
        loop.harness.commands = ExplorationCommands(loop.harness, handlers)
        model = context.roles['decision_maker']
        loop.harness.policy = ModelActorPolicy(context.backends[model.backend_id], model)
        context.exploration_directives = improvement_directives(tuple(handlers))
        return loop
    return build
