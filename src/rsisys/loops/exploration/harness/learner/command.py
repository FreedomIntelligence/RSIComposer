"""Opt-in full-job delegation, independent of the built-in local RL sequencer."""
from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4

from rsisys.experience.records import ExperienceRecord
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import ApplyCandidateSpec, TrainSpec
from ..commands import CommandHandler
from . import ActorTrainingRequest


class TrainingCommandHandler(CommandHandler):
    def prepare(self, harness, state):
        spec = state.directive.payload
        harness.commands.validate_training_payload(spec, TrainSpec,
            apply=isinstance(spec, TrainSpec) and spec.apply_result)
        base = spec.base or (spec.origin.initial_model if spec.origin else getattr(harness.policy, 'model', None))
        if not isinstance(base, ModelRef):
            raise ValueError('Training requires a base model')
        if spec.apply_result and base != getattr(harness.policy, 'model', None):
            raise ValueError('Direct application requires the current role model as initializer; '
                             'use separate system candidate registration and application for another initializer')
        state.request = ActorTrainingRequest(base, state.directive.input_refs,
                                             spec.recipe_ref, method=spec.method, origin=spec.origin)

    def validate_update(self, harness, state, request):
        backend = harness.learner
        validate = getattr(backend, 'validate_update', None)
        update = getattr(backend, 'update', None)
        if not callable(validate) or not callable(update):
            raise NotImplementedError('Training backend does not support live updates')
        if state.job is None:
            raise RuntimeError('Live training updates require a confirmed submitted job')
        from rsisys.loops.contracts.backend import BackendCapabilityProvider, validate_update_capabilities
        if isinstance(backend, BackendCapabilityProvider):
            validate_update_capabilities(backend.capabilities(), tuple(request.changes))
        validate(state.job, deepcopy(request))

    def update(self, harness, state, request):
        # The backend must confirm application and deduplicate the full request.
        # It owns the allowed fields; this adapter promises no generic hot tuning.
        harness.learner.update(state.job, deepcopy(request))

    def advance(self, harness, state, action_space):
        # Reuse submission, polling, cancellation, evidence and model-application
        # semantics. The backend owns rollout and updates inside this one job.
        harness.commands.advance_job(state, training=True)


class ApplyCandidateHandler(CommandHandler):
    """Opt-in application of a retained result, independent of learner jobs."""

    def prepare(self, harness, state):
        spec = state.directive.payload
        if not isinstance(spec, ApplyCandidateSpec):
            raise ValueError('apply_candidate requires ApplyCandidateSpec')
        state.request = harness.learning.candidate_result(spec.candidate_id)

    def cancel_before_start(self, harness, state, action_space):
        return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED,
                        'Cancelled before candidate application',
                        state.directive.simulated or state.request.simulated,
                        candidates=(deepcopy(state.request.candidate),))

    def advance(self, harness, state, action_space):
        result, directive = state.request, state.directive
        if state.record is None:
            status = ExecutionStatus.CANCELLED if state.cancel_requested else ExecutionStatus.COMPLETED
            if not state.cancel_requested:
                state.phase = 'applying'
                harness.commands.apply_candidate(result.candidate.candidate_id,
                                                    directive_id=directive.directive_id)
            # Freeze identity and content before persistence. A lost write response
            # retries only the record, even if cancellation arrives after binding.
            state.record = ExperienceRecord(
                str(uuid4()), 'runtime_feedback', 'actor', directive.directive_id,
                f'apply_candidate {status.value}', directive.simulated or result.simulated,
                source_refs=tuple(ref.artifact_id for ref in result.candidate.changes.values()),
                details={'mode': directive.mode, 'status': status.value,
                         'candidate_id': result.candidate.candidate_id,
                         'base': asdict(result.base), 'model': asdict(result.model)},
            )
            state.phase = 'recording'
        harness.experience.record(deepcopy(state.record))
        return Feedback(
            directive.directive_id, ExecutionStatus(state.record.details['status']),
            state.record.summary, state.record.simulated,
            training_metrics=deepcopy(result.metrics),
            artifact_refs=tuple(deepcopy(result.candidate.changes).values()),
            evidence_refs=(state.record.record_id,), candidates=(deepcopy(result.candidate),),
            adopted_candidate_ids=(result.candidate.candidate_id,)
                if state.record.details['status'] == ExecutionStatus.COMPLETED.value else (),
        )
