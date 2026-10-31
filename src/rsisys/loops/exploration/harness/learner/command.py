"""Opt-in full-job delegation, independent of the built-in local RL sequencer."""
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import TrainSpec
from ..commands import CommandHandler
from . import ActorTrainingRequest


class TrainingCommandHandler(CommandHandler):
    def prepare(self, harness, state):
        spec = state.directive.payload
        harness.commands.validate_training_payload(spec, TrainSpec,
            apply=isinstance(spec, TrainSpec) and spec.apply_result)
        base = spec.base or getattr(harness.policy, 'model', None)
        if not isinstance(base, ModelRef):
            raise ValueError('Training requires a base model')
        state.request = ActorTrainingRequest(base, state.directive.input_refs,
                                             spec.recipe_ref, method=spec.method)

    def advance(self, harness, state, action_space):
        # Reuse submission, polling, cancellation, evidence and model-application
        # semantics. The backend owns rollout and updates inside this one job.
        harness.commands.advance_job(state, training=True)
