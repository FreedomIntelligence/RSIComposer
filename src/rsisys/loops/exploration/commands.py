"""Exploration-owned command lifecycle; runtime only transports progress/feedback."""

from copy import deepcopy
from dataclasses import dataclass, replace

from rsisys.loops.contracts.coordination import ExplorationProgress
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef, ModelUpdatable
from rsisys.loops.contracts.requests import SFTSpec, RLSpec, EvaluationSpec
from .learner import ActorTrainingRequest
from .evaluation import EvaluationRequest
from .rl import RLPlan


@dataclass
class CommandState:
    directive: Directive
    phase: str = 'accepted'
    request: ActorTrainingRequest | EvaluationRequest | None = None
    job: JobRef | None = None
    feedback: Feedback | None = None
    cancel_requested: bool = False


class ExplorationCommands:
    def __init__(self, loop):
        self.loop = loop
        self.active: str | None = None
        # In-process receipts support retries. They are not a durable job journal.
        self.receipts: dict[str, CommandState] = {}

    @property
    def busy(self):
        return self.active is not None

    def accept(self, directive: Directive) -> ExplorationProgress:
        if directive.directive_id in self.receipts:
            state = self.receipts[directive.directive_id]
            if directive != state.directive:
                raise ValueError('Directive ID reused with different content')
            return self.inspect(directive.directive_id)
        loop = self.loop
        if (self.busy or loop.state.active_directive_id is not None or loop.training.busy
                or loop.evaluation.busy or (loop.state.rl is not None and not loop.state.rl.terminal)):
            raise RuntimeError('Exploration is busy')
        state = CommandState(deepcopy(directive))
        self.receipts[directive.directive_id] = state
        self.active = directive.directive_id
        try:
            self._prepare(state)
        except BaseException as exc:
            self._fail(state, exc)
            raise
        return self.inspect(directive.directive_id)

    def _prepare(self, state):
        directive, loop = state.directive, self.loop
        payload = directive.payload
        if directive.mode == 'execute':
            if payload is not None:
                raise ValueError('Execute does not accept a learning/evaluation payload')
        elif directive.mode in ('sft', 'rl'):
            if loop.learner is None:
                raise NotImplementedError('Training directives require a learner')
            expected = SFTSpec if directive.mode == 'sft' else RLSpec
            if not isinstance(payload, expected):
                raise ValueError(f'{directive.mode} requires {expected.__name__}')
            if directive.mode == 'rl' or payload.apply_result:
                if not isinstance(loop.policy, ModelUpdatable):
                    raise TypeError('This command requires a model-updatable policy')
            if directive.mode == 'sft':
                base = payload.base or getattr(loop.policy, 'model', None)
                if not isinstance(base, ModelRef):
                    raise ValueError('SFT requires a base model')
                state.request = ActorTrainingRequest(base, directive.input_refs, payload.recipe_ref)
        elif directive.mode == 'evaluate':
            if loop.evaluation.backend is None:
                raise NotImplementedError('Evaluation directives require an evaluator')
            if not isinstance(payload, EvaluationSpec):
                raise ValueError('Evaluation requires EvaluationSpec')
            state.request = EvaluationRequest(payload.candidate, payload.protocol_ref)
        else:
            raise NotImplementedError(f'Unsupported exploration mode: {directive.mode}')

    def inspect(self, directive_id):
        state = self.receipts[directive_id]
        return deepcopy(ExplorationProgress(directive_id, state.phase, state.directive, state.job, state.feedback))

    def _finish(self, state, feedback):
        state.feedback = deepcopy(feedback)
        state.phase = feedback.status.value
        self.loop.state.last_feedback = deepcopy(feedback)

    def _fail(self, state, exc):
        self._finish(state, Feedback(state.directive.directive_id,
            ExecutionStatus.FAILED if isinstance(exc, Exception) else ExecutionStatus.CANCELLED,
            'Exploration command failed', state.directive.simulated,
            error=f'{type(exc).__name__}: {exc}'))

    def advance(self, directive_id):
        state = self.receipts[directive_id]
        if state.feedback is not None:
            return self.inspect(directive_id)
        directive, loop = state.directive, self.loop
        if state.cancel_requested and state.phase == 'accepted':
            self._finish(state, Feedback(directive_id, ExecutionStatus.CANCELLED,
                                        'Cancelled before execution', directive.simulated))
        elif directive.mode == 'execute':
            try:
                self._finish(state, loop._run_episode(directive))
            except BaseException as exc:
                feedback = loop.state.last_feedback
                if feedback is not None and feedback.directive_id == directive_id:
                    self._finish(state, feedback)
                else:
                    self._fail(state, exc)
                raise
        elif directive.mode == 'rl':
            self._advance_rl(state)
        else:
            self._advance_job(state)
        return self.inspect(directive_id)

    def _advance_job(self, state):
        loop, directive = self.loop, state.directive
        training = directive.mode == 'sft'
        if state.phase in ('accepted', 'submitting'):
            # Submission may have succeeded remotely before raising. Retain request/key.
            state.phase = 'submitting'
            submit = loop._submit_learning if training else loop._submit_evaluation
            state.job = submit(state.request, idempotency_key=f'directive:{directive.directive_id}')
            state.phase = 'running'
            return
        if training:
            poll = loop.cancel_learning if state.cancel_requested else loop.poll_learning
        else:
            poll = loop.cancel_evaluation if state.cancel_requested else loop.poll_evaluation
        outcome = poll(state.job)
        if not outcome.terminal:
            return
        result = outcome.result
        status = (ExecutionStatus.CANCELLED if state.cancel_requested or outcome.status == JobStatus.CANCELLED
                  else ExecutionStatus.COMPLETED if outcome.status == JobStatus.SUCCEEDED else ExecutionStatus.FAILED)
        error = outcome.error
        application_error = None
        if training and result and directive.payload.apply_result and not state.cancel_requested:
            try:
                loop._apply_learning(state.job)
            except Exception as exc:
                status, error, application_error = ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}', exc
        records = loop.state.training_record_ids if training else loop.state.evaluation_record_ids
        self._finish(state, Feedback(
            directive.directive_id, status, f'{directive.mode} {status.value}',
            directive.simulated or outcome.simulated or bool(result and result.simulated),
            training_metrics=result.metrics if training and result else {},
            quality_metrics=result.metrics if not training and result else {},
            artifact_refs=tuple(result.candidate.changes.values()) if training and result else (),
            evidence_refs=(records[state.job],),
            candidates=(result.candidate,) if training and result else (),
            jobs=(state.job,), error=error,
        ))
        if application_error is not None:
            raise application_error

    def _advance_rl(self, state):
        loop, directive = self.loop, state.directive
        if state.phase == 'accepted':
            loop._start_rl(RLPlan(replace(directive, mode='execute', payload=None),
                                  directive.payload.recipe_ref, directive.payload.updates))
            state.phase = 'running'
            return
        failure = None
        try:
            if state.cancel_requested and not loop.state.rl.cancel_requested:
                rl = loop._cancel_rl()
            else:
                rl = loop._step_rl()
        except BaseException as exc:
            rl = deepcopy(loop.state.rl)
            if not rl.terminal:
                raise  # Uncertain submit/record/poll: retry the same phase, never restart rollout.
            failure = exc
        state.job = rl.job
        if rl.terminal:
            status = ExecutionStatus(rl.phase)
            records = tuple(loop.state.training_record_ids[job] for job in rl.jobs
                            if job in loop.state.training_record_ids)
            self._finish(state, Feedback(
                directive.directive_id, status, f'RL {rl.phase}: {rl.updates_completed} updates',
                directive.simulated or any(f.simulated for f in rl.feedback)
                or any(r.simulated for r in rl.results) or rl.simulated,
                artifact_refs=tuple(a for f in rl.feedback for a in f.artifact_refs)
                + tuple(a for r in rl.results for a in r.candidate.changes.values()),
                evidence_refs=tuple(e for f in rl.feedback for e in f.evidence_refs) + records,
                candidates=tuple(r.candidate for r in rl.results), jobs=tuple(rl.jobs), error=rl.error,
            ))
        if failure is not None:
            raise failure

    def cancel(self, directive_id):
        state = self.receipts[directive_id]
        if state.feedback is None:
            state.cancel_requested = True
        return self.inspect(directive_id)

    def acknowledge(self, directive_id):
        state = self.receipts[directive_id]
        if state.feedback is None:
            raise RuntimeError('Cannot acknowledge unfinished exploration')
        if self.active == directive_id:
            self.active = None
