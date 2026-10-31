"""Exploration-owned command lifecycle; runtime only transports progress/feedback."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from types import MappingProxyType
from hashlib import sha256
from uuid import uuid4

from rsisys.experience.records import ExperienceRecord
from rsisys.loops.contracts.coordination import ExplorationProgress
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef, ModelUpdatable
from rsisys.loops.contracts.requests import SFTSpec, RLSpec, EvaluationSpec, PromptSpec
from rsisys.loops.contracts.training import ModelActivationPending
from .learner import ActorTrainingRequest
from .evaluation import EvaluationRequest
from .rl import RLPlan


@dataclass
class CommandState:
    directive: Directive
    phase: str = 'accepted'
    request: object | None = None
    job: JobRef | None = None
    feedback: Feedback | None = None
    cancel_requested: bool = False
    record: ExperienceRecord | None = None
    data: dict[str, object] = field(default_factory=dict)
    started: bool = False

    @property
    def idempotency_key(self) -> str:
        return f'directive:{self.directive.directive_id}'


class CommandHandler:
    """Explicit command implementation; state survives in-process retries.

    prepare validates/freezes the request without starting external work.
    advance makes one bounded step and returns Feedback when terminal, or None
    while waiting. An exception without terminal feedback retains the receipt.
    After work starts, advance must honor state.cancel_requested. Use the stable
    state.idempotency_key for submissions whose response may be lost.
    """

    def prepare(self, harness, state: CommandState) -> None:
        raise NotImplementedError

    def advance(self, harness, state: CommandState, action_space) -> Feedback | None:
        raise NotImplementedError

    def cancel_before_start(self, harness, state: CommandState, action_space) -> Feedback | None:
        return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED,
                        'Cancelled before execution', state.directive.simulated)


class _BuiltinHandler(CommandHandler):
    def __init__(self, prepare, advance, *, cancel_before_start=None):
        self._prepare, self._advance, self._cancel = prepare, advance, cancel_before_start

    def prepare(self, harness, state):
        self._prepare(state)

    def advance(self, harness, state, action_space):
        return self._advance(state, action_space)

    def cancel_before_start(self, harness, state, action_space):
        if self._cancel is not None:
            return self._cancel(state, action_space)
        return super().cancel_before_start(harness, state, action_space)


class ExplorationCommands:
    def __init__(self, loop, handlers: Mapping[str, CommandHandler] | None = None):
        self.loop = loop
        self.active: str | None = None
        # In-process receipts support retries. They are not a durable job journal.
        self.receipts: dict[str, CommandState] = {}
        defaults = {
            'execute': _BuiltinHandler(self._prepare_execute, self._advance_execute),
            'configure_prompt': _BuiltinHandler(self._prepare_prompt,
                lambda state, actions: self._advance_prompt(state),
                cancel_before_start=lambda state, actions: self._advance_prompt(state)),
            'sft': _BuiltinHandler(self._prepare_sft,
                lambda state, actions: self.advance_job(state, training=True)),
            'rl': _BuiltinHandler(self._prepare_rl, self._advance_rl),
            'evaluate': _BuiltinHandler(self._prepare_evaluation,
                lambda state, actions: self.advance_job(state, training=False)),
        }
        for mode, handler in (handlers or {}).items():
            if not isinstance(mode, str) or not mode.strip():
                raise ValueError('Command handler modes must be nonempty strings')
            if not isinstance(handler, CommandHandler):
                raise TypeError('Handlers must implement CommandHandler')
            defaults[mode] = handler
        self.builtin_modes = tuple(mode for mode in defaults if mode not in (handlers or {}))
        self.customized = bool(handlers)
        self.handlers = MappingProxyType(defaults)

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
        if (self.busy or loop.state.active_directive_id is not None or loop.state.pending_episode is not None or loop.training.busy
                or loop.evaluation.busy or (loop.state.rl is not None and not loop.state.rl.terminal)):
            raise RuntimeError('Exploration is busy')
        state = CommandState(deepcopy(directive))
        self.receipts[directive.directive_id] = state
        self.active = directive.directive_id
        try:
            if directive.mode not in self.handlers:
                raise NotImplementedError(f"Unsupported exploration mode: {directive.mode}")
            self.handlers[directive.mode].prepare(loop, state)
        except BaseException as exc:
            self._fail(state, exc)
            raise
        return self.inspect(directive.directive_id)

    @staticmethod
    def _prepare_execute(state):
        if state.directive.payload is not None:
            raise ValueError('Execute does not accept a system command payload')

    @staticmethod
    def _prepare_prompt(state):
        if not isinstance(state.directive.payload, PromptSpec):
            raise ValueError('configure_prompt requires PromptSpec')

    def validate_training_payload(self, payload, expected, *, apply) -> None:
        """Validate learner, payload type and optional model-update support.

        Handler support for the default harness only; starts no external work.
        """
        if self.loop.learner is None:
            raise NotImplementedError('Training directives require a learner')
        if not isinstance(payload, expected):
            raise ValueError(f'Training requires {expected.__name__}')
        if apply and not isinstance(self.loop.policy, ModelUpdatable):
            raise TypeError('This command requires a model-updatable policy')

    def _prepare_sft(self, state):
        payload = state.directive.payload
        self.validate_training_payload(payload, SFTSpec, apply=isinstance(payload, SFTSpec) and payload.apply_result)
        base = payload.base or getattr(self.loop.policy, 'model', None)
        if not isinstance(base, ModelRef):
            raise ValueError('SFT requires a base model')
        if not state.directive.input_refs:
            raise ValueError('The built-in SFT command requires data artifacts; use train for backend-owned inputs')
        state.request = ActorTrainingRequest(base, state.directive.input_refs, payload.recipe_ref)

    def _prepare_rl(self, state):
        self.validate_training_payload(state.directive.payload, RLSpec, apply=True)

    def _prepare_evaluation(self, state):
        if self.loop.evaluation.backend is None:
            raise NotImplementedError('Evaluation directives require an evaluator')
        payload = state.directive.payload
        if not isinstance(payload, EvaluationSpec):
            raise ValueError('Evaluation requires EvaluationSpec')
        state.request = EvaluationRequest(payload.candidate, payload.protocol_ref, state.directive.input_refs)

    def inspect(self, directive_id):
        state = self.receipts[directive_id]
        return deepcopy(ExplorationProgress(directive_id, state.phase, state.directive, state.job, state.feedback))

    def _finish(self, state, feedback):
        if not isinstance(feedback, Feedback) or feedback.directive_id != state.directive.directive_id:
            raise ValueError('Command feedback must match its directive')
        feedback = replace(feedback, simulated=feedback.simulated or state.directive.simulated)
        state.feedback = deepcopy(feedback)
        state.phase = feedback.status.value
        self.loop.state.last_feedback = deepcopy(feedback)

    def _fail(self, state, exc):
        self._finish(state, Feedback(state.directive.directive_id,
            ExecutionStatus.FAILED if isinstance(exc, Exception) else ExecutionStatus.CANCELLED,
            'Exploration command failed', state.directive.simulated,
            error=f'{type(exc).__name__}: {exc}'))

    def advance(self, directive_id, action_space):
        state = self.receipts[directive_id]
        if state.feedback is not None:
            return self.inspect(directive_id)
        handler = self.handlers[state.directive.mode]
        if state.cancel_requested and not state.started:
            feedback = handler.cancel_before_start(self.loop, state, action_space)
        else:
            # An uncertain first advance may already have started external work.
            state.started = True
            feedback = handler.advance(self.loop, state, action_space)
        if feedback is not None:
            self._finish(state, feedback)
        return self.inspect(directive_id)

    def _advance_execute(self, state, action_space):
        state.phase = 'executing'
        try:
            self._finish(state, self.loop._run_episode(state.directive, action_space))
        except BaseException as exc:
            if self.loop.state.pending_episode is not None:
                raise  # Save system feedback before exposing a terminal receipt.
            feedback = self.loop.state.last_feedback
            if feedback is not None and feedback.directive_id == state.directive.directive_id:
                self._finish(state, feedback)
            else:
                self._fail(state, exc)
            raise

    def _advance_prompt(self, state):
        loop, directive = self.loop, state.directive
        if state.record is None:
            before = loop.state.instructions
            after = before if state.cancel_requested else directive.payload.instructions
            status = ExecutionStatus.CANCELLED if state.cancel_requested else ExecutionStatus.COMPLETED

            def describe(instructions):
                return {'sha256': sha256(instructions.encode('utf-8')).hexdigest(),
                        'characters': len(instructions), 'configured': bool(instructions)}

            # Freeze both identity and payload before any write. A lost record response
            # retries this receipt, never reapplies the command or rebuilds its evidence.
            state.record = ExperienceRecord(
                str(uuid4()), 'runtime_feedback', 'actor', directive.directive_id,
                f'configure_prompt {status.value}', directive.simulated,
                details={'mode': directive.mode, 'status': status.value,
                         'before': describe(before), 'after': describe(after)},
            )
            if not state.cancel_requested:
                loop.state.instructions = after
            state.phase = 'recording'
        loop.experience.record(deepcopy(state.record))
        self._finish(state, Feedback(
            directive.directive_id, ExecutionStatus(state.record.details['status']),
            state.record.summary, directive.simulated,
            evidence_refs=(state.record.record_id,),
        ))

    def advance_job(self, state: CommandState, *, training: bool) -> None:
        """Advance a prepared job for a default-harness handler in place.

        Select training or evaluation with ``training``. Reuse the state's
        request/idempotency key, honor cancellation, record evidence and apply
        training results when requested. Terminal feedback is written to state,
        not returned; uncertain work retains the receipt for a later advance.
        """
        loop, directive = self.loop, state.directive
        if state.phase in ('accepted', 'submitting'):
            # Submission may have succeeded remotely before raising. Retain request/key.
            state.phase = 'submitting'
            submit = loop._submit_learning if training else loop._submit_evaluation
            state.job = submit(state.request, idempotency_key=state.idempotency_key)
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
            except ModelActivationPending:
                raise  # Retry application of this job without a terminal receipt.
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

    def _advance_rl(self, state, action_space):
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
                rl = loop._step_rl(action_space)
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
