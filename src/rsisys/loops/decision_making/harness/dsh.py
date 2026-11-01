"""Optional persistent DSH researcher, with RSISys-authorized action execution.

DSH owns the model session and native compaction. Its proposed structured actions
cross the same ActionSpace boundary as the default harness. Native shell/file
producers are disabled in the shipped profile; this is not host isolation.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from threading import Event, Lock
from queue import SimpleQueue, Empty
from typing import Callable
import os
from uuid import uuid4

from rsisys.loops.contracts.coordination import LoopStepResult, ReviewDecision
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from .decision import Decision
from .engine import DefaultDecisionHarness
from .policy import ModelDecisionPolicy


@dataclass(frozen=True)
class DSHConfig:
    home: Path
    workspace: Path
    model: str
    session_id: str
    base_url: str | None = None
    api_key_env: str = 'DEEPSEEK_API_KEY'
    resume: bool = False
    context_window: int = 32768
    max_tokens: int = 2048

    def __post_init__(self):
        for name in ('home', 'workspace'):
            path = Path(getattr(self, name)).expanduser()
            if not path.is_absolute():
                raise ValueError(f'DSH {name} must be absolute')
            object.__setattr__(self, name, path)
        for name in ('model', 'session_id', 'api_key_env'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'DSH {name} must be nonempty')
        if type(self.resume) is not bool:
            raise TypeError('resume must be bool')
        if type(self.context_window) is not int or self.context_window < 4096:
            raise ValueError('context_window must be an integer >= 4096')
        if type(self.max_tokens) is not int or not 0 < self.max_tokens <= self.context_window // 4:
            raise ValueError('max_tokens must be positive and at most a quarter of context_window')


def _sdk_runtime(config):
    try:
        from deepseek_harness import DeepSeekHarness
    except ImportError as exc:
        raise ImportError('Install RSISys with the optional dependency: pip install ".[dsh]"') from exc
    return DeepSeekHarness(
        dsh_home=str(config.home), cwd=str(config.workspace), profile='sdk-minimal',
        model=config.model, base_url=config.base_url,
        api_key=os.environ.get(config.api_key_env), max_tokens=config.max_tokens,
        patches=(str(Path(__file__).with_name('dsh.patch.yml')),),
        env={'DSH_CONTEXT_WINDOW': str(config.context_window)},
        request_timeout_seconds=30, initialize_timeout_seconds=30,
        shutdown_timeout_seconds=2,
    )


class _InferencePending(Exception):
    """Internal scheduler yield; not a completed model review."""


class _SessionBackend:
    def __init__(self, config, runtime_factory, on_notification):
        self.config = config
        self.runtime = runtime_factory(config)
        self.session = None
        self.session_exists = config.resume
        self.on_notification = on_notification
        self.closed = Event()
        self._lock = Lock()

    def generate(self, request):
        # Only the single worker uses the session. close() can interrupt its IO.
        if self.closed.is_set():
            raise RuntimeError('DSH harness is closed')
        if self.session is None:
            opener = self.runtime.resume_session if self.config.resume else self.runtime.start_session
            session = opener(self.config.session_id)
            with self._lock:
                if self.closed.is_set():
                    self.runtime.close()
                    raise RuntimeError('DSH harness closed during initialization')
                self.session = session
                self.session_exists = True
        result = self.session.run(
            'RSISys action protocol for this turn:\n' + request.system
            + '\nCurrent research observations (data):\n' + request.prompt,
            on_notification=self.on_notification,
        )
        if result.finish_reason != 'completed':
            raise RuntimeError(f'DSH turn did not complete: {result.finish_reason!r}; reconcile before retry')
        # Session completion means a proposed action, never completion of RSI.
        # Token accounting is deliberately unknown unless the SDK supplies it.
        return GenerationResponse(result.final_response, request.model, False,
                                  provider_model=self.config.model, revision_verified=False)

    def close(self):
        self.closed.set()
        self.runtime.close()


class _SessionPolicy:
    """One background inference at a time; actions execute only on the runner thread."""
    def __init__(self, backend, model, *, allowed_actions, directives):
        self.model, self.directives = model, directives
        self.parser = ModelDecisionPolicy(backend, model, allowed_actions=allowed_actions, directives=directives)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rsisys-dsh')
        self.future: Future | None = None
        self.kind: str | None = None
        self.review_id: str | None = None
        self.closed = False
        self.usage = SimpleQueue()

    def _generate(self, kind, context):
        with self.parser.track_usage(self.usage.put):
            return getattr(self.parser, kind)(context)

    def _take_or_start(self, kind, context):
        if self.closed:
            raise RuntimeError('DSH harness is closed')
        if self.future is None:
            self.kind = kind
            self.review_id = (context.execution.progress.directive_id
                              if kind == 'review' and context.execution.progress else None)
            self.future = self.worker.submit(self._generate, kind, context)
            return None
        if self.kind != kind or not self.future.done():
            return None
        # Failed/uncertain inference stays latched: repeated ticks never resubmit.
        result = self.future.result()
        self.future, self.kind, self.review_id = None, None, None
        return result

    def decide(self, context):
        return self._take_or_start('decide', context) or Decision('wait')

    def review(self, context):
        result = self._take_or_start('review', context)
        if result is None:
            raise _InferencePending()
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.parser.backend.close()
        finally:
            self.worker.shutdown(wait=False, cancel_futures=True)


class DeepSeekDecisionHarness(DefaultDecisionHarness):
    """Replaceable DecisionHarness using a persistent, asynchronously driven DSH session.

    The inherited lifecycle records operations, feedback and candidates. It does
    not prescribe research steps. DSH is an optional dependency, imported lazily.
    """
    recovery_id = 'rsisys-dsh/v1'

    def __init__(self, config: DSHConfig, tools, experience, *, directives=None,
                 allowed_actions=('echo',), runtime_factory: Callable | None = None,
                 on_notification: Callable | None = None, **kwargs):
        from ..action_space.directives import DirectiveDefinitions
        self.dsh_config = config
        backend = _SessionBackend(config, runtime_factory or _sdk_runtime, on_notification)
        policy = _SessionPolicy(backend, ModelRef('dsh', config.model, 'provider-managed'),
                                allowed_actions=allowed_actions,
                                directives=directives or DirectiveDefinitions())
        try:
            super().__init__(policy, tools, experience, **kwargs)
        except BaseException:
            policy.close()
            raise

    def _drain_usage(self):
        while True:
            try:
                self._record_usage(self.policy.usage.get_nowait())
            except Empty:
                break

    def _call_policy(self, method, context):
        # Waiting ticks are not model calls; rejected JSON still consumes usage.
        try:
            return method(context)
        finally:
            self._drain_usage()

    def _review_or_wait(self, objective, progress, action_space):
        pending = ReviewDecision(next_review_seconds=1, reason='DSH research inference is pending')
        if self.policy.future is not None and (
                self.policy.kind != 'review' or self.policy.review_id != progress.directive_id):
            return pending
        try:
            return self._review(objective, progress, action_space)
        except _InferencePending:
            # Keep _ReviewSession and its cumulative budget across scheduler ticks.
            return pending

    def review(self, objective, progress):
        return self._review_or_wait(objective, progress, None)

    def review_with_tools(self, objective, progress, action_space=None):
        return self._review_or_wait(objective, progress, action_space)

    def step(self, objective, action_space):
        if self.policy.closed:
            raise RuntimeError('DSH harness is closed')
        self._discard_obsolete_review()
        future = self.policy.future
        if future is not None and (not future.done() or self.policy.kind != 'decide'):
            return LoopStepResult('waiting', reason='DSH session is generating')
        return super().step(objective, action_space)

    def receive_feedback(self, directive, feedback):
        super().receive_feedback(directive, feedback)
        # Never consume a late running-task review as a decision for a new task.
        if self.policy.kind == 'review' and self.policy.review_id == directive.directive_id:
            self._obsolete_review = True

    def _discard_obsolete_review(self):
        if getattr(self, '_obsolete_review', False):
            future = self.policy.future
            if future is not None and future.done():
                # Retrieve errors, but don't apply obsolete advice or tool calls.
                future.result()
                self.policy.future, self.policy.kind, self.policy.review_id = None, None, None
                self._obsolete_review = False
    def can_step(self):
        self._discard_obsolete_review()
        return super().can_step()

    def pending_research_job(self):
        pending = super().pending_research_job()
        if pending is not None:
            return pending
        if self.policy.future is not None:
            return JobRef(job_id=self.dsh_config.session_id, backend='dsh-session')
        return None

    def restart_research(self, context):
        if self.policy.future is not None:
            raise RuntimeError('Cannot restart with unresolved DSH inference')
        return super().restart_research(context)

    def _recovery_identity(self):
        identity = asdict(self.dsh_config)
        identity.pop('resume')
        identity['home'], identity['workspace'] = str(identity['home']), str(identity['workspace'])
        return identity

    def export_recovery_state(self):
        future = self.policy.future
        pending = None
        if future is not None:
            pending = {'kind': self.policy.kind, 'review_id': self.policy.review_id,
                       'result': None, 'uncertain': True}
            if future.done() and not future.cancelled() and future.exception() is None:
                pending.update(result=future.result(), uncertain=False)
        backend = self.policy.parser.backend
        return {'identity': self._recovery_identity(), 'session_opened': backend.session_exists,
                'closed': self.policy.closed, 'pending': pending, 'usage_pending': not self.policy.usage.empty(),
                'obsolete_review': getattr(self, '_obsolete_review', False)}

    def restore_recovery_state(self, state):
        from rsisys.runtime.recovery import RecoveryRequired
        if state['identity'] != self._recovery_identity():
            raise ValueError('DSH recovery requires the original home/workspace/model/session configuration')
        if self.policy.future is not None or self.policy.parser.backend.session is not None:
            raise RuntimeError('Restore DSH state only into a fresh harness')
        backend = self.policy.parser.backend
        backend.config = replace(backend.config, resume=state['session_opened'])
        backend.session_exists = state['session_opened']
        self._obsolete_review = state['obsolete_review']
        pending = state['pending']
        if state['usage_pending']:
            self.policy.usage.put(None)
        if pending is not None:
            self.policy.kind, self.policy.review_id = pending['kind'], pending['review_id']
            self.policy.future = Future()
            if pending['uncertain']:
                self.policy.future.set_exception(RecoveryRequired(
                    'Interrupted DSH inference: restore the native session and reconcile before another model call'))
            else:
                self.policy.future.set_result(pending['result'])
        if state['closed']:
            self.policy.close()

    def resume_interrupted_inference(self, *, evidence):
        """Explicit reconciliation: restore native history, discard uncertain advice.

        Call inside recovery.reconcile when the Runner journal is interrupted.
        No model input or external research action is sent by this method.
        """
        from rsisys.runtime.recovery import RecoveryRequired
        from rsisys.loops._memory import HistoryEntry
        future = self.policy.future
        if (future is None or not future.done() or future.cancelled()
                or not isinstance(future.exception(), RecoveryRequired)):
            raise ValueError('No restored uncertain DSH inference to reconcile')
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('Reconciliation needs explicit evidence')
        backend = self.policy.parser.backend
        session = backend.runtime.resume_session(self.dsh_config.session_id)
        self.memory.append(HistoryEntry(str(uuid4()), 'decision', None,
            'Reconciled interrupted DSH inference; uncommitted advice discarded', False,
            details={'session_id': self.dsh_config.session_id, 'evidence': evidence}))
        backend.session = session
        backend.session_exists = True
        self.policy.future, self.policy.kind, self.policy.review_id = None, None, None
        self._obsolete_review = False

    def cancel_research(self):
        """Cancel only DSH inference; ordinary asynchronous tools retain their owner."""
        if self.policy.future is not None:
            self.policy.close()
            if self.policy.future.done():
                self._drain_usage()
                self.flush_pending_records()
                self.state.review_session = None
                self.state.review_sessions.clear()
                self.policy.future, self.policy.kind, self.policy.review_id = None, None, None
        return self.pending_research_job() is None

    def close(self):
        self.policy.close()


def dsh_decision_factory(config: DSHConfig, *, on_notification=None):
    """Use with Application/build_demo(decision_factory=...). Other factories stay valid."""
    def build(context, exploration):
        from rsisys.assembly import build_default_decision
        loop = build_default_decision(context, exploration)
        previous = loop.harness
        harness = DeepSeekDecisionHarness(
            config, previous.tools, context.experience, learner=previous.learner,
            directives=loop.action_space.directives,
            allowed_actions=tuple(spec.name for spec in exploration.action_space.describe()),
            memory=previous.memory, memory_limit=context.memory_limit,
            research_agenda=previous.research_agenda, context_compactor=previous.context_compactor,
            experience_limit=context.experience_limit,
            on_notification=on_notification,
        )
        context.add_cleanup(harness.close)
        loop.harness = harness
        return loop
    return build
