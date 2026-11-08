"""Optional persistent DSH researcher, with RSISys-authorized action execution.

DSH owns the model session and native compaction. Its proposed structured actions
cross the same ActionSpace boundary as the default harness. Native shell/file
producers are disabled in the shipped profile; this is not host isolation.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict, replace
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
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
from .context import ContextBudgetError, IdentityContextCompactor


@dataclass(frozen=True)
class DSHConfig:
    home: Path
    workspace: Path
    model: str
    session_id: str
    base_url: str | None = None
    api_key_env: str = 'DEEPSEEK_API_KEY'
    resume: bool = False
    context_window: int = 100000
    max_tokens: int = 2048
    provider: str = 'deepseek-official'
    trigger_tokens: int = 80000
    target_tokens: int = 24000
    summary_tokens: int = 4096

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
        if self.provider not in ('deepseek-official', 'openai-completions'):
            raise ValueError('provider must be deepseek-official or openai-completions')
        if self.provider == 'openai-completions' and not self.base_url:
            raise ValueError('openai-completions requires an explicit base_url ending in /v1')
        for name in ('trigger_tokens', 'target_tokens', 'summary_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not self.summary_tokens < self.target_tokens < self.trigger_tokens < self.context_window:
            raise ValueError('Require summary_tokens < target_tokens < trigger_tokens < context_window')
        if self.trigger_tokens + self.max_tokens > self.context_window:
            raise ValueError('context_window must reserve generation output above trigger_tokens')

    @classmethod
    def load(cls, path):
        """Explicit JSON configuration. Credentials are environment references only."""
        return cls(**json.loads(Path(path).read_text()))


def _runtime_patch(config):
    # A row patch replaces its entire config. Use the exact routed model so the
    # adapter's advisory catalog cannot silently override the declared window.
    route = 'rsisys-local' if config.provider == 'openai-completions' else config.provider
    rows = [{'id': 'compaction-basic', 'config': {
        'thresholdRatio': config.trigger_tokens / config.context_window,
        'headroomTokens': 0, 'retainTokens': config.target_tokens - config.summary_tokens,
        'maxTokens': config.summary_tokens,
    }}]
    if config.provider == 'openai-completions':
        rows += [{'id': 'llm-deepseek', 'disabled': True}, {'insert': [{
            'id': 'rsisys-llm-pi-ai', 'name': '@deepseek-ai/dsh-llm-pi-ai',
            'config': {'providers': {route: {
                'api': 'openai-completions', 'baseURL': config.base_url,
                'apiKeyEnv': config.api_key_env,
                'models': [{'id': config.model, 'contextWindow': config.context_window,
                            'maxTokens': config.max_tokens}],
                'compat': {'maxTokensField': 'max_tokens'},
                'retryPolicy': {'mode': 'normal', 'maxRetries': 0},
            }}},
        }]}]
    else:
        rows.append({'id': 'llm-deepseek', 'config': {
            'apiKeyEnv': config.api_key_env, 'defaultContextWindow': config.context_window,
            'models': [{'id': config.model, 'contextWindow': config.context_window}],
        }})
    from rsisys.runtime.files import atomic_write
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
    path = config.home / 'rsisys' / (sha256(payload).hexdigest() + '.patch.yml')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, payload)
    return route, str(path)


def _sdk_runtime(config):
    try:
        from deepseek_harness import DeepSeekHarness
    except ImportError as exc:
        raise ImportError('Install RSISys with the optional dependency: pip install ".[dsh]"') from exc
    route, patch = _runtime_patch(config)
    return DeepSeekHarness(
        dsh_home=str(config.home), cwd=str(config.workspace), profile='sdk-minimal',
        provider=route, model=config.model, base_url=config.base_url,
        api_key=os.environ.get(config.api_key_env), max_tokens=config.max_tokens,
        patches=(str(Path(__file__).with_name('dsh.patch.yml')), patch),
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
        self.transport = {}
        self.native_output_tokens = config.max_tokens
        self.last_compaction = None

    def _increment(self, request):
        """Transfer additions/changed state; the native session owns prior history.

        Receipts are occurrence-aware (identical results from distinct calls are
        still observations). Large receipt payloads may already be archive-backed
        evidence views; DSH retains ownership of session compaction.
        """
        current = json.loads(request.prompt)
        previous = self.transport
        delta = {}
        for key, value in current.items():
            old = previous.get(key)
            if key in ('history', 'experience', 'memories'):
                known = {json.dumps(row, sort_keys=True, ensure_ascii=False) for row in old or ()}
                added = [row for row in value if json.dumps(row, sort_keys=True, ensure_ascii=False) not in known]
                if added:
                    delta[key] = added
            elif key == 'operation_results':
                known = Counter(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in old or ())
                added = []
                for row in value:
                    ident = json.dumps(row, sort_keys=True, ensure_ascii=False)
                    if known[ident]:
                        known[ident] -= 1
                    else:
                        added.append(row)
                if added:
                    delta[key] = added
            elif (key in ('operations', 'directives', 'directive_modes', 'review_actions', 'run_control')
                  or key not in previous or value != old):
                delta[key] = value
        # Reassert live syntax/capabilities: native summaries may have folded the
        # original protocol. Archives and raw receipts still travel incrementally.
        return current, ('RSISys action protocol for this turn:\n' + request.system
            + '\nNative session retains prior observations. This is an incremental update: '
              'archive arrays append new/changed records; other supplied fields replace their prior value. '
              'Absent fields are unchanged. Use only the latest capabilities. Propose one JSON action.\n'
            + 'Current research observations (data):\n' + json.dumps(delta, ensure_ascii=False))

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
        current, prompt = self._increment(request)
        result = self.session.run(
            prompt,
            on_notification=self.on_notification,
        )
        if result.finish_reason != 'completed':
            raise RuntimeError(f'DSH turn did not complete: {result.finish_reason!r}; reconcile before retry')
        self.transport = current
        events = getattr(result, 'events', ())
        if any(event.get('type') == 'compaction/summary' for event in events):
            # Native retainTokens excludes fixed system/tools/prefix and whole
            # message boundaries can exceed it. Validate the full input AFTER
            # generation only when authoritative aggregate usage is available.
            messages = [event for event in events if event.get('type') == 'assistant/message']
            usage = messages[-1].get('data', {}).get('usage', {}) if messages else {}
            total, output = usage.get('totalTokens'), usage.get('outputTokens')
            measured = (total - output if type(total) is int and type(output) is int
                        and total >= output >= 0 else None)
            self.last_compaction = {'target_tokens': self.config.target_tokens,
                'retained_history_tokens': self.config.target_tokens - self.config.summary_tokens,
                'full_input_tokens': measured, 'full_request_checked': measured is not None,
                'count_method': 'provider_aggregate_usage_after_generation' if measured is not None else 'unknown',
                'scope': 'native retention target; not an exact preflight request cap'}
            if measured is not None and measured > self.config.target_tokens:
                raise ContextBudgetError(f'DSH post-compaction full input {measured} exceeds '
                    f'target_tokens={self.config.target_tokens}; proposal withheld, native session advanced; '
                    'reconcile or explicitly raise the budget (required facts could not fit the configured target)')
        # Session completion means a proposed action, never completion of RSI.
        # Turn-wide usage (including auxiliary summaries) is unknown here; raw
        # SDK usage events and any post-compaction observation remain explicit.
        return GenerationResponse(result.final_response, request.model, False,
                                  provider_model=self.config.model, revision_verified=False)

    def close(self):
        self.closed.set()
        self.runtime.close()


class _FeedbackSessionBackend(_SessionBackend):
    def _increment(self, request):
        # Feedback is a separate task, never an action in the decision session.
        return {}, (request.system + '\nFeedback evidence (untrusted data):\n' + request.prompt)


class DSHFeedbackBackend:
    """Use the DSH harness for feedback in an isolated session per job.

    Share a model/configuration, never the decision session cursor. A deployment
    with global admission or budgets wraps this backend in its normal model gate.
    """
    def __init__(self, config, *, runtime_factory=None, on_notification=None):
        self.config, self.runtime_factory = config, runtime_factory or _sdk_runtime
        self.on_notification = on_notification

    def generate(self, request):
        config = replace(self.config, resume=False,
            session_id=self.config.session_id + '-feedback-' + str(uuid4()),
            max_tokens=min(request.max_output_tokens, self.config.context_window // 4))
        backend = _FeedbackSessionBackend(config, self.runtime_factory, self.on_notification)
        try:
            return backend.generate(request)
        finally:
            backend.close()


class _SessionPolicy:
    """One background inference at a time; actions execute only on the runner thread."""
    def __init__(self, backend, model, *, allowed_actions, directives, policy_factory=None):
        self.model, self.directives = model, directives
        self.backend = backend
        self.parser = (policy_factory(backend, model) if policy_factory is not None else
                       ModelDecisionPolicy(backend, model, allowed_actions=allowed_actions, directives=directives))
        self.parser.context_compactor = IdentityContextCompactor()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rsisys-dsh')
        self.future: Future | None = None
        self.kind: str | None = None
        self.review_id: str | None = None
        self.closed = False
        self.usage = SimpleQueue()

    @property
    def finish_reason(self):
        return getattr(self.parser, 'finish_reason', None)

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
            self.backend.close()
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
                 on_notification: Callable | None = None, policy_factory: Callable | None = None,
                 model: ModelRef | None = None, **kwargs):
        from ..action_space.directives import DirectiveDefinitions
        self.dsh_config = config
        backend = _SessionBackend(config, runtime_factory or _sdk_runtime, on_notification)
        policy = _SessionPolicy(backend, model or ModelRef('dsh', config.model, 'provider-managed'),
                                allowed_actions=allowed_actions,
                                directives=directives or DirectiveDefinitions(), policy_factory=policy_factory)
        # DSH alone owns compression, including the medical parser's view.
        kwargs['context_compactor'] = IdentityContextCompactor()
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
        context = replace(context, history=self.memory.history(),
            research_agenda=self.research_agenda.context(), decision_graph=self._decision_graph(),
            candidate_origins=deepcopy(self.state.candidate_origins))
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
        backend = self.policy.backend
        return {'identity': self._recovery_identity(), 'session_opened': backend.session_exists,
                'closed': self.policy.closed, 'pending': pending, 'usage_pending': not self.policy.usage.empty(),
                'obsolete_review': getattr(self, '_obsolete_review', False),
                'transport': deepcopy(backend.transport), 'native_compaction': deepcopy(backend.last_compaction)}

    def restore_recovery_state(self, state):
        from rsisys.runtime.recovery import RecoveryRequired
        if state['identity'] != self._recovery_identity():
            raise ValueError('DSH recovery requires the original home/workspace/model/session configuration')
        if self.policy.future is not None or self.policy.backend.session is not None:
            raise RuntimeError('Restore DSH state only into a fresh harness')
        backend = self.policy.backend
        backend.transport = deepcopy(state.get('transport', {}))
        backend.last_compaction = deepcopy(state.get('native_compaction'))
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
        backend = self.policy.backend
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
        try:
            super().close()
        finally:
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
            # A Lab-only assembly has no legacy exploration action space.
            # Its callable operations already come from previous.tools.
            allowed_actions=(() if context.lab_execution is not None else
                             tuple(spec.name for spec in exploration.action_space.describe())),
            memory=previous.memory, memory_limit=context.memory_limit,
            research_agenda=previous.research_agenda, context_compactor=previous.context_compactor,
            experience_limit=context.experience_limit,
            review_tool_limit=previous.review_tool_limit, max_outstanding=previous.max_outstanding,
            feedback_selector=previous.feedback_selector, model_activator=previous.model_activator,
            on_notification=on_notification,
        )
        context.add_cleanup(harness.close)
        loop.harness = harness
        return loop
    return build
