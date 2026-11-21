"""Optional persistent DSH researcher, with RSIComposer-authorized action execution.

DSH owns the model session and native compaction. Its proposed structured actions
cross the same ActionSpace boundary as the default harness. Native shell/file
producers are disabled in the shipped profile; this is not host isolation.
"""
import json
import os
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Lock
from uuid import uuid4

from rsicomposer.harnesses.decision.context import (
    ContextBudgetError,
    IdentityContextCompactor,
)
from rsicomposer.harnesses.decision.policy import ModelDecisionPolicy
from rsicomposer.loops.contracts.decision import Decision
from rsicomposer.loops.contracts.jobs import JobRef
from rsicomposer.loops.contracts.model_service import GenerationResponse, ModelRef


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
    from rsicomposer.runtime.storage.files import atomic_write
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
    path = config.home / 'rsicomposer' / (sha256(payload).hexdigest() + '.patch.yml')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, payload)
    return route, str(path)


def _sdk_runtime(config):
    try:
        from deepseek_harness import DeepSeekHarness
    except ImportError as exc:
        raise ImportError('Install RSIComposer with the optional dependency: pip install ".[dsh]"') from exc
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
            elif (key in ('operations', 'run_control')
                  or key not in previous or value != old):
                delta[key] = value
        # Reassert live syntax/capabilities: native summaries may have folded the
        # original protocol. Archives and raw receipts still travel incrementally.
        return current, ('RSIComposer action protocol for this turn:\n' + request.system
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
    def __init__(self, backend, model, *, policy_factory=None):
        self.model = model
        self.backend = backend
        self.parser = (policy_factory(backend, model) if policy_factory is not None else
                       ModelDecisionPolicy(backend, model))
        self.parser.context_compactor = IdentityContextCompactor()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rsisys-dsh')
        self.future: Future | None = None
        self.kind: str | None = None
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
            self.future = self.worker.submit(self._generate, kind, context)
            return None
        if self.kind != kind or not self.future.done():
            return None
        # Failed/uncertain inference stays latched: repeated ticks never resubmit.
        result = self.future.result()
        self.future, self.kind = None, None
        return result

    def can_decide(self):
        if self.closed:
            raise RuntimeError('DSH policy is closed')
        return self.future is None or (self.future.done() and self.kind == 'decide')

    def decide(self, context):
        return self._take_or_start('decide', context) or Decision('wait')


    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.backend.close()
        finally:
            self.worker.shutdown(wait=False, cancel_futures=True)


class DeepSeekResearchPolicy(_SessionPolicy):
    """DSH inference lifecycle, independent of run execution and tools."""
    recovery_id = 'rsisys-dsh-policy/v2'
    asynchronous_usage = True

    def __init__(self, config, harness, *, runtime_factory=None, on_notification=None,
                 policy_factory=None, model=None):
        self.dsh_config, self.harness = config, harness
        self._restored_metadata = None
        backend = _SessionBackend(config, runtime_factory or _sdk_runtime, on_notification)
        super().__init__(backend, model or ModelRef('dsh', config.model, 'provider-managed'),
                         policy_factory=policy_factory)

    def decision_metadata(self, ident):
        from rsicomposer.host.ledger import DecisionResearchMetadata
        if self._restored_metadata is not None and self._restored_metadata[0] == ident:
            return DecisionResearchMetadata.from_record(deepcopy(self._restored_metadata[1]))
        read = getattr(self.parser, 'decision_metadata', None)
        return read(ident) if callable(read) else DecisionResearchMetadata()

    def drain_usage(self):
        result = []
        while True:
            try:
                result.append(self.usage.get_nowait())
            except Empty:
                return tuple(result)

    def pending_research_job(self):
        if self.future is not None:
            return JobRef(job_id=self.dsh_config.session_id, backend='dsh-session')
        return None

    def update_model(self, model):
        if self.future is not None:
            raise RuntimeError('Cannot rebind a model with unresolved DSH inference')
        self.parser.update_model(model)
        self.model = model

    def drain_research(self):
        if self.future is None:
            return True
        if not self.future.done():
            return False
        # Complete the accepted inference without applying its uncommitted advice.
        self.future.result()
        self.future, self.kind = None, None
        self._restored_metadata = None
        return True

    def cancel_research(self):
        if self.future is not None:
            self.close()
            if self.future.done():
                self.future, self.kind = None, None
        return self.future is None


    def _recovery_identity(self):
        identity = asdict(self.dsh_config)
        identity.pop('resume')
        identity['home'], identity['workspace'] = str(identity['home']), str(identity['workspace'])
        return identity

    def state(self):
        future = self.future
        pending = None
        if future is not None:
            pending = {'kind': self.kind,
                       'result': None, 'uncertain': True}
            if future.done() and not future.cancelled() and future.exception() is None:
                result = future.result()
                pending.update(result=result, uncertain=False,
                    metadata=asdict(self.decision_metadata(result.decision_id)))
        backend = self.backend
        usages = self.drain_usage()
        for usage in usages:
            self.usage.put(usage)
        return {'identity': self._recovery_identity(), 'session_opened': backend.session_exists,
                'closed': self.closed, 'pending': pending, 'usage_pending': usages,
                'transport': deepcopy(backend.transport), 'native_compaction': deepcopy(backend.last_compaction)}

    def restore(self, state):
        from rsicomposer.runtime.state.recovery import RecoveryRequired
        if state['identity'] != self._recovery_identity():
            raise ValueError('DSH recovery requires the original home/workspace/model/session configuration')
        if self.future is not None or self.backend.session is not None:
            raise RuntimeError('Restore DSH state only into a fresh harness')
        backend = self.backend
        backend.transport = deepcopy(state.get('transport', {}))
        backend.last_compaction = deepcopy(state.get('native_compaction'))
        backend.config = replace(backend.config, resume=state['session_opened'])
        backend.session_exists = state['session_opened']
        pending = state['pending']
        if not isinstance(state['usage_pending'], tuple):
            raise ValueError('DSH usage checkpoint requires exact pending responses')
        for usage in state['usage_pending']:
            self.usage.put(usage)
        if pending is not None:
            self.kind = pending['kind']
            self.future = Future()
            if pending['uncertain']:
                self.future.set_exception(RecoveryRequired(
                    'Interrupted DSH inference: restore the native session and reconcile before another model call'))
            else:
                from rsicomposer.host.ledger import DecisionResearchMetadata
                metadata = DecisionResearchMetadata.from_record(pending['metadata'])
                self._restored_metadata = (pending['result'].decision_id, asdict(metadata))
                self.future.set_result(pending['result'])
        if state['closed']:
            self.close()

    def resume_interrupted_inference(self, *, evidence):
        """Explicit reconciliation: restore native history, discard uncertain advice.

        Call inside recovery.reconcile when the Runner journal is interrupted.
        No model input or external research action is sent by this method.
        """
        from rsicomposer.harnesses.memory import HistoryEntry
        from rsicomposer.runtime.state.recovery import RecoveryRequired
        future = self.future
        if (future is None or not future.done() or future.cancelled()
                or not isinstance(future.exception(), RecoveryRequired)):
            raise ValueError('No restored uncertain DSH inference to reconcile')
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('Reconciliation needs explicit evidence')
        backend = self.backend
        session = backend.runtime.resume_session(self.dsh_config.session_id)
        self.harness.memory.append(HistoryEntry(str(uuid4()), 'decision', None,
            'Reconciled interrupted DSH inference; uncommitted advice discarded', False,
            details={'session_id': self.dsh_config.session_id, 'evidence': evidence}))
        backend.session = session
        backend.session_exists = True
        self.future, self.kind = None, None
        self._restored_metadata = None


def dsh_decision_factory(config: DSHConfig, *, on_notification=None):
    def build(context, execution):
        from rsicomposer.component_context import controller_wiring
        from rsicomposer.harnesses.defaults import build_decision_parts
        from rsicomposer.harnesses.decision.engine import ResearchControllerHarness
        from rsicomposer.harnesses.decision.public_policy import ResearchPolicyAdapter
        from rsicomposer.loops.decision_making.loop import DecisionMakingLoop
        parts = build_decision_parts(context, execution)
        owner = controller_wiring(context)
        harness = ResearchControllerHarness(local_tools=parts.tools, experience=owner.experience,
            memory=parts.memory, memory_limit=owner.memory_limit, research_agenda=parts.agenda,
            experience_limit=owner.experience_limit, context_compactor=IdentityContextCompactor())
        policy = DeepSeekResearchPolicy(config, harness, on_notification=on_notification)
        harness.add_cleanup(policy.close)
        return DecisionMakingLoop(ResearchPolicyAdapter(policy, harness), harness, parts.action_space,
                                  host_execution=owner.host_execution, lab_execution=execution, model=policy.model)
    return build
