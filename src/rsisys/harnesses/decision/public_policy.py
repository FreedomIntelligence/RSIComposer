"""Adapt the default harness view to the public decision input window."""
import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict

from rsisys.loops.contracts.decision import DecisionInput


class DecisionInputAdapter:
    """The rich DecisionContext stays here, outside third-party policy contracts.

    context and references are optional rendering callbacks over the default
    harness view. The default text retains its observed facts without requiring
    consuming policies to understand Memory, HistoryEntry or research_state.
    """
    def __init__(self, *, context=None, references=None):
        if ((context is not None and not callable(context))
                or (references is not None and not callable(references))):
            raise TypeError('Decision input renderers must be callable')
        self.context, self.references = context, references

    def __call__(self, view):
        return self._build(view)


    def _build(self, view):
        capabilities = tuple({'kind': 'invoke', **asdict(spec)} for spec in view.operations)
        if self.context is None:
            data = asdict(view)
            for field in ('objective', 'operations', 'directive_modes', 'review_actions'):
                data.pop(field, None)
            text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        else:
            text = self.context(view)
        return DecisionInput(view.objective, text, capabilities,
            self.references(view) if self.references is not None else (), feedbacks=view.feedbacks,
            pending=view.pending, progresses=view.progresses, budget=view.run_control.get('budget', {}))



class ResearchPolicyAdapter:
    """Drive a native research policy through the public Controller input port."""
    def __init__(self, implementation, harness):
        self.implementation, self.harness = implementation, harness

    @property
    def model(self):
        return self.implementation.model

    @property
    def backend(self):
        return self.implementation.backend

    def share_backend(self, wrap):
        share = getattr(self.implementation, 'share_backend', None)
        if not callable(share):
            raise ValueError('Session/custom policies require an explicit feedback backend')
        return share(wrap)

    @property
    def asynchronous_usage(self):
        return getattr(self.implementation, 'asynchronous_usage', False)

    def can_decide(self):
        ready = getattr(self.implementation, 'can_decide', None)
        return ready() if callable(ready) else True

    def decide(self, window):
        from contextlib import ExitStack
        if not isinstance(window, DecisionInput):
            raise TypeError('ResearchPolicyAdapter requires DecisionInput')
        context = deepcopy(self.harness.prepared_context)
        with ExitStack() as stack:
            owner = getattr(self.implementation, 'context_scope', None)
            if callable(owner):
                stack.enter_context(owner(self.harness.context_compactor, recent_limit=self.harness.memory_limit))
            else:
                context = self.harness.context_compactor.compact(context, recent_limit=self.harness.memory_limit)
            return self.implementation.decide(context)

    @contextmanager
    def track_usage(self, record):
        method = getattr(self.implementation, 'track_usage', None)
        if callable(method):
            with method(record):
                yield
        else:
            yield

    def decision_metadata(self, ident):
        from rsisys.host.ledger import DecisionResearchMetadata
        read = getattr(self.implementation, 'decision_metadata', None)
        return read(ident) if callable(read) else DecisionResearchMetadata()

    def drain_usage(self):
        drain = getattr(self.implementation, 'drain_usage', None)
        return drain() if callable(drain) else ()

    def pending_research_job(self):
        read = getattr(self.implementation, 'pending_research_job', None)
        return read() if callable(read) else None

    def drain_research(self):
        drain = getattr(self.implementation, 'drain_research', None)
        return drain() if callable(drain) else None

    def cancel_research(self):
        cancel = getattr(self.implementation, 'cancel_research', None)
        return cancel() if callable(cancel) else False

    def update_model(self, model):
        return self.implementation.update_model(model)

    def state(self):
        return self.implementation.state()

    def restore(self, state):
        return self.implementation.restore(state)

    def close(self):
        close = getattr(self.implementation, 'close', None)
        if callable(close):
            close()
