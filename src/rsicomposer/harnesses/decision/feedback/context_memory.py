"""Context provider adapters for ControllerMemory implementations."""
from copy import deepcopy

from rsicomposer.loops.contracts.feedback_context import ContextQuery, EvidenceEvent
from rsicomposer.loops.contracts.receipts import receipt_task


def _callable_identity(function):
    return {'module': getattr(function, '__module__', type(function).__module__),
            'name': getattr(function, '__qualname__', type(function).__qualname__)}


class CallbackContextMemory:
    """Adapt a function/SDK without requiring journal or publication methods.

    on_feedback is optional and should return promptly (it can enqueue its own
    work). Model workflows can retain the existing accounted processor/gateway;
    supplying a context renderer does not grant additional inference allowance.
    """
    def __init__(self, render, *, on_feedback=None, configuration=None):
        if not callable(render) or (on_feedback is not None and not callable(on_feedback)):
            raise TypeError('Context render and optional on_feedback must be callable')
        self.render, self.observe = render, on_feedback
        self._configuration = deepcopy(configuration or {})

    def context(self, query: ContextQuery) -> str:
        return self.render(query)

    def configuration(self):
        return {'provider': 'callback', 'render': _callable_identity(self.render),
                'observe': _callable_identity(self.observe) if self.observe else None,
                'settings': deepcopy(self._configuration)}


def validate_context_memory(provider):
    if callable(getattr(provider, 'context', None)):
        return
    raise TypeError('Research requires an active context/memory provider: context(query) -> str')


def feedback_context_event(row):
    """Translate a captured receipt once; never expose the private worker row."""
    raw = row['raw_input']
    receipt = raw['receipt']
    task = receipt_task(receipt)
    result = receipt['result']
    native = task.get('source') == 'lab'
    identity = deepcopy(receipt.get('identity', {}))
    if receipt.get('feedback_delivery') is not None:
        identity['feedback_delivery'] = deepcopy(receipt['feedback_delivery'])
    for name in ('decision_id', 'issuing_model'):
        if task.get(name) is not None:
            identity.setdefault(name, deepcopy(task[name]))
    scope = raw.get('research_context', {})
    from rsicomposer.harnesses.decision.feedback.evidence import admitted_candidate_evidence
    packet = deepcopy(row['input'])
    evidence = admitted_candidate_evidence(row)
    identity['feedback_model'] = deepcopy(row['model'])
    identity['receipt_digest'] = row['receipt_digest']
    if scope.get('controller_version') is not None:
        identity['controller_version'] = scope['controller_version']
    for name, key in (('decision_id', 'decision_id'), ('issuing_model', 'issuing_decision_model')):
        if identity.get(name) is None and scope.get(key) is not None:
            identity[name] = deepcopy(scope[key])
    return EvidenceEvent(
        event_id=row['job_id'], kind='receipt', decision_id=identity.get('decision_id'),
        task_id=task['task_id'],
        status=result['status'], output=deepcopy(result.get('output') if native else result),
        request=deepcopy(receipt['operation'].get('input') if native else
                         receipt['operation']),
        references=deepcopy(result.get('references', ()) if native else
                            result.get('artifacts', ())),
        source='lab' if native else 'operation',
        identity=identity, evidence_packet=packet,
        source_refs=tuple(packet['source_refs']), candidate_evidence=deepcopy(evidence))


class ContextMemoryBinding:
    """Internal bridge from the feedback journal to a ControllerMemory provider."""
    def __init__(self, provider):
        validate_context_memory(provider)
        self.provider = provider

    @property
    def observer(self):
        observer = getattr(self.provider, 'observe', None)
        return observer if callable(observer) else None

    def context(self, query):
        value = self.provider.context(deepcopy(query))
        if not isinstance(value, str):
            raise TypeError('Context/memory provider must return context text (str)')
        return value

    def record(self, row):
        return {'event_id': row['job_id'],
                'notification': deepcopy(row.get('context_delivery', {'status': 'not_requested'})),
                'scope': 'feedback_notification; current context is rendered for each decision'}
