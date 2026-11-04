"""Context provider adapters; journal compatibility stays on this side only."""
from copy import deepcopy

from rsisys.loops.contracts.feedback_context import ContextQuery, FeedbackContextEvent
from rsisys.loops.contracts.receipts import receipt_task
from rsisys.runtime.json_tools import encoded


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
    if callable(getattr(provider, 'context', None)) or callable(provider):
        return
    # Compatibility is isolated here, not the public provider protocol.
    if all(callable(getattr(provider, method, None))
           for method in ('record', 'working_memory', 'memory')):
        return
    raise TypeError('Research requires an active context/memory provider: context(query) -> str')


def feedback_context_event(row):
    """Translate a captured receipt once; never expose the private worker row."""
    raw = row.get('raw_input', row['input'])
    receipt = raw['receipt']
    task = receipt_task(receipt)
    result = receipt.get('feedback', receipt.get('result'))
    native = task.get('source') == 'lab'
    identity = deepcopy(receipt.get('identity', {}))
    if receipt.get('feedback_delivery') is not None:
        identity['feedback_delivery'] = deepcopy(receipt['feedback_delivery'])
    for name in ('decision_id', 'issuing_model'):
        if task.get(name) is not None:
            identity.setdefault(name, deepcopy(task[name]))
    scope = raw.get('research_context', {})
    for name, key in (('decision_id', 'decision_id'), ('issuing_model', 'issuing_decision_model')):
        if identity.get(name) is None and scope.get(key) is not None:
            identity[name] = deepcopy(scope[key])
    return FeedbackContextEvent(
        event_id=row['job_id'], task_id=task.get('task_id') or task['directive_id'],
        status=result['status'], output=deepcopy(result.get('output') if native else result),
        request=deepcopy(receipt['operation'].get('input') if native else
                         receipt.get('directive', receipt.get('operation'))),
        references=deepcopy(result.get('references', ()) if native else
                            result.get('artifact_refs', result.get('artifacts', ()))),
        source='lab' if native else ('exploration' if 'directive' in receipt else 'operation'),
        identity=identity)


class ContextMemoryBinding:
    """Internal bridge. Only legacy providers receive journal-shaped arguments."""
    def __init__(self, provider):
        validate_context_memory(provider)
        self.provider = provider
        self.legacy = not (callable(getattr(provider, 'context', None)) or callable(provider))

    @property
    def observer(self):
        observer = None if self.legacy else getattr(self.provider, 'observe', None)
        return observer if callable(observer) else None

    def context(self, query, *, jobs=(), attempts=(), active_decision_ids=(), policy=None):
        if self.legacy:
            value = encoded(self.working_memory(jobs, attempts,
                active_decision_ids=active_decision_ids, policy=policy))
        else:
            render = getattr(self.provider, 'context', None)
            value = (render if callable(render) else self.provider)(deepcopy(query))
        if not isinstance(value, str):
            raise TypeError('Context/memory provider must return context text (str)')
        return value

    def record(self, row):
        if self.legacy:
            value = self.provider.record(deepcopy(row))
            if not isinstance(value, dict):
                raise TypeError('Legacy context/memory record must be a JSON object')
            return deepcopy(value)
        return {'event_id': row['job_id'],
                'notification': deepcopy(row.get('context_delivery', {'status': 'not_requested'})),
                'scope': 'feedback_notification; current context is rendered for each decision'}

    def working_memory(self, jobs, attempts, *, active_decision_ids=(), policy=None):
        value = self.provider.working_memory(deepcopy(jobs), deepcopy(attempts),
            active_decision_ids=tuple(active_decision_ids), policy=policy)
        if not isinstance(value, dict):
            raise TypeError('Legacy context/memory view must be a JSON object')
        return deepcopy(value)

    def publication(self, row):
        return self.provider.memory(deepcopy(row)) if self.legacy else None
