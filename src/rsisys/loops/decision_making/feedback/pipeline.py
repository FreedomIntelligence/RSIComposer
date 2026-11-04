"""Compose feedback products without giving feedback execution authority.

Memory processing is required (the journal organizer needs no inference).
Harness proposals are optional and independently observable. Trajectory
extraction remains a journal product, never a generated training target.
"""
from copy import deepcopy
import json

from .components import JournalContextMemory
from .learning import validate_proposals, checked_processor_result
from .evidence import admitted_candidate_evidence


class FeedbackPipeline:
    actual_exchanges_read_path = '/feedback_response/products'

    def __init__(self, *, memory_processor=None, harness_proposals=None):
        self.memory_processor, self.harness_proposals = memory_processor, harness_proposals
        for component in (memory_processor, harness_proposals):
            if component is None:
                continue
            if any(not callable(getattr(component, method, None))
                   for method in ('configuration', 'process', 'recover')):
                raise TypeError('Feedback stages require configuration/process/recover')
            if type(component.max_calls_per_job) is not int or component.max_calls_per_job < 1:
                raise ValueError('Feedback stages require a positive call reservation')
        self.max_calls_per_job = max(1, sum(c.max_calls_per_job for c in
            (memory_processor, harness_proposals) if c is not None))

    def configuration(self):
        return {'provider': 'feedback_pipeline',
            'memory': self.memory_processor.configuration() if self.memory_processor else {'provider': 'journal'},
            'harness_proposals': self.harness_proposals.configuration() if self.harness_proposals else None,
            'trajectory_mode': 'extraction_only',
            'construction_owner': 'main_meta_agent_decision'}

    @staticmethod
    def _journal_memory():
        return {'status': 'completed', 'error': None, 'charged_calls': 0,
                'proposals': [], 'attempt_updates': [], 'response': {'provider': 'journal'}}

    @staticmethod
    def _checked(result, job, allowance, kind):
        result = checked_processor_result(result, allowance)
        charged = result['charged_calls']
        if result.get('status') == 'completed':
            try:
                proposals = validate_proposals(json.dumps({'proposals': result.get('proposals', [])}),
                    job['input']['source_refs'], candidate_evidence=admitted_candidate_evidence(job))
                if any(p['kind'] != kind for p in proposals):
                    raise ValueError('Feedback stage produced a different product')
                if kind == 'harness_change' and any(p.get('subject') not in ('meta_agent', 'lab') for p in proposals):
                    raise ValueError('Harness proposals must name the meta_agent or lab target')
            except (ValueError, TypeError, KeyError) as exc:
                return {'status': 'failed', 'error': str(exc), 'charged_calls': charged,
                        'proposals': [], 'response': result}
        return result

    def _run(self, component, job, backend, limits, allowance, kind):
        if allowance == 0:
            return {'status': 'budget_exhausted', 'error': 'No host calls remain for this product',
                    'charged_calls': 0, 'proposals': []}
        try:
            result = component.process(deepcopy(job), backend, {**limits, 'max_calls': allowance})
        except Exception as exc:
            result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                      'charged_calls': allowance, 'proposals': []}
        return self._checked(result, job, allowance, kind)

    def _combine(self, memory, harness, *, simulated=False):
        products = {'context_memory': memory}
        if harness is not None:
            products['harness_proposals'] = harness
        proposals = deepcopy(memory.get('proposals', [])) if memory['status'] == 'completed' else []
        if harness is not None and harness['status'] == 'completed':
            if len(proposals) + len(harness.get('proposals', [])) > 6:
                harness = {**harness, 'status': 'failed', 'error': 'Combined proposal limit exceeded'}
                products['harness_proposals'] = harness
            else:
                proposals.extend(deepcopy(harness.get('proposals', [])))
        return {'status': memory['status'], 'error': memory.get('error'),
            'proposals': proposals, 'attempt_updates': deepcopy(memory.get('attempt_updates', [])),
            'charged_calls': sum(p['charged_calls'] for p in products.values()),
            'simulated': simulated or any(p.get('simulated', False) for p in products.values()),
            'products': {name: {key: p.get(key) for key in ('status', 'error', 'charged_calls')}
                         for name, p in products.items()},
            'response': {'provider': 'feedback_pipeline', 'products': products}}

    def process(self, job, backend, limits):
        allowance = min(self.memory_processor.max_calls_per_job, limits['max_calls']) if self.memory_processor else 0
        memory = (self._run(self.memory_processor, job, backend, limits, allowance, 'memory')
                  if self.memory_processor else self._journal_memory())
        harness = None
        if self.harness_proposals is not None:
            if memory['status'] != 'completed':
                harness = {'status': 'not_started', 'error': 'Memory processing did not complete',
                           'charged_calls': 0, 'proposals': []}
            else:
                remaining = limits['max_calls'] - memory['charged_calls']
                allowance = min(self.harness_proposals.max_calls_per_job, remaining)
                harness = self._run(self.harness_proposals, job, backend, limits, allowance, 'harness_change')
        return self._combine(memory, harness, simulated=job['simulated'])

    def recover(self, job):
        memory = self.memory_processor.recover(deepcopy(job)) if self.memory_processor else self._journal_memory()
        if memory is None:
            return None
        reserved = job['reserved_calls']
        allowance = min(self.memory_processor.max_calls_per_job, reserved) if self.memory_processor else 0
        memory = self._checked(memory, job, allowance, 'memory')
        harness = None
        if self.harness_proposals is not None:
            if memory['status'] != 'completed':
                harness = {'status': 'not_started', 'error': 'Memory processing did not complete',
                           'charged_calls': 0, 'proposals': []}
            else:
                allowance = min(self.harness_proposals.max_calls_per_job, reserved - memory['charged_calls'])
                recovery_error = None
                try:
                    harness = self.harness_proposals.recover(deepcopy(job)) if allowance else {
                        'status': 'budget_exhausted', 'error': 'No host calls remain for this product',
                        'charged_calls': 0, 'proposals': []}
                except Exception as exc:
                    harness, recovery_error = None, f'{type(exc).__name__}: {exc}'
                if harness is None:
                    # Memory can already be committed when the owner crashes.
                    # Preserve it; never replay an uncertain optional inference.
                    harness = {'status': 'interrupted', 'error': recovery_error or
                               'Proposal outcome unknown; no automatic replay',
                               'charged_calls': reserved - memory['charged_calls'], 'proposals': []}
                else:
                    harness = self._checked(harness, job, allowance, 'harness_change')
        return self._combine(memory, harness, simulated=job['simulated'])


_DEFAULT_MEMORY = object()


def enable_feedback_pipeline(loop, *, context_memory=_DEFAULT_MEMORY, memory_processor=None,
                             harness_proposals=None, **options):
    """Explicit assembly: active memory, default extraction, optional proposals."""
    from ..harness.learning_harness import enable_feedback_learning
    if context_memory is _DEFAULT_MEMORY:
        context_memory = JournalContextMemory()
    processor = FeedbackPipeline(memory_processor=memory_processor, harness_proposals=harness_proposals)
    return enable_feedback_learning(loop, context_memory=context_memory, processor=processor, **options)
