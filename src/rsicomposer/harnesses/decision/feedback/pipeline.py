"""Private host orchestration of independently bounded product components.

This is not a FeedbackProcessor: only its child components receive the public
process(event, model) interface. Journal state and accounting stay in this host.
"""
from copy import deepcopy
from pathlib import Path

from rsicomposer.harnesses.decision.feedback.evidence import admitted_candidate_evidence
from rsicomposer.harnesses.decision.feedback.proposal_validation import (
    bind_targets, checked_processor_result, validate_proposals,
)
from rsicomposer.harnesses.decision.feedback.stages import FeedbackStageBinding


class FeedbackPipeline:
    actual_exchanges_read_path = '/feedback_response/products'

    def __init__(self, *, memory_processor=None, proposer=None, proposal_requires_memory=False):
        self.memory_processor, self.proposer = memory_processor, proposer
        self.proposal_targets = bind_targets(getattr(proposer, 'proposal_targets', None)) if proposer is not None else {}
        if type(proposal_requires_memory) is not bool:
            raise TypeError('proposal_requires_memory must be a bool')
        self.proposal_requires_memory = proposal_requires_memory
        self.memory_binding = FeedbackStageBinding(memory_processor)
        self.proposal_binding = FeedbackStageBinding(self.proposer)
        self.max_calls = self.memory_binding.max_calls + self.proposal_binding.max_calls

    def configuration(self):
        return {'provider': 'feedback_pipeline',
            'memory': self.memory_binding.configuration() if self.memory_processor is not None else None,
            'proposer': self.proposal_binding.configuration() if self.proposer is not None else None,
            'proposal_targets': [rule.configuration() for rule in self.proposal_targets.values()],
            'proposal_requires_memory': self.proposal_requires_memory,
            'trajectory_mode': 'archive_only', 'dataset_construction_owner': 'lab',
            'construction_owner': 'main_meta_agent_decision'}

    def _checked(self, result, job, allowance, binding):
        result = checked_processor_result(result, allowance)
        charged = result['charged_calls']
        if result.get('status') == 'completed' and binding is self.proposal_binding:
            try:
                result['proposals'] = validate_proposals(result.get('output'), job['input']['source_refs'],
                    targets=self.proposal_targets, candidate_evidence=admitted_candidate_evidence(job))
            except (ValueError, TypeError, KeyError) as exc:
                return {'status': 'failed', 'error': str(exc), 'charged_calls': charged,
                        'proposals': [], 'response': result}
        return result

    def _run(self, binding, job, backend, limits, allowance, product):
        if allowance == 0 and binding.max_calls > 0:
            return {'status': 'budget_exhausted', 'error': 'No host calls remain for this product',
                    'charged_calls': 0, 'proposals': []}
        try:
            directory = limits.get('exchange_directory')
            path = Path(directory) / (product + '.json') if directory else None
            result = binding.run(job, backend, {**limits, 'max_calls': allowance}, path=path)
        except Exception as exc:
            result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                      'charged_calls': allowance, 'proposals': []}
        return self._checked(result, job, allowance, binding)

    def _combine(self, memory, proposal, *, simulated=False):
        products = {'context_memory': memory} if memory is not None else {}
        if proposal is not None:
            products['proposals'] = proposal
        proposals = deepcopy(proposal['proposals']) if proposal is not None and proposal['status'] == 'completed' else []
        completed = not products or any(p['status'] == 'completed' for p in products.values())
        failed = next((p for p in products.values() if p['status'] != 'completed'), {})
        return {'status': 'completed' if completed else failed['status'], 'error': failed.get('error'),
            'proposals': proposals, 'attempt_updates': deepcopy(memory.get('attempt_updates', [])) if memory is not None else [],
            'outputs': {name: deepcopy(p['output']) for name, p in products.items()
                        if p['status'] == 'completed' and 'output' in p},
            'charged_calls': sum(p['charged_calls'] for p in products.values()),
            'simulated': simulated or any(p.get('simulated', False) for p in products.values()),
            'products': {name: {key: p.get(key) for key in ('status', 'error', 'charged_calls')}
                         for name, p in products.items()},
            'response': {'provider': 'feedback_pipeline', 'products': products}}

    def run(self, job, backend, limits, *, path=None):
        allowance = min(self.memory_binding.max_calls, limits['max_calls'])
        memory = (self._run(self.memory_binding, job, backend, limits, allowance, 'context_memory')
                  if self.memory_processor is not None else None)
        proposal = None
        if self.proposer is not None:
            if self.proposal_requires_memory and (memory is None or memory['status'] != 'completed'):
                proposal = {'status': 'not_started', 'error': 'Memory processing did not complete',
                           'charged_calls': 0, 'proposals': []}
            else:
                remaining = limits['max_calls'] - (memory['charged_calls'] if memory is not None else 0)
                allowance = min(self.proposal_binding.max_calls, remaining)
                proposal = self._run(self.proposal_binding, job, backend, limits, allowance, 'proposals')
        return self._combine(memory, proposal, simulated=job['simulated'])

    def recover_job(self, job, *, path=None):
        reserved = job['reserved_calls']
        allowance = min(self.memory_binding.max_calls, reserved)
        memory = (self._recover(self.memory_binding, job, allowance,
                                path.with_name('context_memory.json') if path is not None else None)
                  if self.memory_processor is not None else None)
        proposal = None
        if self.proposer is not None:
            if self.proposal_requires_memory and (memory is None or memory['status'] != 'completed'):
                proposal = {'status': 'not_started', 'error': 'Memory processing did not complete',
                           'charged_calls': 0, 'proposals': []}
            else:
                allowance = min(self.proposal_binding.max_calls, reserved - (memory['charged_calls'] if memory is not None else 0))
                proposal = self._recover(self.proposal_binding, job, allowance,
                    path.with_name('proposals.json') if path is not None else None)
        result = self._combine(memory, proposal, simulated=job['simulated'])
        if any(p['status'] == 'interrupted' for p in result['products'].values()):
            result['charged_calls'] = reserved
            result['response']['accounting'] = 'unknown_stage_full_job_reservation_retained'
        return result

    def _recover(self, binding, job, allowance, path):
        if allowance == 0 and binding.max_calls > 0:
            return {'status': 'budget_exhausted', 'error': 'No host calls remain for this product',
                    'charged_calls': 0, 'proposals': []}
        try:
            result = binding.recover_job({**deepcopy(job), 'reserved_calls': allowance}, path=path)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        else:
            if result is not None:
                return self._checked(result, job, allowance, binding)
            error = 'Product outcome unknown; no automatic replay'
        return {'status': 'interrupted', 'error': error, 'charged_calls': allowance, 'proposals': []}


def enable_feedback_pipeline(loop, *, context_memory, memory_processor=None, proposer=None, **options):
    """Explicit assembly: active memory, default extraction, optional proposals."""
    from rsicomposer.harnesses.decision.learning_harness import enable_feedback_learning
    processor = FeedbackPipeline(memory_processor=memory_processor, proposer=proposer)
    return enable_feedback_learning(loop, context_memory=context_memory, processor=processor, **options)
