"""Private host orchestration of independently bounded product components.

This is not a FeedbackProcessor: only its child components receive the public
process(event, model) interface. Journal state and accounting stay in this host.
"""
import json
from copy import deepcopy
from pathlib import Path

from rsisys.harnesses.decision.feedback.evidence import admitted_candidate_evidence
from rsisys.harnesses.decision.feedback.proposal_validation import checked_processor_result, validate_proposals
from rsisys.harnesses.decision.feedback.stages import FeedbackStageBinding


class FeedbackPipeline:
    actual_exchanges_read_path = '/feedback_response/products'

    def __init__(self, *, memory_processor=None, harness_proposals=None, improvement_proposals=None,
                 proposal_requires_memory=False):
        if harness_proposals is not None and improvement_proposals is not None:
            raise ValueError('Select one improvement proposal stage')
        self.memory_processor, self.harness_proposals = memory_processor, harness_proposals
        self.improvement_proposals = improvement_proposals
        self.proposer = improvement_proposals if improvement_proposals is not None else harness_proposals
        self.proposal_product = 'improvement_proposals' if improvement_proposals is not None else 'harness_proposals'
        self.proposal_kinds = ('harness_change', 'policy_training') if improvement_proposals is not None else ('harness_change',)
        if type(proposal_requires_memory) is not bool:
            raise TypeError('proposal_requires_memory must be a bool')
        self.proposal_requires_memory = proposal_requires_memory
        self.memory_binding = FeedbackStageBinding(memory_processor)
        self.proposal_binding = FeedbackStageBinding(self.proposer)
        self.max_calls = self.memory_binding.max_calls + self.proposal_binding.max_calls

    def configuration(self):
        return {'provider': 'feedback_pipeline',
            'memory': self.memory_binding.configuration() if self.memory_processor is not None else None,
            'harness_proposals': self.proposal_binding.configuration() if self.harness_proposals is not None else None,
            'improvement_proposals': self.proposal_binding.configuration() if self.improvement_proposals is not None else None,
            'proposal_requires_memory': self.proposal_requires_memory,
            'trajectory_mode': 'archive_only', 'dataset_construction_owner': 'lab',
            'construction_owner': 'main_meta_agent_decision'}

    @staticmethod
    def _checked(result, job, allowance, kind):
        result = checked_processor_result(result, allowance)
        charged = result['charged_calls']
        if result.get('status') == 'completed' and kind != 'memory':
            try:
                output = result.get('output')
                if not isinstance(output, dict) or set(output) != {'proposals'}:
                    raise ValueError('Proposal product must contain only proposals')
                proposals = validate_proposals(json.dumps(output),
                    job['input']['source_refs'], candidate_evidence=admitted_candidate_evidence(job))
                kinds = (kind,) if isinstance(kind, str) else kind
                if any(p['kind'] not in kinds for p in proposals):
                    raise ValueError('Feedback stage produced a different product')
                if any((p['kind'] == 'harness_change' and p.get('subject') not in ('meta_agent', 'lab'))
                       or (p['kind'] == 'policy_training' and p.get('subject') != 'meta_agent') for p in proposals):
                    raise ValueError('Improvement proposals must name the proper target')
                result['proposals'] = proposals
            except (ValueError, TypeError, KeyError) as exc:
                return {'status': 'failed', 'error': str(exc), 'charged_calls': charged,
                        'proposals': [], 'response': result}
        return result

    def _run(self, binding, job, backend, limits, allowance, kind, product):
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
        return self._checked(result, job, allowance, kind)

    def _combine(self, memory, harness, *, simulated=False):
        products = {'context_memory': memory} if memory is not None else {}
        if harness is not None:
            products[self.proposal_product] = harness
        proposals = deepcopy(memory.get('proposals', [])) if memory is not None and memory['status'] == 'completed' else []
        if harness is not None and harness['status'] == 'completed':
            proposals.extend(deepcopy(harness.get('proposals', [])))
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
        memory = (self._run(self.memory_binding, job, backend, limits, allowance, 'memory', 'context_memory')
                  if self.memory_processor is not None else None)
        harness = None
        if self.proposer is not None:
            if self.proposal_requires_memory and (memory is None or memory['status'] != 'completed'):
                harness = {'status': 'not_started', 'error': 'Memory processing did not complete',
                           'charged_calls': 0, 'proposals': []}
            else:
                remaining = limits['max_calls'] - (memory['charged_calls'] if memory is not None else 0)
                allowance = min(self.proposal_binding.max_calls, remaining)
                harness = self._run(self.proposal_binding, job, backend, limits, allowance,
                                    self.proposal_kinds, self.proposal_product)
        return self._combine(memory, harness, simulated=job['simulated'])

    def recover_job(self, job, *, path=None):
        reserved = job['reserved_calls']
        allowance = min(self.memory_binding.max_calls, reserved)
        memory = (self._recover(self.memory_binding, job, allowance, 'memory',
                                path.with_name('context_memory.json') if path is not None else None)
                  if self.memory_processor is not None else None)
        harness = None
        if self.proposer is not None:
            if self.proposal_requires_memory and (memory is None or memory['status'] != 'completed'):
                harness = {'status': 'not_started', 'error': 'Memory processing did not complete',
                           'charged_calls': 0, 'proposals': []}
            else:
                allowance = min(self.proposal_binding.max_calls, reserved - (memory['charged_calls'] if memory is not None else 0))
                harness = self._recover(self.proposal_binding, job, allowance, self.proposal_kinds,
                    path.with_name(self.proposal_product + '.json') if path is not None else None)
        result = self._combine(memory, harness, simulated=job['simulated'])
        if any(p['status'] == 'interrupted' for p in result['products'].values()):
            result['charged_calls'] = reserved
            result['response']['accounting'] = 'unknown_stage_full_job_reservation_retained'
        return result

    def _recover(self, binding, job, allowance, kinds, path):
        if allowance == 0 and binding.max_calls > 0:
            return {'status': 'budget_exhausted', 'error': 'No host calls remain for this product',
                    'charged_calls': 0, 'proposals': []}
        try:
            result = binding.recover_job({**deepcopy(job), 'reserved_calls': allowance}, path=path)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        else:
            if result is not None:
                return self._checked(result, job, allowance, kinds)
            error = 'Product outcome unknown; no automatic replay'
        return {'status': 'interrupted', 'error': error, 'charged_calls': allowance, 'proposals': []}


def enable_feedback_pipeline(loop, *, context_memory, memory_processor=None,
                             harness_proposals=None, improvement_proposals=None, **options):
    """Explicit assembly: active memory, default extraction, optional proposals."""
    from rsisys.harnesses.decision.learning_harness import enable_feedback_learning
    processor = FeedbackPipeline(memory_processor=memory_processor, harness_proposals=harness_proposals,
                                 improvement_proposals=improvement_proposals)
    return enable_feedback_learning(loop, context_memory=context_memory, processor=processor, **options)
