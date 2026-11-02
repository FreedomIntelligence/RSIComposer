"""Harness-independent feedback learning lifecycle and adapter contract.

The service knows receipts, model identities, learning proposals and publication;
it never accesses a harness policy, state layout or context compactor.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Protocol

from rsisys.loops._memory import Memory
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.runtime.json_tools import encoded, result_id


@dataclass(frozen=True)
class FeedbackEvidence:
    history: tuple = ()
    evaluations: tuple = ()
    research_context: dict = field(default_factory=dict)
    grouping_context: dict | None = None


class FeedbackAdapter(Protocol):
    def model(self) -> ModelRef: ...
    def evidence(self, directive, feedback) -> FeedbackEvidence: ...
    def publish(self, memory: Memory) -> None: ...
    def decision_ids(self): ...
    def install(self, service): ...


class FeedbackLearningService:
    def __init__(self, learning, adapter: FeedbackAdapter):
        self.learning, self.adapter = learning, adapter

    def receive_feedback(self, directive, feedback):
        evidence = self.adapter.evidence(directive, feedback)
        if not isinstance(evidence, FeedbackEvidence):
            raise TypeError('Feedback adapter must return FeedbackEvidence')
        self.learning.capture(directive, feedback, self.adapter.model(), evidence.history,
            evaluations=evidence.evaluations, research_context=evidence.research_context,
            grouping_context=evidence.grouping_context)
        self.refresh()

    def refresh(self):
        self.learning.poll()
        self._publish()

    def attempts(self):
        return self.learning.attempts(self.adapter.decision_ids())

    @staticmethod
    def _memory(row):
        proposals = [p for p in row['proposals'] if p['kind'] == 'memory']
        interpretations = row.get('attempt_revisions', [])
        if not proposals and not interpretations:
            return None
        receipt = row.get('raw_input', row['input'])['receipt']
        feedback, directive = receipt['feedback'], receipt['directive']
        # These fields are copied from receipts, not rewritten by the model.
        facts = {key: feedback[key] for key in ('directive_id', 'status', 'quality_metrics',
            'training_metrics', 'execution_metrics', 'error', 'evidence_refs', 'adopted_candidate_ids')}
        facts.update(mode=directive['mode'], candidate_ids=[c['candidate_id'] for c in feedback['candidates']])
        return Memory('feedback-learning:' + row['job_id'], 'episodic',
            encoded({'receipt_facts': facts, 'interpretations': proposals, 'learned_by': row['model'],
                     'research_context': row['input'].get('research_context', {}),
                     'attempt_interpretations': interpretations,
                     'simulated': row['simulated'], 'authority': 'unverified_interpretations',
                     'read_operation': 'feedback_learning_read', 'job_id': row['job_id']}),
            tuple(row['input']['source_refs']),
            applicability=encoded({'objective': directive['objective'], 'experiment': directive['directive_id'],
                                   'scope': 'This experiment and its original model/protocol; not current global state'}),
            status='unverified')

    def _publish(self):
        for ident, row in list(self.learning.state['jobs'].items()):
            if row['status'] != 'completed' or row.get('published'):
                continue
            memory = self._memory(row)
            if memory is not None:
                self.adapter.publish(memory)
            state = deepcopy(self.learning.state)
            state['jobs'][ident].update(published=True,
                memory_digest=result_id(asdict(memory)) if memory is not None else None)
            # If persistence fails, retry publishing the same deterministic ID.
            self.learning._commit(state)

    def close(self, *, timeout=30):
        self.learning.close(timeout=timeout)
        self._publish()
