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
from .learning import feedback_text_preview, validate_publication


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
        evidence = validate_publication(row)
        proposals = [p for p in row['proposals'] if p['kind'] == 'memory']
        interpretations = row.get('attempt_revisions', [])
        if not proposals and not interpretations:
            return None
        receipt = row.get('raw_input', row['input'])['receipt']
        feedback, directive = receipt['feedback'], receipt['directive']
        # Facts come from raw receipts; only error prose uses the same bounded,
        # source-linked preview as learning admission, never a generated summary.
        facts = {key: feedback[key] for key in ('directive_id', 'status', 'quality_metrics',
            'training_metrics', 'execution_metrics', 'error', 'evidence_refs', 'adopted_candidate_ids')}
        facts['error'], omission = feedback_text_preview(feedback['error'], job_id=row['job_id'],
                                                        evidence_path='/receipt/feedback/error')
        facts.update(mode=directive['mode'], candidate_ids=[c['candidate_id'] for c in feedback['candidates']])
        content = {'receipt_facts': facts, 'interpretations': proposals, 'learned_by': row['model'],
                   'research_context': row['input'].get('research_context', {}),
                   'attempt_interpretations': interpretations,
                   'simulated': row['simulated'], 'authority': 'unverified_interpretations',
                   'read_operation': 'feedback_learning_read', 'job_id': row['job_id']}
        if evidence is not None:
            content['candidate_evidence'] = evidence
            content['evaluation_validation'] = 'structured_attribution_only; prose_remains_unverified'
        if omission:
            content['receipt_text_omissions'] = [omission]
        return Memory('feedback-learning:' + row['job_id'], 'episodic',
            encoded(content),
            tuple(row['input']['source_refs']),
            applicability=encoded({'objective': directive['objective'], 'experiment': directive['directive_id'],
                                   'scope': 'This experiment and its original model/protocol; not current global state'}),
            status='unverified')

    def _publish(self):
        for ident, row in list(self.learning.state['jobs'].items()):
            if row['status'] != 'completed' or row.get('published'):
                continue
            try:
                memory = self._memory(row)
            except (ValueError, TypeError, KeyError) as exc:
                # Preserve the original response for audit. Rejected prose must
                # not leak through proposal/attempt views while publication is
                # blocked, including restored legacy jobs.
                state = deepcopy(self.learning.state)
                state['jobs'][ident].update(status='failed', error='Publication validation failed: ' + str(exc),
                    publication_error=str(exc), proposals=[], attempt_updates=[], attempt_revisions=[])
                self.learning._commit(state)
                continue
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
