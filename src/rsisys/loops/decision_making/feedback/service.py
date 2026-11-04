"""Harness-independent feedback learning lifecycle and adapter contract.

The service knows receipts, model identities, learning proposals and publication;
it never accesses a harness policy, state layout or context compactor.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Protocol

from rsisys.loops._memory import Memory
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.receipts import ExecutionReceipt
from rsisys.runtime.json_tools import result_id
from .learning import validate_publication
from .components import journal_memory
from .knowledge import proposal_records


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
        return self.observe(ExecutionReceipt.from_feedback(directive, feedback), evidence=evidence)

    def observe(self, receipt: ExecutionReceipt, *, evidence=None):
        """Common observation entry; any harness/Lab may supply its own evidence."""
        if not isinstance(receipt, ExecutionReceipt):
            raise TypeError('Expected ExecutionReceipt')
        if evidence is None:
            provider = getattr(self.adapter, 'receipt_evidence', None)
            evidence = provider(receipt) if callable(provider) else FeedbackEvidence()
        if not isinstance(evidence, FeedbackEvidence):
            raise TypeError('Feedback adapter must return FeedbackEvidence')
        ident = self.learning.capture_receipt(receipt, self.adapter.model(), evidence.history,
            evaluations=evidence.evaluations, research_context=evidence.research_context,
            grouping_context=evidence.grouping_context)
        self.refresh()
        return ident

    def refresh(self):
        self.learning.poll()
        self._publish()

    def attempts(self):
        return self.learning.attempts(self.adapter.decision_ids())

    def working_memory(self, *, active_decision_ids=(), policy=None):
        """Harness-neutral research knowledge; publication is not execution."""
        return self.learning.working_memory(active_decision_ids=active_decision_ids, policy=policy)

    def trajectories(self, *, research_attempt_id=None):
        """Captured public trajectory segments, before any training-data recipe."""
        return self.learning.trajectories(research_attempt_id=research_attempt_id)

    def extract_trajectory(self, job_id):
        return self.learning.extract_trajectory(job_id)

    def proposals(self, *, target='harness'):
        records = proposal_records(self.learning.ordered_jobs(), target=target)
        provider = getattr(self.adapter, 'proposal_decisions', None)
        decisions = provider() if callable(provider) else ()
        for record in records:
            record['decision_links'] = [deepcopy(d) for d in decisions
                if record['proposal_id'] in d.get('feedback_proposal_ids', ())]
        return records

    def resolve_harness_proposals(self, proposal_ids):
        records = {r['proposal_id']: r for r in self.proposals()}
        resolved = []
        for ident in proposal_ids:
            if ident not in records or records[ident]['subject'] not in ('meta_agent', 'lab'):
                raise ValueError('Unknown or untargeted harness proposal: ' + ident)
            row = records[ident]
            resolved.append({key: deepcopy(row[key]) for key in
                ('proposal_id', 'job_id', 'subject', 'source_snapshot')})
            resolved[-1]['proposal_digest'] = result_id(row['proposal'])
        return tuple(resolved)

    # Compatibility helper; live publication uses the configured organizer.
    _memory = staticmethod(journal_memory)

    def _publish(self):
        for ident, row in list(self.learning.state['jobs'].items()):
            if row['status'] != 'completed' or row.get('published'):
                continue
            try:
                validate_publication(row)
            except (ValueError, TypeError, KeyError) as exc:
                # Preserve the original response for audit. Rejected prose must
                # not leak through proposal/attempt views while publication is
                # blocked, including restored legacy jobs.
                state = deepcopy(self.learning.state)
                state['jobs'][ident].update(status='failed', error='Publication validation failed: ' + str(exc),
                    publication_error=str(exc), proposals=[], attempt_updates=[], attempt_revisions=[])
                self.learning._commit(state)
                continue
            # Validate before invoking any organizer. Provider/delivery errors
            # remain retryable and never repeat the underlying execution/model.
            memory = self.learning.context_memory.memory(deepcopy(row))
            if memory is not None:
                if not isinstance(memory, Memory):
                    raise TypeError('Context/memory publication must be a Memory or None')
                self.adapter.publish(memory)
            state = deepcopy(self.learning.state)
            state['jobs'][ident].update(published=True,
                memory_digest=result_id(asdict(memory)) if memory is not None else None)
            # If persistence fails, retry publishing the same deterministic ID.
            self.learning._commit(state)

    def close(self, *, timeout=30):
        self.learning.close(timeout=timeout)
        self._publish()
