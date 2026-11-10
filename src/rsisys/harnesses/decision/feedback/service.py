"""Harness-independent feedback learning lifecycle and adapter contract.

The service knows receipts, model identities, learning proposals and publication;
it never accesses a harness policy, state layout or context compactor.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Protocol

from rsisys.harnesses.decision.feedback.components import journal_memory
from rsisys.harnesses.decision.feedback.knowledge import proposal_records
from rsisys.harnesses.decision.feedback.proposal_validation import validate_publication
from rsisys.harnesses.memory import Memory
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.receipts import ExecutionReceipt
from rsisys.runtime.storage.json_tools import result_id


@dataclass(frozen=True)
class FeedbackEvidence:
    history: tuple = ()
    evaluations: tuple = ()
    research_context: dict = field(default_factory=dict)
    grouping_context: dict | None = None


class FeedbackAdapter(Protocol):
    """Minimal host integration; legacy evidence/publication hooks are optional."""
    def model(self) -> ModelRef: ...
    def install(self, service): ...


class FeedbackLearningService:
    def __init__(self, learning, adapter: FeedbackAdapter):
        self.learning, self.adapter = learning, adapter

    def receive_feedback(self, directive, feedback):
        provider = getattr(self.adapter, 'evidence', None)
        evidence = provider(directive, feedback) if callable(provider) else FeedbackEvidence()
        return self.observe(ExecutionReceipt.from_feedback(directive, feedback), evidence=evidence)

    def observe_output(self, output=None, *, task_id, status, evidence=None, **envelope):
        """Accept native Lab output; the caller supplies executor metadata only.

        Related evidence is optional. No default-harness history, candidate
        catalog, Directive or Feedback conversion is required for a new Lab.
        """
        receipt = ExecutionReceipt.from_output(task_id, output, status=status, **envelope)
        return self.observe(receipt, evidence=FeedbackEvidence() if evidence is None else evidence)

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
        return ident

    def archive(self, receipt: ExecutionReceipt, *, evidence=None):
        """Durably capture a public receipt without scheduling interpretation."""
        if not isinstance(receipt, ExecutionReceipt):
            raise TypeError('Expected ExecutionReceipt')
        if self.learning.trajectory_archive is None:
            return None
        evidence = FeedbackEvidence() if evidence is None else evidence
        if not isinstance(evidence, FeedbackEvidence):
            raise TypeError('Feedback adapter must return FeedbackEvidence')
        return self.learning.capture_receipt(receipt, self.adapter.model(), evidence.history,
            evaluations=evidence.evaluations, research_context=evidence.research_context,
            grouping_context=evidence.grouping_context, archive_only=True)

    def refresh(self):
        # No waiting: collect already-finished work, including inline executors.
        # Receipt delivery itself never invokes model processing/publication.
        self.learning.notify_context()
        self.learning.notify_archive()
        self.learning.poll()
        self.learning.poll(dispatch=False)
        self._publish()

    def attempts(self):
        provider = getattr(self.adapter, 'decision_ids', None)
        return self.learning.attempts(provider() if callable(provider) else ())

    def context(self, query=None, **options):
        """Text context for the next decision, without a publication object."""
        return self.learning.context(query, **options)

    def working_memory(self, *, active_decision_ids=(), policy=None):
        """Harness-neutral research knowledge; publication is not execution."""
        return self.learning.working_memory(active_decision_ids=active_decision_ids, policy=policy)

    def trajectories(self, *, research_attempt_id=None):
        """Captured public trajectory segments, before any training-data recipe."""
        return self.learning.trajectories(research_attempt_id=research_attempt_id)

    def extract_trajectory(self, job_id):
        return self.learning.extract_trajectory(job_id)

    def trajectory_snapshot(self, job_id):
        """Immutable raw capture for Lab-side dataset construction."""
        return self.learning.trajectory_snapshot(job_id)

    def outputs(self, *, product=None, job_id=None):
        """Native products with host provenance; no business schema conversion."""
        records = []
        for row in self.learning.ordered_jobs():
            if job_id is not None and row['job_id'] != job_id:
                continue
            values = row.get('outputs', {})
            if 'output' in row:
                values = {**values, 'feedback': row['output']}
            for name, value in values.items():
                if product is not None and name != product:
                    continue
                path = '/output' if name == 'feedback' and 'output' in row else '/outputs/' + name
                records.append({'event_id': row['job_id'], 'product': name,
                    'output': deepcopy(value), 'authority': 'unverified_product', 'applied': False,
                    'read_path': {'operation': 'feedback_learning_read', 'job_id': row['job_id'],
                                  'evidence_path': path}})
        return records

    def retry_publication(self, job_id):
        """Explicit delivery retry; caller ensures its destination is idempotent.

        This never repeats the underlying Lab action or inference. Failed and
        interrupted external side effects are not automatically replayed.
        """
        self.learning._assert_owner()
        if self.learning._closed:
            raise RuntimeError('Feedback learning is closed')
        state = deepcopy(self.learning.state)
        row = state['jobs'][job_id]
        if row['status'] != 'completed' or row.get('published'):
            raise ValueError('Only a completed, unpublished product can be retried')
        row.pop('publication_delivery', None)
        self.learning._commit(state)

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
        return self._resolve_proposals(proposal_ids, records)

    def resolve_improvement_proposals(self, proposal_ids):
        records = {r['proposal_id']: r for r in self.proposals(target=None)
                   if r['kind'] in ('harness_change', 'policy_training')}
        return self._resolve_proposals(proposal_ids, records)

    @staticmethod
    def _resolve_proposals(proposal_ids, records):
        resolved = []
        for ident in proposal_ids:
            if ident not in records or records[ident]['subject'] not in ('meta_agent', 'lab'):
                raise ValueError('Unknown or untargeted improvement proposal: ' + ident)
            row = records[ident]
            resolved.append({key: deepcopy(row[key]) for key in
                ('proposal_id', 'job_id', 'subject', 'source_snapshot')})
            resolved[-1]['kind'] = row['kind']
            resolved[-1]['proposal_digest'] = result_id(row['proposal'])
        return tuple(resolved)

    # Compatibility helper; live publication uses the configured organizer.
    _memory = staticmethod(journal_memory)

    def _publish(self):
        for ident, row in list(self.learning.state['jobs'].items()):
            if row['status'] != 'completed' or row.get('published'):
                continue
            if row.get('publication_delivery', {}).get('status') in ('failed', 'interrupted', 'running'):
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
            state = deepcopy(self.learning.state)
            state['jobs'][ident]['publication_delivery'] = {'status': 'running'}
            self.learning._commit(state)
            try:
                memory = self.learning.context_binding.publication(row)
                if memory is not None:
                    if not isinstance(memory, Memory):
                        raise TypeError('Legacy context/memory publication must be a Memory or None')
                    publish = getattr(self.adapter, 'publish', None)
                    if not callable(publish):
                        raise TypeError('Legacy Memory publication requires adapter.publish')
                    publish(memory)
            except Exception as exc:
                state = deepcopy(self.learning.state)
                state['jobs'][ident].update(publication_error=f'{type(exc).__name__}: {exc}',
                    publication_delivery={'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
                self.learning._commit(state)
                if self.learning.failure_policy.publication == 'raise':
                    raise
                continue
            state = deepcopy(self.learning.state)
            state['jobs'][ident].update(published=True,
                publication_error=None, publication_delivery={'status': 'delivered'},
                memory_digest=result_id(asdict(memory)) if memory is not None else None)
            # If persistence fails, retry publishing the same deterministic ID.
            self.learning._commit(state)

    def close(self, *, timeout=30):
        self.learning.notify_archive()
        self.learning.close(timeout=timeout)
        self._publish()

    def pause(self, *, timeout=30):
        """Settle submitted work, preserve queued work, and stop admissions."""
        self.learning.notify_archive()
        self.learning.close(timeout=timeout, preserve_queued=True)
        self._publish()
