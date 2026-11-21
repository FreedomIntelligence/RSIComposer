"""Harness-independent feedback learning lifecycle and adapter contract.

The service knows receipts, model identities, learning proposals and publication;
it never accesses a harness policy, state layout or context compactor.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

from rsicomposer.harnesses.decision.feedback.knowledge import proposal_records
from rsicomposer.harnesses.decision.feedback.proposal_validation import validate_publication
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.loops.contracts.receipts import ExecutionReceipt
from rsicomposer.runtime.storage.json_tools import result_id


@dataclass(frozen=True)
class FeedbackEvidence:
    history: tuple = ()
    evaluations: tuple = ()
    research_context: dict = field(default_factory=dict)
    grouping_context: dict | None = None


class FeedbackAdapter(Protocol):
    """Explicit host integration; evidence and projection hooks are optional."""
    def model(self) -> ModelRef: ...
    def install(self, service): ...


class FeedbackLearningService:
    def __init__(self, learning, adapter: FeedbackAdapter):
        self.learning, self.adapter = learning, adapter
        from rsicomposer.method_state import ResearchMemoryState
        self.memory_state = ResearchMemoryState(learning.context_memory)

    @property
    def memory(self):
        return self.learning.context_memory

    def project_window(self, window):
        from rsicomposer.runtime.capabilities import optional_method
        project = optional_method(self.adapter, 'project_window')
        return project(window) if project is not None else window

    def state(self):
        from rsicomposer.method_state import FeedbackJournalState
        return {'feedback': FeedbackJournalState(self).capture(), 'memory': self.memory_state.capture()}

    def restore(self, state):
        from rsicomposer.method_state import FeedbackJournalState
        if not isinstance(state, dict) or set(state) != {'feedback', 'memory'}:
            raise ValueError('Invalid Harness integration state')
        FeedbackJournalState(self).restore(state['feedback'])
        self.memory_state.restore(state['memory'])

    def export_version(self):
        from rsicomposer.runtime.capabilities import optional_method
        export = optional_method(self.memory, 'export_version') or optional_method(self.memory, 'export')
        if export is None:
            raise TypeError('Controller Memory version capture requires export_version() or export()')
        return deepcopy(export())

    def seed_version(self, state, *, history=()):
        from rsicomposer.runtime.capabilities import optional_method
        seed = optional_method(self.memory, 'seed_version')
        if seed is None:
            raise TypeError('Controller Memory version inheritance requires seed_version(state)')
        if history:
            seed_history = optional_method(self.adapter.harness, 'seed_history')
            if seed_history is None:
                raise TypeError('Controller Harness must explicitly support seed_history for inherited evidence')
            current = optional_method(self.adapter.harness, 'history')
            existing = tuple(current()) if current is not None else ()
            if existing != tuple(history):
                if existing:
                    raise ValueError('Candidate history differs from its approved Controller history')
                seed_history(deepcopy(history))
        if seed(deepcopy(state)) is not None:
            raise TypeError('Controller Memory seed_version must return None')


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

    def context(self, query=None):
        """Text context for the next decision, without a publication object."""
        return self.learning.context(query)

    def trajectories(self, *, research_attempt_id=None):
        """Captured public trajectory segments, before any training-data recipe."""
        return self.learning.trajectories(research_attempt_id=research_attempt_id)

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
                # blocked, including restored invalid jobs.
                state = deepcopy(self.learning.state)
                state['jobs'][ident].update(status='failed', error='Publication validation failed: ' + str(exc),
                    publication_error=str(exc), proposals=[], attempt_updates=[], attempt_revisions=[])
                self.learning._commit(state)
                continue
            # Validated products are read through the journal; Memory receives
            # receipts through observe(), not a second published record.
            state = deepcopy(self.learning.state)
            state['jobs'][ident].update(published=True, publication_error=None,
                                        publication_delivery={'status': 'delivered'})
            self.learning._commit(state)

    def validate_snapshot(self):
        """Require portable state to have no uncommitted background work."""
        learning = self.learning
        if learning._futures or any(row['status'] in ('queued', 'running')
                                    for row in learning.state['jobs'].values()):
            raise RuntimeError('Portable snapshots require drained feedback jobs')
        for row in learning.state['jobs'].values():
            if any(row.get(name, {}).get('status') == 'running'
                   for name in ('context_delivery', 'archive_delivery', 'publication_delivery')):
                raise RuntimeError('Portable snapshots require settled feedback notifications')
            if not row.get('archive_only') and (learning.context_observer or learning.context_binding.observer) is not None and not row.get('context_delivery'):
                raise RuntimeError('Portable snapshots require delivered feedback notifications')
            if learning.archive_binding.requires_delivery and not row.get('archive_delivery'):
                raise RuntimeError('Portable snapshots require delivered archive notifications')
            if row['status'] == 'completed' and not row.get('published') and not row.get('publication_delivery'):
                raise RuntimeError('Portable snapshots require settled feedback publication')

    def close(self, *, timeout=30):
        self.learning.notify_archive()
        self.learning.close(timeout=timeout)
        self._publish()

    def pause(self, *, timeout=30):
        """Settle submitted work, preserve queued work, and stop admissions."""
        self.learning.notify_archive()
        self.learning.close(timeout=timeout, preserve_queued=True)
        self._publish()
