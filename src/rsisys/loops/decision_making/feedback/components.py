"""Replaceable feedback products, independent of a particular agent harness.

Context organization must be configured. Public trajectory archival defaults to
a journal-derived index and may be disabled. Both receive detached snapshots;
the execution/delivery journal remains separate from these derived products.
"""
from rsisys.loops._memory import Memory
from rsisys.loops.contracts.feedback_context import ContextMemory
from rsisys.loops.contracts.feedback_modules import TrajectoryArchive, FeedbackProcessor
from rsisys.loops.contracts.receipts import receipt_task, receipt_facts
from rsisys.runtime.json_tools import encoded
from .knowledge import context_record, working_memory, trajectory_records, trajectory_segment, trajectory_snapshot


# Historical import name; the public contract now has one context method.
ContextMemoryOrganizer = ContextMemory


def journal_memory(row):
    """Default compatibility publication, not a requirement on other organizers."""
    # Learning owns output validation; import at call time to avoid coupling its
    # construction to the default organizer's publication implementation.
    from .learning import feedback_text_preview, validate_publication
    evidence = validate_publication(row)
    proposals = [p for p in row['proposals'] if p['kind'] == 'memory']
    interpretations = row.get('attempt_revisions', [])
    receipt = row.get('raw_input', row['input'])['receipt']
    feedback, directive = receipt_facts(receipt), receipt_task(receipt)
    facts = {key: feedback[key] for key in ('status', 'quality_metrics',
        'training_metrics', 'execution_metrics', 'error', 'evidence_refs', 'adopted_candidate_ids')}
    facts.update({key: feedback[key] for key in ('directive_id', 'task_id', 'decision_id', 'source')
                  if key in feedback})
    error_path = '/receipt/feedback/error' if 'feedback' in receipt else '/receipt/result/error'
    facts['error'], omission = feedback_text_preview(feedback['error'], job_id=row['job_id'],
                                                    evidence_path=error_path)
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
    return Memory('feedback-learning:' + row['job_id'], 'episodic', encoded(content),
        tuple(row['input']['source_refs']),
        applicability=encoded({'objective': directive['objective'],
                               'task_id': directive.get('task_id') or directive.get('directive_id'),
                               'experiment': directive.get('directive_id'),
                               'scope': 'This task and its original evidence/model/protocol; not current global state'}),
        status='unverified')


class JournalContextMemory:
    """Journal implementation, exposed through the internal compatibility adapter.

    These three methods are journal-specific, not requirements on new providers.
    """
    record = staticmethod(context_record)
    working_memory = staticmethod(working_memory)
    memory = staticmethod(journal_memory)


class JournalTrajectoryArchive:
    """Default optional archive index over the existing durable journal."""
    records = staticmethod(trajectory_records)
    extract = staticmethod(trajectory_segment)
    snapshot = staticmethod(trajectory_snapshot)
