"""Derived research knowledge, independent of harness state and conversation format.

The learning journal is authoritative for interpretations and their revisions;
execution receipts remain authoritative for facts. This module owns no second
store and makes no model calls. A harness chooses a bounded resident view, while
all three improvement targets remain available through explicit reads.
"""
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

from rsisys.loops.contracts.receipts import receipt_facts, receipt_task
from rsisys.runtime.storage.json_tools import EvidencePathError, encoded, result_id

TARGETS = {'memory': 'context_memory', 'harness_change': 'harness',
           'training_data': 'weights', 'policy_training': 'weights', 'system_change': 'system'}
SUBJECTS = ('meta_agent', 'lab', 'unspecified')


@dataclass(frozen=True)
class WorkingMemoryPolicy:
    """Host-selected residency limits, not research/experiment budgets."""
    max_bytes: int = 12000
    recent: int = 8
    summary_characters: int = 600

    def __post_init__(self):
        if (type(self.max_bytes) is not int or self.max_bytes < 2048
                or type(self.recent) is not int or self.recent < 1
                or type(self.summary_characters) is not int or self.summary_characters < 1):
            raise ValueError('Invalid working memory policy')


def ordered_jobs(jobs):
    rows = list(jobs)
    return [row for _, row in sorted(enumerate(rows),
        key=lambda pair: (pair[1].get('capture_index', -1), pair[0]))]


def interpretation_read(job_id, path):
    return {'operation': 'feedback_learning_read', 'job_id': job_id, 'evidence_path': path}


def _preview(text, maximum):
    return text if len(text) <= maximum else text[:maximum] + ' [excerpt; read original]'


def _snapshot(job):
    return job.get('raw_input', job['input'])


def _identity(job):
    raw = _snapshot(job)
    task = receipt_task(raw['receipt'])
    scope = raw.get('research_context', {})
    identity = raw['receipt'].get('identity', {})
    return {'task_id': task.get('task_id') or task.get('directive_id'),
            'decision_id': identity.get('decision_id') or task.get('decision_id') or scope.get('decision_id'),
            **{name: identity.get(name) for name in ('lab_id', 'run_id', 'agent_id')},
            'issuing_model': deepcopy(identity.get('issuing_model') or task.get('issuing_model')
                                      or scope.get('issuing_decision_model'))}


def context_record(job):
    """Required factual context organization, independent of optional proposals.

    A missing/failed interpretation does not erase an observed execution result.
    Long text remains exact here; a harness separately selects bounded previews.
    """
    raw = _snapshot(job)
    facts = receipt_facts(raw['receipt'])
    return {'record_id': 'context:' + job['job_id'], 'job_id': job['job_id'],
        **_identity(job), 'authority': 'execution_observation',
        'interpretation_status': job['status'],
        'facts': {name: deepcopy(facts.get(name)) for name in (
            'status', 'summary', 'simulated', 'error', 'quality_metrics', 'training_metrics',
            'execution_metrics', 'evidence_refs', 'adopted_candidate_ids')},
        'interpretations': [deepcopy(p) for p in job['proposals']
            if job['status'] == 'completed' and p['kind'] == 'memory'
            and p.get('subject', 'unspecified') != 'lab'],
        'interpretation_authority': 'unverified',
        'source_snapshot': interpretation_read(job['job_id'], ''),
        'read_path': interpretation_read(job['job_id'], '/context_memory')}


def trajectory_records(jobs, *, research_attempt_id=None):
    """Index every captured segment; no training labels or complete-chat claims."""
    records = []
    for job in ordered_jobs(jobs):
        raw = _snapshot(job)
        scope = raw.get('research_context', {})
        attempts = list(dict.fromkeys([*([scope['research_attempt_id']]
            if scope.get('research_attempt_id') else []),
            *(a['attempt_id'] for a in job.get('attempt_revisions', []))]))
        if research_attempt_id is not None and research_attempt_id not in attempts:
            continue
        records.append({'trajectory_id': 'trajectory:' + job['job_id'], 'job_id': job['job_id'],
            **_identity(job), 'research_attempt_ids': attempts,
            'authority': 'archived_public_trajectory', 'feedback_status': job['status'],
            'execution_status': receipt_facts(raw['receipt'])['status'],
            'feedback_model': None if job.get('archive_only') else deepcopy(job['model']),
            'capture_index': job.get('capture_index'), 'snapshot_digest': result_id(raw),
            'source_snapshot': interpretation_read(job['job_id'], ''),
            'captured_history_records': len(raw.get('history', [])),
            'captured_decision_records': sum(r.get('kind') == 'decision' for r in raw.get('history', [])),
            'feedback_request': (interpretation_read(job['job_id'], '/feedback_request')
                                 if job.get('request') is not None else None),
            'feedback_response': (interpretation_read(job['job_id'], '/feedback_response')
                                  if job.get('response') is not None else None),
            'coverage': 'captured_decision_records_execution_and_feedback',
            'original_meta_agent_exchange': 'not_captured_by_this_observer',
            'extraction_read_path': interpretation_read(job['job_id'], '/trajectory_segment'),
            'complete_conversation': False, 'training_ready': False, 'applied': False})
    return records


def trajectory_segment(job):
    """Extract recorded public material verbatim; add no generated supervision."""
    raw = _snapshot(job)
    index, = trajectory_records([job])
    return json.loads(encoded({**index, 'schema_version': 1,
        'research_context': deepcopy(raw.get('research_context', {})),
        'history': deepcopy(raw.get('history', [])),
        'execution_receipt': deepcopy(raw['receipt']),
        'evaluations': deepcopy(raw.get('evaluations', [])),
        'feedback': {'model': deepcopy(job['model']), 'status': job['status'],
            'request': deepcopy(job.get('request')), 'response': deepcopy(job.get('response')),
            'products': deepcopy(job.get('products', {})), 'error': job.get('error')},
        'scope': 'captured_public_segment; original meta-agent exchange is not captured by this observer'}))


def trajectory_snapshot(job):
    """Immutable captured facts, independent of later memory/proposal results."""
    raw = deepcopy(_snapshot(job))
    return json.loads(encoded({'schema_version': 1,
        'trajectory_id': 'trajectory:' + job['job_id'], 'job_id': job['job_id'],
        **_identity(job), 'capture_index': job.get('capture_index'),
        'authority': 'archived_public_trajectory', 'snapshot_digest': result_id(raw),
        'capture': raw, 'complete_conversation': False, 'training_ready': False,
        'coverage': {'decision_records': 'captured_history_only',
                     'execution_receipt': 'original', 'original_meta_agent_exchange': 'not_captured'},
        'dataset_construction_owner': 'lab'}))


def proposal_records(jobs, *, target=None, research_attempt_id=None):
    """Separate improvement targets without claiming validation or application.

    Legacy system_change proposals stay explicitly separate: actor/system edits
    must not silently become researcher harness changes or weight supervision.
    """
    if target is not None and target not in TARGETS.values():
        raise ValueError('Unknown feedback target')
    records = []
    for job in ordered_jobs(jobs):
        if job['status'] != 'completed':
            continue
        scope = job['input'].get('research_context', {})
        attempt_ids = list(dict.fromkeys([*([scope['research_attempt_id']]
            if scope.get('research_attempt_id') else []),
            *(a['attempt_id'] for a in job.get('attempt_revisions', []))]))
        if research_attempt_id is not None and research_attempt_id not in attempt_ids:
            continue
        for index, proposal in enumerate(job['proposals']):
            lane = TARGETS[proposal['kind']]
            if target is not None and lane != target:
                continue
            records.append({'proposal_id': job['job_id'] + ':' + str(index),
                'target': lane, 'kind': proposal['kind'], 'status': 'proposed',
                'subject': proposal.get('subject', 'unspecified'),
                'authority': 'unverified_interpretation', 'training_ready': False, 'applied': False,
                'job_id': job['job_id'], 'research_attempt_ids': attempt_ids,
                'feedback_model': deepcopy(job['model']),
                'source_snapshot': interpretation_read(job['job_id'], ''),
                'proposal': deepcopy(proposal),
                'read_path': interpretation_read(job['job_id'], '/proposals/' + str(index))})
    return records


def working_memory(jobs, attempts, *, active_decision_ids=(), policy=None):
    """Latest Attempt interpretations plus small ungrouped observation cards.

    An Attempt replaces its older summaries in this view. Related per-receipt
    prose is not injected again. Unassigned feedback stays explicitly ungrouped;
    shared titles and temporal adjacency never create membership. Large content
    and source/member lists are recovered through the journal, not truncated
    into invented facts. Aggregate encoded size is checked for every addition.
    """
    policy = WorkingMemoryPolicy() if policy is None else policy
    if not isinstance(policy, WorkingMemoryPolicy):
        raise TypeError('working memory policy must be a WorkingMemoryPolicy')
    jobs = [j for j in ordered_jobs(jobs) if not j.get('archive_only')]
    def attention(job):
        return (job['status'] not in ('completed', 'archived') or bool(job.get('grouping_error'))
                or any(p.get('status') != 'completed' for p in job.get('products', {}).values())
                or any(job.get(key, {}).get('status') in ('failed', 'interrupted') for key in
                       ('context_delivery', 'archive_delivery', 'publication_delivery')))
    rank = {j['job_id']: i for i, j in enumerate(jobs)}
    active = set(active_decision_ids)
    attempts = sorted(attempts, key=lambda a: (
        bool(active.intersection(a['member_decision_ids'])),
        rank.get(a['provenance']['job_id'], -1), a['attempt_id']))
    memberships = {member for a in attempts for member in a['member_decision_ids']}
    explicit_attempts = {j['input'].get('research_context', {}).get('research_attempt_id')
        for j in jobs if j['input'].get('research_context', {}).get('research_attempt_id')}
    # Report all target counts, but only context_memory prose belongs in the
    # resident research memory. Harness/weight proposals require explicit reads.
    counts = Counter(TARGETS[p['kind']] for j in jobs if j['status'] == 'completed'
                     for p in j['proposals'])
    state = {'schema_version': 2, 'authority': 'derived_research_memory',
        'interpretation_authority': 'unverified', 'attempts': [], 'findings': [],
        'learning_attention': [],
        'targets': {name: {'proposals': counts[name], 'applied': False}
                    for name in dict.fromkeys(TARGETS.values())},
        'totals': {'attempts': len(attempts), 'learning_jobs': len(jobs),
                   'explicit_attempts': len(explicit_attempts),
                   'learning_attention': sum(attention(j) for j in jobs)},
        'read_path': {'operation': 'feedback_learning_read', 'view': 'attempts'},
        'proposal_read_path': {'operation': 'feedback_learning_read', 'view': 'proposals'},
        'residency': {'max_bytes': policy.max_bytes, 'omitted_cards': 0},
        'note': 'Working selection, not complete history or verified conclusions. '
                'Missing records are not negative results. Read exact revisions, '
                'proposals and execution evidence before reusing a claim.'}

    def add(collection, card):
        state[collection].append(card)
        # Reserve for final digest and omission counters.
        if len(encoded(state).encode('utf-8')) + 160 <= policy.max_bytes:
            return True
        state[collection].pop()
        state['residency']['omitted_cards'] += 1
        return False

    for attempt in reversed(attempts[-policy.recent:]):
        add('attempts', {'attempt_id': attempt['attempt_id'],
            'title': _preview(attempt['title'], 160),
            'summary': _preview(attempt['summary'], policy.summary_characters),
            'revision': attempt['revision'], 'member_count': len(attempt['member_decision_ids']),
            'source_count': len(attempt['source_refs']), 'simulated': attempt['simulated'],
            'authority': 'unverified_interpretation',
            'read_path': {'operation': 'feedback_learning_read', 'view': 'attempts',
                          'research_attempt_id': attempt['attempt_id']}})
    state['attempts'].reverse()
    ungrouped = []
    for job in jobs:
        scope = job['input'].get('research_context', {})
        grouped = (scope.get('decision_id') in memberships
                   or bool(job.get('attempt_revisions'))
                   or any(a['attempt_id'] == scope.get('research_attempt_id') for a in attempts))
        if not grouped:
            ungrouped.append(job)
    for job in reversed(ungrouped[-policy.recent:]):
        proposals = []
        for index, proposal in enumerate(job['proposals'] if job['status'] == 'completed' else []):
            if proposal['kind'] != 'memory' or proposal.get('subject', 'unspecified') == 'lab':
                continue
            if len(encoded(proposal).encode('utf-8')) <= 1024:
                proposals.append(deepcopy(proposal))
            else:
                proposals.append({'kind': 'memory',
                    'content': _preview(proposal['content'], 240),
                    'authority': 'unverified_excerpt',
                    'read_path': interpretation_read(job['job_id'], '/proposals/' + str(index))})
        card = {'job_id': job['job_id'], 'simulated': job['simulated'],
            'authority': 'unverified_interpretation' if proposals else 'execution_observation',
            'interpretation_grouping': 'unassigned',
            'learned_by': deepcopy(job['model']),
            'research_context': {key: deepcopy(job['input'].get('research_context', {}).get(key))
                for key in ('decision_id', 'research_attempt_id', 'research_attempt_title')},
            'source_refs': deepcopy(job['input']['source_refs'][:2]),
            'source_ref_count': len(job['input']['source_refs']), 'proposals': proposals,
            'read_path': interpretation_read(job['job_id'], '/proposals')}
        if not proposals:
            # The required factual record exists for every job, but do not
            # repeat it inside an existing interpretation card as well.
            observation = context_record(job)
            facts = observation['facts']
            card.update(interpretation_status=job['status'], observation={
                'task_id': observation['task_id'], 'status': facts['status'],
                'summary_preview': _preview(facts['summary'], policy.summary_characters),
                'error_preview': _preview(str(facts['error']), 160) if facts['error'] else None,
                'read_path': observation['read_path']})
        add('findings', card)
    state['findings'].reverse()
    for job in reversed([j for j in jobs if attention(j)][-policy.recent:]):
        add('learning_attention', {'job_id': job['job_id'], 'status': job['status'],
            'error_preview': _preview(str(job.get('error') or job.get('grouping_error') or job.get('publication_error') or ''), 160),
            'product_statuses': {name: product.get('status') for name, product in job.get('products', {}).items()},
            'delivery_statuses': {name: job[name].get('status') for name in
                ('context_delivery', 'archive_delivery', 'publication_delivery') if job.get(name)},
            'read_path': {'operation': 'feedback_learning_read', 'job_id': job['job_id']}})
    state['residency']['omitted_cards'] = (len(attempts) + len(ungrouped)
        + state['totals']['learning_attention']
        - sum(len(state[name]) for name in ('attempts', 'findings', 'learning_attention')))
    state['revision'] = result_id(state)
    assert len(encoded(state).encode('utf-8')) <= policy.max_bytes
    return state


def interpretation_page(value, *, offset=0, limit=8, expected_digest=None, metadata=None,
                        authority='unverified_interpretations'):
    """Exact JSON-text pages, pinned across updates when the caller supplies SHA."""
    digest = result_id(value)
    if expected_digest is not None and expected_digest != digest:
        raise EvidencePathError('Research knowledge changed; restart pagination')
    text = encoded(value)
    metadata = {**(metadata or {}), 'sha256': digest, 'encoding': 'json', 'offset': offset,
        'total_characters': len(text), 'authority': authority,
        'training_ready': False, 'applied': False}
    chunk = text[offset:offset + limit * 128]
    def page(size):
        end = offset + size
        return {**metadata, 'text': chunk[:size],
                'next_offset': end if end < len(text) else None}
    size = len(chunk)
    while size > 1 and len(encoded(page(size)).encode('utf-8')) > 6000:
        size //= 2
    return page(size)
