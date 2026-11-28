"""Derived research knowledge, independent of harness state and conversation format.

The learning journal is authoritative for interpretations and their revisions;
execution receipts remain authoritative for facts. This module owns no second
store and makes no model calls. A harness chooses a bounded resident view; installed
improvement targets remain available through explicit reads.
"""
import json
from copy import deepcopy

from rsicomposer.loops.contracts.receipts import receipt_facts, receipt_task
from rsicomposer.runtime.storage.json_tools import EvidencePathError, encoded, result_id


def ordered_jobs(jobs):
    return sorted(jobs, key=lambda row: row['capture_index'])


def interpretation_read(job_id, path):
    return {'operation': 'feedback_learning_read', 'job_id': job_id, 'evidence_path': path}


def _identity(job):
    raw = job['raw_input']
    task = receipt_task(raw['receipt'])
    scope = raw.get('research_context', {})
    identity = raw['receipt'].get('identity', {})
    return {'task_id': task.get('task_id') or task.get('directive_id'),
            'decision_id': identity.get('decision_id') or task.get('decision_id') or scope.get('decision_id'),
            **{name: identity.get(name) for name in ('lab_id', 'run_id', 'agent_id')},
            'issuing_model': deepcopy(identity.get('issuing_model') or task.get('issuing_model')
                                      or scope.get('issuing_decision_model'))}


def trajectory_records(jobs, *, research_attempt_id=None):
    """Index every captured segment; no training labels or complete-chat claims."""
    records = []
    for job in ordered_jobs(jobs):
        raw = job['raw_input']
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
            'capture_index': job['capture_index'], 'snapshot_digest': result_id(raw),
            'source_snapshot': interpretation_read(job['job_id'], ''),
            'captured_history_records': len(raw.get('history', [])),
            'captured_decision_records': sum(r.get('kind') == 'decision' for r in raw.get('history', [])),
            'feedback_request': (interpretation_read(job['job_id'], '/feedback_request')
                                 if job.get('request') is not None else None),
            'feedback_response': (interpretation_read(job['job_id'], '/feedback_response')
                                  if job.get('response') is not None else None),
            'coverage': 'captured_decision_records_execution_and_feedback',
            'original_meta_agent_exchange': 'not_captured_by_this_observer',
            'extraction_read_path': interpretation_read(job['job_id'], '/trajectory_snapshot'),
            'complete_conversation': False, 'training_ready': False, 'applied': False})
    return records


def trajectory_snapshot(job):
    """Immutable captured facts, independent of later memory/proposal results."""
    raw = deepcopy(job['raw_input'])
    return json.loads(encoded({'schema_version': 1,
        'trajectory_id': 'trajectory:' + job['job_id'], 'job_id': job['job_id'],
        **_identity(job), 'capture_index': job['capture_index'],
        'authority': 'archived_public_trajectory', 'snapshot_digest': result_id(raw),
        'capture': raw, 'complete_conversation': False, 'training_ready': False,
        'coverage': {'decision_records': 'captured_history_only',
                     'execution_receipt': 'original', 'original_meta_agent_exchange': 'not_captured'},
        'dataset_construction_owner': 'lab'}))


def proposal_records(jobs, *, targets, target=None, research_attempt_id=None):
    """Project published interpretations into their installed query categories."""
    if target is not None and target not in {rule.target for rule in targets.values()}:
        raise ValueError('Unknown feedback target')
    records = []
    for job in ordered_jobs(jobs):
        if job['status'] != 'completed' or not job.get('published'):
            continue
        scope = job['input'].get('research_context', {})
        attempt_ids = list(dict.fromkeys([*([scope['research_attempt_id']]
            if scope.get('research_attempt_id') else []),
            *(a['attempt_id'] for a in job.get('attempt_revisions', []))]))
        if research_attempt_id is not None and research_attempt_id not in attempt_ids:
            continue
        for index, proposal in enumerate(job['proposals']):
            lane = targets[proposal['kind']].target
            if target is not None and lane != target:
                continue
            records.append({'proposal_id': job['job_id'] + ':' + str(index),
                'target': lane, 'kind': proposal['kind'], 'status': 'proposed',
                'subject': proposal['subject'],
                'authority': 'unverified_interpretation', 'training_ready': False, 'applied': False,
                'job_id': job['job_id'], 'research_attempt_ids': attempt_ids,
                'feedback_model': deepcopy(job['model']),
                'source_snapshot': interpretation_read(job['job_id'], ''),
                'proposal': deepcopy(proposal),
                'read_path': interpretation_read(job['job_id'], '/proposals/' + str(index))})
    return records


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
