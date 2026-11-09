"""Pure producer-inbox projection for the acting agent's explicit Memory work.

No note text, model calls, hidden writes or automatic success assessments.
Dispositions live in AgentMemory's ordinary revision journal.
"""
from dataclasses import asdict

from rsisys.harnesses.decision.observations import failed_observation, task_observation
from rsisys.runtime.storage.json_tools import result_id


def pending_review(memory):
    history = memory._history()
    sources = memory._sources()
    indexed = {row.record_id: row for row in history}

    def key(row):
        # Identical re-delivery is evidence once, not a fresh obligation on
        # every poll. Different results from the same job remain distinct.
        return result_id({'receipt': row.details.get('result', row.details.get('feedback', asdict(row))),
                          'inherited': bool(row.details.get('continuation_origin'))})

    considered, dispositions = set(), {}
    for revision in memory._state['revisions']:
        if revision['operation'] not in ('memory_write', 'memory_review'):
            continue
        args = revision['arguments']
        label = 'written' if revision['operation'] == 'memory_write' else args['disposition']
        dispositions[label] = dispositions.get(label, 0) + 1
        for ref in args['source_refs']:
            # A result alias can be re-delivered later. The write's recorded
            # provenance, not the latest alias resolver, owns that occurrence.
            provenance = (revision['memories'][args['memory_id']]['provenance'].get(ref)
                          if revision['operation'] == 'memory_write' else sources.get(ref))
            if provenance and provenance['history_id'] in indexed:
                considered.add(key(indexed[provenance['history_id']]))
    producers = {}
    for row in indexed.values():
        if task_observation(row) and not row.details.get('continuation_origin'):
            producers.setdefault(key(row), row)
    pending = [row for digest, row in producers.items() if digest not in considered]
    # Navigation only: exact matching native failures, not a model-written
    # diagnosis or a semantic claim that two failures share a cause.
    repeats, seen_failures = {}, set()
    for row in indexed.values():
        if failed_observation(row) and not row.details.get('continuation_origin'):
            result = row.details.get('result', row.details.get('feedback', {}))
            signature = result_id({'operation': row.details.get('operation', {}).get('name'),
                                   'error': result.get('error'), 'summary': row.summary})
            data = result.get('data') or {}
            job_id = data.get('job_id') if isinstance(data, dict) else None
            # Repeated terminal polls describe one failed job, but separate
            # failed invocations without a job are distinct behavior events.
            occurrence = ('job', job_id) if isinstance(job_id, str) and job_id else ('record', row.record_id)
            if (signature, occurrence) in seen_failures:
                continue
            seen_failures.add((signature, occurrence))
            repeats.setdefault(signature, []).append(row)
    operations = [row.details.get('operation', {}).get('name') for row in history
                  if row.kind == 'operation' and not row.details.get('continuation_origin')]
    return {'pending_count': len(pending), 'producer_count': len(producers),
        'considered_count': len(producers.keys() & considered),
        'memory_count': len(memory._state['memories']), 'dispositions': dispositions,
        'explicit_search_calls': operations.count('memory_search'),
        'explicit_read_calls': operations.count('memory_read'),
        'usage_verified': False,
        'recurring_failures': [{'signature': signature, 'count': len(rows),
            'operation': rows[-1].details.get('operation', {}).get('name'),
            'source_refs': ['history:decision_maker:' + row.record_id for row in rows[-4:]],
            'instruction': 'Assess operational reuse, not only scientific novelty. Search/read prior notes; '
                'write/update a source-cited prevention procedure, link an existing note, or explain why '
                'it will not help future actions. A note is not proof that behavior changed.'}
            for signature, rows in [(k, v) for k, v in repeats.items() if len(v) > 1][-8:]],
        'items': [{'record_id': row.record_id, 'source_sha256': result_id(asdict(row)),
                   'read_path': {'operation': 'research_archive', 'collection': 'history',
                                 'record_id': row.record_id, 'evidence_path': None, 'limit': 8}}
                  for row in pending[:8]]}
