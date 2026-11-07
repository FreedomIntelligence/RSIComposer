"""Pure producer-inbox projection for the acting agent's explicit Memory work.

No note text, model calls, hidden writes or automatic success assessments.
Dispositions live in AgentMemory's ordinary revision journal.
"""
from dataclasses import asdict

from rsisys.runtime.json_tools import result_id
from .observations import task_observation


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
    operations = [row.details.get('operation', {}).get('name') for row in history
                  if row.kind == 'operation' and not row.details.get('continuation_origin')]
    return {'pending_count': len(pending), 'producer_count': len(producers),
        'considered_count': len(producers.keys() & considered),
        'memory_count': len(memory._state['memories']), 'dispositions': dispositions,
        'explicit_search_calls': operations.count('memory_search'),
        'explicit_read_calls': operations.count('memory_read'),
        'usage_verified': False,
        'items': [{'record_id': row.record_id, 'source_sha256': result_id(asdict(row)),
                   'read_path': {'operation': 'research_archive', 'collection': 'history',
                                 'record_id': row.record_id, 'evidence_path': None, 'limit': 8}}
                  for row in pending[:8]]}
