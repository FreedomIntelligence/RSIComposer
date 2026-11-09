"""Versioned, evidence-linked interpretations; never an execution ledger."""
from copy import deepcopy
from rsisys.runtime.storage.json_tools import result_id
from .evidence import claim_schema, validate_claims


def attempt_schema(context, refs, candidate_evidence=None):
    decisions = [r['decision_id'] for r in context.get('decisions', [])]
    known = [r['attempt_id'] for r in context.get('attempts', [])]
    fields = {
        'attempt_id': {'enum': [None, *known]},
        'base_revision': {'type': 'integer', 'minimum': 0},
        'title': {'type': 'string', 'minLength': 1, 'maxLength': 500},
        'summary': {'type': 'string', 'minLength': 1, 'maxLength': 2400},
        'member_decision_ids': {'type': 'array', 'minItems': 1, 'maxItems': 32,
                                'items': {'enum': decisions} if decisions else {'type': 'string'}},
        'source_refs': {'type': 'array', 'minItems': 1, 'maxItems': 32,
                        'items': {'enum': list(refs)}},
    }
    if candidate_evidence is not None:
        fields['evaluation_claims'] = claim_schema(candidate_evidence)
    return {'type': 'array', 'maxItems': 4 if decisions else 0, 'items': {
        'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}}


def validate_attempt_updates(updates, context, refs, candidate_evidence=None):
    if not isinstance(updates, list) or len(updates) > 4:
        raise ValueError('Expected at most four attempt interpretations')
    decisions = {r['decision_id']: r for r in context.get('decisions', [])}
    known = {r['attempt_id']: r for r in context.get('attempts', [])}
    seen = set()
    for update in updates:
        fields = {'attempt_id', 'base_revision', 'title', 'summary', 'member_decision_ids', 'source_refs'}
        if candidate_evidence is not None:
            fields.add('evaluation_claims')
        if not isinstance(update, dict) or set(update) != fields:
            raise ValueError('Invalid attempt interpretation fields')
        ident, revision = update['attempt_id'], update['base_revision']
        if ident is not None and (not isinstance(ident, str) or ident not in known):
            raise ValueError('Unknown interpretation attempt ID')
        if type(revision) is not int or revision != (known[ident]['revision'] if ident else 0):
            raise ValueError('Attempt base revision does not match supplied evidence')
        for key, maximum in (('title', 500), ('summary', 2400)):
            if not isinstance(update[key], str) or not update[key].strip() or len(update[key]) > maximum:
                raise ValueError('Invalid attempt ' + key)
        members, sources = update['member_decision_ids'], update['source_refs']
        if (not isinstance(members, list) or not 1 <= len(members) <= 32
                or any(not isinstance(m, str) or m not in decisions for m in members)
                or len(set(members)) != len(members) or seen.intersection(members)):
            raise ValueError('Unknown or duplicate attempt member decision')
        if (not isinstance(sources, list) or not 1 <= len(sources) <= 32
                or any(not isinstance(s, str) or s not in refs for s in sources)
                or len(set(sources)) != len(sources)
                or any('decision:' + m not in sources for m in members)):
            raise ValueError('Attempt must cite supplied evidence for every member')
        if candidate_evidence is not None:
            validate_claims(update, candidate_evidence)
        seen.update(members)
    return deepcopy(updates)


def apply_attempt_updates(row, current):
    """Optimistic concurrency: all updates or none, with immutable revisions."""
    latest = {r['attempt_id']: deepcopy(r) for r in current}
    owners = {member: r['attempt_id'] for r in current for member in r['member_decision_ids']}
    revisions = []
    for index, update in enumerate(row.get('attempt_updates', [])):
        ident = update['attempt_id'] or 'attempt-' + result_id([row['job_id'], index])[:24]
        previous = latest.get(ident)
        if update['base_revision'] != (previous['revision'] if previous else 0):
            raise ValueError('Stale attempt interpretation; newer revision preserved')
        if any(m in owners and owners[m] != ident for m in update['member_decision_ids']):
            raise ValueError('Conflicting attempt membership; previous grouping preserved')
        entry = {key: deepcopy(update[key]) for key in ('title', 'summary', 'member_decision_ids', 'source_refs')}
        if 'evaluation_claims' in update:
            entry['evaluation_claims'] = deepcopy(update['evaluation_claims'])
        if previous:
            # A bounded update adds observed members; omitted earlier evidence is
            # retained, not erased or required to fit in every future model input.
            for key in ('member_decision_ids', 'source_refs'):
                entry[key] = list(dict.fromkeys([*previous[key], *entry[key]]))
        entry.update(attempt_id=ident, revision=update['base_revision'] + 1,
            provenance={'job_id': row['job_id'], 'model': deepcopy(row['model']),
                        'input_digest': result_id(row['input'])},
            simulated=bool(row['simulated'] or previous and previous['simulated']))
        latest[ident] = entry
        owners.update({m: ident for m in entry['member_decision_ids']})
        revisions.append(entry)
    return revisions


def attempt_view(jobs, decision_ids=()):
    revisions = [deepcopy(r) for job in jobs for r in job.get('attempt_revisions', [])]
    # JSON persistence sorts journal keys; presentation must not depend on that
    # incidental order or reinterpret a later revision as an earlier one.
    revisions.sort(key=lambda r: (r['attempt_id'], r['revision']))
    latest = {}
    for row in revisions:
        if row['revision'] > latest.get(row['attempt_id'], {}).get('revision', 0):
            latest[row['attempt_id']] = row
    assigned = {m for row in latest.values() for m in row['member_decision_ids']}
    return {'schema_version': 1, 'authority': 'model_interpretation',
            'items': list(latest.values()), 'revisions': revisions,
            'ungrouped_decision_ids': [m for m in decision_ids if m not in assigned]}
