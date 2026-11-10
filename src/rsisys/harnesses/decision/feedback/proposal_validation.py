"""Validate feedback proposals and accounting before journal publication."""
from copy import deepcopy

from .attempts import attempt_schema, validate_attempt_updates
from .evidence import admitted_candidate_evidence, claim_schema, validate_claims
from .knowledge import SUBJECTS
from rsisys.runtime.storage.files import json_bytes
from rsisys.runtime.storage.json_tools import encoded, load_object as _load_object

LEGACY_KINDS = ('memory', 'training_data', 'harness_change', 'system_change')
KINDS = (*LEGACY_KINDS, 'policy_training')
FIELDS = ('kind', 'content', 'applicability', 'source_refs', 'validation_plan')
CANDIDATE_INSTRUCTIONS = (
    'candidate_evidence binds evaluations to full candidate IDs, separately from the current receipt. '
    'Every proposal/attempt update requires evaluation_claims: [] for hypotheses or observations without metric claims; '
    'otherwise enumerate every factual candidate metric claim with candidate_id, evaluation_ref, protocol_ref, '
    'metrics (exact names/values), simulated, and cite evaluation_ref in source_refs. '
    'Use completed supplied evaluations only. Never transfer metrics between candidates or protocols. '
    'Training completion, parents, adjacency and prior interpretations prove no candidate quality. '
    'not_observed_in_snapshot is snapshot-scoped absence; not_supplied is omitted evidence. '
    'Keep prose consistent with claims; structural validation does not verify prose or claim completeness. ')

def proposal_schema(refs, grouping_context=None, candidate_evidence=None, *, kinds=LEGACY_KINDS):
    properties = {
        'kind': {'enum': list(KINDS)},
        'subject': {'enum': list(SUBJECTS)},
        'content': {'type': 'string', 'minLength': 1, 'maxLength': 2400},
        'applicability': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'source_refs': {'type': 'array', 'minItems': 1, 'maxItems': 16,
                        'items': {'enum': list(refs)}},
        'validation_plan': {'type': 'string', 'minLength': 1, 'maxLength': 1200},
    }
    if candidate_evidence is not None:
        properties['evaluation_claims'] = claim_schema(candidate_evidence)
    branches = []
    for kind in kinds:
        fields = {**properties, 'kind': {'const': kind}}
        if kind == 'training_data':
            fields['training_example'] = {'type': 'object', 'properties': {
                'input': {'type': 'string', 'minLength': 1, 'maxLength': 4000},
                'target': {'type': 'string', 'minLength': 1, 'maxLength': 4000},
                'target_status': {'const': 'unverified'}},
                'required': ['input', 'target', 'target_status'], 'additionalProperties': False}
        branches.append({'type': 'object', 'properties': fields, 'required': list(fields),
                         'additionalProperties': False})
    fields = {'proposals': {'type': 'array', 'maxItems': 6, 'items': {'oneOf': branches}}}
    if grouping_context is not None:
        fields['attempt_updates'] = attempt_schema(grouping_context, refs, candidate_evidence)
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def validate_proposals(text, refs, *, candidate_evidence=None, maximum=6):
    payload = _load_object(text)
    if (set(payload) != {'proposals'} or not isinstance(payload['proposals'], list)
            or (maximum is not None and len(payload['proposals']) > maximum)):
        raise ValueError('Expected at most six learning proposals')
    for row in payload['proposals']:
        if not isinstance(row, dict) or row.get('kind') not in KINDS:
            raise ValueError('Invalid learning proposal fields or kind')
        fields = set(FIELDS) | ({'training_example'} if row['kind'] == 'training_data' else set())
        if 'subject' in row:
            fields.add('subject')
            if row['subject'] not in SUBJECTS:
                raise ValueError('Invalid feedback proposal subject')
        if candidate_evidence is not None:
            fields.add('evaluation_claims')
        if set(row) != fields:
            raise ValueError('Invalid learning proposal fields')
        if row['kind'] == 'training_data':
            sample = row['training_example']
            if (not isinstance(sample, dict) or set(sample) != {'input', 'target', 'target_status'}
                    or sample['target_status'] != 'unverified'
                    or any(not isinstance(sample[k], str) or not sample[k].strip()
                           or len(sample[k]) > 4000 for k in ('input', 'target'))):
                raise ValueError('Training example must be a bounded, unverified input/target pair')
        for key, maximum in (('content', 2400), ('applicability', 1000), ('validation_plan', 1200)):
            if not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > maximum:
                raise ValueError(f'Invalid proposal {key}')
        sources = row['source_refs']
        if (not isinstance(sources, list) or not 1 <= len(sources) <= 16
                or any(not isinstance(ref, str) or ref not in refs for ref in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError('Proposal cites unknown or duplicate evidence')
        if candidate_evidence is not None:
            validate_claims(row, candidate_evidence)
    return payload['proposals']


def validate_publication(row):
    """Recheck saved output against admitted raw evidence before exposing it.

    Structured attribution is checked; free text is still an unverified model
    interpretation. Legacy candidate-related outputs require the new contract.
    """
    evidence = admitted_candidate_evidence(row)
    validate_proposals(encoded({'proposals': row['proposals']}), row['input']['source_refs'],
                       candidate_evidence=evidence, maximum=None)
    validate_attempt_updates(row.get('attempt_updates', []), row['input'].get('grouping_context', {}),
                             row['input']['source_refs'], evidence)
    if evidence is not None:
        for revision in row.get('attempt_revisions', ()):
            validate_claims(revision, evidence)
    return evidence


def checked_processor_result(result, reserved):
    """Validate product/accounting metadata without granting journal ownership."""
    fields = {'status', 'error', 'charged_calls', 'proposals', 'attempt_updates',
              'response', 'simulated', 'grouping_error', 'products', 'output', 'outputs'}
    error = None
    if not isinstance(result, dict):
        error = 'Feedback processor must return a result object'
    elif type(result.get('charged_calls')) is not int or not 0 <= result['charged_calls'] <= reserved:
        error = 'Invalid native workflow accounting; full reservation retained'
    elif result.keys() - fields:
        error = 'Feedback processor cannot overwrite host journal fields'
    elif result.get('status') not in ('completed', 'failed', 'input_too_large',
                                      'budget_exhausted', 'cancelled', 'interrupted'):
        error = 'Feedback processor must return a terminal status'
    elif 'simulated' in result and type(result['simulated']) is not bool:
        error = 'Feedback processor simulated flag must be a boolean'
    try:
        json_bytes(result)
    except (ValueError, TypeError, OverflowError):
        error = 'Feedback processor result must be finite JSON data'
    if error:
        return {'status': 'failed', 'error': error, 'charged_calls': reserved,
                'proposals': [], 'attempt_updates': []}
    value = deepcopy(result)
    value.setdefault('proposals', [])
    value.setdefault('attempt_updates', [])
    if value['status'] != 'completed':
        value.update(proposals=[], attempt_updates=[])
    return value


