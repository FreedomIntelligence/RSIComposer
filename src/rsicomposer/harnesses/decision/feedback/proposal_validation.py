"""Validate feedback proposals and accounting before journal publication."""
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

from .attempts import validate_attempt_updates
from .evidence import admitted_candidate_evidence, claim_schema, validate_claims
from rsicomposer.runtime.storage.files import json_bytes
from rsicomposer.runtime.storage.json_tools import encoded

FIELDS = ('kind', 'subject', 'content', 'applicability', 'source_refs', 'validation_plan')


@dataclass(frozen=True, slots=True)
class ProposalTarget:
    """A trusted proposer declaration, not an execution or deployment capability."""
    kind: str
    target: str  # Query category; subject names the component to which a row is attributed.
    subjects: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.subjects, tuple) or not self.subjects:
            raise ValueError('Proposal targets need a nonempty tuple of subjects')
        for name in (self.kind, self.target, *self.subjects):
            if not isinstance(name, str) or not name.strip() or name != name.strip() or len(name) > 128:
                raise ValueError('Proposal kinds, targets and subjects must be bounded nonempty names')
        if 'unspecified' in self.subjects or len(set(self.subjects)) != len(self.subjects):
            raise ValueError('Proposal subjects must be explicit and distinct')

    def configuration(self):
        return {'kind': self.kind, 'target': self.target, 'subjects': list(self.subjects)}


HARNESS_CHANGE = ProposalTarget('harness_change', 'harness', ('meta_agent', 'lab'))
POLICY_TRAINING = ProposalTarget('policy_training', 'weights', ('meta_agent',))
BUILTIN_IMPROVEMENTS = (HARNESS_CHANGE, POLICY_TRAINING)


def bind_targets(declared):
    """Freeze the installed processor's target rules; never read declarations from a result."""
    if not isinstance(declared, (tuple, list)) or not declared:
        raise ValueError('An installed proposer must declare nonempty proposal_targets')
    by_kind = {}
    for rule in declared:
        if not isinstance(rule, ProposalTarget):
            raise TypeError('proposal_targets must contain ProposalTarget declarations')
        if rule.kind in by_kind:
            raise ValueError('Duplicate proposal kind: ' + rule.kind)
        if rule.kind in ('training_data', 'memory', 'system_change'):
            raise ValueError('Memory, training records and system edits are not improvement task proposals')
        for builtin in BUILTIN_IMPROVEMENTS:
            if rule.kind == builtin.kind and (rule.target != builtin.target
                    or not set(rule.subjects) <= set(builtin.subjects)):
                raise ValueError('Cannot redefine a built-in improvement target: ' + rule.kind)
        by_kind[rule.kind] = rule
    return MappingProxyType(by_kind)


def proposal_schema(refs, *, targets, candidate_evidence=None):
    properties = {
        'content': {'type': 'string', 'minLength': 1, 'maxLength': 2400},
        'applicability': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'source_refs': {'type': 'array', 'minItems': 1, 'maxItems': 16,
                        'items': {'enum': list(refs)}},
        'validation_plan': {'type': 'string', 'minLength': 1, 'maxLength': 1200},
    }
    if candidate_evidence is not None:
        properties['evaluation_claims'] = claim_schema(candidate_evidence)
    branches = []
    for rule in targets.values():
        fields = {**properties, 'kind': {'const': rule.kind}, 'subject': {'enum': list(rule.subjects)}}
        branches.append({'type': 'object', 'properties': fields, 'required': list(fields),
                         'additionalProperties': False})
    fields = {'proposals': {'type': 'array', 'maxItems': 6, 'items': {'oneOf': branches}}}
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def validate_proposals(payload, refs, *, targets, candidate_evidence=None):
    if (not isinstance(payload, dict) or set(payload) != {'proposals'} or not isinstance(payload['proposals'], list)
            or len(payload['proposals']) > 6):
        raise ValueError('Expected at most six learning proposals')
    for row in payload['proposals']:
        if not isinstance(row, dict) or not isinstance(row.get('kind'), str) or row['kind'] not in targets:
            raise ValueError('Invalid learning proposal fields or kind')
        fields = set(FIELDS)
        if row.get('subject') not in targets[row['kind']].subjects:
            raise ValueError('Invalid feedback proposal subject for ' + row['kind'])
        if candidate_evidence is not None:
            fields.add('evaluation_claims')
        if set(row) != fields:
            raise ValueError('Invalid learning proposal fields')
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


def validate_publication(row, targets):
    """Recheck saved output against admitted raw evidence before exposing it.

    Structured attribution is checked; free text is still an unverified model
    interpretation. Target rules are pinned by the installed method, not the saved output.
    """
    evidence = admitted_candidate_evidence(row)
    product = {'proposals': row['proposals']}
    validate_proposals(product, row['input']['source_refs'], candidate_evidence=evidence, targets=targets)
    if encoded(row.get('outputs', {}).get('proposals', {'proposals': []})) != encoded(product):
        raise ValueError('Published proposals differ from the processor output')
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
