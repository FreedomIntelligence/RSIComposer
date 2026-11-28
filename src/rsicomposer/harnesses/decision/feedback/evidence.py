"""Candidate-bound evaluation evidence, independent of generated prose.

This checks structured attribution, not the semantics of an interpretation's
free text. A citation to training, a decision, or a prior interpretation cannot
establish evaluation metrics for a candidate.
"""
from copy import deepcopy
from math import isfinite

from rsicomposer.loops.contracts.receipts import receipt_task
from rsicomposer.runtime.storage.json_tools import result_id


def candidate_evidence(raw, indices):
    """Build from immutable receipts, using only admitted evaluation records.

    Absence is explicitly scoped to the captured evaluation snapshot. An
    omitted evaluation is distinguished from one absent from that snapshot.
    Neither status claims anything about an artifact's unread contents.
    """
    receipt = raw['receipt']
    task = receipt_task(receipt)
    current = []
    records = raw.get('evaluations', ())
    sources = []
    for index in indices:
        record = records[index]
        sources.append({**deepcopy(record), 'source_ref': 'evaluation:' + result_id(record),
                        'raw_ref': '/evaluations/' + str(index)})
    # Only the delivering Host marks a measured operation. The same fields in
    # an ordinary tool result or in a Lab's opaque native output carry no claim.
    if task.get('measured_evaluation') is True:
        data = receipt['result']['data']
        current.append(data['candidate_id'])
        sources.append({'candidate_id': data['candidate_id'], 'call_id': receipt['operation']['call_id'],
            'operation': receipt['operation']['name'], 'protocol_ref': data['protocol_ref'],
            'input_refs': list(data['input_refs']), 'status': receipt['result']['status'],
            'quality_metrics': deepcopy(data['quality_metrics']), 'simulated': task['simulated'],
            'source_ref': 'feedback:' + result_id(receipt), 'raw_ref': '/receipt'})
    if not current and not records and not sources:
        return None
    known = {r['candidate_id'] for r in records}
    known.update(s['candidate_id'] for s in sources)
    candidates = []
    for ident in dict.fromkeys([*current, *(s['candidate_id'] for s in sources)]):
        refs = list(dict.fromkeys(s['source_ref'] for s in sources if s['candidate_id'] == ident))
        candidates.append({'candidate_id': ident, 'evaluation_refs': refs,
            'evaluation_coverage': 'supplied' if refs else (
                'not_supplied' if ident in known else 'not_observed_in_snapshot')})
    return {'version': 1, 'scope': 'captured_evaluation_snapshot',
            'receipt_mode': task['mode'], 'current_candidate_ids': current,
            'candidates': candidates, 'evaluations': sources}


def claim_schema(evidence):
    if not evidence['evaluations']:
        return {'type': 'array', 'maxItems': 0, 'items': {'type': 'object',
                'properties': {}, 'required': [], 'additionalProperties': False}}
    fields = {
        'candidate_id': {'type': 'string'},
        'evaluation_ref': {'type': 'string'},
        'protocol_ref': {'type': 'string'},
        'metrics': {'type': 'object', 'minProperties': 1, 'additionalProperties': {'type': 'number'}},
        'simulated': {'type': 'boolean'},
    }
    return {'type': 'array', 'maxItems': 16,
            'items': {'type': 'object', 'properties': fields,
                      'required': list(fields), 'additionalProperties': False}}


def validate_claims(item, evidence):
    """Validate exact source/subject/protocol/metric bindings, never NLP guesses.

    Empty claims are valid for hypotheses or non-evaluation observations. They
    do not verify free text or certify that all prose claims were enumerated.
    """
    claims = item.get('evaluation_claims')
    if not isinstance(claims, list) or len(claims) > 16:
        raise ValueError('Candidate evidence requires structured evaluation_claims (possibly empty)')
    sources = {s['source_ref']: s for s in evidence['evaluations']}
    seen = set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {
                'candidate_id', 'evaluation_ref', 'protocol_ref', 'metrics', 'simulated'}:
            raise ValueError('Invalid evaluation claim fields')
        ref = claim['evaluation_ref']
        if not isinstance(ref, str) or ref not in sources or ref not in item['source_refs']:
            raise ValueError('Evaluation claim must cite its supplied evaluation source')
        if ref in seen:
            raise ValueError('Duplicate evaluation claim source; combine its metrics')
        seen.add(ref)
        record = sources[ref]
        if (claim['candidate_id'] != record['candidate_id']
                or claim['protocol_ref'] != record['protocol_ref']
                or type(claim['simulated']) is not bool or claim['simulated'] != record['simulated']
                or record['status'] != 'completed'):
            raise ValueError('Evaluation claim candidate, protocol, simulation or status mismatch')
        metrics = claim['metrics']
        if (not isinstance(metrics, dict) or not metrics
                or any(type(value) not in (int, float) or (type(value) is float and not isfinite(value))
                       or key not in record['quality_metrics']
                       or type(record['quality_metrics'][key]) not in (int, float)
                       or value != record['quality_metrics'][key] for key, value in metrics.items())):
            raise ValueError('Evaluation claim metrics differ from the cited candidate receipt')


def admitted_candidate_evidence(row):
    """Rebuild at acceptance/publication rather than trusting a saved catalog."""
    packet = row['input']
    raw = row['raw_input']
    if result_id(raw['receipt']) != row['receipt_digest']:
        raise ValueError('Captured receipt differs from its journal digest')
    stream = packet.get('evidence_packaging', {}).get('streams', {}).get('/evaluations', {})
    indices = stream.get('supplied_indices', list(range(len(packet.get('evaluations', ())))))
    if (not isinstance(indices, list) or any(type(i) is not int or i < 0
            or i >= len(raw.get('evaluations', ())) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('Invalid admitted evaluation indices')
    evidence = candidate_evidence(raw, indices)
    if result_id(packet.get('evaluations', ())) != result_id([raw['evaluations'][i] for i in indices]):
        raise ValueError('Admitted evaluations differ from captured records')
    if result_id(evidence) != result_id(packet.get('candidate_evidence')):
        raise ValueError('Missing or changed candidate evidence requires fresh validation')
    if evidence is not None and any(s['source_ref'] not in packet['source_refs']
                                   for s in evidence['evaluations']):
        raise ValueError('Candidate evaluation source is outside admitted references')
    return evidence
