"""Read authoritative nested Lab receipts, rather than Controller self-reports."""
import json
import math

from rsicomposer.api import ArtifactRef


def trial_score(store, receipt, *, version_id, candidate_ref, allow_simulated, expected_steps, start_id=None):
    data = receipt.data
    terminal = data.get('validation', {})
    if (terminal.get('status') != 'finished' or terminal.get('shutdown_complete') is not True
            or terminal.get('stop_reason') != 'researcher_finished'
            or data.get('candidate_ref') != candidate_ref or data.get('parent_call_id') != receipt.call_id):
        raise ValueError('Child did not finish with matched candidate evidence')
    budget = terminal.get('allocated_budget')
    if (not isinstance(budget, dict) or 'seconds' not in budget
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in budget.values())):
        raise ValueError('Child did not report a usable allocated budget')
    ref = ArtifactRef(**data['evidence_ref'])
    if ref.kind != 'nested-run-evidence':
        raise ValueError('Unexpected nested evidence kind')
    ledger = json.loads(store.read(ref))
    if ledger['parent_call_id'] != receipt.call_id or ledger['child_run_id'] != data['child_run_id']:
        raise ValueError('Nested evidence identity mismatch')
    selected, seed, scores, attempts = [], [], {}, []
    mode = None
    for event in ledger['events']:
        req = event.get('request') or {}
        name = req.get('name')
        if event.get('source') != 'lab' or name not in ('math.seed', 'math.propose', 'math.select', 'math.evaluate'):
            continue
        output = event.get('output') or {}
        args, payload = req.get('arguments', {}), output.get('data', {})
        if req.get('call_id') != output.get('call_id') or event.get('identity', {}).get(
                'issuing_controller_version') != version_id or output.get('status') != 'completed':
            raise ValueError('Unexpected issuing Controller or unsuccessful child operation')
        is_simulated = output.get('simulated')
        if type(is_simulated) is not bool or (is_simulated and not allow_simulated):
            raise ValueError('Child evidence simulation flag does not match experiment policy')
        if mode is None:
            mode = is_simulated
        elif is_simulated != mode:
            raise ValueError('Mixed real and simulated receipts')
        if name == 'math.seed':
            seed.append(payload['candidate_id'])
        elif name == 'math.select':
            if payload['candidate_id'] != args.get('candidate'):
                raise ValueError('Child selection identity mismatch')
            selected.append(payload['candidate_id'])
        elif name == 'math.propose':
            attempts.append((args['operator'], args['profile']))
        elif args.get('split') == 'validation':
            key, value, count = payload['candidate_id'], payload['score'], payload['count']
            protocol = payload.get('protocol_id')
            quality = payload.get('quality_metrics', {})
            if (key != args['candidate'] or type(value) not in (float, int) or not math.isfinite(value)
                    or not 0 <= value <= 1 or type(count) is not int or count < 1
                    or not isinstance(protocol, str) or not protocol or payload.get('scored') is not True
                    or quality.get('accuracy') != value or type(quality.get('correct')) is not int
                    or quality['correct'] / count != value):
                raise ValueError('Unusable development score')
            point = {'score': value, 'count': count, 'protocol': protocol}
            # Repeated observations may legitimately vary for a real model. Use
            # the initial observation per immutable candidate, like the policy.
            scores.setdefault(key, point)
    if (len(seed) != 1 or len(selected) != 1 or len(attempts) != expected_steps
            or (start_id is not None and seed[0] != start_id)
            or seed[0] not in scores or selected[0] not in scores):
        raise ValueError('Child did not complete the declared matched experiment')
    base, winner = scores[seed[0]], scores[selected[0]]
    if (base['count'], base['protocol']) != (winner['count'], winner['protocol']):
        raise ValueError('Within-trial protocols differ')
    return {'baseline': base['score'], 'selected_score': winner['score'], 'delta': winner['score'] - base['score'],
            'start_id': seed[0], 'selected_id': selected[0], 'count': base['count'], 'protocol': base['protocol'],
            'attempts': [list(x) for x in attempts], 'simulated': mode, 'evidence_ref': data['evidence_ref'],
            'allocated_budget': budget, 'budget_used': data['budget_used']}


def comparable(a, b):
    return all(a[key] == b[key] for key in ('start_id', 'baseline', 'count', 'protocol', 'simulated',
                                             'allocated_budget'))
