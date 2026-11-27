"""Bounded, replayable policy choices for a multi-term MetaRSI experiment.

A term is a task-improvement budget, not one LoRA optimizer step. Vertical
experiments change one operator's proposal profile; a scheduler experiment
reorders eligible operators without changing their policies or work horizon.
"""
from copy import deepcopy
import math


CROSS_TERM_TARGETS = ('schedule', 'controller_harness')
VERTICAL_TARGETS = {'data_policy': 'data', 'harness_policy': 'harness', 'model_policy': 'model'}
DIAGNOSES = ('format_gap', 'training_ready', 'reasoning_gap', 'solved', 'no_signal')


def campaign_config(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) - {'terms', 'track', 'routing', 'protocol', 'policy_seed'}:
        raise ValueError('Unknown MetaRSI campaign options')
    terms = config.get('terms', 1)
    track = config.get('track', 'adaptive')
    routing = config.get('routing', 'ordered')
    protocol = config.get('protocol', 'aime')
    seed = config.get('policy_seed', 23)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Campaign policy_seed must be an unsigned 32-bit integer')
    if type(terms) is not int or not 1 <= terms <= 8:
        raise ValueError('MetaRSI campaigns need an integer term count between 1 and 8')
    if (track not in ('adaptive', 'frozen', 'trial', 'reference')
            or routing not in ('ordered', 'diagnostic', 'uniform') or protocol != 'aime'):
        raise ValueError('Unsupported MetaRSI campaign track, routing or protocol')
    if track in ('trial', 'reference') and terms != 1:
        raise ValueError('Nested campaign trials and the reference run exactly one term')
    return {'terms': terms, 'track': track, 'routing': routing, 'protocol': protocol, 'policy_seed': seed}


def diagnose(signal, available):
    if signal is None:
        return 'no_signal'
    count, accuracy, invalid = (signal[k] for k in ('count', 'accuracy', 'invalid_format'))
    if (type(count) is not int or count < 1 or type(invalid) is not int or not 0 <= invalid <= count
            or type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1):
        raise ValueError('Invalid candidate diagnosis')
    if accuracy == 1:
        return 'solved'
    if invalid * 2 >= count:
        return 'format_gap'
    if 'model' in available:
        return 'training_ready'
    return 'reasoning_gap'


def route(eligible, basis):
    """Stable rank: selected evidence can change priority, never admissibility."""
    priority = {'format_gap': 'harness', 'training_ready': 'model', 'reasoning_gap': 'data'}.get(basis)
    return sorted(eligible, key=lambda operator: operator != priority) if priority else list(eligible)


def change_order(order, available, attempts):
    """Build one deterministic horizontal challenger from observed receipts."""
    pool = [op for op in order if op in available]
    if len(pool) < 2:
        return None
    # Try a previously productive or less recently attempted eligible operator
    # before the incumbent lead. No reward from a protected test is available.
    history = {op: [] for op in pool}
    for index, row in enumerate(attempts):
        if row['operator'] in history:
            history[row['operator']].append((index, row))

    def preference(op):
        rows = history[op]
        gain = max((r.get('gain', 0.) for _, r in rows if r.get('selected')), default=0.)
        return (-max(0., gain), rows[-1][0] if rows else -1, pool.index(op))

    lead = min(pool[1:], key=preference)
    return [lead, *(op for op in order if op != lead)]


def vertical_patch(profiles, operator):
    """A declarative, one-field policy rewrite. The contract is outside the diff."""
    if operator not in VERTICAL_TARGETS.values():
        raise ValueError('Unknown vertical target')
    incumbent = profiles.get(operator, 'compact')
    challenger = 'detailed' if incumbent == 'compact' else 'compact'
    return {**deepcopy(profiles), operator: challenger}, {
        'contract': 'meta-rsi-proposal-policy/aime-v1',
        'surface': f'operators.{operator}.proposal_profile',
        'before': incumbent, 'after': challenger,
    }
