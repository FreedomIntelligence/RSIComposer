"""Fixed paired development selection; target-model policy cannot edit this contract."""
import math

from . import aime
from .engine import digest

PROTOCOL = 'meta-rsi/aime-paired-release-v1'


def validate_release_options(options):
    if not isinstance(options, dict) or options.keys() - {'max_regression_fraction', 'min_gain'}:
        raise ValueError('Invalid paired release contract')
    defaults = {'max_regression_fraction': 1., 'min_gain': 0.}
    options = {**defaults, **options}
    if any(type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1 for n in options.values()):
        raise ValueError('Invalid paired release threshold')
    return options


def make_measurement(candidate, protocol, records):
    """Extract the only item-level signal used by the Python-owned selector."""
    if not aime.candidate_id(candidate) or not aime.sha256_text(protocol):
        raise ValueError('Unbound paired release measurement')
    items = {digest(row['id']): row['correct'] for row in records}
    if not items or len(items) != len(records) or any(type(status) is not bool for status in items.values()):
        raise ValueError('Invalid paired development outcome vector')
    return {'candidate': candidate, 'protocol_id': protocol, 'items': items}


def compare(before, after, options, *, equivalent_reconciliation=False):
    options = validate_release_options(options)
    if (not isinstance(before, dict) or not isinstance(after, dict) or any(
            set(x) != {'candidate', 'protocol_id', 'items'} or not aime.candidate_id(x['candidate']) or
            not aime.sha256_text(x['protocol_id']) or not isinstance(x['items'], dict) or not x['items'] or
            any(not aime.sha256_text(key) or type(value) is not bool for key, value in x['items'].items())
            for x in (before, after)) or before['protocol_id'] != after['protocol_id'] or
            before['items'].keys() != after['items'].keys() or before['candidate'] == after['candidate']):
        raise ValueError('Paired comparison requires distinct candidates on the same fixed development binding')
    count = len(before['items'])
    won = sum(after['items'][key] and not value for key, value in before['items'].items())
    lost = sum(value and not after['items'][key] for key, value in before['items'].items())
    delta = (won - lost) / count
    same_accuracy_simplification = (equivalent_reconciliation is True and delta == 0. and options['min_gain'] == 0.)
    reject = ('regression_limit' if lost / count > options['max_regression_fraction'] else
              'no_strict_development_gain' if delta <= options['min_gain'] and not same_accuracy_simplification else None)
    return {'protocol': PROTOCOL, 'protocol_id': before['protocol_id'],
            'incumbent': before['candidate'], 'candidate': after['candidate'],
            'before_sha256': digest(before), 'after_sha256': digest(after), 'contract_sha256': digest(options),
            'count': count, 'gained': won, 'regressed': lost, 'delta': delta,
            'eligible': reject is None, 'reason': reject,
            **({'basis': 'equivalent_reconciliation'} if same_accuracy_simplification else {})}
