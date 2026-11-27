"""Label-free, state-bound Diagnose contract for the opt-in AIME kernel.

The Host supplies already-observed AIME verifier outcomes from its pinned
cohorts. This module projects those outcomes and validates one reply; it does
not call a model, load a task, execute an operator, or authorize a write. The
Lab must use the pinned target for the Diagnose call and enforce protected
checks independently of anything the target says.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import re

from rsicomposer.runtime.storage.files import json_bytes

from .kernel_evidence import Operator, WriteAction, operator_card


PROTOCOL = 'meta-rsi/aime-diagnosis/v1'
MAX_HYPOTHESES = 3
MAX_RESPONSE_BYTES = 4096
_MAX_COUNT = (1 << 31) - 1
_SCOPES = ('train', 'held', 'search', 'validation')
_DEV_ROLES = ('search', 'validation')
_SHA256 = re.compile('[0-9a-f]{64}', re.ASCII)

_PURPOSES = {
    'data': 'Nominate what a bounded batch of new independently verified training tasks should exercise.',
    'harness': 'Nominate what a prompt-only patch and matched replay should test.',
    'model': 'Nominate what a fresh adapter fit on eligible cumulative verified data should consolidate or retain.',
}
_FOCUS = {
    'data': {
        'format': ('valid_output_demonstrations',),
        'knowledge': ('new_verified_task_variants',),
        'reasoning': ('representative_derivation_coverage',),
        'verification': ('independently_checked_demonstrations',),
        'retention': ('preserve_verified_coverage',),
    },
    'harness': {
        'format': ('clarify_answer_format', 'reduce_prompt_pressure'),
        'knowledge': ('demonstrate_candidate_method',),
        'reasoning': ('surface_stepwise_strategy',),
        'verification': ('prompt_for_independent_check',),
        'retention': ('preserve_working_guidance',),
    },
    'model': {
        'format': ('protect_output_format_in_training',),
        'knowledge': ('fit_verified_cumulative_data',),
        'reasoning': ('consolidate_verified_derivations',),
        'verification': ('protect_verifier_checked_training',),
        'retention': ('favor_conservative_update',),
    },
}
_SIGNALS = {
    'format': ('format_failure', 'truncated'),
    'knowledge': ('verified_failure',),
    'reasoning': ('verified_failure',),
    'verification': ('verified_failure', 'verification_failure'),
    'retention': ('verified_held',),
}
_DEV_FIELDS = {'count', 'correct', 'invalid_format', 'truncated', 'prompt_overlength'}
_RECORD_STATUS = {
    'correct': (True, True, 'verified_held'),
    'incorrect': (True, False, 'verified_failure'),
    'invalid_format': (False, False, 'format_failure'),
    'truncated': (False, False, 'truncated'),
    'verification_failed': (False, False, 'verification_failure'),
}
_METRICS = ('verified_held', 'verified_failure', 'format_failure', 'knowledge_failure',
            'reasoning_failure', 'verification_failure', 'truncated', 'prompt_overlength')
_SYSTEM = (
    'Diagnose AIME observations as the target model; do not propose a patch yet. '
    'Return exactly one JSON object with request_sha256, hypotheses, protected_checks. '
    'Copy request_sha256 and protected_checks exactly. Hypotheses is a ranked array of '
    'zero to three distinct objects {"rank":1,"gap":"…","focus":"…","evidence":["scope.metric"]}; '
    'use consecutive ranks starting at 1. Choose each gap and focus from allowed_hypotheses, '
    'with one to three distinct references to positive observed counts from evidence. '
    'For format cite format_failure or truncated; for knowledge or reasoning cite verified_failure; '
    'for verification cite verified_failure or verification_failure; for retention cite verified_held. '
    'To reduce prompt pressure cite truncated. Null means unmeasured, not zero. '
    'verified_held means an observed correct check, not proven mastery; verified_failure means a parsed '
    'answer rejected by the verifier, not a finding of absent knowledge, faulty reasoning or inadequate checking. '
    'Treat these as hypotheses to probe, not causal conclusions or authority to change protected surfaces. '
    'If evidence is insufficient return an empty hypotheses array. No extra fields, prose, tasks, or answers.'
)


def _hash(value):
    return sha256(json_bytes(value)).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _count(value, maximum=_MAX_COUNT):
    return type(value) is int and 0 <= value <= maximum


def _dev_counts(raw):
    # The Lab checks source candidate/cohort identity before projecting this receipt.
    # Deliberately do not copy arbitrary fields, including IDs, rates, or nested data.
    if type(raw) is not dict or not _DEV_FIELDS <= raw.keys() or not _count(raw['count']) or not raw['count']:
        raise ValueError('Invalid AIME development diagnostic')
    n, correct, invalid, truncated, overlength = (raw[k] for k in (
        'count', 'correct', 'invalid_format', 'truncated', 'prompt_overlength'))
    if (not all(_count(x, n) for x in (correct, invalid, truncated, overlength))
            or correct + invalid + truncated > n or overlength > truncated):
        raise ValueError('Inconsistent AIME development diagnostic')
    # The existing AIME development receipt partitions finished evaluations into
    # correct, malformed, truncated, and (by subtraction) parsed-wrong. It does
    # not measure verification outages: the evaluation would fail instead.
    return dict(count=n, verified_held=correct, verified_failure=n - correct - invalid - truncated,
                format_failure=invalid, knowledge_failure=None, reasoning_failure=None,
                verification_failure=None, truncated=truncated, prompt_overlength=overlength)


def _record_counts(records):
    if type(records) not in (list, tuple) or not _count(len(records)):
        raise ValueError('Invalid AIME diagnosis records')
    if not records:
        return None
    counts = dict.fromkeys(_METRICS, 0)
    counts.update(knowledge_failure=None, reasoning_failure=None)
    for row in records:
        if type(row) is not dict or type(row.get('status')) is not str or row['status'] not in _RECORD_STATUS:
            raise ValueError('Invalid AIME diagnosis record status')
        parsed, correct, metric = _RECORD_STATUS[row['status']]
        if row.get('parsed') is not parsed or row.get('correct') is not correct:
            raise ValueError('Inconsistent AIME diagnosis record verdict')
        if row.get('finish_reason') == 'prompt_budget':
            if metric != 'truncated':
                raise ValueError('Inconsistent AIME diagnosis record finish')
            counts['prompt_overlength'] += 1
        counts[metric] += 1
    return {'count': len(records), **counts}


def project_evidence(*, development=None, train_records=(), held_records=()):
    """Project only fixed AIME aggregate counters from already-verified Host data.

    Pass existing aime.metrics receipts by role; a held cohort is optional and may
    contain either passing or failing replays of previously successful material.
    Record inputs use MathLab.record status/parsed/correct/finish_reason (or the
    explicit verification_failed status from Host instrumentation). No per-item
    fields are serialized. Knowledge and reasoning failures are unmeasured with
    these checks, even when a parsed answer was conclusively rejected.
    """
    if development is None:
        development = {}
    if type(development) is not dict or development.keys() - set(_DEV_ROLES):
        raise ValueError('Expected only search and validation diagnostics')
    return {'train': _record_counts(train_records), 'held': _record_counts(held_records),
            **{role: _dev_counts(development[role]) if role in development else None for role in _DEV_ROLES}}


def _safe_evidence(evidence):
    # A caller cannot bypass the projection by handing proposal_messages a raw
    # or augmented dict. Reconstruct a bounded primitive snapshot for hashing.
    if type(evidence) is not dict or evidence.keys() != set(_SCOPES):
        raise ValueError('Invalid AIME diagnosis evidence scopes')
    result = {}
    for scope, item in evidence.items():
        if item is None:
            result[scope] = None
            continue
        if (type(item) is not dict or item.keys() != {'count', *_METRICS}
                or not _count(item['count']) or not item['count']
                or item['knowledge_failure'] is not None or item['reasoning_failure'] is not None):
            raise ValueError('Invalid AIME diagnosis aggregate')
        count = item['count']
        partition = ('verified_held', 'verified_failure', 'format_failure', 'truncated')
        if scope in _DEV_ROLES:
            if item['verification_failure'] is not None:
                raise ValueError('AIME development does not instrument verification failure')
        else:
            partition += ('verification_failure',)
        if (not all(_count(item[key], count) for key in (*partition, 'prompt_overlength'))
                or sum(item[key] for key in partition) != count or item['prompt_overlength'] > item['truncated']):
            raise ValueError('Inconsistent AIME diagnosis aggregate')
        result[scope] = dict(item)
    return result


def _request(operator, state_sha256, evidence):
    card = operator_card(operator)
    if type(state_sha256) is not str or _SHA256.fullmatch(state_sha256) is None:
        raise ValueError('Invalid AIME diagnosis state identity')
    view = dict(protocol=PROTOCOL, operator=card.operator, purpose=_PURPOSES[card.operator], state_sha256=state_sha256,
                evidence=_safe_evidence(evidence), action=card.action, allowed_hypotheses=_FOCUS[card.operator],
                protected_checks=card.protected_surfaces)
    return {**view, 'request_sha256': _hash({'system': _SYSTEM, 'user': view})}


def proposal_messages(operator, *, state_sha256, evidence):
    """Build a fresh target-model Diagnose request bound to exact Host state."""
    view = _request(operator, state_sha256, evidence)
    return [{'role': 'system', 'content': _SYSTEM}, {'role': 'user', 'content': _json(view)}]


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A ranked typed probe direction, never a claim of a proven cause."""

    rank: int
    gap: str
    focus: str
    evidence: tuple[str, ...]

    def to_dict(self):
        return dict(rank=self.rank, gap=self.gap, focus=self.focus, evidence=list(self.evidence))


@dataclass(frozen=True, slots=True)
class Diagnosis:
    operator: Operator
    action: WriteAction
    state_sha256: str
    request_sha256: str
    hypotheses: tuple[Hypothesis, ...]
    protected_checks: tuple[str, ...]

    def to_dict(self):
        return dict(protocol=PROTOCOL, operator=self.operator, action=self.action, state_sha256=self.state_sha256,
                    request_sha256=self.request_sha256, hypotheses=[item.to_dict() for item in self.hypotheses],
                    protected_checks=list(self.protected_checks))

    @property
    def sha256(self):
        return _hash(self.to_dict())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate field')
        result[key] = value
    return result


def _reject_number(_):
    raise ValueError('Unexpected number')


def _rank_int(token):
    # Ranks are the only integers in a reply, and are single digit.
    if len(token) != 1:
        raise ValueError('Invalid rank')
    return int(token)


def _hypothesis(raw, rank, view):
    if (type(raw) is not dict or raw.keys() != {'rank', 'gap', 'focus', 'evidence'}
            or type(raw['rank']) is not int or raw['rank'] != rank):
        raise ValueError('Invalid AIME diagnosis hypothesis shape or rank')
    gap, focus, refs = (raw[key] for key in ('gap', 'focus', 'evidence'))
    allowed = view['allowed_hypotheses']
    if (type(gap) is not str or gap not in allowed or type(focus) is not str or focus not in allowed[gap]
            or type(refs) is not list or not 1 <= len(refs) <= 3 or not all(type(ref) is str for ref in refs)
            or len(set(refs)) != len(refs)):
        raise ValueError('Invalid AIME diagnosis hypothesis fields')
    signals = ('truncated',) if focus == 'reduce_prompt_pressure' else _SIGNALS[gap]
    observed = {f'{scope}.{metric}' for scope, item in view['evidence'].items() if item is not None
                for metric in signals if item[metric] is not None and item[metric] > 0}
    if not observed.issuperset(refs):
        raise ValueError('AIME diagnosis referenced unavailable or unrelated evidence')
    return Hypothesis(rank, gap, focus, tuple(refs))


def parse(operator, text, *, state_sha256, evidence):
    """Validate one strict JSON reply; do not grant or verify any capability."""
    view = _request(operator, state_sha256, evidence)
    if type(text) is not str or not 0 < len(text) <= MAX_RESPONSE_BYTES:
        raise ValueError('Invalid AIME diagnosis response length or type')
    try:
        if len(text.encode('utf-8')) > MAX_RESPONSE_BYTES:
            raise ValueError('Oversized response')
        raw = json.loads(text, object_pairs_hook=_unique_object, parse_int=_rank_int,
                         parse_float=_reject_number, parse_constant=_reject_number)
    except (ValueError, RecursionError):
        raise ValueError('AIME diagnosis must be one strict bounded JSON object') from None
    if (type(raw) is not dict or raw.keys() != {'request_sha256', 'hypotheses', 'protected_checks'}
            or raw['request_sha256'] != view['request_sha256']
            or raw['protected_checks'] != list(view['protected_checks'])
            or type(raw['hypotheses']) is not list or len(raw['hypotheses']) > MAX_HYPOTHESES):
        raise ValueError('AIME diagnosis response does not match its request or protected checks')
    hypotheses = tuple(_hypothesis(item, i, view) for i, item in enumerate(raw['hypotheses'], start=1))
    if len({(item.gap, item.focus) for item in hypotheses}) != len(hypotheses):
        raise ValueError('Duplicate AIME diagnosis hypothesis')
    return Diagnosis(operator, view['action'], state_sha256, view['request_sha256'], hypotheses, view['protected_checks'])
