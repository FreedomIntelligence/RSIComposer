"""Pure scheduling over verified train IDs, and a bounded typed counting witness.

The Lab owns admission and competence evidence. Neither a row's presence nor a
correct training completion alone establishes the independently already-held
lane. A monitoring ID is a target for a fresh retention check, not evidence of
retention, and it still receives its one training exposure in this plan.
"""
from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
import json
import re

from .aime_domain_policy import FAMILY, canonical_task


CURRICULUM_RULE = 'aime-verified-lanes-v1'
BATCH_SIZE = 16
FRONTIER = 'frontier'
RECOVERED = 'recovered'
ALREADY_HELD = 'already_held'
REPLAY = 'replay'
_LANES = (FRONTIER, RECOVERED, ALREADY_HELD, REPLAY)

SUBSET_RESIDUE_CERTIFICATE_RULE = 'aime-subset-residue-dp-histogram-v1'
SUBSET_RESIDUE_CERTIFICATE_INSTRUCTION = (
    'For a subset-residue question, return exactly two lines. First write "Residue histogram: " '
    'followed by a JSON array of counts for every residue from 0 through modulus minus 1, '
    'in that order, using the subset size in the question. On the second line write only '
    'the requested count in one \\boxed{...}.'
)
_CERTIFICATE_FIELDS = {'rule', 'spec_sha256', 'histogram'}
_COMPLETION = re.compile(r'Residue histogram: (\[[0-9, \t]+\])\r?\n\\boxed\{([0-9]{1,3})\}')
_MAX_WITNESS_CHARS = 512  # At most 31 residue counts of at most C(14, 7) each.
_TASK_ID_PREFIX = f'aime-{FAMILY}-'
_TASK_ID_SUFFIX = re.compile(r'((?:[0-9a-f]{2}){11,17})-([0-9a-f]{64})')


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return sha256(_canonical_json(value).encode('ascii')).hexdigest()


def plan_curriculum(
    row_lanes: Mapping[str, str], *, seed: int, round_number: int, min_retention: int = 1,
) -> dict:
    """Return the exact single-pass ordering and private retention monitor IDs.

    Rank verified IDs independently inside each lane using a stable seeded
    hash. Spread each lane proportionally across the fit using its normalized
    midpoint positions; resolve ties in frontier, recovered, already-held,
    replay order. The last global batch is short when needed: no repeat/drop.

    Pick exactly the requested minimum number of monitors from the already-held
    lane with a separate hash domain. The caller must explicitly lower the
    quota (e.g. to zero for a no-held control) or supply more independent held
    evidence if the minimum cannot be met. Replay is coverage-only and never
    upgrades unclassified verified rows to already held.
    """
    if (not isinstance(row_lanes, Mapping)
            or any(type(value) is not int or not 0 <= value < 2**63 for value in (seed, round_number))
            or type(min_retention) is not int or min_retention < 0):
        raise ValueError('Invalid AIME curriculum lanes, seed, round, or retention minimum')
    by_lane = {lane: [] for lane in _LANES}
    validated_lanes = {}
    for ident, lane in row_lanes.items():
        if (type(ident) is not str or not ident.strip() or ident in validated_lanes
                or type(lane) is not str or lane not in by_lane):
            raise ValueError('Invalid or duplicate verified AIME curriculum ID or lane')
        by_lane[lane].append(ident)
        validated_lanes[ident] = lane
    held = by_lane[ALREADY_HELD]
    if min_retention > len(held):
        raise ValueError('AIME curriculum needs more independently already-held rows for the retention minimum')

    def ranked(ids, purpose):
        return sorted(ids, key=lambda ident: (_digest([CURRICULUM_RULE, purpose, seed, round_number, ident]), ident))

    positions = []
    for priority, (lane, ids) in enumerate(by_lane.items()):
        for index, ident in enumerate(ranked(ids, f'train:{lane}')):
            positions.append((Fraction(2 * index + 1, 2 * len(ids)), priority, ident))
    ordered = [ident for _, _, ident in sorted(positions)]
    return {
        'rule': CURRICULUM_RULE, 'batch_size': BATCH_SIZE, 'seed': seed, 'round_number': round_number,
        'row_lanes_sha256': _digest(validated_lanes), 'min_retention': min_retention,
        'ordered_row_ids': ordered,
        'batches': [ordered[i:i + BATCH_SIZE] for i in range(0, len(ordered), BATCH_SIZE)],
        'retention_monitor_ids': ranked(held, 'retention-monitor')[:min_retention],
    }


def check_curriculum(
    plan: dict, row_lanes: Mapping[str, str], *, seed: int, round_number: int, min_retention: int = 1,
) -> None:
    """Reject any persisted plan not exactly reproduced from the verified lanes."""
    expected = plan_curriculum(row_lanes, seed=seed, round_number=round_number, min_retention=min_retention)
    if (type(plan) is not dict or plan != expected
            or any(type(plan[key]) is not type(value) for key, value in expected.items())):
        raise ValueError('AIME curriculum plan differs from the verified lane schedule')


def subset_residue_spec_from_id(row_id: str) -> dict | None:
    """Decode a canonical typed train ID without running the reference enumerator.

    An unrelated ID has no typed spec; a malformed ID in this version's typed
    namespace fails closed, including an alias or changed digest.
    """
    if not isinstance(row_id, str) or not row_id.startswith(_TASK_ID_PREFIX):
        return None
    match = _TASK_ID_SUFFIX.fullmatch(row_id[len(_TASK_ID_PREFIX):])
    if match is None:
        raise ValueError('Invalid encoded AIME subset-residue identity')
    size, modulus, residue, *values = bytes.fromhex(match[1])
    spec = {'family': FAMILY, 'values': values, 'size': size, 'modulus': modulus, 'residue': residue}
    if canonical_task(spec) != spec or _digest(spec) != match[2]:
        raise ValueError('AIME subset-residue identity is not canonical or its digest changed')
    return spec


def _subset_residue_histogram(task):
    # dp[j][r] counts j-element subsets of the values processed so far with
    # sum r modulo modulus. Descending cardinality uses each new value once.
    size, modulus = task['size'], task['modulus']
    dp = [[0] * modulus for _ in range(size + 1)]
    dp[0][0] = 1
    for index, value in enumerate(task['values'], 1):
        for count in range(min(index, size), 0, -1):
            destination, source = dp[count], dp[count - 1]
            for residue, ways in enumerate(source):
                destination[(residue + value) % modulus] += ways
    return dp[size]


def _certificate(task, histogram):
    return {'rule': SUBSET_RESIDUE_CERTIFICATE_RULE, 'spec_sha256': _digest(task), 'histogram': histogram}


def derive_subset_residue_certificate(spec: dict) -> dict:
    """Derive the histogram on the canonical prompt parameters, without enumeration."""
    task = canonical_task(spec)
    return _certificate(task, _subset_residue_histogram(task))


def _check_certificate(task, certificate):
    if (type(certificate) is not dict or certificate.keys() != _CERTIFICATE_FIELDS
            or type(certificate['rule']) is not str or certificate['rule'] != SUBSET_RESIDUE_CERTIFICATE_RULE
            or type(certificate['spec_sha256']) is not str or certificate['spec_sha256'] != _digest(task)):
        raise ValueError('Invalid AIME subset-residue certificate rule or task identity')
    histogram = certificate['histogram']
    if (type(histogram) is not list or len(histogram) != task['modulus']
            or any(type(count) is not int or count < 0 for count in histogram)
            or histogram != _subset_residue_histogram(task)):
        raise ValueError('AIME subset-residue certificate histogram does not match the recomputed DP')
    return histogram[task['residue']]


def check_subset_residue_certificate(spec: dict, certificate: dict) -> int:
    """Recompute and check every residue count, returning the requested count.

    Only this typed mathematical assertion is checked, not freeform reasoning.
    """
    return _check_certificate(canonical_task(spec), certificate)


def render_subset_residue_completion(spec: dict, certificate: dict) -> str:
    """Render a checked Lab-side witness in the terminal-box answer format."""
    count = check_subset_residue_certificate(spec, certificate)
    if not 0 <= count <= 999:
        raise ValueError('Subset-residue certificate answer does not fit the AIME format')
    return f"Residue histogram: {_canonical_json(certificate['histogram'])}\n\\boxed{{{count:03d}}}"


def check_subset_residue_completion(spec: dict, completion: str) -> dict:
    """Check a two-line witness for the canonical prompt; return its bound certificate.

    A correct box is insufficient. This deliberately does not interpret a model's
    preceding prose, hidden reasoning, Markdown, or an answer-only response.
    """
    if type(completion) is not str or len(completion) > _MAX_WITNESS_CHARS:
        raise ValueError('Invalid AIME subset-residue completion length or type')
    match = _COMPLETION.fullmatch(completion.strip())
    if match is None:
        raise ValueError('AIME subset-residue completion must be a two-line histogram and terminal box')
    try:
        histogram = json.loads(match[1])
    except ValueError:
        raise ValueError('Invalid JSON histogram in AIME subset-residue completion') from None
    task = canonical_task(spec)
    certificate = _certificate(task, histogram)
    if int(match[2]) != _check_certificate(task, certificate):
        raise ValueError('AIME subset-residue completion box disagrees with its checked histogram')
    return certificate
