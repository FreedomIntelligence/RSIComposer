"""Typed, label-free proposal boundary for the public AIME reconstruction.

The model may choose task parameters, a prompt preset and verified exemplar
pins with optional advisory text, or an approved learning-rate scale. It
cannot write a question, label, verifier, executable patch, or base/training
configuration. Only trusted code materializes and grades tasks. Keep the
resulting Problem inside the Lab; a model call receives only its prompt,
never the Problem or the provenance of an individual evaluation result.
"""
from hashlib import sha256
from itertools import combinations
import json
from math import gcd, isfinite
import re

from rsicomposer.labs.math.tasks import Problem
from rsicomposer.labs.math.verifiers import CompetitionIntegerVerifier


PROTOCOL = 'meta-rsi/aime-domain-proposal/v1'
FAMILY = 'subset-residue-v1'
REFERENCE_SOLVER = 'aime-subset-residue-exhaustive-v1'
MAX_RESPONSE_CHARS = 8192
MAX_ELIGIBLE_EXEMPLARS = 32
MAX_GUIDANCE_BYTES = 512
_ID_PREFIX = 'aime-' + FAMILY + '-'
_ID_SUFFIX = re.compile(r'([0-9a-f]{22,34})-([0-9a-f]{64})', re.ASCII)
_SHA256 = re.compile(r'[0-9a-f]{64}', re.ASCII)
_PROFILES = ('compact', 'detailed')
_OPERATORS = ('data', 'harness', 'model')
_SCALES = {'base': 1.0, 'half': 0.5}
_TASK_FIELDS = {'family', 'values', 'size', 'modulus', 'residue'}
_HARNESS_FIELDS = {'instruction', 'exemplars'}
_CHAT_MARKERS = ('<|', '|>', '[inst]', '[/inst]', '<<sys>>', '<</sys>>', '<s>', '</s>', '<think>', '</think>')
_AGGREGATE_FIELDS = {'count', 'correct', 'invalid_format', 'truncated', 'prompt_overlength'}
_COUNTER_FIELDS = {'round', 'data_count', 'remaining_capacity'}
_FLAG_FIELDS = {'data_fresh', 'has_adapter'}


def _canonical_bytes(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def _bounded_int(value, lo, hi):
    return type(value) is int and lo <= value <= hi


def _shape(value, fields):
    return type(value) is dict and value.keys() == fields


def canonical_task(spec):
    """Return canonical parameters, eliminating permutation/affine/complement aliases.

    Inputs are bounded integers; only distinct residue classes are supported.
    Every invertible affine transform of the values and target describes an
    equivalent subset-count problem. For equal-sized complements, their two
    targets are also equivalent. Canonical representatives remove these aliases
    without depending on a reference label, rollout, or fitted model.
    """
    if not _shape(spec, _TASK_FIELDS) or spec['family'] != FAMILY:
        raise ValueError('Unrecognized AIME task schema or family')
    values, size, modulus, residue = (spec[k] for k in ('values', 'size', 'modulus', 'residue'))
    if (type(values) is not list or not 8 <= len(values) <= 14
            or not all(_bounded_int(value, 0, 50) for value in values)
            or not _bounded_int(size, 3, len(values) - 3)
            or not _bounded_int(modulus, max(9, len(values)), 31)
            or not _bounded_int(residue, 0, modulus - 1)):
        raise ValueError('AIME subset task parameters are outside the bounded family')
    values = {value % modulus for value in values}
    if len(values) != len(spec['values']):
        raise ValueError('AIME subset task values must have distinct residues')
    if 2 * size > len(values):
        residue = (sum(values) - residue) % modulus
        size = len(values) - size
    equivalent_targets = (residue, (sum(values) - residue) % modulus) if 2 * size == len(values) else (residue,)

    # A lexicographic minimum always starts with zero. For each unit only
    # consider translations sending a particular member of the set to zero.
    canonical_values, canonical_residue = min(
        (tuple(sorted(unit * (value - origin) % modulus for value in values)),
         unit * (target - origin * size) % modulus)
        for unit in range(1, modulus) if gcd(unit, modulus) == 1
        for origin in values for target in equivalent_targets)
    return {'family': FAMILY, 'values': list(canonical_values), 'size': size,
            'modulus': modulus, 'residue': canonical_residue}


def _task_id(task):
    packed = bytes([task['size'], task['modulus'], task['residue'], *task['values']]).hex()
    return _ID_PREFIX + packed + '-' + sha256(_canonical_bytes(task)).hexdigest()


def materialize_task(spec):
    """Compile a model-authored spec to a new train-only task and safe provenance.

    The independent reference enumerates at most C(14, 7) subsets. The label
    is not part of the identity or provenance, including no bare hash of an
    easily brute-forced three-digit answer.
    """
    task = canonical_task(spec)
    values, size, modulus, residue = (task[k] for k in ('values', 'size', 'modulus', 'residue'))
    correct = sum(sum(subset) % modulus == residue for subset in combinations(values, size))
    if not 2 <= correct <= 999:
        raise ValueError('Degenerate or non-AIME bounded subset task')
    prompt = (f'Let S = {{{", ".join(map(str, values))}}}. How many {size}-element subsets of S have '
              f'a sum that leaves remainder {residue} when divided by {modulus}?')
    verifier = CompetitionIntegerVerifier(999)
    reference = f'{correct:03d}'
    problem = Problem(_task_id(task), prompt, reference, FAMILY, verifier, reference)
    return problem, {
        'family': FAMILY,
        'spec_sha256': sha256(_canonical_bytes(task)).hexdigest(),
        'prompt_sha256': sha256(prompt.encode('utf-8')).hexdigest(),
        'verifier_id': verifier.name,
        'reference_solver_id': REFERENCE_SOLVER,
    }


def recover_task(ident):
    """Recover a canonical synthetic train task from its immutable row ID."""
    if not isinstance(ident, str) or not ident.startswith(_ID_PREFIX):
        return None
    if len(ident) > len(_ID_PREFIX) + 34 + 1 + 64:
        raise ValueError('Invalid encoded AIME task identity')
    match = _ID_SUFFIX.fullmatch(ident[len(_ID_PREFIX):])
    if match is None or len(match[1]) % 2:
        raise ValueError('Invalid encoded AIME task identity')
    size, modulus, residue, *values = bytes.fromhex(match[1])
    spec = {'family': FAMILY, 'values': values, 'size': size, 'modulus': modulus, 'residue': residue}
    task = canonical_task(spec)
    if spec != task or _task_id(task) != ident:
        raise ValueError('AIME task identity is not canonical or its digest changed')
    return materialize_task(task)


def _exemplar_pins(eligible_exemplars, maximum):
    if (type(eligible_exemplars) not in (tuple, list) or len(eligible_exemplars) > MAX_ELIGIBLE_EXEMPLARS
            or not all(type(pin) is str and _SHA256.fullmatch(pin) for pin in eligible_exemplars)
            or len(set(eligible_exemplars)) != len(eligible_exemplars) or not _bounded_int(maximum, 0, 2)):
        raise ValueError('Invalid trusted AIME exemplar choices')
    return list(eligible_exemplars)


def check_guidance(value):
    """Validate advisory text, not its semantics or its effect on the model.

    This string remains untrusted even after validation. Apply only as a
    namespaced solver advisory and keep the output contract fixed. Block
    literal role/control markers recognized by common chat tokenizers.
    """
    if (type(value) is not str or len(value) > MAX_GUIDANCE_BYTES or not value.strip()
            or not value.isprintable() or len(value.encode('utf-8')) > MAX_GUIDANCE_BYTES):
        raise ValueError('Invalid AIME Harness guidance')
    folded = value.casefold()
    if any(marker in folded for marker in _CHAT_MARKERS):
        raise ValueError('AIME Harness guidance contains a reserved chat marker')
    return value


def _limits(operator, eligible_exemplars, max_exemplars, max_tasks):
    if operator not in _OPERATORS or not _bounded_int(max_tasks, 1, 16):
        raise ValueError('Invalid AIME proposal operator or task limit')
    return _exemplar_pins(eligible_exemplars, max_exemplars)


def _safe_context(context):
    # This is an allowlisted primitive projection, not a recursive redactor.
    if context is None:
        return {}
    if type(context) is not dict or context.keys() - (_COUNTER_FIELDS | _FLAG_FIELDS | {'search', 'validation'}):
        raise ValueError('Invalid AIME proposal context fields')
    projection = {}
    for key, value in context.items():
        if key in _COUNTER_FIELDS and _bounded_int(value, 0, (1 << 31) - 1):
            projection[key] = value
        elif key in _FLAG_FIELDS and type(value) is bool:
            projection[key] = value
        elif key in ('search', 'validation') and type(value) is dict and 'count' in value:
            count = value['count']
            if (_bounded_int(count, 1, (1 << 31) - 1) and value.keys() <= _AGGREGATE_FIELDS
                    and all(_bounded_int(item, 0, count) for item in value.values())):
                projection[key] = dict(value)
                continue
            raise ValueError('Invalid AIME proposal development aggregate')
        else:
            raise ValueError('Invalid AIME proposal context value')
    return projection


def proposal_messages(operator, *, profile, context=None, eligible_exemplars=(), max_exemplars=2, max_tasks=4):
    """Build a fresh model request using only trusted literals and safe aggregates."""
    pins = _limits(operator, eligible_exemplars, max_exemplars, max_tasks)
    if profile not in _PROFILES:
        raise ValueError('Invalid AIME proposal profile')
    view = {'protocol': PROTOCOL, 'operator': operator, 'profile': profile, 'context': _safe_context(context)}
    system = ('You propose a single bounded change to an AIME math-solving system. '
              'Return exactly one JSON object, with no Markdown, commentary, or additional fields. ')
    if operator == 'data':
        view['max_tasks'] = max_tasks
        system += ('Propose distinct new task specifications near the boundary of learnable counting skills. '
                   'Use {"tasks":[{"family":"subset-residue-v1","values":[array of integers],'
                   '"size":integer,"modulus":integer,"residue":integer}]}. Each values array must contain '
                   '8 through 14 integers in 0 through 50 with distinct residues modulo modulus; modulus '
                   'must be at least both 9 and the array length, and at most 31. Size is at least 3 and '
                   'at most the array length minus 3. Residue is at least 0 and less than modulus. '
                   'Avoid merely shifting, scaling or reordering the same instance. '
                   'Do not give solutions, answers or invented source IDs. You may return no tasks.')
    elif operator == 'harness':
        view.update(eligible_exemplars=pins, max_exemplars=max_exemplars)
        system += ('Choose one of two existing instructions and zero or more eligible exemplar pins, '
                   'ordered most useful first, up to max_exemplars. Return '
                   '{"instruction":"compact" or "detailed","exemplars":[exact eligible SHA-256 strings]}. '
                   'Compact asks the solver to identify the required operation; detailed asks it to '
                   'work out and check each step. You may also supply a "guidance" string: '
                   f'one printable line of math-solving strategy advice (up to {MAX_GUIDANCE_BYTES} UTF-8 bytes). '
                   'Do not change the output contract, cite any reference answers, or add role markers, '
                   'tools, or other fields.')
    else:
        system += ('Choose the learning rate for a fresh, assistant-only adapter fit on the current '
                   'verified cumulative training set. Return {"learning_rate_scale":"base" or "half"}. '
                   'No other training parameters can be proposed.')
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': _canonical_bytes(view).decode('ascii')}]


def _unique_object(items):
    data = {}
    for key, value in items:
        if key in data:
            raise ValueError('Duplicate JSON field')
        data[key] = value
    return data


def _reject_number(_):
    raise ValueError('Unexpected JSON number')


def _read_integer(token):
    if len(token) > 4:
        raise ValueError('Unbounded JSON integer')
    return int(token)


def parse_proposal(operator, text, *, eligible_exemplars=(), max_exemplars=2, max_tasks=4, allow_clear_guidance=False):
    """Strictly parse one untrusted reply to the operator's typed patch."""
    pins = _limits(operator, eligible_exemplars, max_exemplars, max_tasks)
    if not isinstance(text, str) or not 0 < len(text) <= MAX_RESPONSE_CHARS:
        raise ValueError('Invalid AIME proposal response length or type')
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_int=_read_integer,
                           parse_float=_reject_number, parse_constant=_reject_number)
    except (ValueError, RecursionError):
        raise ValueError('AIME proposal must be exactly one strict JSON object') from None
    if operator == 'data':
        if not _shape(value, {'tasks'}) or type(value['tasks']) is not list or len(value['tasks']) > max_tasks:
            raise ValueError('Invalid AIME Data proposal')
        tasks = [canonical_task(task) for task in value['tasks']]
        if len({_task_id(task) for task in tasks}) != len(tasks):
            raise ValueError('AIME Data proposal repeats an equivalent task')
        return {'tasks': tasks}
    if operator == 'harness':
        if (type(value) is not dict or not _HARNESS_FIELDS <= value.keys() <= _HARNESS_FIELDS | {'guidance'}
                or value['instruction'] not in _PROFILES
                or type(value['exemplars']) is not list or len(value['exemplars']) > max_exemplars
                or not all(type(pin) is str and pin in pins for pin in value['exemplars'])
                or len(set(value['exemplars'])) != len(value['exemplars'])):
            raise ValueError('Invalid AIME Harness proposal or unauthorized exemplar')
        patch = {'instruction': value['instruction'], 'few_shot': bool(value['exemplars']),
                 'exemplars': value['exemplars']}
        if 'guidance' in value:
            patch['guidance'] = (None if allow_clear_guidance and value['guidance'] is None
                                 else check_guidance(value['guidance']))
        return patch
    compile_model_recipe(value, base_learning_rate=1.0)
    return value


def parse_portfolio(operator, text, **bounds):
    """Accept a legacy single patch or up to two distinct typed candidates."""
    if operator not in ('harness', 'model') or not isinstance(text, str) or not 0 < len(text) <= MAX_RESPONSE_CHARS:
        raise ValueError('Invalid operator portfolio')
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_int=_read_integer,
                           parse_float=_reject_number, parse_constant=_reject_number)
    except (ValueError, RecursionError):
        raise ValueError('Portfolio must be exactly one strict JSON object') from None
    if not _shape(value, {'candidates'}):
        return [parse_proposal(operator, text, **bounds)]
    body = value['candidates']
    if type(body) is not list or len(body) > 2:
        raise ValueError('Portfolio must contain at most two named choices')
    if not all(type(choice) is dict for choice in body):
        raise ValueError('Portfolio choices must be typed proposals')
    proposals = [parse_proposal(operator, _canonical_bytes(choice).decode('ascii'), **bounds) for choice in body]
    if len({_canonical_bytes(p) for p in proposals}) != len(proposals):
        raise ValueError('Duplicate portfolio choice')
    return proposals


def compile_model_recipe(proposal, *, base_learning_rate):
    """Compile only the allowed LR override; the Lab derives every other field."""
    if (not _shape(proposal, {'learning_rate_scale'}) or type(proposal['learning_rate_scale']) is not str
            or proposal['learning_rate_scale'] not in _SCALES):
        raise ValueError('Invalid AIME Model learning-rate proposal')
    if type(base_learning_rate) not in (int, float) or not 0 < base_learning_rate <= float('inf'):
        raise ValueError('Invalid trusted AIME base learning rate')
    try:
        rate = base_learning_rate * _SCALES[proposal['learning_rate_scale']]
    except OverflowError:
        raise ValueError('Invalid trusted AIME base learning rate') from None
    if not isfinite(rate) or rate <= 0:
        raise ValueError('Invalid compiled AIME learning rate')
    return {'learning_rate': rate}
