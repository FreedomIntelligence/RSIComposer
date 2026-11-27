"""A small, proof-carrying replacement for the unreleased AIME synthesis author.

Only known train questions feed it. Construction changes what is asked and
supplies an independent numeric answer; the model never writes its labels.
"""
from dataclasses import replace
from hashlib import sha256
import re


KINDS = ('rollout', 'affine')
_MAX_SOURCE_LENGTH = 3500
_TAG = re.compile(r'^aime-affine-v1-([0-9a-f]{24})-([2357])-([1-9][0-9]{0,2})$')
_MULTIPLIERS = (2, 3, 5, 7)


def source_key(problem):
    return sha256(problem.id.encode()).hexdigest()[:24]


def construct(problem, mul, offset):
    if mul not in _MULTIPLIERS or not 1 <= offset <= 999 or len(problem.prompt) > _MAX_SOURCE_LENGTH:
        raise ValueError('Invalid bounded AIME synthesis parameters')
    x = problem.verifier.parse_reference(problem.answer)
    if type(x) is not int or not 0 <= x <= 999:
        raise ValueError('Only AIME-compatible integer train sources may be synthesized')
    ident = f'aime-affine-v1-{source_key(problem)}-{mul}-{offset}'
    question = (f'In the source problem below, let x be the numerical answer. '
                f'Find the remainder when {mul}x + {offset} is divided by 1000. '
                'Return the remainder, not x.\n\nSource problem:\n' + problem.prompt)
    result = (mul * x + offset) % 1000
    derived = replace(problem, id=ident, prompt=question, answer=f'{result:03d}', expected=f'{result:03d}')
    provenance = {'source_id': problem.id, 'rule': 'aime-affine-v1', 'multiplier': mul, 'offset': offset}
    return derived, provenance


def propose(problem, seed, round_number):
    reference = problem.verifier.parse_reference(problem.answer)
    if len(problem.prompt) > _MAX_SOURCE_LENGTH or type(reference) is not int or not 0 <= reference <= 999:
        return None
    payload = f'{seed}:{round_number}:{problem.id}'.encode()
    value = int.from_bytes(sha256(payload).digest()[:8], 'big')
    return construct(problem, _MULTIPLIERS[value % len(_MULTIPLIERS)], 1 + (value // 4) % 999)


def recover(ident, sources):
    match = _TAG.fullmatch(ident)
    if match is None:
        return None
    source, mul, offset = match.groups()
    if source not in sources:
        raise ValueError('Synthesized AIME record is not linked to the pinned train cohort')
    return construct(sources[source], int(mul), int(offset))
