"""veRL reward callback using the verifier declared by MathLab's train split."""

import json
from functools import lru_cache

from rsicomposer.api import build_verifier


@lru_cache(maxsize=32)
def _verifier(spec):
    return build_verifier(json.loads(spec))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source != 'rsicomposer-math':
        raise ValueError('Unexpected MathLab reward source')
    return float(_verifier(extra_info['verifier_json']).is_correct(solution_str, json.loads(ground_truth)))
