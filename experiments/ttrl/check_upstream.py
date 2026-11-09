"""Check GRPO advantages against the exact pinned upstream function on CPU."""
import argparse
import ast
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path


def main():
    import numpy as np
    import torch

    from rsisys.labs.math_verifiers import StrictNumericVerifier
    from rsisys.labs.ttrl import majority_rewards
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('upstream', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '5806e119789132fcb43552beaf2532d7d0447213':
        raise ValueError('Expected pinned TTRL revision')
    path = args.upstream / 'verl/verl/trainer/ppo/core_algos.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_grpo_outcome_advantage')
    function.decorator_list = []
    namespace = {'torch': torch, 'np': np, 'defaultdict': defaultdict}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    rng, cases = random.Random(7309), 0
    for size in (2, 3, 4, 8, 16, 32):
        for _ in range(100):
            texts = [rng.choice([r'\boxed{1}', r'\boxed{2}', 'invalid']) for _ in range(size)]
            result = majority_rewards(texts, StrictNumericVerifier())
            rewards = torch.tensor(result['rewards']).reshape(size, 1)
            masks = torch.ones_like(rewards)
            expected, _ = namespace['compute_grpo_outcome_advantage'](rewards.clone(), masks, np.array(['one'] * size))
            if not torch.allclose(expected.flatten(), torch.tensor(result['advantages']), atol=1e-6):
                raise AssertionError('GRPO advantage mismatch')
            cases += 1
    args.output.write_text(json.dumps({'revision': revision, 'advantage_cases': cases,
        'passed': True, 'scope': 'group-normalized advantages; not full verl/backend equivalence'}, indent=2))


if __name__ == '__main__':
    main()
