"""Run TTRL scoring on a Python main thread."""
import json
import subprocess
import sys

from rsicomposer.api import build_verifier, final_boxed


def score_groups(groups, verifier):
    # Isolated verifiers (Math-Verify uses SIGALRM for up to two seconds per
    # parse and verification) grade on a child main thread. Start one process
    # per reward call and send only final answer boxes.
    boxed = [[final_boxed(completion) or '' for completion in group] for group in groups]
    return _score({'verifier': verifier, 'groups': boxed}, sum(map(len, boxed)))


def score_official_pairs(pairs, root):
    return _score({'upstream_root': str(root), 'pairs': pairs}, len(pairs))


def _score(request, count):
    timeout = 10 + 4 * count
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'rsicomposer.algorithms.ttrl._reward_worker'],
            input=json.dumps(request), capture_output=True, text=True, check=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'TTRL reward worker exceeded its {timeout}-second deadline') from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'TTRL reward worker exited with code {error.returncode}') from error
    return json.loads(result.stdout)


def main():
    request = json.load(sys.stdin)
    if 'pairs' in request:
        from .upstream import TTRLReward
        verifier = TTRLReward(request['upstream_root'])
        results = verifier.score_many(request['pairs'])
    else:
        from .lab import _majority_rewards_many
        results = _majority_rewards_many(request['groups'], build_verifier(request['verifier']))
    json.dump(results, sys.stdout, allow_nan=False)


if __name__ == '__main__':
    main()
