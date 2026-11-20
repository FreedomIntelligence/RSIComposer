"""Run a round of reference TTRL rewards on a Python main thread."""
import json
import subprocess
import sys

from rsisys.api import build_verifier, final_boxed


def score_groups(groups, verifier):
    # Isolated verifiers (Math-Verify uses SIGALRM for up to two seconds per
    # parse and verification) grade on a child main thread. Start one process
    # per reward call and send only final answer boxes.
    boxed = [[final_boxed(completion) or '' for completion in group] for group in groups]
    timeout = 10 + 4 * sum(len(group) for group in boxed)
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'rsisys.algorithms.ttrl._reward_worker'],
            input=json.dumps({'verifier': verifier, 'groups': boxed}), capture_output=True, text=True, check=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'TTRL reward worker exceeded its {timeout}-second deadline') from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'TTRL reward worker exited with code {error.returncode}') from error
    return json.loads(result.stdout)


def main():
    from .lab import _majority_rewards_many
    request = json.load(sys.stdin)
    json.dump(_majority_rewards_many(request['groups'], build_verifier(request['verifier'])),
              sys.stdout, allow_nan=False)


if __name__ == '__main__':
    main()
