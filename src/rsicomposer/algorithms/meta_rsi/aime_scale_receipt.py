"""Verify code identity and aggregate optimizer evidence for the scale probe."""
import json
import math
from pathlib import Path

from rsicomposer.api import file_hash

from .engine import digest


def check_frozen_code():
    package = Path(__file__).resolve().parents[2]
    files = sorted(package.rglob('*.py'))
    if any(file.is_symlink() for file in files):
        raise ValueError('Probe source must not use symlinks')
    return digest({file.relative_to(package).as_posix(): file_hash(file) for file in files})


def verify_training_receipt(path, selected, checkpoint, *, steps, labels=None, full_token_count=None,
                            world_size=1, global_batch_size=None, allow_uneven_tail=False, learning_rate=None):
    raw = Path(path).read_bytes()
    receipt = json.loads(raw)
    coverage = receipt.get('coverage') or {}
    distinct = len({row['problem_id'] for row in selected})
    expected = steps * labels if labels is not None else full_token_count
    selection = ('full-causal-assistant-including-eos-v1' if labels is None
                 else 'uniform-causal-assistant-including-eos-v1')
    batch = world_size if global_batch_size is None else global_batch_size
    samples = len(selected)
    tail = samples - (steps - 1) * batch
    if (type(expected) is not int or expected < samples
            or not 0 < tail <= batch or (not allow_uneven_tail and samples % world_size)
            or len({row['id'] for row in selected}) != samples
            or receipt.get('steps') != steps or receipt.get('microsteps') != (samples + world_size - 1) // world_size
            or (learning_rate is not None and receipt.get('learning_rate') != learning_rate)
            or receipt.get('samples') != samples or receipt.get('supervised_tokens_per_step') != labels
            or (batch > world_size and receipt.get('final_batch_size') != tail)
            or (world_size > 1 and (receipt.get('world_size') != world_size or receipt.get('examples_seen') != samples
                                    or receipt.get('effective_batch_size') != batch
                                    or receipt.get('microsteps_unit') != 'global_waves'
                                    or receipt.get('loss_normalization') != 'global_completion_tokens'
                                    or receipt.get('rank_weights_identical') is not True))
            or receipt.get('supervision_label_selection') != selection
            or receipt.get('completion_tokens_seen') != expected
            or coverage.get('distinct_problem_ids_seen') != distinct
            or coverage.get('distinct_problem_ids_available') != distinct
            or coverage.get('distinct_rollout_ids_seen') != samples
            or coverage.get('distinct_rollout_ids_available') != samples
            or type(coverage.get('input_tokens_seen')) is not int or coverage['input_tokens_seen'] < expected):
        raise ValueError('An arm did not execute the declared matched optimizer budget and coverage')
    for name in ('last_loss', 'token_weighted_loss', 'elapsed_seconds'):
        x = receipt.get(name)
        if type(x) not in (float, int) or not math.isfinite(x) or x < 0:
            raise ValueError('An arm did not publish a valid numerical training receipt')
    return {**receipt, 'checkpoint': checkpoint, 'training_receipt_sha256': file_hash(path)}
