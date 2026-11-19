"""Pinned GEPA sampling, mutation and fixed-spend protocol."""

from functools import lru_cache
from pathlib import Path
import sys

from ..shared.experiment import author_checkout


SOURCE_COMMIT = '99c3b0ac3b4750cde6a2a88547ed4450930d681b'


@lru_cache(maxsize=4)
def upstream(source_dir):
    source = author_checkout(source_dir, SOURCE_COMMIT, 'src/gepa') / 'src'
    sys.path.insert(0, str(source))
    try:
        import gepa
        from gepa.core.data_loader import ensure_loader
        from gepa.gepa_utils import select_program_candidate_from_pareto_front
        from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
        from gepa.strategies.instruction_proposal import InstructionProposalSignature
    finally:
        sys.path.remove(str(source))
    if not Path(gepa.__file__).resolve().is_relative_to(source):
        raise RuntimeError('GEPA helpers resolved outside the frozen author checkout')
    return (ensure_loader, select_program_candidate_from_pareto_front,
            EpochShuffledBatchSampler, InstructionProposalSignature)


@lru_cache(maxsize=4)
def merge_api(source_dir):
    upstream(source_dir)
    from gepa import optimize
    from gepa.core.adapter import EvaluationBatch
    return optimize, EvaluationBatch


def budget(config):
    p, d = config['parameters'], config['data']
    return (d['validation_count'] + p['rounds'] * (2 * p['minibatch_size'] + d['validation_count'])
            + 2 * d['test_count'], p['rounds'])


def valid_instruction(instruction):
    return isinstance(instruction, str) and 0 < len(instruction.strip()) <= 1600 and len(instruction) <= 1600
