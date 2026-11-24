"""Load the official GEPA optimizer from its pinned checkout."""
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from ..shared.experiment import author_checkout


ARTIFACT_COMMIT = 'cbefbc1aa0f43dd39874ec4bf42211365dbda42e'
DSPY_COMMIT = '62dc3b634d7dc0c4889abcf905cb4c391ea6b396'


@lru_cache(maxsize=4)
def aime_protocol(artifact_dir, dspy_dir):
    artifact = author_checkout(artifact_dir, ARTIFACT_COMMIT, 'gepa_artifact')
    fork = author_checkout(dspy_dir, DSPY_COMMIT, 'dspy')
    sys.path[:0] = [str(artifact), str(fork)]
    try:
        import dspy
        from gepa_artifact.benchmarks import AIME
    finally:
        del sys.path[:2]
    if (not Path(dspy.__file__).resolve().is_relative_to(fork)
            or not Path(AIME.__file__).resolve().is_relative_to(artifact)):
        raise RuntimeError('AIME protocol resolved outside the frozen author checkouts')
    predictor, = AIME.program_cot.named_predictors()
    return AIME, predictor[0], predictor[1], dspy.ChatAdapter(), dspy


def paper_modules(artifact_dir):
    """Isolate the artifact's module globals so each run can bind its own I/O."""
    root = Path(artifact_dir).resolve() / 'gepa_artifact' / 'gepa'

    def load(name):
        spec = spec_from_file_location(f'gepa_artifact.gepa._rsi_{name}', root / f'{name}.py')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load('gepa'), load('gepa_utils')
