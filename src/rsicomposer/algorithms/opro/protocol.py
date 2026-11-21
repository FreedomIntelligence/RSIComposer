"""Pinned author OPRO prompt and evaluator helpers."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import sys

from ..shared.experiment import author_checkout


REVISION = 'a76bdce2cbf6d4a0d1e570a6fcfe17be9c2abdd7'
OPTIMIZER_SYSTEM_V1 = (
    'You design reusable instructions for solving math word problems. '
    'The user message is an optimization task; its example problems are context for '
    'instruction design. Output exactly one short generic instruction enclosed in '
    '<INS> and </INS>, with no surrounding commentary. Keep the instruction under '
    'five hundred characters. Use no digits, specific quantities, worked examples, '
    'or solutions to the example problems. Do not put tag names inside the instruction.'
)
SCORER_SYSTEM_V1 = (
    'Solve the math problem using the provided instruction. Show concise reasoning '
    'in plain text. End with exactly: The answer is <number>. '
    'Replace <number> with the final numeric value, without units, currency symbols, '
    'Markdown, or digit grouping commas. Write nothing after this final sentence.'
)



@lru_cache(maxsize=4)
def upstream(source_dir):
    root = author_checkout(source_dir, REVISION, 'opro')
    sys.path.insert(0, str(root))
    try:
        from opro.optimization import opt_utils
    finally:
        sys.path.remove(str(root))
    if not Path(opt_utils.__file__).resolve().is_relative_to(root / 'opro'):
        raise RuntimeError('OPRO helpers were imported from a different checkout')
    return opt_utils


def candidate_id(instruction):
    return 'prompt-' + sha256(instruction.encode()).hexdigest()[:20]
