"""The official competition bundle: AIME 2024, AIME 2025 and AIMO3 Reference, from local JSONL."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .verifiers import CompetitionIntegerVerifier


@dataclass(frozen=True)
class BenchmarkSpec:
    maximum: int
    groups: tuple[tuple[str, int], ...]


BENCHMARKS = {
    'aime2024': BenchmarkSpec(999, (('I', 15), ('II', 15))),
    'aime2025': BenchmarkSpec(999, (('I', 15), ('II', 15))),
    'aimo3_reference': BenchmarkSpec(99999, (('reference', 10),)),
}


@dataclass(frozen=True)
class MathProblem:
    id: str
    prompt: str
    answer: int
    group: str
    number: int


@dataclass(frozen=True)
class MathBenchmark:
    name: str
    sha256: str
    problems: tuple[MathProblem, ...]


def load_benchmark(manifest_path, name, *, manifest=None):
    """Check the exact bytes being parsed and validate full benchmark coverage."""
    spec = BENCHMARKS[name]
    manifest_path = Path(manifest_path)
    if manifest is None:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entry = manifest['benchmarks'][name]
    if manifest['schema_version'] != 1 or entry['file'] != f'{name}.jsonl':
        raise ValueError(f'Invalid benchmark manifest entry for {name}')
    data = (manifest_path.parent / entry['file']).read_bytes()
    digest = sha256(data).hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'{name} dataset hash does not match its manifest')

    verifier = CompetitionIntegerVerifier(spec.maximum)
    problems = []
    for line in data.decode('utf-8').splitlines():
        row = json.loads(line)
        if row['benchmark'] != name:
            raise ValueError(f'Unexpected benchmark on row {len(problems) + 1}')
        answer = verifier.parse_reference(row['answer'])
        if (answer is None or not isinstance(row['id'], str) or not row['id'].strip()
                or not isinstance(row['prompt'], str) or not row['prompt'].strip()
                or not isinstance(row['group'], str) or type(row['number']) is not int):
            raise ValueError(f'Invalid benchmark row {len(problems) + 1} in {name}')
        problems.append(MathProblem(row['id'], row['prompt'], answer, row['group'], row['number']))

    actual = Counter((p.group, p.number) for p in problems)
    expected = {(group, number): 1 for group, count in spec.groups for number in range(1, count + 1)}
    if (actual != expected or len(problems) != entry['count']
            or len({p.id for p in problems}) != len(problems)
            or len({p.prompt.strip() for p in problems}) != len(problems)):
        raise ValueError(f'{name} has missing or duplicate problems')
    return MathBenchmark(name, digest, tuple(problems))


def official_problems(root, entry, *, budgets=True, max_output_tokens=32768):
    """Selected benchmarks of the official bundle, as one sealed split's problems.

    entry may list `benchmarks` (default: all three) and a `max_output_tokens`. Each
    benchmark's verifier is part of its protocol and is fixed here, not by the Lab.
    Returns (problems, path, benchmarks by name).
    """
    from .tasks import Problem, checked_bytes, manifest_path, math_benchmark

    if not {'benchmarks_manifest', 'sha256'} <= set(entry) or set(entry) - {
            'benchmarks_manifest', 'sha256', 'benchmarks', 'max_output_tokens'}:
        raise ValueError('The official math preset requires benchmarks_manifest and sha256')
    selected = entry.get('benchmarks', list(BENCHMARKS))
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or set(selected) - set(BENCHMARKS)):
        raise ValueError(f'benchmarks must be a nonempty list of: {sorted(BENCHMARKS)}')
    if not budgets and 'max_output_tokens' in entry:
        raise ValueError('This Lab generates with its own budget; remove max_output_tokens from the source')
    tokens = None if not budgets else entry.get('max_output_tokens', max_output_tokens)
    if budgets and (type(tokens) is not int or tokens < 1):
        raise ValueError('max_output_tokens must be a positive integer')
    path = manifest_path(root, entry['benchmarks_manifest'])
    manifest = json.loads(checked_bytes(path, entry['sha256']))
    if (not isinstance(manifest, dict) or not isinstance(manifest.get('benchmarks'), dict)
            or set(manifest['benchmarks']) != set(BENCHMARKS)):
        raise ValueError('The official bundle manifest must describe AIME 2024, AIME 2025 and AIMO3 Reference')
    rows, benchmarks = [], {}
    for name in sorted(selected):
        spec = BENCHMARKS[name]
        verifier = CompetitionIntegerVerifier(spec.maximum)
        rows.extend(Problem(row.id, row.prompt, row.answer, name, verifier, str(row.answer), tokens)
                    for row in load_benchmark(path, name, manifest=manifest).problems)
        benchmarks[name] = math_benchmark(name, {'name': 'competition_integer', 'maximum': spec.maximum},
                                          verifier, tokens, {'bundle': str(path), 'sha256': entry['sha256']})
    return tuple(rows), path, benchmarks
