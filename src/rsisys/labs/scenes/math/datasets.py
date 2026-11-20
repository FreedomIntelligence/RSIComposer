"""Freeze named math evaluation splits and keep formal benchmarks together."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from unicodedata import normalize

from rsisys.loops.contracts.data import DatasetBinding

from .benchmarks import BENCHMARKS, load_benchmark
from rsisys.labs.verifiers import CompetitionIntegerVerifier, build_verifier


@dataclass(frozen=True)
class Problem:
    id: str
    prompt: str
    answer: int | str
    benchmark: str
    verifier: object
    expected: str  # The reference as the verifier normalizes it, for records.


def checked_bytes(path, digest):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f'Math dataset hash does not match the manifest: {path}')
    return data


def _path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Math dataset paths must be nonempty strings')
    return root / value


def _official(root, entry):
    if set(entry) != {'benchmarks_manifest', 'sha256'}:
        raise ValueError('The official math preset requires benchmarks_manifest and sha256')
    path = _path(root, entry['benchmarks_manifest'])
    manifest = json.loads(checked_bytes(path, entry['sha256']))
    if (not isinstance(manifest, dict) or not isinstance(manifest.get('benchmarks'), dict)
            or set(manifest['benchmarks']) != set(BENCHMARKS)):
        raise ValueError('Official evaluation requires AIME 2024, AIME 2025 and AIMO3 Reference together')
    rows = []
    for name, spec in BENCHMARKS.items():
        benchmark = load_benchmark(path, name, manifest=manifest)
        verifier = CompetitionIntegerVerifier(spec.maximum)
        rows.extend(Problem(row.id, row.prompt, row.answer, name, verifier, str(row.answer))
                    for row in benchmark.problems)
    return tuple(rows)


def _jsonl(root, entry):
    required = {'file', 'sha256', 'count'}
    if not required <= set(entry) or set(entry) - required - {'verifier', 'maximum'}:
        raise ValueError('A math JSONL split requires file, sha256 and count')
    path = _path(root, entry['file'])
    data = checked_bytes(path, entry['sha256'])
    verifier_name = entry.get('verifier', 'competition_integer')
    # Any shared verifier; maximum is the competition_integer option.
    verifier = build_verifier({'name': verifier_name,
                               **({'maximum': entry['maximum']} if 'maximum' in entry else {})})

    problems = []
    for line, value in enumerate(data.decode('utf-8').splitlines(), 1):
        if not value.strip():
            continue
        row = json.loads(value)
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line}: expected a problem object')
        ident, prompt = row.get('id'), row.get('prompt')
        answer, benchmark = row.get('answer'), row.get('benchmark', 'math')
        if not all(isinstance(text, str) and text.strip() for text in (ident, prompt, benchmark)):
            raise ValueError(f'{path}:{line}: expected nonempty id, prompt and benchmark')
        expected = verifier.parse_reference(answer)
        if expected is None:
            raise ValueError(f'{path}:{line}: invalid answer for {verifier_name}')
        problems.append(Problem(ident, prompt, answer, benchmark, verifier, str(expected)))
    if type(entry['count']) is not int or not problems or len(problems) != entry['count']:
        raise ValueError(f'{path}: nonempty math dataset must match manifest count')
    return tuple(problems)


def load_splits(path, roles):
    path = Path(path).resolve()
    manifest = path.read_bytes()
    content = json.loads(manifest)
    if (not isinstance(content, dict) or content.get('schema_version') != 1
            or not isinstance(content.get('splits'), dict) or set(content['splits']) != set(roles)):
        raise ValueError('Math dataset manifest must match split_roles and use schema_version 1')
    problems, bindings = {}, []
    ids, questions = set(), set()
    for split, entry in content['splits'].items():
        if not isinstance(entry, dict):
            raise ValueError(f'{split} must have a math dataset manifest entry')
        sealed = roles[split].after_finish_only
        if 'benchmarks_manifest' in entry:
            if not sealed:
                raise ValueError('The official math preset requires a sealed split')
            rows = _official(path.parent, entry)
            source = _path(path.parent, entry['benchmarks_manifest'])
        else:
            rows = _jsonl(path.parent, entry)
            source = _path(path.parent, entry['file'])
        problems[split] = rows
        bindings.append(DatasetBinding(
            name='math', split=split, role='sealed' if sealed else 'dev',
            source=str(source), schema_version='1', content_digest=entry['sha256'],
            row_ids=tuple(row.id for row in rows),
        ))
        for row in rows:
            question = ' '.join(normalize('NFKC', row.prompt).split()).casefold()
            if row.id in ids or question in questions:
                raise ValueError(f'Duplicate math ID or question in evaluation split {split}: {row.id}')
            ids.add(row.id)
            questions.add(question)
    return problems, tuple(bindings), sha256(manifest).hexdigest()
