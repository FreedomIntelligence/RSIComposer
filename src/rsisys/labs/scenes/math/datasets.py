"""Freeze named math evaluation splits and keep formal benchmarks together."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from unicodedata import normalize

from rsisys.api import DatasetBinding

from .benchmarks import BENCHMARKS, load_benchmark
from .verifiers import CompetitionIntegerVerifier, MathVerifyVerifier, final_boxed


@dataclass(frozen=True)
class Problem:
    id: str
    prompt: str
    answer: int | str
    benchmark: str
    verifier: object


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
        raise ValueError('A sealed math split requires benchmarks_manifest and sha256')
    path = _path(root, entry['benchmarks_manifest'])
    manifest = json.loads(checked_bytes(path, entry['sha256']))
    if (not isinstance(manifest, dict) or not isinstance(manifest.get('benchmarks'), dict)
            or set(manifest['benchmarks']) != set(BENCHMARKS)):
        raise ValueError('Official evaluation requires AIME 2024, AIME 2025 and AIMO3 Reference together')
    rows = []
    for name, spec in BENCHMARKS.items():
        benchmark = load_benchmark(path, name, manifest=manifest)
        verifier = CompetitionIntegerVerifier(spec.maximum)
        rows.extend(Problem(row.id, row.prompt, row.answer, name, verifier)
                    for row in benchmark.problems)
    return tuple(rows)


def _development(root, entry):
    required = {'file', 'sha256', 'count'}
    if not required <= set(entry) or set(entry) - required - {'verifier', 'maximum'}:
        raise ValueError('A development split requires file, sha256 and count')
    path = _path(root, entry['file'])
    data = checked_bytes(path, entry['sha256'])
    verifier_name = entry.get('verifier', 'competition_integer')
    if verifier_name == 'competition_integer':
        verifier = CompetitionIntegerVerifier(entry.get('maximum', 99999))
    elif verifier_name == 'math_verify' and 'maximum' not in entry:
        verifier = MathVerifyVerifier()
    else:
        raise ValueError('Invalid development verifier or maximum')

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
        if verifier_name == 'competition_integer':
            answer = verifier.parse_gold(answer)
            valid = answer is not None
        else:
            if type(answer) is int:
                answer = str(answer)
            if isinstance(answer, str) and answer.strip():
                answer = final_boxed(answer) or r'\boxed{' + answer + '}'
                valid = verifier.parse(answer) is not None
            else:
                valid = False
        if not valid:
            raise ValueError(f'{path}:{line}: invalid answer for {verifier_name}')
        problems.append(Problem(ident, prompt, answer, benchmark, verifier))
    if type(entry['count']) is not int or not problems or len(problems) != entry['count']:
        raise ValueError(f'{path}: nonempty development dataset must match manifest count')
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
        loader = _official if sealed else _development
        problems[split] = rows = loader(path.parent, entry)
        source = _path(path.parent, entry['benchmarks_manifest' if sealed else 'file'])
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
