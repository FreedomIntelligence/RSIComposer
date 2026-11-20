"""Frozen math task splits, bound by a scene manifest of hashed files.

The scene manifest names each split's source; the Lab's split_roles assign the
host role. A split is either one JSONL file or, for a sealed split, the
official competition bundle (benchmarks.py):

    {"schema_version": 1, "splits": {
        "dev":      {"file": "dev.jsonl", "sha256": "...", "count": 128,
                     "verifier": "marked_numeric", "fields": {"prompt": "question"}},
        "official": {"benchmarks_manifest": "benchmarks/manifest.json", "sha256": "..."}}}

Rows have id, prompt, answer and an optional benchmark (default "math"), which
groups per-benchmark metrics. fields renames those keys for files that spell
them differently. verifier selects the split's answer verifier by name
(verifiers.py); the default accepts competition integers through 99,999.

load_splits checks every hash and count, rejects repeated IDs or normalized
prompts within or across splits, and returns the DatasetBindings that the host
records with the Lab's identity.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from unicodedata import normalize

from rsisys.loops.contracts.data import DatasetBinding

from .verifiers import build_verifier


FIELDS = ('id', 'prompt', 'answer', 'benchmark')
DEFAULT_BENCHMARK = 'math'
DEFAULT_VERIFIER = 'competition_integer'


@dataclass(frozen=True)
class Problem:
    id: str
    prompt: str
    answer: object
    benchmark: str
    verifier: object
    expected: str  # The reference as the verifier normalizes it, for records.


def checked_bytes(path, digest):
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f'Math dataset hash does not match the manifest: {path}')
    return data


def manifest_path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Math dataset paths must be nonempty strings')
    return Path(root) / value


def _fields(entry):
    fields = entry.get('fields', {})
    if (not isinstance(fields, dict) or set(fields) - set(FIELDS)
            or any(not isinstance(name, str) or not name.strip() for name in fields.values())):
        raise ValueError('fields must rename id, prompt, answer or benchmark to row keys')
    keys = {name: fields.get(name, name) for name in FIELDS}
    if len(set(keys.values())) != len(keys):
        raise ValueError('fields must map id, prompt, answer and benchmark to distinct row keys')
    return keys


def jsonl_problems(root, entry):
    """Read one hashed JSONL split; every answer must be valid for the split's verifier."""
    required = {'file', 'sha256', 'count'}
    if not required <= set(entry) or set(entry) - required - {'verifier', 'maximum', 'fields'}:
        raise ValueError('A math JSONL split requires file, sha256 and count')
    path = manifest_path(root, entry['file'])
    data = checked_bytes(path, entry['sha256'])
    spec = entry.get('verifier', DEFAULT_VERIFIER)
    if 'maximum' in entry:
        if not isinstance(spec, str):
            raise ValueError('Give maximum inside the verifier object')
        spec = {'name': spec, 'maximum': entry['maximum']}
    verifier = build_verifier(spec)
    keys = _fields(entry)
    problems = []
    for line, value in enumerate(data.decode('utf-8').splitlines(), 1):
        if not value.strip():
            continue
        row = json.loads(value)
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line}: expected a problem object')
        ident, prompt = row.get(keys['id']), row.get(keys['prompt'])
        answer, benchmark = row.get(keys['answer']), row.get(keys['benchmark'], DEFAULT_BENCHMARK)
        if not all(isinstance(text, str) and text.strip() for text in (ident, prompt, benchmark)):
            raise ValueError(f"{path}:{line}: expected nonempty {keys['id']}, {keys['prompt']} and {keys['benchmark']}")
        expected = verifier.parse_reference(answer)
        if expected is None:
            raise ValueError(f'{path}:{line}: invalid answer for {verifier.name}')
        problems.append(Problem(ident, prompt, answer, benchmark, verifier, str(expected)))
    if type(entry['count']) is not int or not problems or len(problems) != entry['count']:
        raise ValueError(f'{path}: nonempty math dataset must match manifest count')
    return tuple(problems), path


def load_splits(path, roles):
    """Bind each split in roles to its manifest entry: (problems by split, bindings, manifest digest)."""
    from .benchmarks import official_problems

    path = Path(path).resolve()
    manifest = path.read_bytes()
    content = json.loads(manifest)
    if (not isinstance(content, dict) or content.get('schema_version') != 1
            or not isinstance(content.get('splits'), dict) or set(content['splits']) != set(roles)):
        raise ValueError('Math dataset manifest must match split_roles and use schema_version 1')
    problems, bindings = {}, []
    ids, prompts = set(), set()
    for split, entry in content['splits'].items():
        if not isinstance(entry, dict):
            raise ValueError(f'{split} must have a math dataset manifest entry')
        sealed = roles[split].after_finish_only
        if 'benchmarks_manifest' in entry:
            if not sealed:
                raise ValueError('The official math preset requires a sealed split')
            rows, source = official_problems(path.parent, entry)
        else:
            rows, source = jsonl_problems(path.parent, entry)
        problems[split] = rows
        bindings.append(DatasetBinding(
            name='math', split=split, role='sealed' if sealed else 'dev',
            source=str(source), schema_version='1', content_digest=entry['sha256'],
            row_ids=tuple(row.id for row in rows),
        ))
        for row in rows:
            prompt = ' '.join(normalize('NFKC', row.prompt).split()).casefold()
            if row.id in ids or prompt in prompts:
                raise ValueError(f'Duplicate math ID or question in evaluation split {split}: {row.id}')
            ids.add(row.id)
            prompts.add(prompt)
    return problems, tuple(bindings), sha256(manifest).hexdigest()
