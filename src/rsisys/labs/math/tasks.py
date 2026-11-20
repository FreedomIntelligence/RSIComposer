"""Frozen math task splits, bound by a manifest of hashed files.

The manifest names each split's source; the Lab's split_roles assign the host
role. Each source mounts one benchmark: a named, frozen problem set together
with its protocol, the answer verifier and the generation budget, recorded with
a protocol_id (rsisys.loops.contracts.benchmark). A split is one source, a list
of sources, or, for a sealed split, selected benchmarks of the official
competition bundle (benchmarks.py):

    {"schema_version": 1, "splits": {
        "dev":      {"file": "dev.jsonl", "sha256": "...", "count": 128, "benchmark": "gsm8k",
                     "verifier": "marked_numeric", "fields": {"prompt": "question"}},
        "mixed":    [{"file": "a.jsonl", "sha256": "...", "count": 10, "benchmark": "a"},
                     {"file": "b.jsonl", "sha256": "...", "count": 10, "benchmark": "b",
                      "verifier": "math_verify", "max_output_tokens": 8192}],
        "official": {"benchmarks_manifest": "benchmarks/manifest.json", "sha256": "...",
                     "benchmarks": ["aime2025"]}}}

Rows have id, prompt and answer; fields renames those keys for files that spell
them differently. A source's benchmark (default "math") names it in metrics; a
benchmark keeps one protocol wherever it is mounted, so two sources with the same
name must agree. verifier selects the answer verifier by name (verifiers.py); the
default accepts competition integers through 99,999. max_output_tokens (default
32768) is the actor's generation budget, for Labs whose actor answers each problem.

load_splits checks every hash and count, rejects repeated IDs or normalized
prompts within or across splits, and returns the DatasetBindings that the host
records with the Lab's identity, and the mounted benchmarks.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from unicodedata import normalize

from rsisys.loops.contracts.benchmark import Benchmark
from rsisys.loops.contracts.data import DatasetBinding

from .verifiers import VERIFIERS, build_verifier


FIELDS = ('id', 'prompt', 'answer')
DEFAULT_BENCHMARK = 'math'
DEFAULT_VERIFIER = 'competition_integer'
DEFAULT_MAX_OUTPUT_TOKENS = 32768
KIND = 'math-qa'


@dataclass(frozen=True)
class Problem:
    id: str
    prompt: str
    answer: object
    benchmark: str
    verifier: object
    expected: str  # The reference as the verifier normalizes it, for records.
    max_output_tokens: int | None = None  # None when the Lab owns generation.


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
        raise ValueError('fields must rename id, prompt or answer to row keys')
    keys = {name: fields.get(name, name) for name in FIELDS}
    if len(set(keys.values())) != len(keys):
        raise ValueError('fields must map id, prompt and answer to distinct row keys')
    return keys


def _budget(entry, budgets, default):
    """The source's generation budget: its own, else the Lab's default; None when the Lab owns generation."""
    if not budgets:
        if 'max_output_tokens' in entry:
            raise ValueError('This Lab generates with its own budget; remove max_output_tokens from the source')
        return None
    value = entry.get('max_output_tokens', default)
    if type(value) is not int or value < 1:
        raise ValueError('max_output_tokens must be a positive integer')
    return value


def math_benchmark(name, spec, verifier, tokens, source):
    """The Benchmark a source mounts: its protocol is the verifier selection and budget."""
    return Benchmark(name, KIND, {'verifier': {'name': spec} if isinstance(spec, str) else spec,
                                  'verifier_name': verifier.name,
                                  **({} if tokens is None else {'max_output_tokens': tokens})}, source)


def jsonl_problems(root, entry, verifiers=VERIFIERS, *, budgets=True, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Read one hashed JSONL source; every answer must be valid for its verifier.

    Returns (problems, path, benchmark).
    """
    required = {'file', 'sha256', 'count'}
    if not required <= set(entry) or set(entry) - required - {
            'verifier', 'maximum', 'fields', 'benchmark', 'max_output_tokens'}:
        raise ValueError('A math JSONL source requires file, sha256 and count')
    path = manifest_path(root, entry['file'])
    data = checked_bytes(path, entry['sha256'])
    spec = entry.get('verifier', DEFAULT_VERIFIER)
    if 'maximum' in entry:
        if not isinstance(spec, str):
            raise ValueError('Give maximum inside the verifier object')
        spec = {'name': spec, 'maximum': entry['maximum']}
    verifier = build_verifier(spec, verifiers=verifiers)
    name = entry.get('benchmark', DEFAULT_BENCHMARK)
    tokens = _budget(entry, budgets, max_output_tokens)
    keys = _fields(entry)
    problems = []
    for line, value in enumerate(data.decode('utf-8').splitlines(), 1):
        if not value.strip():
            continue
        row = json.loads(value)
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line}: expected a problem object')
        ident, prompt = row.get(keys['id']), row.get(keys['prompt'])
        answer = row.get(keys['answer'])
        if not all(isinstance(text, str) and text.strip() for text in (ident, prompt)):
            raise ValueError(f"{path}:{line}: expected nonempty {keys['id']} and {keys['prompt']}")
        expected = verifier.parse_reference(answer)
        if expected is None:
            raise ValueError(f'{path}:{line}: invalid answer for {verifier.name}')
        problems.append(Problem(ident, prompt, answer, name, verifier, str(expected), tokens))
    if type(entry['count']) is not int or not problems or len(problems) != entry['count']:
        raise ValueError(f'{path}: nonempty math dataset must match manifest count')
    return tuple(problems), path, math_benchmark(name, spec, verifier, tokens, {'file': str(path), 'sha256': entry['sha256']})


def load_splits(path, roles, *, verifiers=VERIFIERS, role_names=None, budgets=True,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Bind each split in roles to its manifest entry.

    Returns (problems by split, bindings, manifest digest, benchmarks by name).
    verifiers is the catalog (name -> factory) that sources select from; role_names
    names each split's role (train, dev or sealed) for its binding. budgets is
    False for Labs that generate with their own budget; otherwise a source that
    sets no max_output_tokens takes max_output_tokens.
    """
    from .benchmarks import official_problems

    path = Path(path).resolve()
    manifest = path.read_bytes()
    content = json.loads(manifest)
    if (not isinstance(content, dict) or content.get('schema_version') != 1
            or not isinstance(content.get('splits'), dict) or set(content['splits']) != set(roles)):
        raise ValueError('Math dataset manifest must match split_roles and use schema_version 1')
    problems, bindings, benchmarks = {}, [], {}
    ids, prompts = set(), set()
    for split, entry in content['splits'].items():
        sources = entry if isinstance(entry, list) else [entry]
        if not sources or any(not isinstance(item, dict) for item in sources):
            raise ValueError(f'{split} must have a math dataset manifest entry')
        sealed = roles[split].after_finish_only
        rows, digests, mounted = [], [], {}
        for item in sources:
            if 'benchmarks_manifest' in item:
                if not sealed:
                    raise ValueError('The official math preset requires a sealed split')
                found, source, official = official_problems(path.parent, item, budgets=budgets,
                                                            max_output_tokens=max_output_tokens)
            else:
                found, source, benchmark = jsonl_problems(path.parent, item, verifiers, budgets=budgets,
                                                          max_output_tokens=max_output_tokens)
                official = {benchmark.name: benchmark}
            rows.extend(found)
            digests.append(item['sha256'])
            for name, benchmark in official.items():
                if mounted.setdefault(name, benchmark) is not benchmark:
                    raise ValueError(f'Benchmark {name!r} is mounted twice in split {split}')
        for name, benchmark in mounted.items():
            if benchmarks.setdefault(name, benchmark).protocol_id != benchmark.protocol_id:
                raise ValueError(f'Benchmark {name!r} is mounted with two protocols; give each its own name')
        rows.sort(key=lambda row: row.benchmark)
        problems[split] = tuple(rows)
        bindings.append(DatasetBinding(
            name='math', split=split, role=(role_names or {}).get(split, 'sealed' if sealed else 'dev'),
            source=str(source), schema_version='1',
            content_digest=digests[0] if len(digests) == 1 else sha256(''.join(digests).encode()).hexdigest(),
            row_ids=tuple(row.id for row in rows),
            provenance={'benchmarks': {name: benchmark.protocol_id for name, benchmark in sorted(mounted.items())}},
        ))
        for row in rows:
            prompt = ' '.join(normalize('NFKC', row.prompt).split()).casefold()
            if row.id in ids or prompt in prompts:
                raise ValueError(f'Duplicate math ID or question in evaluation split {split}: {row.id}')
            ids.add(row.id)
            prompts.add(prompt)
    return problems, tuple(bindings), sha256(manifest).hexdigest(), dict(sorted(benchmarks.items()))
