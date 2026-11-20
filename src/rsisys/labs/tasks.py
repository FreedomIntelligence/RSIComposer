"""Frozen question-answer task splits, bound by a manifest of hashed files.

A task manifest names each split's source; the Lab's split_roles assign the
host role. Every split is one JSONL file by default:

    {"schema_version": 1, "splits": {
        "dev":    {"file": "dev.jsonl", "sha256": "...", "count": 128,
                   "verifier": "strict_numeric", "fields": {"prompt": "question"}},
        "sealed": {"file": "test.jsonl", "sha256": "...", "count": 500}}}

Rows have id, prompt, answer and an optional group (for per-group metrics).
fields renames those keys for files that spell them differently. verifier
selects the split's AnswerVerifier (see rsisys.labs.verifiers). A Lab can add
loaders for other entry forms, keyed by an entry field, such as the math
scene's official benchmark bundle.

load_task_manifest checks every hash and count, rejects repeated IDs or
normalized prompts within or across splits, and returns DatasetBindings that
the host records with the Lab's identity.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from unicodedata import normalize

from rsisys.loops.contracts.data import DatasetBinding

from .verifiers import build_verifier


TASK_FIELDS = ('id', 'prompt', 'answer', 'group')


@dataclass(frozen=True)
class TaskItem:
    id: str
    prompt: str
    answer: object
    group: str
    verifier: object
    expected: str  # The reference as the verifier normalizes it, for records.


@dataclass(frozen=True)
class TaskSplits:
    items: Mapping[str, tuple[TaskItem, ...]]
    bindings: tuple[DatasetBinding, ...]
    manifest_sha256: str


# A loader reads one manifest entry: (manifest directory, entry, sealed) -> (items, source path, digest).
TaskLoader = Callable[[Path, dict, bool], tuple[tuple[TaskItem, ...], Path, str]]


def checked_bytes(path, digest):
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f'Task data hash does not match the manifest: {path}')
    return data


def manifest_path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Task data paths must be nonempty strings')
    return Path(root) / value


def _fields(entry, defaults):
    fields = entry.get('fields', {})
    if (not isinstance(fields, dict) or set(fields) - set(TASK_FIELDS)
            or any(not isinstance(name, str) or not name.strip() for name in fields.values())):
        raise ValueError('fields must rename id, prompt, answer or group to row keys')
    keys = {name: fields.get(name, defaults.get(name, name)) for name in TASK_FIELDS}
    if len(set(keys.values())) != len(keys):
        raise ValueError('fields must map id, prompt, answer and group to distinct row keys')
    return keys


def jsonl_tasks(root, entry, sealed, *, default_group='task', default_verifier='strict_numeric',
                default_fields=None):
    """Read one hashed JSONL split; every answer must be valid for the split's verifier."""
    required = {'file', 'sha256', 'count'}
    if not required <= set(entry) or set(entry) - required - {'verifier', 'maximum', 'fields'}:
        raise ValueError('A JSONL task split requires file, sha256 and count')
    path = manifest_path(root, entry['file'])
    data = checked_bytes(path, entry['sha256'])
    spec = entry.get('verifier', default_verifier)
    if 'maximum' in entry:
        if not isinstance(spec, str):
            raise ValueError('Give maximum inside the verifier object')
        spec = {'name': spec, 'maximum': entry['maximum']}
    verifier = build_verifier(spec)
    keys = _fields(entry, default_fields or {})
    items = []
    for line, value in enumerate(data.decode('utf-8').splitlines(), 1):
        if not value.strip():
            continue
        row = json.loads(value)
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line}: expected a task object')
        ident, prompt = row.get(keys['id']), row.get(keys['prompt'])
        answer, group = row.get(keys['answer']), row.get(keys['group'], default_group)
        if not all(isinstance(text, str) and text.strip() for text in (ident, prompt, group)):
            raise ValueError(f"{path}:{line}: expected nonempty {keys['id']}, {keys['prompt']} and {keys['group']}")
        expected = verifier.parse_reference(answer)
        if expected is None:
            raise ValueError(f'{path}:{line}: invalid answer for {verifier.name}')
        items.append(TaskItem(ident, prompt, answer, group, verifier, str(expected)))
    if type(entry['count']) is not int or not items or len(items) != entry['count']:
        raise ValueError(f'{path}: nonempty task file must match the manifest count')
    return tuple(items), path, entry['sha256']


def load_task_manifest(path, roles, *, name, loaders=None, jsonl=jsonl_tasks):
    """Bind each split in roles to its manifest entry; loaders map an entry key to a loader."""
    loaders = dict(loaders or {})
    path = Path(path).resolve()
    manifest = path.read_bytes()
    content = json.loads(manifest)
    if (not isinstance(content, dict) or content.get('schema_version') != 1
            or not isinstance(content.get('splits'), dict) or set(content['splits']) != set(roles)):
        raise ValueError('Task manifest must match split_roles and use schema_version 1')
    items, bindings = {}, []
    ids, prompts = set(), set()
    for split, entry in content['splits'].items():
        if not isinstance(entry, dict):
            raise ValueError(f'{split} must have a task manifest entry')
        sealed = roles[split].after_finish_only
        loader = next((loaders[key] for key in loaders if key in entry), jsonl)
        rows, source, digest = loader(path.parent, entry, sealed)
        items[split] = rows
        bindings.append(DatasetBinding(
            name=name, split=split, role='sealed' if sealed else 'dev',
            source=str(source), schema_version='1', content_digest=digest,
            row_ids=tuple(row.id for row in rows),
        ))
        for row in rows:
            prompt = ' '.join(normalize('NFKC', row.prompt).split()).casefold()
            if row.id in ids or prompt in prompts:
                raise ValueError(f'Duplicate task ID or prompt in split {split}: {row.id}')
            ids.add(row.id)
            prompts.add(prompt)
    return TaskSplits(items, tuple(bindings), sha256(manifest).hexdigest())
