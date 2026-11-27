"""Opt-in preparation of small GSM8K cohorts for MetaRSI (no keys or models).

Run `python -m rsicomposer.algorithms.meta_rsi.prepare_gsm8k --output-dir PATH`.
Pass the resulting manifest paths to MetaRSI as absolute paths; references
inside each manifest are relative to it so the whole bundle can be moved.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPException
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unicodedata import normalize
from urllib.request import urlopen

from rsicomposer.labs.math.verifiers import MarkedNumericVerifier


REPOSITORY = 'https://github.com/openai/grade-school-math'
REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
RAW_ROOT = f'https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}'
DEFAULT_COUNT = 8
VERIFIER = 'marked_numeric'
PARENT_MANIFEST = 'manifest.json'
META_MANIFEST = 'meta-manifest.json'
PARENT_SPLITS = ('train', 'search', 'validation', 'test')
META_SPLITS = ('train', 'meta_search', 'meta_validation')
DEV_SPLITS = ('search', 'validation', 'meta_search', 'meta_validation')


@dataclass(frozen=True)
class _Source:
    path: str
    sha256: str
    max_bytes: int
    count: int | None = None

    @property
    def url(self):
        return f'{RAW_ROOT}/{self.path}'


_SOURCES = {
    'train': _Source('grade_school_math/data/train.jsonl',
                     '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465', 5_000_000, 7473),
    'test': _Source('grade_school_math/data/test.jsonl',
                    '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14', 1_000_000, 1319),
    'license': _Source('LICENSE',
                       '86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b', 4096),
}


def _verified_bytes(source, source_dir):
    label = source.url if source_dir is None else str(source_dir / source.path)
    try:
        if source_dir is None:
            stream = urlopen(source.url, timeout=30)
        else:
            path = source_dir / source.path
            if not path.is_file():
                raise OSError('expected a regular file')
            stream = path.open('rb')
        with stream as handle:
            data = handle.read(source.max_bytes + 1)
    except (OSError, HTTPException) as error:
        raise ValueError(f'Could not read pinned GSM8K source {label}: {error}') from error
    if len(data) > source.max_bytes:
        raise ValueError(f'Pinned GSM8K source {label} exceeds its {source.max_bytes}-byte limit')
    actual = sha256(data).hexdigest()
    if actual != source.sha256:
        raise ValueError(f'GSM8K SHA-256 mismatch for {label}: expected {source.sha256}, got {actual}')
    return data


def _parse_rows(split, data, prompts):
    source = _SOURCES[split]
    lines = data.splitlines()
    if len(lines) != source.count:
        raise ValueError(f'GSM8K {split} row count mismatch: expected {source.count}, got {len(lines)}')
    verifier = MarkedNumericVerifier()
    rows = []
    for index, line in enumerate(lines):
        label = f'GSM8K {split} row {index + 1}'
        try:
            row = json.loads(line.decode('utf-8'))
        except ValueError as error:
            raise ValueError(f'{label}: expected UTF-8 JSON') from error
        if not isinstance(row, dict) or set(row) != {'question', 'answer'}:
            raise ValueError(f'{label}: expected question and answer fields')
        prompt, answer = row['question'], row['answer']
        if (not isinstance(prompt, str) or not prompt.strip() or not isinstance(answer, str)
                or not answer.rstrip().rsplit('\n', 1)[-1].startswith('#### ')
                or verifier.parse_reference(answer) is None):
            raise ValueError(f'{label}: expected a nonempty prompt and a marked numeric answer')
        # Match MathLab's question identity and check across both upstream splits.
        key = ' '.join(normalize('NFKC', prompt).split()).casefold()
        if key in prompts:
            raise ValueError(f'{label}: duplicate normalized prompt (also in {prompts[key]})')
        prompts[key] = label
        rows.append({'id': f'gsm8k-{split}-{index:04d}', 'prompt': prompt, 'answer': answer})
    return rows


def _frozen_order(split, count):
    # A stable ordering independent of Python's random implementation.
    return sorted(range(count), key=lambda index: sha256(f'{REVISION}:{split}:{index}'.encode('ascii')).digest())


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def _write_bundle(directory, rows, selections, source_bytes):
    entries, splits, manifest_digests = {}, {}, {}
    for name, (source, indices) in selections.items():
        filename = f'{name}.jsonl'
        data = ''.join(json.dumps(rows[source][index], separators=(',', ':')) + '\n'
                       for index in indices).encode('utf-8')
        (directory / filename).write_bytes(data)
        details = {'file': filename, 'sha256': sha256(data).hexdigest(), 'count': len(indices)}
        entries[name] = {**details, 'benchmark': 'gsm8k', 'verifier': VERIFIER}
        splits[name] = {**details, 'source_split': source, 'source_indices': indices}
    for filename, split_names in ((PARENT_MANIFEST, PARENT_SPLITS), (META_MANIFEST, META_SPLITS)):
        data = _json_bytes({'schema_version': 1, 'splits': {name: entries[name] for name in split_names}})
        (directory / filename).write_bytes(data)
        manifest_digests[filename] = {'sha256': sha256(data).hexdigest()}
    (directory / 'LICENSE').write_bytes(source_bytes['license'])
    (directory / 'provenance.json').write_bytes(_json_bytes({
        'schema_version': 1,
        'dataset': 'gsm8k',
        'repository': REPOSITORY,
        'revision': REVISION,
        'sources': {name: {'path': source.path, 'url': source.url, 'sha256': source.sha256,
                           'bytes': len(source_bytes[name]),
                           **({'count': source.count} if source.count is not None else {})}
                    for name, source in _SOURCES.items()},
        'selection': {'index_base': 0, 'ordering': 'ascending SHA-256 of ASCII revision:source_split:source_index',
                      'train_allocation': ['train', *DEV_SPLITS]},
        'splits': splits,
        'manifests': manifest_digests,
    }))


def prepare(output_dir, *, train_count=DEFAULT_COUNT, dev_count=DEFAULT_COUNT,
            sealed_count=DEFAULT_COUNT, source_dir=None):
    """Publish a verified bundle in a new directory; return absolute parent and trial manifest paths."""
    train_limit, test_limit = _SOURCES['train'].count, _SOURCES['test'].count
    for name, count, limit in (('train_count', train_count, train_limit),
                               ('dev_count', dev_count, train_limit // len(DEV_SPLITS)),
                               ('sealed_count', sealed_count, test_limit)):
        if type(count) is not int or not 1 <= count <= limit:
            raise ValueError(f'{name} must be an integer in [1, {limit}]')
    if train_count + len(DEV_SPLITS) * dev_count > train_limit:
        raise ValueError(f'train_count + {len(DEV_SPLITS)} * dev_count cannot exceed {train_limit} upstream train rows')

    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output directory already exists: {output}')
    local = Path(source_dir).expanduser() if source_dir is not None else None
    sources = {key: _verified_bytes(source, local) for key, source in _SOURCES.items()}
    prompts = {}
    rows = {split: _parse_rows(split, sources[split], prompts) for split in ('train', 'test')}
    order = _frozen_order('train', train_limit)
    selections = {'train': ('train', order[:train_count])}
    for offset, name in enumerate(DEV_SPLITS):
        start = train_count + offset * dev_count
        selections[name] = ('train', order[start:start + dev_count])
    selections['test'] = ('test', _frozen_order('test', test_limit)[:sealed_count])

    # Publish only a complete bundle. Keep the staging directory on the same filesystem.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.gsm8k-prep-') as temporary:
        staging = Path(temporary) / 'bundle'
        staging.mkdir()
        _write_bundle(staging, rows, selections, sources)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Output directory already exists: {output}')
        staging.rename(output)
    return output / PARENT_MANIFEST, output / META_MANIFEST


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare pinned, keyless GSM8K cohorts for MetaRSI.')
    parser.add_argument('--output-dir', required=True, type=Path, help='new directory for the prepared bundle')
    parser.add_argument('--train-count', type=int, default=DEFAULT_COUNT, help='shared train rows (default: 8)')
    parser.add_argument('--dev-count', type=int, default=DEFAULT_COUNT, help='rows per development cohort (default: 8)')
    parser.add_argument('--sealed-count', type=int, default=DEFAULT_COUNT, help='sealed upstream test rows (default: 8)')
    parser.add_argument('--source-dir', type=Path,
                        help='optional offline copy of the upstream checkout, verified against the same pins')
    args = parser.parse_args(argv)
    try:
        parent, meta = prepare(args.output_dir, train_count=args.train_count, dev_count=args.dev_count,
                               sealed_count=args.sealed_count, source_dir=args.source_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f'GSM8K preparation failed: {error}\n')
    print(f'Parent manifest: {parent}')
    print(f'Meta-trial manifest: {meta}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
