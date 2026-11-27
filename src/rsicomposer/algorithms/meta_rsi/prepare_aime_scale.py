"""Prepare a private, development-only AIME pool-scaling bundle.

Run: uv run --with pyarrow==23.0.1 python -m rsicomposer.algorithms.meta_rsi.prepare_aime_scale
     --fetch --output-dir PATH

Use --source-dir for a local copy of the same verified inputs, or --list-sources
for the two input pins. Keep inputs and bundles out of published artifacts.
"""

import argparse
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from . import prepare_aime as aime


SMALL_TRAIN_COUNT = aime.DEFAULT_COUNT
DEFAULT_TRAIN_COUNT = 256
DEV_COUNT = 60
MANIFEST = 'manifest.json'
# Deliberate allowlist: the full AIME preparer also reads evaluation materials
# unrelated to this experiment. Do not use its bulk reader, packer or writer.
_SOURCE_NAMES = ('math_train', 'aime_dev')
_LICENSE_NOTE = ('Keep raw snapshots and prepared corpora local and out of source control and releases. '
                 'Dataset-card licenses do not establish permission to redistribute underlying contest material.')


def _read_sources(source_dir):
    try:
        reader = aime._parquet_reader()
    except ValueError as error:
        raise ValueError('Install PyArrow for this conversion: uv run --with pyarrow==23.0.1 python -m '
                         'rsicomposer.algorithms.meta_rsi.prepare_aime_scale ...') from error
    rows = {}
    for name in _SOURCE_NAMES:
        source = aime._SOURCES[name]
        table = reader(BytesIO(aime._verified_bytes(source, source_dir)), columns=list(source.columns))
        if table.num_rows != source.count:
            raise ValueError(f'{name} row count mismatch: expected {source.count}, got {table.num_rows}')
        rows[name] = table.to_pylist()
    return rows


def _select(rows, train_count):
    history = aime._history(rows['aime_dev'])
    blocked = set()
    for row in history:
        fingerprint = aime._identity(row['prompt'])
        if not fingerprint or fingerprint in blocked:
            raise ValueError(f"Historical AIME has an empty or duplicate normalized prompt at {row['id']}")
        blocked.add(fingerprint)
    # Screen against all six historical exams, not just the evaluation cohort.
    train, _ = aime._train(rows['math_train'], blocked)
    if len(train) < train_count:
        hint = (f'; use --train-count {len(train)} for the largest available pool'
                if len(train) >= SMALL_TRAIN_COUNT else '')
        raise ValueError(f'Only {len(train)} MATH rows qualify for this bundle ({train_count} requested){hint}')
    return train[:train_count], aime._order(history)[:DEV_COUNT]


def _write_bundle(directory, train, dev):
    entries = {}
    for split, rows, benchmark, topic in (
        ('train', train, 'math-counting-and-probability-train', 'counting_and_probability'),
        ('dev', dev, 'aime-2022-2024-dev', 'mixed'),
    ):
        entry = aime._entry(directory / f'{split}.jsonl', rows, benchmark)
        entries[split] = {**entry, 'topics': {topic: len(rows)}}
    (directory / MANIFEST).write_bytes(aime._json_bytes({'schema_version': 1, 'splits': entries}))


def prepare(output_dir, *, train_count=DEFAULT_TRAIN_COUNT, source_dir=None, fetch=False):
    """Publish a new local bundle and return its absolute manifest path.

    Train is selected in the original revision-and-ID hash order: its first
    eight rows are the small-pool control. Dev is the same hash-ordered prefix
    of sixty 2022–24 AIME problems at every pool size. Prompt normalization
    detects exact matches after formatting changes, not paraphrases.
    """
    if bool(fetch) == (source_dir is not None):
        raise ValueError('Choose exactly one of --fetch or --source-dir')
    maximum = aime._SOURCES['math_train'].count
    if type(train_count) is not int or not SMALL_TRAIN_COUNT <= train_count <= maximum:
        raise ValueError(f'train_count must be an integer in [{SMALL_TRAIN_COUNT}, {maximum}]')
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output directory already exists: {output}')
    rows = _read_sources(Path(source_dir).expanduser() if source_dir is not None else None)
    train, dev = _select(rows, train_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.aime-scale-') as temporary:
        staging = Path(temporary) / 'bundle'
        staging.mkdir(mode=0o700)
        _write_bundle(staging, train, dev)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Output directory already exists: {output}')
        staging.rename(output)
    return output / MANIFEST


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, help='new local (not vendored) bundle directory')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fetch', action='store_true', help='opt in to the two pinned HTTP fetches')
    source.add_argument('--source-dir', type=Path, help='offline inputs under the paths from --list-sources')
    source.add_argument('--list-sources', action='store_true', help='show the two input pins without fetching')
    parser.add_argument('--train-count', type=int, default=DEFAULT_TRAIN_COUNT,
                        help=f'training pool size (default: {DEFAULT_TRAIN_COUNT}; min: {SMALL_TRAIN_COUNT})')
    args = parser.parse_args(argv)
    if args.list_sources:
        print(aime._json_bytes({'sources': {name: aime._SOURCES[name].provenance() for name in _SOURCE_NAMES},
                                'licensing': _LICENSE_NOTE}).decode('utf-8'), end='')
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required to prepare a bundle')
    try:
        manifest = prepare(args.output_dir, train_count=args.train_count, source_dir=args.source_dir, fetch=args.fetch)
    except (ValueError, OSError) as error:
        parser.exit(1, f'AIME scale preparation failed: {error}\n')
    print(f'Development scale manifest: {manifest}')
    print(_LICENSE_NOTE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
