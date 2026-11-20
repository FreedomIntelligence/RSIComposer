"""Compose installed Harbor task manifests into one frozen experiment cohort."""
from hashlib import sha256
import json
from pathlib import Path
import re


def _path(value):
    return isinstance(value, str) and bool(value.strip())


def validate_benchmark_config(config):
    if ('manifest' in config) == ('benchmarks' in config):
        raise ValueError('Coding Lab requires exactly one of manifest or benchmarks')
    if 'manifest' in config:
        if not _path(config['manifest']):
            raise ValueError('Coding Lab manifest must be a path')
        return
    benchmarks = config['benchmarks']
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError('benchmarks must be a nonempty mapping of names to manifest entries')
    for name, entry in benchmarks.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError(f'Invalid coding benchmark name: {name!r}')
        if not isinstance(entry, dict) or set(entry) != {'manifest'} or not _path(entry['manifest']):
            raise ValueError(f'Benchmark {name!r} requires exactly one manifest path')


def _read_manifest(path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError(f'Expected a JSON object in coding manifest: {path}')
    return manifest, sha256(raw).hexdigest()


def _splits(manifest):
    return manifest.get('splits', {k: manifest[k] for k in ('search', 'heldout') if k in manifest})


def load_task_manifests(config):
    if 'manifest' in config:
        source = Path(config['manifest']).resolve()
        manifest, _ = _read_manifest(source)
        return manifest, _splits(manifest)

    sources, merged = {}, {}
    for name, entry in sorted(config['benchmarks'].items()):
        path = Path(entry['manifest']).resolve()
        manifest, digest = _read_manifest(path)
        splits = _splits(manifest)
        if not isinstance(splits, dict) or not splits:
            raise ValueError(f'Benchmark {name!r} requires nonempty named task splits')
        for split, rows in sorted(splits.items()):
            if not isinstance(rows, list) or not rows:
                raise ValueError(f'Benchmark {name!r} split {split!r} requires a nonempty task list')
            if any(not isinstance(row, dict) or row.get('benchmark') != name for row in rows):
                raise ValueError(f'Each row in {path} must declare benchmark {name!r}')
            merged.setdefault(split, []).extend(rows)
        sources[name] = {'manifest': str(path), 'sha256': digest}
    return {'splits': merged, 'benchmarks': sources}, merged
