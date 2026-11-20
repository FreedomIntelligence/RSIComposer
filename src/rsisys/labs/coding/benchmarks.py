"""Coding benchmarks a CodingLab mounts: each lists its Harbor tasks and owns its budgets.

A benchmark entry names the manifest of its frozen tasks and may set the
protocol: step, token and time budgets and thinking. Unset budgets take the
defaults below and are recorded with the benchmark, so protocol_id (see
rsisys.loops.contracts.benchmark) identifies exactly how its tasks are scored.
The Lab never applies a budget of its own to a benchmark.

    [labs.coding.benchmarks.swebench-verified]
    manifest = "/data/swebench-verified.json"
    max_steps = 48
    [labs.coding.benchmarks."terminal-bench-2.1"]
    manifest = "/data/terminal-bench-2.1.json"
    max_steps = 96
    agent_timeout = 3600

Splits with the same name are merged in benchmark name order; each task row
keeps its benchmark, whose protocol it is run under.
"""
from hashlib import sha256
import json
from pathlib import Path

from rsisys.loops.contracts.benchmark import mount_benchmarks

KIND = 'coding-task'
PROTOCOL_DEFAULTS = {'max_steps': 48, 'max_tokens': 4096, 'total_tokens': 98304,
                     'agent_timeout': 1800, 'verifier_timeout': 1800, 'thinking': False}


def validate_benchmark_config(config):
    benchmarks = mount_benchmarks(config.get('benchmarks'), kind=KIND, source_keys=('manifest',),
                                  protocol_defaults=PROTOCOL_DEFAULTS)
    for benchmark in benchmarks:
        for key, default in PROTOCOL_DEFAULTS.items():
            value = benchmark.protocol[key]
            if type(value) is not type(default) or (type(default) is int and value < 1):
                raise ValueError(f'Benchmark {benchmark.name!r} {key} must be a '
                                 + ('boolean' if type(default) is bool else 'positive integer'))
    return benchmarks


def _read_manifest(path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError(f'Expected a JSON object in coding manifest: {path}')
    return manifest, sha256(raw).hexdigest()


def _splits(manifest):
    return manifest.get('splits', {k: manifest[k] for k in ('search', 'heldout') if k in manifest})


def load_task_manifests(config):
    """(combined manifest, merged splits, mounted benchmarks) for a validated config."""
    benchmarks = validate_benchmark_config(config)
    sources, merged = {}, {}
    for benchmark in benchmarks:
        name, path = benchmark.name, Path(benchmark.source['manifest']).resolve()
        manifest, digest = _read_manifest(path)
        splits = _splits(manifest)
        if not isinstance(splits, dict) or not splits:
            raise ValueError(f'Benchmark {name!r} requires nonempty named task splits')
        found = False
        for split, rows in sorted(splits.items()):
            if not isinstance(rows, list) or not rows:
                raise ValueError(f'Benchmark {name!r} split {split!r} requires a nonempty task list')
            if any(not isinstance(row, dict) or not isinstance(row.get('benchmark'), str) for row in rows):
                raise ValueError(f'Each row in {path} must declare its benchmark')
            # A manifest may hold several benchmarks' tasks (a tranche does); this one keeps its own.
            own = [row for row in rows if row['benchmark'] == name]
            if own:
                merged.setdefault(split, []).extend(own)
                found = True
        if not found:
            raise ValueError(f'Benchmark {name!r} has no tasks in {path}')
        sources[name] = {'manifest': str(path), 'sha256': digest, **benchmark.identity,
                         'protocol': benchmark.protocol}
    return {'splits': merged, 'benchmarks': sources}, merged, benchmarks
