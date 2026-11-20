"""The official math benchmark bundle as a task-manifest entry loader."""
import json

from rsisys.labs.tasks import TaskItem, checked_bytes, manifest_path
from rsisys.labs.verifiers import CompetitionIntegerVerifier

from .benchmarks import BENCHMARKS, load_benchmark


def official_tasks(root, entry, sealed):
    """AIME 2024, AIME 2025 and AIMO3 Reference together, for a sealed split."""
    if not sealed:
        raise ValueError('The official math preset requires a sealed split')
    if set(entry) != {'benchmarks_manifest', 'sha256'}:
        raise ValueError('The official math preset requires benchmarks_manifest and sha256')
    path = manifest_path(root, entry['benchmarks_manifest'])
    manifest = json.loads(checked_bytes(path, entry['sha256']))
    if (not isinstance(manifest, dict) or not isinstance(manifest.get('benchmarks'), dict)
            or set(manifest['benchmarks']) != set(BENCHMARKS)):
        raise ValueError('Official evaluation requires AIME 2024, AIME 2025 and AIMO3 Reference together')
    rows = []
    for name, spec in BENCHMARKS.items():
        benchmark = load_benchmark(path, name, manifest=manifest)
        verifier = CompetitionIntegerVerifier(spec.maximum)
        rows.extend(TaskItem(row.id, row.prompt, row.answer, name, verifier, str(row.answer))
                    for row in benchmark.problems)
    return tuple(rows), path, entry['sha256']
