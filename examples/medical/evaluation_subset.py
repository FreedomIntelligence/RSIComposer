"""Host-owned, reproducible subsets of already pinned evaluation datasets.

Selection depends only on a declared seed, benchmark and sample ID. Original
JSONL bytes and source order are preserved; neither answers nor model outputs
influence membership. Keep closed manifests and rows outside the Lab.
"""
import hashlib
import json
from pathlib import Path

from .official_protocol import official_inputs


def freeze_evaluation_subset(groups, destination, *, count=500,
                             seed='rsisys-medical-eval500-v1'):
    if type(count) is not int or count < 1 or not isinstance(seed, str) or not seed.strip():
        raise ValueError('Declare a positive subset count and nonempty seed')
    # Validate complete pinned inputs before creating any destination.
    official_inputs(groups)
    prepared = []
    for group in groups:
        benchmark = group['benchmark_id']
        raw = Path(group['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != group['sha256']:
            raise ValueError('Parent dataset changed during subset preparation')
        lines = [line for line in raw.splitlines(keepends=True) if line.strip()]
        if len(lines) < count:
            raise ValueError('Parent dataset has fewer rows than the requested fixed subset')
        ids = [json.loads(line)['id'] for line in lines]
        if any(not isinstance(ident, str) or not ident for ident in ids):
            raise ValueError('Subset selection requires nonempty string sample IDs')
        ranks = [hashlib.sha256(json.dumps([seed, benchmark, ident],
                     ensure_ascii=False, separators=(',', ':')).encode()).digest()
                 for ident in ids]
        chosen = set(sorted(range(len(lines)), key=lambda i: (ranks[i], ids[i]))[:count])
        indices = sorted(chosen)
        content = b''.join(lines[i] for i in indices)
        prepared.append((benchmark, content, {
            'benchmark_id': benchmark, 'parent_sha256': group['sha256'],
            'parent_rows': len(lines), 'rows': count,
            'sha256': hashlib.sha256(content).hexdigest(),
            'selected_indices': indices, 'selected_ids': [ids[i] for i in indices]}))
    destination = Path(destination).resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    result, manifest_groups = [], []
    for benchmark, content, record in prepared:
        path = destination / (benchmark + '.jsonl')
        path.write_bytes(content)
        path.chmod(0o600)
        result.append({'benchmark_id': benchmark, 'path': str(path), 'sha256': record['sha256']})
        manifest_groups.append({**record, 'path': path.name})
    manifest = {'schema': 'fixed-medical-evaluation-subset-v1',
                'selection': 'sha256-seed-benchmark-id; preserve-parent-order',
                'seed': seed, 'count_per_benchmark': count, 'groups': manifest_groups}
    manifest_path = destination / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    manifest_path.chmod(0o600)
    official_inputs(result)
    return result, manifest
