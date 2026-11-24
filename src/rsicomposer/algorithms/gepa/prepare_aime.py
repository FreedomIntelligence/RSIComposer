"""Freeze the author AIME benchmark into a MathLab manifest."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from .protocol import aime_protocol


def prepare(destination, artifact_dir, dspy_dir):
    official, *_ = aime_protocol(artifact_dir, dspy_dir)
    benchmark = official.AIMEBench()
    test = list(benchmark.test_set[:30])
    if (len(benchmark.train_set) != 45 or len(benchmark.val_set) != 45 or len(test) != 30
            or [row.problem for row in benchmark.test_set] != [row.problem for row in test] * 5):
        raise ValueError('Unexpected author AIME split sizes or test repetitions')
    destination.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split, examples in (('search', benchmark.train_set), ('validation', benchmark.val_set), ('test', test)):
        rows = [{'id': f'{split}-{index:02d}', 'prompt': row.problem, 'answer': str(row.answer),
                 **({'solution': row.solution} if split == 'search' else {})}
                for index, row in enumerate(examples)]
        contents = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
        file = f'{split}.jsonl'
        (destination / file).write_bytes(contents)
        splits[split] = {'file': file, 'sha256': sha256(contents).hexdigest(), 'count': len(rows),
                         'benchmark': 'AIME2025' if split == 'test' else 'AIME2022-2024',
                         'verifier': 'gepa_aime_integer'}
    (destination / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'splits': splits}, indent=2) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--dspy-dir', required=True)
    args = parser.parse_args()
    prepare(args.destination, args.artifact_dir, args.dspy_dir)
