"""Freeze pre-existing GSM8K train/dev/test JSONL without score-based selection."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source_dir, output_dir, *, train=64, dev=32, test=64):
    source, output = Path(source_dir), Path(output_dir)
    if any(type(n) is not int or n < 1 for n in (train, dev, test)):
        raise ValueError('Split sizes must be positive integers')
    selected, manifest = {}, {'selection': 'first-N-in-existing-frozen-order-before-evaluation', 'splits': {}}
    for split, size in (('train', train), ('dev', dev), ('test', test)):
        path = source / (split + '.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) < size:
            raise ValueError('Insufficient ' + split + ' rows')
        selected[split] = rows[:size]
        manifest['splits'][split] = {'count': size, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        for key in ('id', 'prompt'):
            if {r[key] for r in selected[left]} & {r[key] for r in selected[right]}:
                raise ValueError('Data splits overlap by ' + key)
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in selected.items():
        path = output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        manifest['splits'][split]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train', type=int, default=64)
    parser.add_argument('--dev', type=int, default=32)
    parser.add_argument('--test', type=int, default=64)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, train=args.train, dev=args.dev, test=args.test)
