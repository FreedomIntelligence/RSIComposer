"""Freeze an AIDE GSM8K inference-program search task from existing JSONL data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def prepare(train, validation, heldout, output, *, train_size=8, validation_size=32, heldout_size=64):
    groups = {}
    sources = {}
    for label, path, size in (('train', train, train_size), ('validation', validation, validation_size),
                              ('heldout', heldout, heldout_size)):
        path = Path(path)
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if type(size) is not int or size < 1 or len(rows) < size:
            raise ValueError('Insufficient ' + label + ' rows')
        selected = rows[:size]
        if any(not all(isinstance(row.get(k), str) and row[k].strip() for k in ('id', 'prompt', 'answer'))
               for row in selected):
            raise ValueError('Invalid ' + label + ' rows')
        groups[label] = selected
        sources[label] = {'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rows), 'selected': size}
    for key in ('id', 'prompt'):
        values = [r[key] for rows in groups.values() for r in rows]
        if len(values) != len(set(values)):
            raise ValueError('Frozen data overlap by ' + key)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    public = output / 'public'
    public.mkdir()
    for label in ('train', 'validation'):
        columns = ['id', 'prompt', 'answer'] if label == 'train' else ['id', 'prompt']
        with (public / (label + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: r[k] for k in columns} for r in groups[label])
    for label in ('validation', 'heldout'):
        (output / (label + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in groups[label]))
    manifest = {'selection': 'fixed prefix before any new evaluation', 'sources': sources,
                'files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in output.rglob('*') if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train', 'validation', 'heldout', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--train-size', type=int, default=8)
    parser.add_argument('--validation-size', type=int, default=32)
    parser.add_argument('--heldout-size', type=int, default=64)
    args = vars(parser.parse_args())
    prepare(**args)


if __name__ == '__main__':
    main()
