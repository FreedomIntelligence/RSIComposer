"""Generate disjoint toy datasets; this format belongs only to this example."""
import argparse
import json
from pathlib import Path


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for name, numbers in [('train', range(20, 84)), ('eval', range(100, 112))]:
        rows = [dict(prompt=f'Return only JSON with key "parity" and value "even" or "odd" for integer {n}.',
                     answer=json.dumps({'parity': 'even' if n % 2 == 0 else 'odd'}))
                for n in numbers]
        target = root / f'{name}.jsonl'
        content = ''.join(json.dumps(row) + '\n' for row in rows)
        if target.exists() and target.read_text() != content:
            raise ValueError(f'Refusing to overwrite different data: {target}')
        target.write_text(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    prepare(parser.parse_args().run_dir)
