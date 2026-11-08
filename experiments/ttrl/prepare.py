"""Split pinned upstream MATH-500 into unlabeled adaptation and transfer sets."""
import argparse
import json
import random
import urllib.request
from hashlib import sha256
from pathlib import Path

REVISION = '5806e119789132fcb43552beaf2532d7d0447213'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--upstream', type=Path)
    p.add_argument('--target', type=int, default=32)
    p.add_argument('--transfer', type=int, default=128)
    p.add_argument('--seed', type=int, default=7309)
    args = p.parse_args()
    if args.target < 1 or args.transfer < 1:
        p.error('target and transfer counts must be positive')
    relative = 'verl/data/MATH-TTT/train.json'
    if args.upstream:
        content = (args.upstream / relative).read_bytes()
    else:
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/PRIME-RL/TTRL/{REVISION}/{relative}', timeout=120) as r:
            content = r.read()
    rows = json.loads(content)
    random.Random(args.seed).shuffle(rows)
    if len(rows) < args.target + args.transfer or len({r['prompt'].strip() for r in rows}) != len(rows):
        raise ValueError('Insufficient unique questions')
    args.output.mkdir(parents=True, exist_ok=False)
    target, transfer = rows[:args.target], rows[args.target:args.target + args.transfer]
    train = [{'id': r['id'], 'prompt': r['prompt']} for r in target]
    evaluation = {key: [{'id': r['id'], 'prompt': r['prompt'], 'answer': r'\boxed{' + r['answer'] + '}'} for r in data]
                  for key, data in [('target', target), ('transfer', transfer)]}
    manifest = {'upstream_revision': REVISION, 'source_sha256': sha256(content).hexdigest(),
                'seed': args.seed, 'target': len(target), 'transfer': len(transfer), 'files': {}}
    for name, data in [('unlabeled.json', train), ('evaluation.json', evaluation)]:
        path = args.output / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        manifest['files'][name] = sha256(path.read_bytes()).hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
