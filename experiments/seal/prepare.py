"""Prepare a pinned upstream SQuAD document subset without ML dependencies."""
import argparse
import json
import random
import urllib.request
from hashlib import sha256
from pathlib import Path

REVISION = '6d9c9f9ee392c6cc618e771f399d436d190f6ca4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--documents', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7309)
    parser.add_argument('--paper', action='store_true', help='Use published disjoint round-1/round-2 documents and the 200-document/974-question evaluation set')
    args = parser.parse_args()
    if args.paper:
        prepare_paper(args)
        return
    if args.documents < 1:
        parser.error('documents must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'upstream_revision': REVISION, 'seed': args.seed, 'source': {}, 'outputs': {}}
    contexts = set()
    for split, filename in [('train', 'squad_train.json'), ('heldout', 'squad_val.json')]:
        relative = 'general-knowledge/data/' + filename
        if args.upstream:
            content = (args.upstream / relative).read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/Continual-Intelligence/SEAL/{REVISION}/{relative}'
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
        manifest['source'][filename] = sha256(content).hexdigest()
        rows = json.loads(content)
        random.Random(args.seed).shuffle(rows)
        subset = []
        for row in rows:
            if row['context'].strip() not in contexts:
                contexts.add(row['context'].strip())
                subset.append(row)
            if len(subset) == args.documents:
                break
        if len(subset) != args.documents:
            raise ValueError('Insufficient disjoint documents')
        path = args.output / f'{split}.json'
        path.write_text(json.dumps(subset, ensure_ascii=False, indent=2))
        manifest['outputs'][path.name] = {'sha256': sha256(path.read_bytes()).hexdigest(),
            'documents': len(subset), 'questions': sum(len(r['questions']) for r in subset)}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def prepare_paper(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'upstream_revision': REVISION, 'profile': 'paper', 'source': {}, 'outputs': {}}
    seen = set()
    for name, relative, count in (
            ('train-round-1', 'train/iter0_train.json', 50),
            ('train-round-2', 'train/iter1_train.json', 50),
            ('heldout', 'eval/base_val.json', 200)):
        relative = 'general-knowledge/data/synthetic_data/' + relative
        if args.upstream:
            content = (args.upstream / relative).read_bytes()
        else:
            with urllib.request.urlopen(f'https://raw.githubusercontent.com/Continual-Intelligence/SEAL/{REVISION}/{relative}', timeout=120) as response:
                content = response.read()
        rows = json.loads(content)[:count]
        # Use only published document identities, never their pre-generated edits.
        rows = [{k: r[k] for k in ('title', 'context', 'questions')} for r in rows]
        contexts = {r['context'].strip() for r in rows}
        if len(rows) != count or len(contexts) != count or seen & contexts:
            raise ValueError('Published document splits changed or overlap')
        seen.update(contexts)
        path = args.output / f'{name}.json'
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        manifest['source'][relative] = sha256(content).hexdigest()
        manifest['outputs'][path.name] = dict(sha256=sha256(path.read_bytes()).hexdigest(),
            documents=count, questions=sum(len(r['questions']) for r in rows))
    if manifest['outputs']['heldout.json']['questions'] != 974:
        raise ValueError('Expected published 974-question evaluation set')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
