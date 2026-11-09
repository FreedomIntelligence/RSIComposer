"""Prepare disjoint MATH pilot partitions and the complete held-out MATH-500."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

try:
    from .prepare_gsm8k import fetch, metadata, write_rows
except ImportError:
    from prepare_gsm8k import fetch, metadata, write_rows

MATH_REPO = 'EleutherAI/hendrycks_math'
MATH_REVISION = '21a5633873b6a120296cce3e2df9d5550074f4a3'
TEST_REPO = 'HuggingFaceH4/MATH-500'
TEST_REVISION = '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'


def partition(rows, test, train_size=128, dev_size=128, shots=4, seed=73):
    if any(type(n) is not int or n < 1 for n in (train_size, dev_size, shots)):
        raise ValueError('Partition sizes must be positive integers')
    seen = {r['prompt'].strip() for r in test}
    unique = []
    for row in rows:
        if row['prompt'].strip() not in seen:
            unique.append(row)
            seen.add(row['prompt'].strip())
    if len(unique) < train_size + dev_size + shots:
        raise ValueError('Not enough unique non-test questions')
    random.Random(seed).shuffle(unique)
    return (unique[:train_size], unique[train_size:train_size + dev_size],
            unique[train_size + dev_size:train_size + dev_size + shots])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--train-samples', type=int, default=128)
    parser.add_argument('--eval-samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=73)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    upstream = metadata(MATH_REPO, MATH_REVISION, dataset=True)
    test_upstream = metadata(TEST_REPO, TEST_REVISION, dataset=True)
    (root / 'math-upstream.json').write_text(json.dumps(upstream, indent=2))
    (root / 'math500-upstream.json').write_text(json.dumps(test_upstream, indent=2))
    files = [r for r in upstream['siblings'] if r['rfilename'].endswith('.parquet')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = list(pool.map(lambda r: fetch(MATH_REPO, MATH_REVISION, r,
                                          root / 'upstream/math', dataset=True), files))
    test_file = next(r for r in test_upstream['siblings'] if r['rfilename'] == 'test.jsonl')
    test_raw = fetch(TEST_REPO, TEST_REVISION, test_file, root / 'upstream/math500', dataset=True)
    import pyarrow.parquet as pq
    rows = []
    for path in sorted((root / 'upstream/math').glob('*/train-*.parquet')):
        subject = path.parent.name
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            rows.append({'id': f'math/{subject}/train/{index}', 'prompt': row['problem'],
                         'answer': row['solution'], 'subject': row['type'], 'level': row['level']})
    test = []
    for row in map(json.loads, (root / 'upstream/math500/test.jsonl').read_text().splitlines()):
        test.append({'id': f'math500/{row["unique_id"]}', 'prompt': row['problem'],
                     'answer': row['solution'], 'subject': row['subject'], 'level': row['level']})
    assert len(rows) == 7500 and len(test) == 500, (len(rows), len(test))
    assert len({r['prompt'].strip() for r in test}) == 500
    train, dev, shots = partition(rows, test, args.train_samples, args.eval_samples, 4, args.seed)
    # Never silently remove unsupported evaluation questions from the denominator.
    from rsisys.labs.math_verifiers import MathVerifyVerifier
    verifier = MathVerifyVerifier()
    invalid = [r['id'] for r in train + dev + shots + test if verifier.parse(r['answer']) is None]
    if invalid:
        raise ValueError(f'Unparseable gold answers; fix verifier before running: {invalid}')
    report = {'dataset': {'repository': MATH_REPO, 'revision': MATH_REVISION, 'raw': raw},
              'heldout': {'repository': TEST_REPO, 'revision': TEST_REVISION, 'raw': test_raw},
              'seed': args.seed, 'verifier': verifier.name,
              'official_test_used_for_training_or_iteration_eval': False}
    for label, items in (('train', train), ('dev', dev), ('few-shot', shots), ('test', test)):
        report[label] = write_rows(root / (label + '.jsonl'), items)
    (root / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'prepared': str(root / 'manifest.json'), 'train': len(train),
                      'dev': len(dev), 'few_shot': len(shots), 'test': len(test)}), flush=True)


if __name__ == '__main__':
    main()
