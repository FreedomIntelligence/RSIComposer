"""Pin/download local Qwen weights and create a deterministic GSM8K pilot split.

The development partition comes only from the official training split. The full
1319-item official test is exported separately and never used for self-training.
Requires pyarrow only when preparing data; no model or dataset code is executed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
from pathlib import Path
import random
import time
import urllib.request

MODEL_REPO = 'Qwen/Qwen2.5-7B'
MODEL_REVISION = 'd149729398750b98c0af14eb82c78cfe92750796'
DATA_REPO = 'openai/gsm8k'
DATA_REVISION = '740312add88f781978c0658806c59bc2815b9866'


def metadata(repo, revision, *, dataset=False):
    kind = 'datasets' if dataset else 'models'
    with urllib.request.urlopen(f'https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true', timeout=30) as stream:
        data = json.load(stream)
    if data['sha'] != revision:
        raise ValueError('Upstream revision mismatch')
    return data


def fetch(repo, revision, row, root, *, dataset=False):
    name = row['rfilename']
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Invalid upstream filename')
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'datasets/' if dataset else ''
    url = f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}'
    expected = row.get('lfs', {}).get('sha256')
    for attempt in range(3):
        try:
            if not target.is_file() or target.stat().st_size != row['size']:
                temporary = target.with_name(target.name + '.partial')
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                    while block := response.read(8 * 1024 * 1024):
                        output.write(block)
                if temporary.stat().st_size != row['size']:
                    raise ValueError('Download length mismatch')
                temporary.replace(target)
            with target.open('rb') as stream:
                digest = file_digest(stream, 'sha256').hexdigest()
            if expected and digest != expected:
                target.unlink()
                raise ValueError('Downloaded LFS SHA256 mismatch')
            result = {'name': name, 'bytes': target.stat().st_size, 'sha256': digest,
                      'upstream_lfs_match': digest == expected if expected else None}
            print(json.dumps(result), flush=True)
            return result
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 + attempt * 3)


def split_train(rows, train_samples, eval_samples, seed):
    if train_samples < 1 or eval_samples < 1 or train_samples + eval_samples > len(rows):
        raise ValueError('Invalid training/development partition sizes')
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    def convert(chosen):
        return [{'id': f'gsm8k/main/train/{index}', 'prompt': rows[index]['question'],
                 'answer': rows[index]['answer']} for index in chosen]
    train, dev = convert(indices[:train_samples]), convert(indices[train_samples:train_samples + eval_samples])
    if {r['prompt'].strip() for r in train} & {r['prompt'].strip() for r in dev}:
        raise ValueError('Duplicate questions cross train/development partitions')
    return train, dev


def write_rows(path, rows):
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {'path': str(path.resolve()), 'count': len(rows), 'sha256': sha256(path.read_bytes()).hexdigest()}


def few_shot_rows(rows, excluded, count=8):
    if type(count) is not int or count < 1:
        raise ValueError('few-shot count must be positive')
    seen = {r['prompt'].strip() for r in excluded}
    selected = []
    for index, row in enumerate(rows):
        if row['question'].strip() in seen:
            continue
        seen.add(row['question'].strip())
        selected.append({'id': f'gsm8k/main/train/{index}', 'prompt': row['question'], 'answer': row['answer']})
        if len(selected) == count:
            return selected
    raise ValueError('Not enough distinct few-shot questions outside target partitions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-repo', default=MODEL_REPO)
    parser.add_argument('--model-revision', default=MODEL_REVISION)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--train-samples', type=int, default=64)
    parser.add_argument('--eval-samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=73)
    args = parser.parse_args()
    model_dir, data_dir = args.model_dir.resolve(), args.data_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=False)
    model_meta = metadata(args.model_repo, args.model_revision)
    (data_dir / 'model-upstream.json').write_text(json.dumps(model_meta, indent=2))
    files = [row for row in model_meta['siblings'] if '/' not in row['rfilename'] and not row['rfilename'].startswith('.')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifests = list(pool.map(lambda row: fetch(args.model_repo, args.model_revision, row, model_dir), files))
    data_meta = metadata(DATA_REPO, DATA_REVISION, dataset=True)
    (data_dir / 'data-upstream.json').write_text(json.dumps(data_meta, indent=2))
    parquet = [row for row in data_meta['siblings'] if row['rfilename'].startswith('main/') and row['rfilename'].endswith('.parquet')]
    raw_manifest = [fetch(DATA_REPO, DATA_REVISION, row, data_dir / 'upstream', dataset=True) for row in parquet]
    import pyarrow.parquet as pq
    rows = pq.read_table(data_dir / 'upstream/main/train-00000-of-00001.parquet').to_pylist()
    test_rows = pq.read_table(data_dir / 'upstream/main/test-00000-of-00001.parquet').to_pylist()
    train, dev = split_train(rows, args.train_samples, args.eval_samples, args.seed)
    test = [{'id': f'gsm8k/main/test/{index}', 'prompt': row['question'], 'answer': row['answer']}
            for index, row in enumerate(test_rows)]
    demonstrations = few_shot_rows(rows, train + dev + test)
    report = {'model': {'repository': args.model_repo, 'revision': args.model_revision, 'path': str(model_dir), 'files': manifests},
        'dataset': {'repository': DATA_REPO, 'revision': DATA_REVISION, 'raw': raw_manifest},
        'seed': args.seed, 'official_test_used_for_training_or_iteration_eval': False,
        'train': write_rows(data_dir / 'train.jsonl', train),
        'dev': write_rows(data_dir / 'dev.jsonl', dev),
        'test': write_rows(data_dir / 'test.jsonl', test)}
    report['few_shot'] = write_rows(data_dir / 'few-shot.jsonl', demonstrations)
    (data_dir / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'prepared': str(data_dir / 'manifest.json'), 'train': len(train), 'dev': len(dev), 'test': len(test)}), flush=True)


if __name__ == '__main__':
    main()
