"""Prepare full pinned official test sets outside the research filesystem.

Public development is MedQA's official dev split minus exact test overlaps. MedXpertQA has no official
development split: its complete 2450-row Text test stays closed. Do not partition
or deduplicate test membership and then call the result a full official score.
Only exact-overlap exclusion and deterministic selection affect training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .task import samples


SOURCES = {
    'medxpertqa_text_input.jsonl': ('1752383562a639319d738e0e83bc76f76c1da039d4f93bece869080b6e6da43f', 2450,
        'TsinghuaC3I/MedXpertQA', '5f1db162d0d2b03d62da2d4fe086076a36105e18'),
    'medqa_input.jsonl': ('e75adf63a3ea1d67f8699a45df01adce4cb8faec3cbe3704721df040620b23ba', 1273,
        'TsinghuaC3I/MedXpertQA', '5f1db162d0d2b03d62da2d4fe086076a36105e18'),
    'medqa-official-dev.jsonl': ('44212909568dcdafe0e6536d402f6f63dbb371cecf9b57e17b2fc16b7165889f', 1272,
        'GBaker/MedQA-USMLE-4-options-hf', '17af9355ef89fba60de966eabaeba797c695f86e'),
    'medqa-official-test.jsonl': ('6104e893a8964aea725fad332aa98d64da43a1b8d210bbae0d357403a39efb10', 1273,
        'GBaker/MedQA-USMLE-4-options-hf', '17af9355ef89fba60de966eabaeba797c695f86e'),
    'phrases_no_exclude_train.jsonl': ('3a65baf760f17c395058d6699315a7ce8aa767c50cb193fea87ceb94dd790324', 10178,
        'GBaker/MedQA-USMLE-4-options', '0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18'),
}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode() + b'\n'


def stem(question):
    return ' '.join(question.rsplit('\nAnswer Choices:', 1)[0].casefold().split())


def convert_test(row, benchmark):
    labels = 'ABCDEFGHIJ' if benchmark == 'medxpertqa' else 'ABCD'
    choices = row['question'].rsplit('\nAnswer Choices:', 1)
    if len(choices) != 2 or not isinstance(row['label'], list) or len(row['label']) != 1:
        raise ValueError('Expected the original official single-answer question')
    if benchmark == 'medxpertqa':
        options = {o['letter']: o['content'] for o in row['options']}
        # The official evaluator sends question verbatim and uses options only
        # to determine its label range. Preserve upstream discrepancies rather
        # than silently rewriting or dropping official test questions.
    elif 'options' in row:
        options = row['options']
    else:
        parts = re.split(r'\(([A-D])\) ', choices[1].strip())
        if parts[0] or ''.join(parts[1::2]) != labels:
            raise ValueError('Cannot losslessly parse official MedQA options')
        options = {key: value.strip() for key, value in zip(parts[1::2], parts[2::2])}
    result = {'id': benchmark + ':' + row['id'], 'benchmark_id': benchmark,
              'question': row['question'], 'options': options, 'gold': row['label'][0]}
    samples(encoded(result))
    return result


def prepare(source_root, public, private, *, train_count=1024):
    source_root, public, private = (Path(p).resolve() for p in (source_root, public, private))
    if (public.exists() or private.exists() or public.is_relative_to(private)
            or private.is_relative_to(public) or public.is_relative_to(source_root)
            or private.is_relative_to(source_root)):
        raise ValueError('Use new, separate public/private outputs outside the source cache')
    if type(train_count) is not int or train_count < 1:
        raise ValueError('Positive training size required')
    upstream = {}
    for name, (sha, count, repository, revision) in SOURCES.items():
        raw = (source_root / name).read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if hashlib.sha256(raw).hexdigest() != sha or len(rows) != count:
            raise ValueError('Pinned official source differs: ' + name)
        upstream[name] = rows
    medqa_native = {r['id']: r for r in upstream['medqa-official-test.jsonl']}
    medqa_test = []
    for r in upstream['medqa_input.jsonl']:
        native = medqa_native[r['id']]
        options = {letter: native['ending' + str(i)] for i, letter in enumerate('ABCD')}
        expected = native['sent1'] + '\nAnswer Choices: ' + ' '.join(f'({k}) {v}' for k, v in options.items())
        if expected != r['question'] or ['ABCD'[native['label']]] != r['label']:
            raise ValueError('Official MedQA prepared and native test differ')
        medqa_test.append(convert_test({**r, 'options': options}, 'medqa'))
    closed = {'test.medxpertqa.jsonl': [convert_test(r, 'medxpertqa') for r in upstream['medxpertqa_text_input.jsonl']],
              'test.medqa.jsonl': medqa_test}
    dev = []
    for r in upstream['medqa-official-dev.jsonl']:
        options = {letter: r['ending' + str(i)] for i, letter in enumerate('ABCD')}
        question = r['sent1'] + '\nAnswer Choices: ' + ' '.join(f'({k}) {v}' for k, v in options.items())
        dev.append(convert_test({'id': r['id'], 'question': question, 'label': ['ABCD'[r['label']]],
                                 'options': options}, 'medqa'))
    protected = {stem(r['question']) for rows in closed.values() for r in rows}
    dev_before = len(dev)
    dev = [r for r in dev if stem(r['question']) not in protected]
    blocked = protected | {stem(r['question']) for r in dev}
    train, seen = [], set(blocked)
    for i, r in enumerate(upstream['phrases_no_exclude_train.jsonl']):
        key = stem(r['question'])
        if key in seen:
            continue
        seen.add(key)
        if r['answer'] != r['options'][r['answer_idx']]:
            raise ValueError('Training answer fields disagree')
        train.append({'id': f'medqa:train-{i:05d}', 'benchmark_id': 'medqa',
            'question': r['question'], 'options': r['options'], 'gold': r['answer_idx']})
    train.sort(key=lambda r: hashlib.sha256(('20261107:' + r['id']).encode()).hexdigest())
    if len(train) < train_count:
        raise ValueError('Insufficient overlap-free training data')
    public_files = {'train.medqa.jsonl': train[:train_count], 'dev.medqa.jsonl': dev}
    def write_files(root, files):
        root.mkdir(parents=True, mode=0o700)
        manifest = {}
        for name, rows in files.items():
            data = b''.join(encoded(r) for r in rows)
            samples(data)
            (root / name).write_bytes(data)
            manifest[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'rows': len(rows)}
        return manifest
    private_manifest = write_files(private, closed)
    public_manifest = write_files(public, public_files)
    task = {'train': 'train.medqa.jsonl', 'development': [
        {'path': 'dev.medqa.jsonl', 'benchmark_id': 'medqa', 'role': 'primary'}]}
    (public / 'task.json').write_bytes(encoded(task))
    # Private source identities and full-test memberships never enter task.json.
    (public / 'manifest.json').write_bytes(encoded({'files': public_manifest, 'task': task,
        'scope': 'MedQA official dev minus exact test overlaps; no MedXpertQA test preview'}))
    manifest = {'sources': SOURCES, 'public_files': public_manifest, 'closed_files': private_manifest,
        'test_membership': 'full official, original order, no test exclusions',
        'training_eligible_rows': len(train), 'training_selected': train_count,
        'public_dev_exact_test_overlaps_excluded': dev_before - len(dev),
        'upstream_choice_metadata_conflicts_preserved': sum(
            ' '.join(r['question'].rsplit('\nAnswer Choices:', 1)[1].split()) !=
            ' '.join(' '.join(f"({o['letter']}) {o['content']}" for o in r['options']).split())
            for r in upstream['medxpertqa_text_input.jsonl']),
        'closed_release_policy': 'aggregate accuracy and n only, at most once per 1800 seconds'}
    (private / 'manifest.json').write_bytes(encoded(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--private-output', required=True)
    parser.add_argument('--train-count', type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.output, args.private_output, train_count=args.train_count)))


if __name__ == '__main__':
    main()
