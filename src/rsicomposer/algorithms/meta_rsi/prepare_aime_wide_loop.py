"""Convert a local dev-only AIME scale run into ordinary MetaRSI manifests.

The parent and child mount the *same* unchanged 256 MATH training originals:
MetaRSI requires this for candidate inheritance. Their development questions
are disjoint: 45 parent (22 search, 23 validation) and 15 child (7 search,
8 validation). Allocate each cohort by exam largest remainder (ties by exam
ID), then rank rows by SHA-256 of a versioned purpose and stable ID. No score
or rollout outcome participates in dev assignment.

Only the prepared manifest, train/dev JSONLs, run plan and four rollout
receipts are inputs. There is no raw-source access, sealed mount, model load,
or fit. Disjointness is checked within these inputs, not against unaccessed
corpora, paraphrases or model-pretraining data. Repartitioning dev does not
turn prior scored evaluations on these questions into a blind holdout. Keep
all outputs private.

The separate bootstrap.json stores the 228 verified rollouts in receipt order.
Pin its raw file SHA-256 when adopting it; --source-plan-sha256 can additionally
pin the input plan's canonical digest as recorded by the scale probe.
"""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from rsicomposer.labs.math.tasks import Problem

from . import aime, aime_scale_probe as probe, prepare_aime as original, prepare_aime_scale as scale
from .engine import digest


BOOTSTRAP = 'bootstrap.json'
ROLLOUT_COUNT = 228
RECEIPTS = tuple(f'wide-rollout-{i:02d}' for i in range(4))
_DOMAIN = 'meta-rsi/aime-wide-loop/v1'
_OUTCOMES = {'length', 'prompt_budget', 'invalid_format', 'correct', 'incorrect'}
_TRAIN_BENCHMARK = 'math-counting-and-probability-train'
_DEV_BENCHMARK = 'aime-2022-2024-dev'


def _sha(raw):
    return sha256(raw).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _plain_directory(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f'Expected a local directory without symlinks: {path}')
    return path


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular local input file without symlinks: {path}')
    return path.read_bytes()


def _json(raw):
    def unique_fields(pairs):
        record = dict(pairs)
        if len(record) != len(pairs):
            raise ValueError('Duplicate JSON field in wide-loop input')
        return record

    def reject_constant(value):
        raise ValueError(f'Non-finite number in wide-loop input: {value}')

    return json.loads(raw, object_pairs_hook=unique_fields, parse_constant=reject_constant)


def _check_row(row, role):
    fields = {'id', 'prompt', 'answer', 'competition', 'source', 'source_id'}
    if role == 'dev':
        fields |= {'problem_number', 'source_url'}
    if (not isinstance(row, dict) or row.keys() != fields
            or not isinstance(row['id'], str) or not isinstance(row['prompt'], str)
            or not original._identity(row['prompt']) or not isinstance(row['answer'], str)
            or not re.fullmatch(r'[0-9]{3}', row['answer']) or type(row['source_id']) is not int):
        raise ValueError(f'Invalid prepared {role} row shape')
    if role == 'train':
        if (row['source'] != 'math_train' or row['competition'] is not None
                or not 0 <= row['source_id'] <= 9999
                or row['id'] != f"{_TRAIN_BENCHMARK}-{row['source_id']:04d}"):
            raise ValueError('Invalid source identity in prepared MATH training rows')
    else:
        url = row['source_url']
        match = original._DEV_URL.fullmatch(url) if isinstance(url, str) else None
        if not match:
            raise ValueError('Expected a historical 2022–2024 development source')
        year, exam, number = match.groups()
        competition = f'aime-{year}-{exam.lower()}'
        if (row['source'] != 'aime_dev' or row['competition'] != competition
                or row['id'] != f'{competition}-{int(number):02d}'
                or type(row['problem_number']) is not int or row['problem_number'] != int(number)):
            raise ValueError('Invalid source identity in prepared historical AIME rows')


def _load_bundle(path):
    if path.name != scale.MANIFEST:
        raise ValueError('Expected the prepared dev-only scale manifest.json')
    _plain_directory(path.parent)
    manifest_raw = _read(path)
    manifest = _json(manifest_raw)
    if (not isinstance(manifest, dict) or manifest.keys() != {'schema_version', 'splits'}
            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1
            or not isinstance(manifest['splits'], dict) or manifest['splits'].keys() != {'train', 'dev'}):
        raise ValueError('Expected exactly prepared train/dev splits, with no sealed role')
    inputs, hashes, files, originals = manifest['splits'], {}, {}, {}
    for role, count, benchmark, topic in (
        ('train', scale.DEFAULT_TRAIN_COUNT, _TRAIN_BENCHMARK, 'counting_and_probability'),
        ('dev', scale.DEV_COUNT, _DEV_BENCHMARK, 'mixed'),
    ):
        entry = inputs[role]
        filename = f'{role}.jsonl'
        if (not isinstance(entry, dict) or entry.keys() != {'file', 'sha256', 'count', 'benchmark', 'verifier', 'topics'}
                or entry['file'] != filename or not _is_sha(entry['sha256']) or type(entry['count']) is not int
                or entry['count'] != count or entry['benchmark'] != benchmark or entry['topics'] != {topic: count}
                or entry['verifier'] != original._VERIFIER.spec):
            raise ValueError(f'Invalid prepared {role} source: expected {count} local {benchmark} rows')
        raw = _read(path.parent / filename)
        if _sha(raw) != entry['sha256']:
            raise ValueError(f'Prepared {role} source SHA-256 must match the manifest')
        rows = [_json(line) for line in raw.splitlines()]
        if len(rows) != count:
            raise ValueError(f'Prepared {role} count must match the manifest')
        for row in rows:
            _check_row(row, role)
        if len({row['source_id'] for row in rows}) != count:
            raise ValueError(f'Prepared {role} source IDs must be distinct')
        originals[role], hashes[role], files[role] = rows, entry['sha256'], raw
    all_rows = originals['train'] + originals['dev']
    if (len({row['id'] for row in all_rows}) != len(all_rows)
            or len({original._identity(row['prompt']) for row in all_rows}) != len(all_rows)):
        raise ValueError('Prepared IDs and normalized prompts must be distinct across train and dev')
    train_entry = {name: value for name, value in inputs['train'].items() if name != 'topics'}
    return originals, files['train'], train_entry, {'manifest_sha256': _sha(manifest_raw), 'split_sha256': hashes}


def _rank(row, purpose):
    return (_sha(f"{_DOMAIN}:{purpose}:{row['id']}".encode('ascii')), row['id'])


def _stratified(rows, count, purpose):
    exams = defaultdict(list)
    for row in rows:
        exams[row['competition']].append(row)
    n = len(rows)
    quota = {exam: divmod(len(group) * count, n) for exam, group in exams.items()}
    extras = count - sum(whole for whole, _ in quota.values())
    bonus = set(sorted(exams, key=lambda exam: (-quota[exam][1], exam))[:extras])
    picked = set()
    for exam, group in exams.items():
        limit = quota[exam][0] + (exam in bonus)
        picked.update(row['id'] for row in sorted(group, key=lambda row: _rank(row, purpose))[:limit])
    ordered = sorted(rows, key=lambda row: _rank(row, f'{purpose}:output'))
    return ([row for row in ordered if row['id'] in picked], [row for row in ordered if row['id'] not in picked])


def _load_plan(run, source, pin):
    raw = _read(run / 'plan.json')
    plan = _json(raw)
    plan_key = digest(plan)
    if pin is not None and pin != plan_key:
        raise ValueError('Source wide run plan does not match its canonical SHA-256 pin')
    expected = {**source, 'profile': probe.PROFILE, 'mode': 'wide_only', 'base_sha256': probe.BASE_SHA256,
                'wide_unique_questions': scale.DEFAULT_TRAIN_COUNT, 'rollouts_per_arm': scale.DEFAULT_TRAIN_COUNT,
                'worker_chunk': probe.CHUNK, 'dev_count': scale.DEV_COUNT, 'training_world_size': 4,
                'training_global_batch_size': 16}
    if (not isinstance(plan, dict) or plan.get('dev_only') is not True
            or any(plan.get(key) != value for key, value in expected.items())
            or not _is_sha(plan.get('source_rollout_plan_sha256'))):
        raise ValueError('The source is not a wide-only global-batch-16 run bound to this prepared bundle')
    values = plan.get('reused_rollout_value_sha256')
    if (not isinstance(values, dict) or values.keys() != set(RECEIPTS)
            or not all(_is_sha(value) for value in values.values())):
        raise ValueError('Expected exactly four frozen wide rollout digests in the source plan')
    return plan, {**source, 'plan_sha256': plan_key, 'plan_file_sha256': _sha(raw),
                  'base_sha256': plan['base_sha256'], 'upstream_rollout_plan_sha256': plan['source_rollout_plan_sha256']}


def _load_receipt(owned, name, plan_key, value_key, chunk):
    raw = _read(owned / f'{name}.json')
    receipt = _json(raw)
    if (not isinstance(receipt, dict) or receipt.keys() != {'key', 'value', 'value_sha256'}
            or receipt['key'] != plan_key or receipt['value_sha256'] != value_key
            or digest(receipt['value']) != value_key):
        raise ValueError(f'Wide rollout receipt does not match its frozen identity: {name}')
    value = receipt['value']
    if (not isinstance(value, dict) or value.keys() != {'attempts', 'stats', 'rows'}
            or type(value['attempts']) is not int or value['attempts'] != len(chunk)
            or not isinstance(value['stats'], dict) or not isinstance(value['rows'], list)):
        raise ValueError(f'Invalid wide rollout receipt shape: {name}')
    stats, rows = value['stats'], value['rows']
    if (stats.keys() - (_OUTCOMES | {'storage_reject'})
            or any(type(count) is not int or count < 0 for count in stats.values())
            or sum(stats.get(outcome, 0) for outcome in _OUTCOMES) != len(chunk)
            or stats.get('correct', 0) - stats.get('storage_reject', 0) != len(rows)):
        raise ValueError(f'Invalid wide rollout accounting: {name}')

    by_id = {problem.id: (i, problem) for i, problem in enumerate(chunk)}
    checked, previous = [], -1
    for row in rows:
        if (not isinstance(row, dict) or row.keys() != {'id', 'problem_id', 'prompt', 'completion'}
                or not isinstance(row['problem_id'], str) or row['problem_id'] not in by_id):
            raise ValueError(f'Wide rollout is not bound to an original in its receipt chunk: {name}')
        index, problem = by_id[row['problem_id']]
        completion = row['completion']
        if (index <= previous or not isinstance(completion, str) or completion != completion.strip()
                or not 0 < len(completion) <= aime.MAX_VERIFIED_COMPLETION_CHARS
                or len(problem.prompt) > aime.MAX_TRAIN_TEXT
                or row['id'] != digest([problem.id, completion])
                or row['prompt'] != probe.prompts((problem,))[0]['prompt']
                or probe.grade(problem, completion) != 'correct'):
            raise ValueError(f'Wide rollout failed order, identity, prompt or answer verification: {name}/{problem.id}')
        checked.append({'id': problem.id, 'prompt': problem.prompt, 'completion': completion})
        previous = index
    return checked, {'file_sha256': _sha(raw), 'value_sha256': value_key}


def _bootstrap(run, train, source, pin):
    plan, evidence = _load_plan(run, source, pin)
    owned = _plain_directory(run / 'owned')
    problems = [Problem(row['id'], row['prompt'], row['answer'], _TRAIN_BENCHMARK, original._VERIFIER, row['answer'])
                for row in train]
    bootstrap, receipts = [], {}
    for i, name in enumerate(RECEIPTS):
        result, receipts[name] = _load_receipt(owned, name, evidence['plan_sha256'],
                                              plan['reused_rollout_value_sha256'][name],
                                              problems[i * probe.CHUNK:(i + 1) * probe.CHUNK])
        bootstrap.extend(result)
    if len(bootstrap) != ROLLOUT_COUNT:
        raise ValueError(f'Expected {ROLLOUT_COUNT} distinct verified wide trajectories, got {len(bootstrap)}')
    accepted = {row['id'] for row in bootstrap}
    return {'schema_version': 1, 'rows': bootstrap, 'failed_problem_ids': [row['id'] for row in train
                                                                           if row['id'] not in accepted],
            'source': {**evidence, 'receipts': receipts}}


def prepare(output_dir, *, manifest, rollouts_from, source_plan_sha256=None):
    """Publish a fresh private bundle; return absolute parent, child and bootstrap paths."""
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output directory already exists: {output}')
    if source_plan_sha256 is not None and not _is_sha(source_plan_sha256):
        raise ValueError('Source plan SHA-256 pin must be 64 lowercase hex characters')
    originals, train_bytes, train, source = _load_bundle(Path(manifest).expanduser().absolute())
    run = _plain_directory(Path(rollouts_from).expanduser().absolute())
    bootstrap = _bootstrap(run, originals['train'], source, source_plan_sha256)
    meta, domain = _stratified(originals['dev'], 15, 'meta-dev')
    search, validation = _stratified(domain, 22, 'domain-search')
    meta_search, meta_validation = _stratified(meta, 7, 'meta-search')

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.aime-wide-loop-') as temporary:
        staging = Path(temporary) / 'bundle'
        staging.mkdir(mode=0o700)
        (staging / train['file']).write_bytes(train_bytes)
        for filename, cohorts in ((original.PARENT_MANIFEST, {'search': search, 'validation': validation}),
                                  (original.META_MANIFEST, {'meta_search': meta_search, 'meta_validation': meta_validation})):
            splits = {'train': train}
            for role, rows in cohorts.items():
                splits[role] = original._entry(staging / f'{role}.jsonl', rows, _DEV_BENCHMARK)
            (staging / filename).write_bytes(original._json_bytes({'schema_version': 1, 'splits': splits}))
        (staging / BOOTSTRAP).write_bytes(original._json_bytes(bootstrap))
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Output directory already exists: {output}')
        staging.rename(output)
    return output / original.PARENT_MANIFEST, output / original.META_MANIFEST, output / BOOTSTRAP


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True, type=Path, help='existing prepared scale manifest.json')
    parser.add_argument('--rollouts-from', required=True, type=Path, help='existing wide-only batch-16 run directory')
    parser.add_argument('--output-dir', required=True, type=Path, help='new private local output directory')
    parser.add_argument('--source-plan-sha256', help='optional canonical SHA-256 pin of the input run plan')
    args = parser.parse_args(argv)
    try:
        parent, meta, bootstrap = prepare(args.output_dir, manifest=args.manifest, rollouts_from=args.rollouts_from,
                                          source_plan_sha256=args.source_plan_sha256)
    except (ValueError, OSError) as error:
        parser.exit(1, f'AIME wide-loop preparation failed: {error}\n')
    print(f'Parent manifest: {parent}')
    print(f'Meta-trial manifest: {meta}')
    print(f'Bootstrap: {bootstrap} (SHA-256: {_sha(bootstrap.read_bytes())})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
