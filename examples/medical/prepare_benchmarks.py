"""Prepare pinned, already-cached medical benchmarks; standard library only.

Run from RSISys with Python 3.12+ (no network, models, tokenizer or trainer):
  python -m examples.medical.prepare_benchmarks --source-root /cache/data \
      --output /research/public --private-output /operator/private

Both output directories must be new, disjoint and outside the source tree. Only
the public directory belongs in the researcher workspace. Private output contains
all remaining eligible MedXpertQA/MedQA test rows, provenance and optional exclusion
identities. Directory permissions are not an isolation boundary for the same Unix
user: the caller must withhold private output AND source caches from researcher
tools/mounts. No private path, membership or digest is published in the task.

Defaults: MedQA train 1024 (512 also supported), primary MedXpertQA 128, guardrail
MedQA 64. Development is an INTERNAL partition of official test, not official dev
or a full-benchmark result. These caches were used by older HuatuoMedRSI runs;
disjointness here does not establish historical freshness. Supply --exclude with
operator-owned domain JSONL rows to exclude known prior memberships everywhere.

Conversion keeps source questions/context and ordered options, maps literal
MedQA answer_idx / MedXpertQA label to gold, checks MedQA answer text, and drops
metadata/answer text from prompts. MedXpertQA's embedded Answer Choices suffix is
removed ONLY when it matches options after whitespace normalization; conflicting
rows are quarantined in private audit. No content, rationale or labels are model
generated here. Upstream MedXpertQA itself uses augmentation/expert review; this
script makes no claim that upstream examples are exclusively human-authored.

Exact dedup uses NFKC, casefold and collapsed whitespace on stems (before choices),
and source IDs. Priority is full MedXpertQA test > full MedQA test > MedQA train,
including rows not selected for public development. Equivalent duplicates keep
the first source occurrence. Across benchmarks, later matching stems are excluded
regardless of the different label/option spaces. Within a benchmark, conflicting
duplicates are all quarantined rather than choosing a gold. This is not semantic
dedup, pretraining contamination detection, or medical correctness verification.

Outputs: domain *.jsonl; run-compatible task.json; frozen_task.json (asdict task
and protocol_ref); hashmanifest.json binding source bytes, conversion, sampling,
public files and frozen task. Private audit separately binds heldout files and
source row lineage. verify_public() checks bytes AND the current frozen task.
Seeded SHA256 ordering selects evaluation independently of train size and labels.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import unicodedata

from examples.medical.task import encoded, prepare, samples


CONVERSION = 'medical-cached-v1'
SOURCES = {
    'medxpertqa_test': {
        'repository': 'TsinghuaC3I/MedXpertQA',
        'revision': '7e7c465a68eb2b866926bfa59c8c9d17a8daba65',
        'path': 'medxpertqa/7e7c465a68eb2b866926bfa59c8c9d17a8daba65/Text-test.jsonl',
        'upstream_path': 'Text/test.jsonl', 'split': 'test', 'rows': 2450,
        'sha256': '07ea977f7e326909b51c883b65f6ad7f399fc6db07408ef3630f4e3048ec71b2',
        'license': 'MIT (upstream project)',
        'reference': 'https://github.com/TsinghuaC3I/MedXpertQA',
    },
    'medqa_test': {
        'repository': 'GBaker/MedQA-USMLE-4-options',
        'revision': '0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18',
        'path': 'medqa-usmle-4-options/0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18/phrases_no_exclude_test.jsonl',
        'upstream_path': 'phrases_no_exclude_test.jsonl', 'split': 'test', 'rows': 1273,
        'sha256': 'c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb',
        'license': 'CC-BY-4.0 (pinned mirror card)',
        'reference': 'https://github.com/jind11/MedQA',
    },
    'medqa_train': {
        'repository': 'GBaker/MedQA-USMLE-4-options',
        'revision': '0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18',
        'path': 'medqa-usmle-4-options/0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18/phrases_no_exclude_train.jsonl',
        'upstream_path': 'phrases_no_exclude_train.jsonl', 'split': 'train', 'rows': 10178,
        'sha256': '3a65baf760f17c395058d6699315a7ce8aa767c50cb193fea87ceb94dd790324',
        'license': 'CC-BY-4.0 (pinned mirror card)',
        'reference': 'https://github.com/jind11/MedQA',
    },
}


def canonical(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def json_bytes(value):
    return encoded(value) + b'\n'


def jsonl_bytes(rows):
    return b''.join(json_bytes(row) for row in rows)


@dataclass(frozen=True)
class Entry:
    row: dict
    source: str
    line: int
    source_id: str | None


class ChoicesConflict(ValueError):
    pass


def convert(raw, source, line):
    benchmark = 'medxpertqa' if source == 'medxpertqa_test' else 'medqa'
    labels = 'ABCDEFGHIJ' if benchmark == 'medxpertqa' else 'ABCD'
    question, options = raw['question'], raw['options']
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Missing source question')
    if (not isinstance(options, dict) or set(options) != set(labels)
            or any(not isinstance(v, str) or not v.strip() for v in options.values())):
        raise ValueError('Source options do not match legal labels')
    options = {label: options[label] for label in labels}
    source_id = raw.get('id')
    if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()):
        raise ValueError('Invalid source ID')
    if benchmark == 'medxpertqa':
        if not source_id or raw.get('images'):
            raise ValueError('Expected identified text-only MedXpertQA')
        gold = raw['label']
        # Do not use casefold here: case changes can alter clinical meaning.
        parts = question.rsplit('\nAnswer Choices:', 1)
        if len(parts) == 2:
            expected = ' '.join(f'({k}) {v}' for k, v in options.items())
            if ' '.join(parts[1].split()) != ' '.join(expected.split()):
                raise ChoicesConflict('Embedded choices conflict with options')
            question = parts[0]
    else:
        gold = raw['answer_idx']
        if gold not in options or raw.get('answer') != options[gold]:
            raise ValueError('MedQA answer text disagrees with answer_idx')
    # Content ID is split-independent when native MedQA has no source ID.
    ident = source_id or sha256(canonical(question).encode()).hexdigest()
    row = dict(id=f'{benchmark}:{ident}', benchmark_id=benchmark,
               question=question, options=options, gold=gold)
    samples(jsonl_bytes([row]))  # Validate against the actual domain contract.
    return Entry(row, source, line, source_id)


def build_plan(source_root, *, train_count=1024, dev_count=128, guardrail_count=64,
               seed=20261101, exclusions=(), run_id=None):
    """Read pinned source bytes and construct a plan in memory; no writes/network.

    exclusions are operator-owned domain JSONL paths, never researcher artifacts.
    All checks happen before output directories are created. Insufficient eligible
    rows or changed source bytes fail rather than shrinking the requested sizes.
    """
    if any(type(n) is not int or n < 1 for n in (train_count, dev_count, guardrail_count)):
        raise ValueError('Requested counts must be positive integers')
    if type(seed) is not int:
        raise ValueError('Seed must be an integer')
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError('run_id must be a nonempty string')
    source_root = Path(source_root).resolve(strict=True)
    blocked_stems, blocked_ids, exclusion_refs = set(), set(), []
    for path in exclusions:
        path = Path(path).resolve(strict=True)
        content = path.read_bytes()
        rows = samples(content)
        blocked_stems.update(canonical(row.question) for row in rows)
        blocked_ids.update(row.id for row in rows)
        # Also accept native IDs when domain IDs were not namespaced previously.
        blocked_ids.update(f'{row.benchmark_id}:{row.id}' for row in rows)
        exclusion_refs.append(dict(path=str(path), sha256=sha256(content).hexdigest(), rows=len(rows)))
    all_ids, all_stems = {}, {}
    pools, audit, counts, source_refs = {}, [], Counter(), {}
    quarantined = set()

    def quarantine(entry):
        key = (entry.source, entry.line)
        if key not in quarantined:
            quarantined.add(key)
            blocked_stems.add(canonical(entry.row['question']))
            blocked_ids.add(entry.row['id'])
            audit.append(dict(source=entry.source, line=entry.line,
                              source_id=entry.source_id, reason='duplicate_conflict'))
            counts[f'{entry.source}:duplicate_conflict'] += 1
    for name, spec in SOURCES.items():
        path = (source_root / spec['path']).resolve(strict=True)
        if not path.is_relative_to(source_root):
            raise ValueError('Source symlink escapes source root')
        content = path.read_bytes()
        if sha256(content).hexdigest() != spec['sha256']:
            raise ValueError(f'Source SHA256 mismatch: {name}')
        native = [json.loads(line) for line in content.decode('utf-8').splitlines() if line.strip()]
        if len(native) != spec['rows']:
            raise ValueError(f'Source row count mismatch: {name}')
        source_refs[name] = {k: v for k, v in spec.items() if k != 'path'}
        pool = []
        for line, raw in enumerate(native, 1):
            try:
                entry = convert(raw, name, line)
            except ChoicesConflict:
                # Block even a quarantined question/ID against training.
                stem = raw['question'].rsplit('\nAnswer Choices:', 1)[0]
                blocked_stems.add(canonical(stem))
                blocked_ids.add(f"medxpertqa:{raw['id']}")
                audit.append(dict(source=name, line=line, source_id=raw['id'], reason='choices_conflict'))
                counts[f'{name}:choices_conflict'] += 1
                continue
            row = entry.row
            stem = canonical(row['question'])
            if stem in blocked_stems or row['id'] in blocked_ids:
                counts[f'{name}:excluded'] += 1
                continue
            matches = [e for e in (all_ids.get(row['id']), all_stems.get(stem)) if e is not None]
            if matches:
                conflict = False
                for previous in matches:
                    other = previous.row
                    if other['benchmark_id'] == row['benchmark_id'] and (
                            canonical(other['question']) != stem or other['options'] != row['options']
                            or other['gold'] != row['gold']):
                        conflict = True
                if conflict:
                    for previous in matches:
                        quarantine(previous)
                    quarantine(entry)
                    continue
                # Remember aliases so later occurrences cannot evade ID dedup.
                all_ids[row['id']] = matches[0]
                counts[f'{name}:duplicate'] += 1
                continue
            all_ids[row['id']] = all_stems[stem] = entry
            pool.append(entry)
        pools[name] = pool
    for name, pool in pools.items():
        pools[name] = sorted((e for e in pool if (e.source, e.line) not in quarantined), key=lambda e: (
            sha256(encoded([CONVERSION, seed, name, e.row['id']])).hexdigest(), e.line))
        counts[f'{name}:eligible'] = len(pools[name])
    mx, guard, train = (pools[n] for n in SOURCES)
    for pool, requested in ((mx, dev_count), (guard, guardrail_count), (train, train_count)):
        if len(pool) <= requested:
            raise ValueError('Insufficient eligible data; require requested rows and a reserve')
    public = {'train.medqa.jsonl': train[:train_count],
              'dev.medxpertqa.jsonl': mx[:dev_count],
              'guardrail.medqa.jsonl': guard[:guardrail_count]}
    private = {'heldout.medxpertqa.jsonl': mx[dev_count:],
               'heldout.medqa.jsonl': guard[guardrail_count:]}
    return dict(public=public, private=private, sources=source_refs, counts=dict(counts), run_id=run_id,
                audit=audit, exclusions=exclusion_refs, source_root=source_root,
                selection=dict(seed=seed, train_count=train_count, dev_count=dev_count,
                               guardrail_count=guardrail_count,
                               method='seeded-sha256-order-v1; independent by source; no gold stratification'))


def write_plan(plan, output, private_output):
    """Materialize once; refuses overwrites, overlapping roots and source writes."""
    output, private_output = Path(output).resolve(), Path(private_output).resolve()
    roots = (output, private_output, plan['source_root'])
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError('Source, public and private directories must be disjoint')
    if output.exists() or private_output.exists():
        raise FileExistsError('Output directories must both be new')
    # Also prohibit containing an operator exclusion source in public output.
    for ref in plan['exclusions']:
        if Path(ref['path']).is_relative_to(output):
            raise ValueError('Public output contains an operator exclusion source')
    output.mkdir(parents=True, mode=0o700)
    private_output.mkdir(parents=True, mode=0o700)
    public_bytes = {name: jsonl_bytes([e.row for e in rows]) for name, rows in plan['public'].items()}
    private_bytes = {name: jsonl_bytes([e.row for e in rows]) for name, rows in plan['private'].items()}
    config = dict(train='train.medqa.jsonl', development=[
        dict(benchmark_id='medxpertqa', role='primary', path='dev.medxpertqa.jsonl'),
        dict(benchmark_id='medqa', role='guardrail', path='guardrail.medqa.jsonl')],
        guardrails=True, max_guardrail_drop=0.02, max_output_tokens=32)
    for name, content in public_bytes.items():
        (output / name).write_bytes(content)
    public_bytes['task.json'] = json_bytes(config)
    frozen = prepare(config, base_directory=output)
    public_bytes['frozen_task.json'] = json_bytes(dict(task=asdict(frozen), protocol_ref=frozen.protocol_ref))
    manifest = dict(schema=CONVERSION, run_id=plan['run_id'], sources=plan['sources'], selection=plan['selection'],
                    implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    partitions={name: dict(source=rows[0].source,
                        source_split=plan['sources'][rows[0].source]['split'],
                        rows=len(rows), kind='sft_train' if name.startswith('train.') else 'custom_development',
                        official_development=False, new_holdout_claim=False)
                        for name, rows in plan['public'].items()},
                    conversion=dict(labels='literal answer_idx / label; no numerical remapping',
                                    context='complete source stem; verified choices suffix removed',
                                    dedup='NFKC+casefold+whitespace stems and split-independent IDs',
                                    generated_content=False),
                    evaluation='custom development from official test; not official dev; historical freshness not established',
                    files={name: sha256(content).hexdigest() for name, content in public_bytes.items()},
                    protocol_ref=frozen.protocol_ref)
    public_bytes['hashmanifest.json'] = json_bytes(manifest)
    for name, content in public_bytes.items():
        (output / name).write_bytes(content)
        (output / name).chmod(0o400)
    lineage = {name: [dict(id=e.row['id'], source=e.source, line=e.line, source_id=e.source_id)
                      for e in rows] for name, rows in {**plan['public'], **plan['private']}.items()}
    private_bytes['hashmanifest.json'] = json_bytes(dict(
        schema=CONVERSION, run_id=plan['run_id'], source_root=str(plan['source_root']), sources=plan['sources'],
        selection=plan['selection'], counts=plan['counts'], quarantined=plan['audit'],
        exclusions=plan['exclusions'], lineage=lineage,
        public_manifest_sha256=sha256(public_bytes['hashmanifest.json']).hexdigest(),
        files={name: sha256(content).hexdigest() for name, content in private_bytes.items()}))
    for name, content in private_bytes.items():
        (private_output / name).write_bytes(content)
        (private_output / name).chmod(0o400)
    verify_public(output)
    return manifest


def verify_public(output):
    output = Path(output).resolve()
    manifest = json.loads((output / 'hashmanifest.json').read_bytes())
    expected = {'train.medqa.jsonl', 'dev.medxpertqa.jsonl', 'guardrail.medqa.jsonl',
                'task.json', 'frozen_task.json'}
    if manifest['schema'] != CONVERSION or set(manifest['files']) != expected:
        raise ValueError('Invalid preparation manifest')
    for name, digest in manifest['files'].items():
        path = output / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Prepared output SHA256 mismatch')
    task = prepare(json.loads((output / 'task.json').read_bytes()), base_directory=output)
    frozen = json.loads((output / 'frozen_task.json').read_bytes())
    if (encoded(frozen) != encoded(dict(task=asdict(task), protocol_ref=task.protocol_ref))
            or manifest['protocol_ref'] != task.protocol_ref):
        raise ValueError('Frozen task identity changed (including relocated paths)')
    return task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--private-output', required=True, type=Path)
    parser.add_argument('--train-count', type=int, default=1024)
    parser.add_argument('--dev-count', type=int, default=128)
    parser.add_argument('--guardrail-count', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--run-id', help='Optional operator run identity, recorded in manifests')
    parser.add_argument('--exclude', action='append', default=[], type=Path,
                        help='Operator-only domain JSONL; exclude all listed IDs and stems')
    args = parser.parse_args(argv)
    plan = build_plan(args.source_root, train_count=args.train_count, dev_count=args.dev_count,
                      guardrail_count=args.guardrail_count, seed=args.seed, exclusions=args.exclude,
                      run_id=args.run_id)
    manifest = write_plan(plan, args.output, args.private_output)
    print(json.dumps(dict(output=str(args.output.resolve()), protocol_ref=manifest['protocol_ref'],
                          selection=manifest['selection']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
