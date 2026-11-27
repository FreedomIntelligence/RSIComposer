"""Opt-in, local preparation of a public AIME protocol for MetaRSI.

Run: uv run --with pyarrow==23.0.1 python -m rsicomposer.algorithms.meta_rsi.prepare_aime
     --fetch --output-dir PATH

Use --source-dir instead of --fetch for an offline copy of the same raw files.
Both modes verify the source bytes; --list-sources prints URLs and offline paths.
PyArrow is a preparation-only dependency, not used by the generated manifests.
Public access and dataset-card licenses are not downstream redistribution rights.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPException
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unicodedata import normalize
from urllib.request import urlopen

from rsicomposer.labs.math.verifiers import CompetitionIntegerVerifier, final_boxed


PARENT_MANIFEST = 'manifest.json'
META_MANIFEST = 'meta-manifest.json'
DEV_SPLITS = ('search', 'validation', 'meta_search', 'meta_validation')
PARENT_SPLITS = ('train', 'search', 'validation', 'test')
META_SPLITS = ('train', 'meta_search', 'meta_validation')
DEFAULT_COUNT = 8
_VERIFIER = CompetitionIntegerVerifier(maximum=999)
_DEV_URL = re.compile(
    r'https://artofproblemsolving\.com/wiki/index\.php/(202[234])_AIME_(I|II)_Problems/Problem_([1-9]|1[0-5])'
)
_KEY_ROOT = 'https://live.poshenloh.com/past-contests/aime'
# SHA-256 of 15 three-digit answers in exam order, one per line with a final LF,
# checked against the LIVE answer keys. The site says it publishes with MAA
# permission; these are not MAA-hosted keys or an MAA license for RSIComposer.
_KEY_SHA256 = {
    'aime-2025-i': 'eeea941d89c324ea72429274e637a6fab010ad243d54191608ddf6de6fa36d57',
    'aime-2025-ii': 'b4c1e981102bf087a27dce0f103f1ba88e07af42ea95123f5fefbc17c9d08b27',
    'aime-2026-i': '48af5fa0946aa4bbde9c78affe356fd669f494f896f2d6df463f7163f3787898',
    'aime-2026-ii': '36b9863632d5e78adb2756c8e2ae8b33be8dd47637da3d8aa89202fbfc75deca',
}
# The published problem asks for the maximum; MathArena asks for the sum.
# Restore the published wording and matching answer together. No problem
# statement is embedded; the small phrase edit is guarded by both hashes.
_KEY_VARIANTS = {
    'aime-2026-ii-10': {
        'pinned_answer': '850', 'published_answer': '340',
        'pinned_prompt_sha256': 'f7b4a98af9dd00a521396033749a2180021a97407016472fc87eb707029e05ec',
        'canonical_prompt_sha256': '6bdae3f09de1c3f5732458b499be935266b99f1139743499744f3f9a9255a9eb',
        'published_problem_url': f'{_KEY_ROOT}/2026II/problem/10',
        'replace_codepoints': [440, 466],
        'replacement': 'greatest possible value',
        'difference': 'MathArena asks for the sum of the admissible values, not their maximum.',
    },
}
_FIGURE_BLOCK = re.compile(
    r'\\begin\{(tikzpicture|asy)\}.*?\\end\{\1\}|\[asy\].*?\[/asy\]', re.DOTALL
)
_FIGURE_REFERENCE = re.compile(r'\b(?:as shown|shown below|depicted below|figure below|following diagram)\b', re.I)


@dataclass(frozen=True)
class _Source:
    repo: str
    revision: str
    path: str
    sha256: str
    size: int
    count: int
    license: str
    columns: tuple[str, ...]

    @property
    def url(self):
        return f'https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.path}'

    @property
    def offline_path(self):
        return f'{self.repo}/{self.path}'

    def provenance(self):
        return {'repository': f'https://huggingface.co/datasets/{self.repo}', 'revision': self.revision,
                'path': self.path, 'url': self.url, 'offline_path': self.offline_path,
                'sha256': self.sha256, 'bytes': self.size, 'upstream_count': self.count,
                'dataset_card_license': self.license}


_PARQUET = 'train-00000-of-00001.parquet'
_SOURCES = {
    'math_train': _Source('EleutherAI/hendrycks_math', '21a5633873b6a120296cce3e2df9d5550074f4a3',
                          f'counting_and_probability/{_PARQUET}',
                          'd37ae017930df41678edee50dcde18c17fb06afaff9eb505c1b95f1002bfcb31',
                          329354, 771, 'MIT', ('problem', 'solution')),
    'aime_dev': _Source('project-numina/aimo-validation-aime', '13f9e12f613e720c2a2b2f345dd04b998a29494d',
                        f'data/{_PARQUET}', '025484a99fea498e7d0c3b0ee42afcbec0176405c19c5dbf557b9f6ca6445675',
                        261038, 90, 'Apache-2.0', ('id', 'problem', 'answer', 'url')),
    'aime_2025': _Source('MathArena/aime_2025', 'c94da77eb22bbd6439e62a323bec18493a421302',
                         f'data/{_PARQUET}', '9f9066ff48ad2e31f9bf1b1ac6d5e80693195f987985f2859f89dd25ffa51c2d',
                         14313, 30, 'CC-BY-NC-SA-4.0', ('problem_idx', 'problem', 'answer')),
    'aime_2026': _Source('MathArena/aime_2026', 'd61f486805306e6e6b5913504c07763a99e1b112',
                         f'data/{_PARQUET}', 'ae72853349a46c5f08666feb42db2d09fb4ac4fd4768c726b29702df21c7768e',
                         10062, 30, 'CC-BY-NC-SA-4.0', ('problem_idx', 'problem', 'answer')),
}
_LICENSE_NOTES = (
    'Keep raw snapshots and prepared corpora local and out of source control and released artifacts.',
    'MathArena lists CC BY-NC-SA 4.0: attribution, noncommercial use and share-alike requirements apply where licensed.',
    'Upstream licenses do not establish permission to redistribute underlying MAA problems, graphics or AoPS content; obtain separate permission where needed.',
    'LIVE by Po-Shen Loh reports permission from MAA for its own archive; that does not grant permission to this project.',
)


def _verified_bytes(source, source_dir):
    location = source.url if source_dir is None else source_dir / source.offline_path
    try:
        with (urlopen(location, timeout=30) if source_dir is None else location.open('rb')) as stream:
            data = stream.read(source.size + 1)
    except (OSError, HTTPException) as error:
        raise ValueError(f'Cannot read pinned AIME preparation source {location}: {error}') from error
    if len(data) > source.size:
        raise ValueError(f'Pinned source {location} exceeds its {source.size}-byte limit')
    actual = sha256(data).hexdigest()
    if len(data) != source.size or actual != source.sha256:
        raise ValueError(f'SHA-256 mismatch for {location}: expected {source.sha256}, got {actual}')
    return data


def _parquet_reader():
    try:
        from pyarrow.parquet import read_table
    except ImportError as error:
        raise ValueError('Install PyArrow only for this conversion: uv run --with pyarrow==23.0.1 python -m '
                         'rsicomposer.algorithms.meta_rsi.prepare_aime ...') from error
    return read_table


def _read_sources(source_dir):
    reader = _parquet_reader()
    rows = {}
    for name, source in _SOURCES.items():
        data = _verified_bytes(source, source_dir)
        table = reader(BytesIO(data), columns=list(source.columns))
        if table.num_rows != source.count:
            raise ValueError(f'{name} row count mismatch: expected {source.count}, got {table.num_rows}')
        rows[name] = table.to_pylist()
    return rows


def _prompt(row, location):
    prompt = row['problem']
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f'{location}: expected a nonempty problem statement')
    return prompt


def _identity(prompt):
    # Compare without diagram source, math delimiters, or whitespace; never
    # mutate the stored prompt. This is a duplicate check, not a paraphrase detector.
    text = _FIGURE_BLOCK.sub('', normalize('NFKC', prompt)).casefold()
    text = re.sub(r'\\(?:left|right)\b|\\[,;! ]|\\[\[\]()]|\$', '', text)
    return ''.join(text.split())


def _figures(prompt):
    found = []
    if r'\begin{tikzpicture}' in prompt:
        found.append('tikz-source')
    if r'\begin{asy}' in prompt or '[asy]' in prompt:
        found.append('asymptote-source')
    if re.search(r'^\|[^\n]+\|\s*$', prompt, re.MULTILINE):
        found.append('ascii-diagram')
    if re.search(r'\\includegraphics|!\[|<img\b|\[img\b', prompt, re.I):
        found.append('image-reference-not-resolved')
    if not found and _FIGURE_REFERENCE.search(prompt):
        found.append('figure-reference-no-recognized-inline-diagram')
    return found


def _answer_key_ref(competition):
    _, year, exam = competition.split('-')
    return f'{_KEY_ROOT}/{year}{exam.upper()}/answers'


def _aime_row(raw, competition, number, source, source_id):
    ident = f'{competition}-{number:02d}'
    answer = _VERIFIER.parse_reference(raw['answer'])
    if answer is None:
        raise ValueError(f'{ident}: expected AIME integer answer in [0, 999]')
    row = {'id': ident, 'prompt': _prompt(raw, ident), 'answer': f'{answer:03d}',
           'competition': competition, 'problem_number': number, 'source': source, 'source_id': source_id}
    if 'url' in raw:
        row['source_url'] = raw['url']
    return row


def _restore_published_variant(row, variant):
    prompt = row['prompt']
    if (row['answer'] != variant['pinned_answer']
            or sha256(prompt.encode('utf-8')).hexdigest() != variant['pinned_prompt_sha256']):
        raise ValueError(f"{row['id']}: unexpected pinned wording or answer for published normalization")
    start, end = variant['replace_codepoints']
    restored = prompt[:start] + variant['replacement'] + prompt[end:]
    if sha256(restored.encode('utf-8')).hexdigest() != variant['canonical_prompt_sha256']:
        raise ValueError(f"{row['id']}: published wording normalization did not match its fingerprint")
    row.update(prompt=restored, answer=variant['published_answer'], statement_variant='published-wording-restored')


def _sealed(rows):
    competitions = {}
    for year in (2025, 2026):
        source = f'aime_{year}'
        by_index = {row['problem_idx']: row for row in rows[source]}
        if (len(rows[source]) != 30 or set(by_index) != set(range(1, 31))
                or any(type(index) is not int for index in by_index)):
            raise ValueError(f'{source}: expected each MathArena problem index 1 through 30 exactly once')
        for session, start in (('i', 1), ('ii', 16)):
            name = f'aime-{year}-{session}'
            exam = [_aime_row(by_index[index], name, index - start + 1, source, index)
                    for index in range(start, start + 15)]
            for row in exam:
                variant = _KEY_VARIANTS.get(row['id'])
                if variant is not None:
                    _restore_published_variant(row, variant)
            key = ''.join(row['answer'] + '\n' for row in exam).encode('ascii')
            if sha256(key).hexdigest() != _KEY_SHA256[name]:
                raise ValueError(f'{name}: answers do not match the independently published key fingerprint')
            competitions[name] = exam
    return competitions


def _history(rows):
    source_ids, result = set(), []
    for raw in rows:
        source_id, url = raw['id'], raw['url']
        match = _DEV_URL.fullmatch(url) if isinstance(url, str) else None
        if not match or type(source_id) is not int or source_id in source_ids:
            raise ValueError('AIME dev data must identify distinct upstream rows from official 2022–2024 exams')
        source_ids.add(source_id)
        year, session, number = match.groups()
        result.append(_aime_row(raw, f'aime-{year}-{session.lower()}', int(number), 'aime_dev', source_id))
    expected = {f'aime-{year}-{session}-{number:02d}' for year in (2022, 2023, 2024)
                for session in ('i', 'ii') for number in range(1, 16)}
    if len(result) != len(expected) or {row['id'] for row in result} != expected:
        raise ValueError('AIME dev must contain every problem from the six 2022–2024 exams exactly once')
    return result


def _order(rows):
    return sorted(rows, key=lambda row: sha256(
        f"{_SOURCES[row['source']].revision}:{row['id']}".encode('ascii')).digest())


def _train(raw_rows, blocked):
    eligible = []
    stats = {'non_aime_integer': 0, 'overlap_with_historical_or_sealed': 0, 'duplicate_train_prompt': 0}
    for index, raw in enumerate(raw_rows):
        ident = f'math-counting-and-probability-train-{index:04d}'
        prompt = _prompt(raw, ident)
        solution = raw['solution']
        box = final_boxed(solution) if isinstance(solution, str) else None
        answer = _VERIFIER.parse_reference(box[len(r'\boxed{'):-1]) if box else None
        if answer is None:
            stats['non_aime_integer'] += 1
        elif _identity(prompt) in blocked:
            stats['overlap_with_historical_or_sealed'] += 1
        else:
            eligible.append({'id': ident, 'prompt': prompt, 'answer': f'{answer:03d}', 'competition': None,
                             'source': 'math_train', 'source_id': index})
    unique, used = [], set()
    for row in _order(eligible):
        fingerprint = _identity(row['prompt'])
        if fingerprint in used:
            stats['duplicate_train_prompt'] += 1
        else:
            unique.append(row)
            used.add(fingerprint)
    return unique, stats


def _pack(data, train_count, dev_count, pilot_sealed_count):
    sealed = _sealed(data)
    history = _history(data['aime_dev'])
    blocked = {}
    for row in [*history, *(row for exam in sealed.values() for row in exam)]:
        fingerprint = _identity(row['prompt'])
        if not fingerprint or fingerprint in blocked:
            raise ValueError(f"AIME corpora have an empty/duplicate normalized prompt at {row['id']}")
        blocked[fingerprint] = row['id']
    # Exclude both versions of any edited statement from the training pool.
    for name in ('aime_2025', 'aime_2026'):
        blocked.update({_identity(row['problem']): name for row in data[name]})
    train, stats = _train(data['math_train'], blocked)
    if len(train) < train_count:
        raise ValueError(f'Only {len(train)} MATH rows meet answer and disjointness criteria; reduce --train-count')
    history_order = _order(history)
    splits = {'train': train[:train_count]}
    for index, split in enumerate(DEV_SPLITS):
        splits[split] = history_order[index * dev_count:(index + 1) * dev_count]
    if pilot_sealed_count is None:
        selected = sealed
    else:
        # Balanced prefix; if the count is very small some exams are absent.
        pool = [row for index in range(15) for exam in sealed.values() for row in exam[index:index + 1]]
        picked = {row['id'] for row in pool[:pilot_sealed_count]}
        selected = {name: [row for row in exam if row['id'] in picked] for name, exam in sealed.items()}
    return splits, sealed, selected, {'eligible': len(train), **stats}


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


def _entry(path, rows, benchmark):
    data = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows).encode('utf-8')
    path.write_bytes(data)
    return {'file': path.name, 'sha256': sha256(data).hexdigest(), 'count': len(rows),
            'benchmark': benchmark, 'verifier': _VERIFIER.spec}


def _write_bundle(directory, split_rows, sealed, selected, train_stats, pilot):
    entries = {split: _entry(directory / f'{split}.jsonl', rows,
                             'math-counting-and-probability-train' if split == 'train' else 'aime-2022-2024-dev')
               for split, rows in split_rows.items()}
    entries['test'] = [_entry(directory / f'test-{name}.jsonl', rows, f'pilot-{name}' if pilot else name)
                       for name, rows in selected.items() if rows]
    manifests = {}
    for filename, splits in ((PARENT_MANIFEST, PARENT_SPLITS), (META_MANIFEST, META_SPLITS)):
        encoded = _json_bytes({'schema_version': 1, 'splits': {split: entries[split] for split in splits}})
        (directory / filename).write_bytes(encoded)
        manifests[filename] = sha256(encoded).hexdigest()
    (directory / 'provenance.json').write_bytes(_json_bytes({
        'schema_version': 1,
        'sources': {name: source.provenance() for name, source in _SOURCES.items()},
        'licensing': _LICENSE_NOTES,
        'answer_key_audit': {
            'published_keys': {name: {'url': _answer_key_ref(name), 'sha256_padded_lf': digest}
                               for name, digest in _KEY_SHA256.items()},
            'matches_after_normalization': 60, 'matches_before_normalization': 60 - len(_KEY_VARIANTS),
            'normalizations': _KEY_VARIANTS,
            'policy': 'Verify against permissioned LIVE keys after restoring the published wording and answer '
                      'for the documented variant. These are not MAA-hosted keys.',
        },
        'selection': {
            'ordering': 'ascending SHA-256 of ASCII upstream_revision:stable_row_id for train and dev',
            'train_filter': 'MATH counting_and_probability train; last balanced boxed/fbox answer a bare AIME integer',
            'train_stats': train_stats,
            'contamination_control': 'distinct source IDs and normalized prompts; eligible MATH excludes any normalized '
                                     'match against all historic and sealed rows, including unselected rows. '
                                     'Paraphrase and model-pretraining contamination are not ruled out.',
            'dev_allocation': list(DEV_SPLITS),
            'sealed': {'mode': 'pilot_subset' if pilot else 'full_public_aime_2025_2026',
                       'authority': 'pinned MathArena AIME snapshots with documented normalization against LIVE',
                       'count': sum(len(rows) for rows in selected.values()), 'full_count': 60},
        },
        'figures': {
            'policy': 'Inline Asymptote, TikZ and ASCII are retained as plain text, not rendered images. '
                      'Prompts are unchanged except for the separately documented wording normalization. '
                      'This is not a byte-exact reproduction of the printed exams.',
            'by_id': {row['id']: figures for rows in [*split_rows.values(), *sealed.values()] for row in rows
                      if (figures := _figures(row['prompt']))},
        },
        'splits': entries,
        'row_ids': {**{name: [row['id'] for row in rows] for name, rows in split_rows.items()},
                    'test': [row['id'] for rows in selected.values() for row in rows]},
        'manifests': manifests,
    }))


def prepare(output_dir, *, train_count=DEFAULT_COUNT, dev_count=DEFAULT_COUNT,
            pilot_sealed_count=None, source_dir=None, fetch=False):
    """Verify, convert and publish a local bundle, returning two absolute manifest paths."""
    if bool(fetch) == (source_dir is not None):
        raise ValueError('Choose exactly one of --fetch or --source-dir')
    counts = [('train_count', train_count, _SOURCES['math_train'].count),
              ('dev_count', dev_count, _SOURCES['aime_dev'].count // len(DEV_SPLITS))]
    if pilot_sealed_count is not None:
        counts.append(('pilot_sealed_count', pilot_sealed_count, 59))
    for name, value, maximum in counts:
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f'{name} must be an integer in [1, {maximum}]')
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output directory already exists: {output}')
    sources = _read_sources(Path(source_dir).expanduser() if source_dir is not None else None)
    bundle = _pack(sources, train_count, dev_count, pilot_sealed_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.aime-prep-') as temporary:
        staging = Path(temporary) / 'bundle'
        staging.mkdir()
        _write_bundle(staging, *bundle, pilot_sealed_count is not None)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Output directory already exists: {output}')
        staging.rename(output)
    return output / PARENT_MANIFEST, output / META_MANIFEST


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, help='new local (not vendored) bundle directory')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fetch', action='store_true', help='opt in to pinned HTTP fetches for permitted local use')
    source.add_argument('--source-dir', type=Path, help='offline raw files under the paths from --list-sources')
    source.add_argument('--list-sources', action='store_true', help='show pins and offline paths without fetching')
    parser.add_argument('--train-count', type=int, default=DEFAULT_COUNT, help='shared train rows (default: 8)')
    parser.add_argument('--dev-count', type=int, default=DEFAULT_COUNT, help='rows per dev cohort (default: 8)')
    parser.add_argument('--pilot-sealed-count', type=int, help='explicit partial pilot: 1–59; default all 60 rows')
    args = parser.parse_args(argv)
    if args.list_sources:
        print(_json_bytes({'sources': {name: record.provenance() for name, record in _SOURCES.items()},
                           'licensing': _LICENSE_NOTES}).decode('utf-8'), end='')
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required to prepare a bundle')
    try:
        parent, meta = prepare(args.output_dir, train_count=args.train_count, dev_count=args.dev_count,
                               pilot_sealed_count=args.pilot_sealed_count, source_dir=args.source_dir,
                               fetch=args.fetch)
    except (ValueError, OSError) as error:
        parser.exit(1, f'AIME preparation failed: {error}\n')
    print(f'Parent manifest: {parent}')
    print(f'Meta-trial manifest: {meta}')
    print('Keep corpora local; card licenses do not grant third-party redistribution rights.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
