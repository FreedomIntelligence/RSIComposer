"""Read-only contingency admission of the original wide AIME round-one Data result.

This is not snapshot restoration or candidate selection. Only the run named by
an explicit path and result pin is considered; no development score is used.
"""
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import stat

from rsicomposer.snapshot.codec import decode

from . import aime, aime_synthesis
from .campaign import campaign_config
from .engine import digest


_SEED_ROWS = 228
_ROUND = 1
_MAX_DOCUMENT = 64 * 1024 * 1024
_MAX_RECEIPT = 256 * 1024
_MAX_JOURNAL = 512 * 1024 * 1024
_PROFILE = 'meta-rsi/v1'
_REQUEST_KEYS = ('parent', 'operator', 'round', 'profile')
_ROLES = {'train', 'search', 'validation'}


def _require(condition, message):
    if not condition:
        raise ValueError('AIME Model stage: ' + message)


def _file(path, limit):
    """Read the bounded contents of one local ordinary file, without following links."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, 'invalid source artifact size or type')
        payload = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        _require(len(payload) == before.st_size == after.st_size
                 and before.st_mtime_ns == after.st_mtime_ns and before.st_ctime_ns == after.st_ctime_ns,
                 'source artifact changed during reading')
    return payload


def _json(raw):
    def checked_object(pairs):
        value = dict(pairs)
        _require(len(value) == len(pairs), 'duplicate source JSON field')
        return value

    def invalid_constant(_value):
        raise ValueError('AIME Model stage: non-finite source JSON number')

    return json.loads(raw, object_pairs_hook=checked_object, parse_constant=invalid_constant)


def _source_report(result, lab, seed_id):
    _require(result['status'] == 'finished' and result['stop_reason'] == 'researcher_finished'
             and result['shutdown_complete'] is True and result['cleanup']['status'] == 'completed',
             'source has not finished with clean shutdown')
    _require(not any(result.get(k) for k in ('sealed_evaluations', 'final_evaluations'))
             and not any(event.get('phase') == 'host_finish' or (event.get('operation') or {}).get(
                 'arguments', {}).get('split') in ('test', 'sealed') for event in result['events']),
             'source has a sealed or final evaluation')
    # Use the embedded launch identity only. Never open either source manifest.
    method = result['framework']['method']
    if method['method'] == 'registered_framework':
        method = method['config']['method']
    config = method['config']
    policy, labs = config['controller'], config['labs']
    _require(method['method'] == 'meta_rsi' and set(labs) == {'math', 'reflect'}
             and not policy.get('trial') and policy.get('round_offset', 0) == 0,
             'unexpected source method or non-main controller')
    for configured, names in ((labs['math'], ('train', 'search', 'validation')),
                              (policy['trial_lab'], ('train', 'meta_search', 'meta_validation'))):
        _require(configured['split_roles'] == dict(zip(('train', 'search', 'validation'), names, strict=True))
                 and 'initial_snapshot' not in configured and 'model_stage' not in configured
                 and 'model_stage' not in configured['wide_loop']
                 and configured['engine'] == 'transformers' and configured['aime']['data_policy'] == 'affine'
                 and configured['model_options']['seed'] == 23
                 and all(configured['wide_loop'][key] == lab.wide[key]
                         for key in ('bootstrap_sha256', 'checkpoint_sha256', 'aggregate_sha256')),
                 'source is not the native development-only wide AIME configuration')
    report = result['controller_report']
    campaign = campaign_config(policy['campaign'])
    _require(report['status'] == 'done' and report.get('error') is None and report['simulated'] is False
             and report['campaign'] == campaign and campaign['track'] == 'adaptive'
             and campaign['policy_seed'] == 23 and report['initial'] == report['selected'] == seed_id,
             'source controller did not finish and release the unchanged native seed')
    rows, terms = report['attempts'], report['terms']
    _require(isinstance(rows, list) and rows and isinstance(terms, list) and len(terms) == campaign['terms'],
             'source is missing executed attempts or term releases')
    for number, row in enumerate(rows, 1):
        transition = row['transition']
        _require(type(row['round']) is int and row['round'] == number
                 and row['operator'] in ('data', 'harness', 'model') and row['profile'] in ('compact', 'detailed')
                 and aime.candidate_id(row['parent']) and (row['candidate'] is None or (
                     aime.candidate_id(row['candidate']) and row['candidate'] != row['parent']))
                 and type(row['data_count']) is int and isinstance(transition, dict)
                 and row['training_performed'] is False and 'training' not in transition
                 and (row['operator'] != 'model' or row['candidate'] is None),
                 'source proposal is malformed or has optimizer training')
    releases, prior, cursor = [], seed_id, 0
    for index, term in enumerate(terms, 1):
        operators = term['operators']
        stop = cursor + len(operators)
        _require(term['term'] == index and term['start_id'] == prior and aime.candidate_id(term['released_id'])
                 and type(term['trained_candidates']) is int and term['trained_candidates'] == 0
                 and term['operator_calls'] == len(operators)
                 and operators == [r['operator'] for r in rows[cursor:stop]]
                 and all(r['term'] == index for r in rows[cursor:stop]), 'inconsistent term releases or optimizer counts')
        prior, cursor = term['released_id'], stop
        releases.append(prior)
    _require(cursor == len(rows) and releases[-1] == seed_id, 'source term releases do not cover the native incumbent')
    # Trial summaries expose requested operators, not training flags. A capable
    # trial without conclusive non-Model operator evidence is not a fallback source.
    for meta in report['meta']:
        for trial in meta.get('trials', []):
            if 'error' in trial and meta['target'] in ('data_policy', 'harness_policy'):
                continue  # these vertical arms are fixed to one non-Model operator
            _require('error' not in trial and isinstance(trial['attempts'], list) and trial['attempts']
                     and all(isinstance(item, list) and len(item) == 2 and item[0] in ('data', 'harness')
                             for item in trial['attempts']), 'source trial cannot exclude optimizer training')
    first = rows[0]
    _require(first['term'] == 1 and first['round'] == _ROUND and first['parent'] == seed_id
             and first['operator'] == 'data' and first['profile'] == policy.get('profiles', {}).get('data', 'compact')
             and first['candidate'] is not None and first['data_fresh'] is True,
             'earliest executed round is not a real Data refresh from the native seed')
    return rows, releases


def _feedback(source):
    """Read no evaluation payloads; project only operation identity from any evaluation request."""
    path = source / 'experience.sqlite3'
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= _MAX_JOURNAL, 'invalid source Host journal')
    for suffix in ('-wal', '-shm', '-journal'):
        sidecar = source / (path.name + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            info = sidecar.lstat()
            _require(stat.S_ISREG(info.st_mode) and (suffix == '-shm' or info.st_size == 0),
                     'source Host journal has uncommitted sidecar state')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as conn:
        _require(conn.execute("""
            SELECT 1 FROM experience WHERE record_id LIKE 'sealed-%'
                OR (kind = 'evaluation' AND json_extract(payload, '$.details.phase') = 'host_finish') LIMIT 1
        """).fetchone() is None, 'source Host journal contains final evaluation')
        rows = conn.execute("""
            SELECT record_id, json_extract(payload, '$.details.operation'),
                   json_extract(payload, '$.details.status'),
                   CASE WHEN json_extract(payload, '$.details.operation') = 'math.evaluate' THEN
                       (SELECT json_extract(field.value, '$[1]')
                        FROM json_each(payload, '$.details.event."$map"') AS field
                        WHERE json_extract(field.value, '$[0]') = 'request' AND length(field.value) <= ?)
                   WHEN length(json_extract(payload, '$.details.event')) <= ? THEN
                       json_extract(payload, '$.details.event')
                   END
            FROM experience WHERE kind = 'runtime_feedback' AND json_extract(payload, '$.details.operation')
                IN ('math.seed', 'math.propose', 'math.select', 'math.evaluate') ORDER BY seq LIMIT 1025
        """, (_MAX_RECEIPT, _MAX_RECEIPT)).fetchall()
    _require(0 < len(rows) <= 1024, 'unexpected source Host feedback count')
    feedback, calls = [], set()
    for record_id, operation, status, serialized in rows:
        _require(isinstance(serialized, str), 'missing source Host feedback or oversized receipt')
        event = decode(_json(serialized))
        if operation == 'math.evaluate':
            _require(event['name'] == operation and event['arguments']['split'] in ('search', 'validation'),
                     'source Host journal has a non-development evaluation')
            continue
        request, output = event['request'], event['output']
        args, call = request['arguments'], request['call_id']
        _require(status == event['status'] == output['status'] == 'completed'
                 and event['source'] == 'lab' and event['event_id'] == record_id and request['name'] == operation
                 and isinstance(call, str) and call and call not in calls
                 and event['task_id'] == output['call_id'] == event['identity']['call_id'] == call
                 and output['simulated'] is False and output['data']['simulated'] is False
                 and output['data']['protocol'] == _PROFILE, 'inconsistent source Host operation receipt')
        calls.add(call)
        feedback.append((operation, args, output['data']))
    return feedback


def _match_host(feedback, rows, releases, seed_id):
    _require(feedback and feedback[0][0] == 'math.seed' and feedback[-1][0] == 'math.select',
             'source Host history does not start with seed and end with release')
    seeds = [(args, data) for op, args, data in feedback if op == 'math.seed']
    _require(len(seeds) == 1 and seeds[0][0] == {} and seeds[0][1]['candidate_id'] == seed_id
             and seeds[0][1]['data_count'] == _SEED_ROWS and seeds[0][1]['training_performed'] is False
             and seeds[0][1]['data_fresh'] is False, 'source Host did not admit the native unstaged seed')
    selected = []
    for op, args, data in feedback:
        if op == 'math.select':
            ident = data['candidate_id']
            _require(args == {'candidate': ident} and data['baseline_id'] == seed_id
                     and data['profile'] == _PROFILE and data['training_performed'] is False,
                     'source Host selection receipt mismatches its request')
            selected.append(ident)
    _require(selected == releases, 'source Host releases disagree with the source controller')
    proposals = [(args, data) for op, args, data in feedback if op == 'math.propose']
    _require(len(proposals) == len(rows), 'source Host proposal count disagrees with the source controller')
    for row, (args, data) in zip(rows, proposals, strict=True):
        _require(args == {key: row[key] for key in _REQUEST_KEYS} and type(args['round']) is int
                 and all(data[key] == value for key, value in args.items())
                 and data['candidate_id'] == row['candidate'] and data.get('reason') == row.get('reason')
                 and type(data['data_count']) is int and data['data_count'] == row['data_count']
                 and data['training_performed'] is False and data['data_fresh'] is row['data_fresh']
                 and digest(data['transition']) == digest(row['transition']),
                 'source Host proposal differs from the report or has optimizer training')
    return proposals


def _match_proposals(root, proposals):
    """Only examine the main Lab's exact train-safe proposal namespace."""
    expected = {args['round']: {'request': args, 'candidate_id': data['candidate_id'],
                'reason': data.get('reason'), 'transition': data['transition']} for args, data in proposals}
    first_hash = None
    for entry in root.glob('proposal-*.json'):
        _require(expected and aime.sha256_text(entry.stem.removeprefix('proposal-')), 'unmatched source Lab proposal')
        raw = _file(entry, _MAX_RECEIPT)
        saved = _json(raw)
        number = saved['request']['round']
        _require(type(number) is int and number in expected and digest(saved) == digest(expected.pop(number)),
                 'source Lab proposal differs from the Host request or result')
        if number == _ROUND:
            first_hash = sha256(raw).hexdigest()
    _require(not expected and first_hash is not None, 'source Lab proposal is missing')
    return first_hash


def _verify_synthesis(lab, rows, profile, audit):
    """Verify provenance of the appended rows independently of the refresh hash."""
    gaps, width = lab._wide_bootstrap['failed_problem_ids'], 8 if profile == 'compact' else 16
    _require(len(gaps) == 28 and len(set(gaps)) == 28, 'invalid native training gap cohort')
    start = (_ROUND * 7) % len(gaps)
    cohort = [gaps[(start + index) % len(gaps)] for index in range(width)]
    synthetic = {}
    for ident in cohort:
        item = aime_synthesis.propose(lab._train_rows[ident], 23, _ROUND)
        if item is not None:
            synthetic[item[0].id] = item
    recovered, receipts = 0, []
    for row in rows:
        ident = row['id']
        if ident in cohort:
            problem = lab._train_rows[ident]
            recovered += 1
        else:
            _require(ident in synthetic, 'appended row is not derived from the round-one training gaps')
            problem, provenance = synthetic[ident]
            receipts.append({'source_sha256': digest(provenance['source_id']), 'rule': provenance['rule'],
                             'item_sha256': digest(ident), 'row_sha256': digest(row)})
        _require(row['prompt'] == problem.prompt and problem.verifier.check(row['completion'], problem.answer).correct,
                 'appended row fails independent training-only numeric verification')
    counts = ('fresh_blind_originals', 'fresh_original_failures', 'verified_original_recoveries',
              'nominated', 'already_held', 'rejected_by_anchor_or_verifier')
    _require(set(audit) == {'rule', 'admitted', *counts} and audit['rule'] == 'observed-train-gap-affine-v1'
             and all(type(audit[k]) is int and audit[k] >= 0 for k in counts)
             and audit['fresh_blind_originals'] == width
             and recovered == audit['verified_original_recoveries']
             and recovered + audit['fresh_original_failures'] <= width
             and audit['nominated'] <= audit['fresh_original_failures']
             and audit['already_held'] + audit['rejected_by_anchor_or_verifier'] + len(receipts) <= audit['nominated']
             and receipts == audit['admitted'], 'inconsistent round-one synthesis admissions')


def load_model_stage(lab, seed, spec):
    """Return (Data candidate, exact refresh proof, aggregate provenance), or raise ValueError."""
    _require(isinstance(spec, dict) and set(spec) == {'source_run', 'result_sha256'}
             and isinstance(spec['source_run'], str) and aime.sha256_text(spec['result_sha256']), 'invalid source pin')
    try:
        source = Path(spec['source_run'])
        destination = lab.root.parent.resolve()
        _require(source.is_absolute() and source == source.resolve(strict=True) and source.is_dir()
                 and source != destination and source not in destination.parents and destination not in source.parents,
                 'source must be a separate canonical absolute run directory')
        _require(lab.wide is not None and set(lab._protocols) == _ROLES and aime.sha256_text(lab.base_digest)
                 and lab._wide_bootstrap['source']['base_sha256'] == lab.base_digest,
                 'destination is not the pinned development-only wide Lab')
        settings = {'max_verified_rows': 288, 'max_completion_chars': aime.MAX_VERIFIED_COMPLETION_CHARS, 'max_exemplars': 2}
        aime.check_components(seed, settings, ('seed',))
        _require(len(seed['data']) == _SEED_ROWS and seed['data'] == lab._wide_bootstrap['rows']
                 and seed['harness'] == {'instruction': 'seed', 'few_shot': False, 'exemplars': [],
                                         'revision': digest('aime-system-prompt/v1')}
                 and aime.sha256_text(seed['model']['checkpoint'])
                 and seed['model']['checkpoint_sha256'] == lab.wide['checkpoint_sha256'],
                 'destination seed does not match the native rows, prompt and adapter')
        lab._verify_data(seed['data'])
        seed_id = lab._id(seed)
        _require(aime.candidate_id(seed_id), 'invalid native seed ID')
        raw = _file(source / 'result.json', _MAX_DOCUMENT)
        _require(sha256(raw).hexdigest() == spec['result_sha256'], 'source result SHA256 does not match the pin')
        rows, releases = _source_report(_json(raw), lab, seed_id)
        feedback = _feedback(source)
        proposals = _match_host(feedback, rows, releases, seed_id)
        root = source / 'meta-rsi'
        _require(not root.is_symlink() and root.is_dir(), 'source Lab directory is missing or symlinked')
        proposal_hash = _match_proposals(root, proposals)
        before = _json(_file(root / f'{seed_id}.json', _MAX_DOCUMENT))
        _require(digest(before) == digest(seed) and lab._id(before) == seed_id, 'source candidate differs from native seed')
        first = rows[0]
        key = first['candidate']
        candidate = _json(_file(root / f'{key}.json', _MAX_DOCUMENT))
        aime.check_components(candidate, settings, ('seed',))
        data = candidate['data']
        _require(lab._id(candidate) == key and len(data) > _SEED_ROWS and len(data) == first['data_count']
                 and data[:_SEED_ROWS] == seed['data'] and candidate['harness'] == seed['harness']
                 and candidate['model'] == seed['model'], 'staged Data must only append to the exact native seed')
        lab._verify_data(data)
        proof = lab._data_proof(key, candidate, _SEED_ROWS)
        transition = first['transition']
        _require(set(transition) == {'edges', 'requires_refresh', 'synthesis', 'verified_added', 'refresh_proof'}
                 and transition['edges'] == [] and transition['requires_refresh'] is False
                 and type(transition['verified_added']) is int and transition['verified_added'] == len(data) - _SEED_ROWS
                 and proof['candidate_id'] == key and proof['source_candidate_id'] == seed_id
                 and proof['train_protocol_id'] == lab._protocols['train']
                 and digest(proof) == digest(transition['refresh_proof']) == digest(_json(_file(
                     root / f'fresh-{key}.json', _MAX_RECEIPT))), 'source refresh proof does not match the exact Data candidate')
        _verify_synthesis(lab, data[_SEED_ROWS:], first['profile'], transition['synthesis'])
        return candidate, proof, {'source_run': str(source), 'result_sha256': spec['result_sha256'],
            'seed_candidate_id': seed_id, 'source_candidate_id': key, 'source_round': _ROUND,
            'proposal_sha256': proposal_hash, 'refresh_proof_sha256': digest(proof),
            'added_sha256': proof['added_sha256'], 'cumulative_sha256': proof['cumulative_sha256'],
            'previous_count': _SEED_ROWS, 'added_count': len(data) - _SEED_ROWS, 'cumulative_count': len(data)}
    except (OSError, sqlite3.Error, KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError,
            RuntimeError):
        # Don't expose unrelated source payloads (e.g. JSON fields) in failures.
        raise ValueError('AIME Model stage: malformed or unavailable source evidence') from None
