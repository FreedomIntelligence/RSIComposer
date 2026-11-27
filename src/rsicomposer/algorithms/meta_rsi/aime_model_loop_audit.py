"""Read-only aggregate audit of development-only, target-model-driven MetaRSI."""
import argparse
from contextlib import closing
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3

from rsicomposer.runtime.state.recovery import DurableRecovery

from . import aime
from .aime_domain_policy import recover_task
from .aime_results import optimizer_evidence
from .aime_wide_loop import load_bootstrap
from .aime_wide_loop_audit import _read_candidate, _training_summary, host_adoptions, host_releases
from .engine import digest, read_checkpoint
from .lab import MetaRSILab, validate_config as validate_lab_config
from .model_controller import NAME
from .model_policy import OPERATORS, PROFILES


_ACTIONS = ('route', 'program', 'update_operator', 'rewrite_operator', 'rewrite_scheduler', 'keep', 'stop')
_PHASES = ('seed', 'advance', 'await_adoption', 'failed', 'done', 'model_route', 'model_vertical', 'model_meta')
_META_STATES = ('kept', 'adoption_requested', 'adopted')
_STAGE_STATES = ('staged', 'continued_with_data', 'consumed_by_model', 'consumed_by_harness',
                 'consumer_produced_no_candidate', 'not_consumed_before_release')
_OPERATIONS = ('math.seed', 'math.evaluate', 'math.diagnose', 'math.propose', 'math.policy', 'math.select',
               'math.compare', 'reflect.build_candidate', 'host.adopt_controller')
_POLICY_FIELDS = ('order', 'diagnostics', 'evidence', 'operator_budget', 'prefer_axis', 'stop', 'routing',
                  'transition_guidance')
_RELEASE_VERSION = re.compile(r'meta-model-adopt-[0-9a-f]{24}')


def _owned(path, *, required=True):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('Audit inputs must be owned regular files')
    if required and not path.is_file():
        raise ValueError('Required audit input has not been published')
    return path.is_file()


def _enum(value, choices):
    return value if type(value) is str and value in choices else None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _sha(value):
    return value if aime.sha256_text(value) else None


def _candidate(value):
    return value if aime.candidate_id(value) else None


def _version(value):
    return value if isinstance(value, str) and _RELEASE_VERSION.fullmatch(value) else None


def _operation_list(items):
    return [item for item in items if _enum(item, OPERATORS) is not None]


def _checked_config(run, config):
    run, config = (Path(p).expanduser().absolute() for p in (run, config))
    if run.is_symlink() or not run.is_dir():
        raise ValueError('Expected an owned model-driven campaign directory')
    _owned(config)
    raw = config.read_bytes()
    launch = json.loads(raw)
    method = launch['framework']['method']
    options = method['config']
    lab = options['labs']['math']
    roles = lab['split_roles']
    if (method['name'] != NAME or Path(launch['data_dir']).expanduser().resolve() != run.resolve()
            or set(options['labs']) != {'math', 'reflect'} or lab.get('model_driven') is not True
            or set(roles) != {'train', 'search', 'validation'} or len(set(roles.values())) != 3
            or options['controller'].get('trial') or options['controller'].get('trial_lab')
            or options['controller'].get('meta_targets') or 'model_stage' in lab.get('wide_loop', {})):
        raise ValueError('Expected meta_rsi_model with a train/development-only parent and no nested trials')
    if lab.get('initial_snapshot') is not None:
        raise ValueError('Expected a native model-driven campaign, not an initial snapshot import')
    validate_lab_config({key: value for key, value in lab.items() if key != 'max_concurrent'})
    return run, method, launch, sha256(raw).hexdigest()


def _check_terminal_identity(result, method):
    identity = result.get('framework', {}).get('method', {})
    if identity.get('method') == 'registered_framework':
        identity = identity.get('config', {}).get('method', {})
    if identity.get('method') != NAME or identity.get('config') != method['config']:
        raise ValueError('Result does not match the model-driven launch config')
    if any(result.get(key) for key in ('sealed_evaluations', 'final_evaluations')) or any(
            event.get('phase') == 'host_finish' or (event.get('operation') or {}).get('arguments', {}).get('split')
            in ('test', 'sealed') for event in result.get('events', [])):
        raise ValueError('Unexpected non-development evaluation in model-driven result')


def _journal(run, *, required):
    path = run / 'experience.sqlite3'
    if not _owned(path, required=required):
        return []
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        if connection.execute("SELECT 1 FROM experience WHERE record_id LIKE 'sealed-%' "
                              "OR kind = 'host_finish' LIMIT 1").fetchone() is not None:
            raise ValueError('Unexpected Host final evaluation in model-driven journal')
        rows = connection.execute("SELECT json_extract(payload, '$.details.operation'), "
            "json_extract(payload, '$.details.status') FROM experience WHERE kind='runtime_feedback' "
            "ORDER BY seq DESC LIMIT 16").fetchall()
    return [{'operation': op, 'status': _enum(status, ('completed', 'failed', 'cancelled', 'running'))}
            for op, status in reversed(rows) if _enum(op, _OPERATIONS)]


def _strategy(state):
    decisions = []
    for row in state.get('policy_decisions', state.get('policy_log', [])):
        outcome = row['outcome']
        proposal = outcome.get('proposal', {}) if outcome.get('accepted') is True else {}
        action = _enum(proposal.get('action'), _ACTIONS)
        entry = {'term': _count(row.get('term')), 'phase': _enum(row.get('phase'), _PHASES),
                 'candidate_id': _candidate(row.get('candidate')), 'contract_accepted': outcome.get('accepted') is True,
                 'action': action, 'request_sha256': _sha(row.get('request_sha256')),
                 'response_sha256': _sha(row.get('response_sha256'))}
        if 'operator' in proposal:
            entry['operator'] = _enum(proposal['operator'], OPERATORS)
        if action == 'program':
            entry['program_operators'] = _operation_list(step['operator'] for step in proposal['steps'])
        if action == 'rewrite_operator':
            entry['changed_fields'] = [k for k in ('profile', 'instruction') if k in proposal['patch']]
            entry['proposed_profile'] = _enum(proposal['patch'].get('profile'), PROFILES)
        if action == 'rewrite_scheduler':
            entry['changed_fields'] = [k for k in _POLICY_FIELDS if k in proposal['patch']]
        decisions.append(entry)
    return {'evidence_kind': 'bounded_contract_and_activation_only', 'matched_performance_trial': False,
            'decisions': decisions, 'schedule_reviews': [
                {'term': _count(row.get('term')), 'disposition': _enum(row.get('decision'), _META_STATES),
                 'version_id': _version(row.get('version_id'))} for row in state.get('meta', [])],
            'program_invalidations': [
                {'term': _count(row.get('term')), 'round': _count(row.get('round')),
                 'operator': _enum(row.get('operator'), OPERATORS), 'cleared_steps': _count(row.get('cleared_steps'))}
                for row in state.get('program_invalidations', [])]}


def _development(state):
    return {'measurement_scope': 'frozen parent development validation; no sealed or causal strategy claim',
            'initial_id': _candidate(state.get('initial')),
            'incumbent_id': _candidate(state.get('selected', state.get('current'))),
            'last_released_id': _candidate(state.get('last_released')),
            'validation': [{'candidate_id': key, **{field: _count(row.get(field)) for field in
                            ('count', 'invalid_format', 'truncated', 'prompt_overlength')},
                            'accuracy': _number(row.get('accuracy')), 'protocol_id': _sha(row.get('protocol_id'))}
                           for key, row in state.get('diagnoses', {}).items() if _candidate(key)],
            'terms': [{**{key: _count(row.get(key)) for key in ('term', 'operator_budget', 'operator_calls',
                                                                'trained_candidates', 'dev_items_scored')},
                       **{key: _number(row.get(key)) for key in ('start_score', 'released_score', 'gain')},
                       'start_id': _candidate(row.get('start_id')), 'released_id': _candidate(row.get('released_id')),
                       'controller_version': _version(row.get('controller_version')),
                       'policy_sha256': _sha(row.get('policy_sha256')), 'order': _operation_list(row.get('order', [])),
                       'operators': _operation_list(row.get('operators', []))} for row in state.get('terms', [])]}


def _proof(root, key, parent, proof, *, base, verify_candidates):
    if (not _candidate(key) or not _candidate(parent) or proof.get('candidate_id') != key
            or proof.get('source_candidate_id') != parent or type(proof.get('previous_count')) is not int
            or type(proof.get('cumulative_count')) is not int or type(proof.get('added_count')) is not int
            or not 0 <= proof['previous_count'] < proof['cumulative_count']
            or proof['added_count'] != proof['cumulative_count'] - proof['previous_count']
            or any(not _sha(proof.get(k)) for k in ('train_protocol_id', 'harness_sha256', 'model_sha256',
                                                   'added_sha256', 'cumulative_sha256'))):
        raise ValueError('Invalid staged Data refresh-proof receipt')
    # Current runs use fresh-; earlier build artifacts also used refresh-.
    records = [value for prefix in ('refresh-', 'fresh-')
               if (value := MetaRSILab._read_owned(root, f'{prefix}{key}.json')) is not None]
    if not records or any(record != proof for record in records):
        raise ValueError('Staged Data receipt does not match the owned refresh proof')
    if verify_candidates:
        before, after = (_read_candidate(root, ident, base) for ident in (parent, key))
        count, data = proof['previous_count'], after['data']
        if (count != len(before['data']) or before['data'] != data[:count] or len(data) != proof['cumulative_count']
                or before['harness'] != after['harness'] or digest(after['harness']) != proof['harness_sha256']
                or before['model'] != after['model'] or digest(after['model']) != proof['model_sha256']
                or digest(data[count:]) != proof['added_sha256'] or digest(data) != proof['cumulative_sha256']):
            raise ValueError('Staged Data does not append to the exact source candidate')


def _regrade_synthetic(root, key, proof, synthesis):
    """Independently grade every admitted synthetic completion without any cohort reads."""
    candidate = MetaRSILab._read_owned(root, f'{key}.json')
    data = candidate['data']
    added = data[proof['previous_count']:]
    if digest(data) != proof['cumulative_sha256'] or digest(added) != proof['added_sha256']:
        raise ValueError('Synthetic regrading inputs do not match the owned refresh proof')
    admitted = synthesis['admitted']
    consolidation = synthesis.get('consolidation', [])
    by_hash = {receipt['row_sha256']: (receipt, lane) for lane, receipts in (
        ('frontier', admitted), ('already_held', consolidation)) for receipt in receipts}
    if len(by_hash) != len(admitted) + len(consolidation):
        raise ValueError('Duplicate synthetic admission evidence')
    regraded, originals, seen = [], 0, set()
    for row in added:
        recovered = recover_task(row['id'])
        if recovered is None:
            originals += 1
            continue
        problem, provenance = recovered
        receipt, lane = by_hash.pop(digest(row), (None, None))
        if (problem.id in seen or receipt is None or receipt.get('item_sha256') != digest(problem.id)
                or any(receipt.get(field) != value for field, value in provenance.items())
                or row['prompt'] != problem.prompt or not problem.verifier.check(row['completion'], problem.answer).correct):
            raise ValueError('Owned synthetic row failed independent admission regrading')
        seen.add(problem.id)
        published_proof = receipt.get('certificate_sha256')
        if published_proof is not None:
            from .aime_curriculum import check_subset_residue_completion, subset_residue_spec_from_id
            tail = '\n'.join(row['completion'].strip().splitlines()[-2:])
            checked = digest(check_subset_residue_completion(subset_residue_spec_from_id(problem.id), tail))
            expected = published_proof[0] if isinstance(published_proof, list) and published_proof else published_proof
            if checked != expected:
                raise ValueError('Structured synthetic witness changed after validation')
        # Do not publish the completion-derived digests or the two blind response hashes.
        regraded.append({'item_sha256': digest(problem.id), **provenance,
                         **({'lane': lane, 'certificate_regraded': True} if published_proof is not None else {})})
    if by_hash or originals != synthesis['verified_original_recoveries']:
        raise ValueError('Synthetic admission receipts do not cover exactly the added rows')
    return regraded


def _attempts(run, state, *, base=None, verify_candidates=False, real=False, wide=False, regrade=False):
    root = run / 'meta-rsi'
    incumbent, staged, rows = state.get('initial'), {}, []
    simulation = state.get('simulated') is True
    for row in state.get('attempts', []):
        op, key, parent = row['operator'], row['candidate'], row['parent']
        transition = row.get('transition') or {}
        entry = {'operator': _enum(op, OPERATORS), 'profile': _enum(row.get('profile'), PROFILES),
                 'term': _count(row.get('term')), 'round': _count(row.get('round')),
                 'parent_id': _candidate(parent), 'candidate_id': _candidate(key),
                 'data_count': _count(row.get('data_count')), 'selected': row.get('selected') is True,
                 'training_performed': row.get('training_performed') is True}
        if 'kernel_evidence' in row:
            from .kernel_evidence import validate_kernel_evidence
            validate_kernel_evidence(row['kernel_evidence'])
            entry['kernel_evidence_sha256'] = row['kernel_evidence']['evidence_sha256']
        if 'paired_selection' in row:
            gate = row['paired_selection']
            entry['paired_selection'] = {**{field: _count(gate.get(field)) for field in ('count', 'gained', 'regressed')},
                                         'eligible': gate.get('eligible') is True, 'delta': _number(gate.get('delta'))}
        actor = transition.get('actor') or (transition.get('synthesis') or {}).get('actor')
        if actor:
            entry['domain_proposal'] = {'contract_accepted': actor.get('accepted') is True,
                **{field: _sha(actor.get(field)) for field in ('request_sha256', 'response_sha256', 'proposal_sha256')}}
        synthesis = transition.get('synthesis')
        if synthesis is not None:
            entry['synthesis'] = {**{k: _count(synthesis.get(k)) for k in ('fresh_blind_originals',
                'fresh_original_failures', 'verified_original_recoveries', 'nominated', 'already_held',
                'rejected_by_anchor_or_verifier')}, 'verified_synthetic_admissions': len(synthesis['admitted']),
                'consolidation_rows': len(synthesis.get('consolidation', []))}
        if key is not None and op == 'data':
            if entry['selected'] or key in state.get('scores', {}) or any(k in row for k in ('search_score', 'validation_score')):
                raise ValueError('Staged Data appeared as a measured or selected system')
            proof = transition.get('refresh_proof', {})
            _proof(root, key, parent, proof, base=base, verify_candidates=verify_candidates)
            if row.get('data_count') != proof['cumulative_count'] or row.get('data_fresh') is not True:
                raise ValueError('Staged Data count does not match its refresh proof')
            staged[key] = proof, incumbent
            entry['stage'] = {'disposition': _enum(row.get('stage_disposition'), _STAGE_STATES),
                'incumbent_id': _candidate(incumbent), 'added_verified_rows': proof['added_count'],
                'refresh_sha256': digest(proof), 'cumulative_data_sha256': proof['cumulative_sha256']}
            if regrade:
                if synthesis is None:
                    raise ValueError('Model-driven Data is missing synthesis evidence')
                regraded = _regrade_synthetic(root, key, proof, synthesis)
                entry['synthesis'].update(independently_regraded_synthetic_rows=len(regraded),
                                          independently_regraded_provenance=regraded)
        elif key is not None and op == 'model':
            proof, original = staged.get(parent, (None, None))
            training = transition.get('training', {})
            checkpoint = training.get('checkpoint')
            if (proof is None or original != incumbent or transition.get('refresh_proof') != proof
                    or training.get('cumulative_count') != proof['cumulative_count'] or not _sha(checkpoint)):
                raise ValueError('Model did not consume the exact Data stage from the measured incumbent')
            payload = read_checkpoint(root / 'models', checkpoint, simulated=simulation)
            recipe = payload['recipe']
            if (payload['samples'] != proof['cumulative_count'] or recipe.get('refresh_sha256') != digest(proof)
                    or recipe.get('train_protocol_id') != proof['train_protocol_id']
                    or recipe.get('label_scope') != 'assistant_only' or training.get('label_scope') != 'assistant_only'
                    or recipe.get('operator_proposal_sha256') != (actor or {}).get('proposal_sha256')
                    or not _sha(recipe.get('operator_proposal_sha256'))):
                raise ValueError('Model checkpoint is not bound to the staged proof and domain proposal')
            if verify_candidates:
                source, fitted = (_read_candidate(root, ident, base) for ident in (parent, key))
                if (fitted['harness'] != source['harness'] or fitted['data'] != source['data']
                        or fitted['model'] == source['model']
                        or fitted['model'] != {'checkpoint': checkpoint, 'checkpoint_sha256': digest(payload)}):
                    raise ValueError('Model did not change exactly the adapter on the staged system')
            entry['fit'] = {'checkpoint': checkpoint, 'refresh_sha256': digest(proof),
                            'training_rows': payload['samples'], 'incumbent_id': _candidate(original)}
            if real and wide:
                evidence = _training_summary(root, training)
                entry['fit'].update({field: _number(evidence.get(field)) for field in (
                    'steps', 'microsteps', 'samples', 'completion_tokens_seen', 'world_size',
                    'effective_batch_size', 'final_batch_size', 'learning_rate', 'training_seed')})
        if key is not None and op != 'data':
            score = _number(row.get('validation_score'))
            before = _number(state.get('scores', {}).get(incumbent))
            if score is not None and (before is None or row.get('gain') != score - before):
                raise ValueError('Challenger gain is not relative to the original incumbent')
            entry['development'] = {'comparator_id': _candidate(incumbent), 'validation_accuracy': score,
                'search_accuracy': _number(row.get('search_score')), 'validation_gain': _number(row.get('gain'))}
            if entry['selected']:
                incumbent = key
        rows.append(entry)
    return rows


def _provenance(method, launch, config_sha256):
    lab = method['config']['labs']['math']
    wide = lab.get('wide_loop')
    options = lab.get('model_options', {})
    trainer = launch.get('resources', {}).get('pools', {}).get('trainer', {}).get('gpus', [])
    result = {'config_sha256': config_sha256, 'method': NAME, 'seed': _count(options.get('seed')),
              'trainer_gpu_ids': [index for index in trainer if _count(index) is not None],
              'global_batch_size': _count(options.get('training_global_batch_size')), 'prior_256': None}
    if wide is None:
        return result, None
    bootstrap = load_bootstrap(wide)
    base = _sha(bootstrap['source'].get('base_sha256'))
    if base is None:
        raise ValueError('Inherited bootstrap did not bind a base checkpoint')
    result['prior_256'] = {'relationship': 'inherited_training_bootstrap_not_a_performance_comparator',
        'source_pool_size': len(bootstrap['rows']) + len(bootstrap['failed_problem_ids']),
        'inherited_verified_rows': len(bootstrap['rows']), 'source_non_admissions': len(bootstrap['failed_problem_ids']),
        'inherited_rows_sha256': digest(bootstrap['rows']),
        **{field: _sha(wide.get(field)) for field in ('bootstrap_sha256', 'checkpoint_sha256', 'aggregate_sha256')},
        'base_sha256': base}
    return result, base


def _summary(run, method, launch, config_sha256, state, *, regrade=False):
    provenance, base = _provenance(method, launch, config_sha256)
    real, wide = state.get('simulated') is False, provenance['prior_256'] is not None
    if wide and state.get('initial') is not None:
        initial = _read_candidate(run / 'meta-rsi', state['initial'], base)
        if (digest(initial['data']) != provenance['prior_256']['inherited_rows_sha256'] or
                initial['model'].get('checkpoint_sha256') != provenance['prior_256']['checkpoint_sha256']):
            raise ValueError('Campaign seed does not match its inherited training bootstrap')
    return {'development_only': True, 'provenance': provenance, 'simulated': state.get('simulated') if type(state.get('simulated')) is bool else None,
            'controller_phase': _enum(state.get('status', state.get('phase')), _PHASES),
            'controller_failed': bool(state.get('error', state.get('failed'))),
            'active_controller_version': _version(state.get('active_version')),
            'strategy': _strategy(state), 'development': _development(state),
            'attempts': _attempts(run, state, base=base, verify_candidates=wide or not real, real=real,
                                  wide=wide, regrade=regrade)}


def report(run, config):
    """Audit terminal result against owned Lab artifacts and Host receipts."""
    run, method, launch, config_sha256 = _checked_config(run, config)
    _owned(run / 'result.json')
    result = json.loads((run / 'result.json').read_bytes())
    _check_terminal_identity(result, method)
    _journal(run, required=True)
    state = result['controller_report']
    summary = _summary(run, method, launch, config_sha256, state, regrade=True)
    summary['independently_regraded_synthetic_rows'] = sum(
        entry.get('synthesis', {}).get('independently_regraded_synthetic_rows', 0) for entry in summary['attempts'])
    released = [term['released_id'] for term in summary['development']['terms']]
    if (released[-1] if released else None) != summary['development']['last_released_id']:
        raise ValueError('Last release does not match the Controller term history')
    adoptions = host_adoptions(run, state)
    if any(not _version(item['version_id']) for item in adoptions):
        raise ValueError('Invalid model-driven Host adoption identity')
    summary['strategy']['host_activated_versions'] = [item['version_id'] for item in adoptions]
    real = state.get('simulated') is False
    releases = host_releases(run, state) if real else []
    summary['development']['host_verified_releases'] = [
        {'candidate_id': _candidate(item['candidate_id']), 'controller_version': _version(item['controller_version'])}
        for item in releases] if real else None
    summary['optimizer_evidence'] = optimizer_evidence(result, run)
    summary.update(run_complete=(result.get('status') == 'finished' and result.get('stop_reason') == 'researcher_finished'
                   and result.get('shutdown_complete') is True and state.get('status') == 'done'
                   and len(released) == method['config']['controller']['campaign'].get('terms', 1)),
                   run_status=_enum(result.get('status'), ('finished', 'waiting', 'failed', 'stopped')),
                   stop_reason=_enum(result.get('stop_reason'), ('researcher_finished', 'budget_exhausted',
                                                                 'decision_steps_exhausted', 'cancelled')))
    return summary


def report_progress(run, config):
    """Audit the committed durable state; the read-only journal can be ahead of it."""
    run, method, launch, config_sha256 = _checked_config(run, config)
    recent = _journal(run, required=False)
    if (run / 'recovery').is_symlink():
        raise ValueError('Audit recovery must be an owned directory')
    summary = {'development_only': True, 'config_sha256': config_sha256, 'live': True,
               'result_present': _owned(run / 'result.json', required=False), 'recent_host_operations': recent}
    if not _owned(run / 'recovery' / 'checkpoint.json', required=False):
        return {**summary, 'committed': None}
    snapshot = DurableRecovery.read(run)
    decision = snapshot['components']['decision']['state']['active']
    state = decision['policy']
    if (Path(snapshot['config']['data_dir']).resolve() != run.resolve() or 'policy_log' not in state
            or 'last_released' not in state):
        raise ValueError('Committed checkpoint is not a model-driven campaign for this run')
    return {**summary, 'generation': snapshot['generation'],
            'committed': _summary(run, method, launch, config_sha256, state)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--live', action='store_true', help='Read aggregate committed progress without resuming')
    options = parser.parse_args()
    export = report_progress if options.live else report
    print(json.dumps(export(options.run, options.config), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
