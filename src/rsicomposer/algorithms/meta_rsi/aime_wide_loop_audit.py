"""Export a verified, aggregate-only report for the wide AIME Host campaign."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from rsicomposer.algorithms.meta_rsi.aime_results import optimizer_evidence, summarize_development
from rsicomposer.algorithms.meta_rsi import aime, aime_synthesis
from rsicomposer.algorithms.meta_rsi.aime_wide_loop import load_bootstrap
from rsicomposer.algorithms.meta_rsi.engine import digest, file_hash
from rsicomposer.algorithms.meta_rsi.lab import PROFILE, MetaRSILab
from rsicomposer.labs.math.tasks import jsonl_problems
from rsicomposer.labs.math.verifiers import VERIFIERS
from rsicomposer.runtime.state.recovery import DurableRecovery
from rsicomposer.snapshot.codec import decode


def host_adoptions(run, controller):
    """Reconcile policy export with the Host's durable activation events."""
    source = Path(run).resolve() / 'experience.sqlite3'
    if source.is_symlink() or not source.is_file():
        raise ValueError('Missing owned Host adoption journal')
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as connection:
        events = [dict(adoption_id=ident, version_id=version) for ident, version in connection.execute(
            "SELECT json_extract(payload, '$.details.event.output.adoption_id'), "
            "json_extract(payload, '$.details.event.output.controller_version') "
            "FROM experience WHERE kind = 'controller_event' "
            "AND json_extract(payload, '$.details.event.kind') = 'controller_adopted' ORDER BY seq")]
    if (any(not isinstance(e['adoption_id'], str) or not isinstance(e['version_id'], str) for e in events)
            or len({e['adoption_id'] for e in events}) != len(events)):
        raise ValueError('Invalid Host adoption events')
    adopted = [item.get('version_id') for item in controller['meta'] if item['decision'] == 'adopted']
    if adopted != [event['version_id'] for event in events] or (
            adopted and controller.get('active_version') != adopted[-1]):
        raise ValueError('Controller report does not match actual Host activations')
    return events


def host_releases(run, controller):
    """Verify durable math.select receipts separately from controller adoption."""
    root = Path(run).resolve()
    source = root / 'experience.sqlite3'
    if source.is_symlink() or not source.is_file():
        raise ValueError('Missing Host release journal')
    receipts = []
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as connection:
        rows = connection.execute("SELECT json_extract(payload, '$.details.event') FROM experience "
            "WHERE kind='runtime_feedback' AND json_extract(payload, '$.details.operation') "
            "IN ('math.select','math.evaluate') ORDER BY seq")
        for (payload,) in rows:
            event = decode(json.loads(payload))
            request, output = event['request'], event['output']
            if request['name'] == 'math.evaluate':
                if request['arguments'].get('split') not in ('search', 'validation'):
                    raise ValueError('Unexpected non-development Host evaluation')
                continue
            if event['status'] != 'completed':
                continue
            key = output['data']['candidate_id']
            if not aime.candidate_id(key):
                raise ValueError('Invalid real Host release ID')
            published = MetaRSILab._read_owned(root / 'meta-rsi', f'selection-{key}.json')
            if (key != request['arguments']['candidate'] or not isinstance(published, dict)
                    or output.get('simulated') is not False or output.get('call_id') != request['call_id']
                    or {'protocol': PROFILE, **published} != output['data']):
                raise ValueError('Unexpected real Host release receipt')
            receipts.append({'candidate_id': key, 'controller_version': event['identity']['issuing_controller_version']})
    released = [term['released_id'] for term in controller['terms']]
    if [item['candidate_id'] for item in receipts] != released:
        raise ValueError('Controller reported releases different from durable Host math.select events')
    return receipts


def _read_candidate(lab_root, key, base_digest):
    if not aime.candidate_id(key):
        raise ValueError('Invalid AIME candidate ID')
    candidate = MetaRSILab._read_owned(lab_root, f'{key}.json')
    if (not isinstance(candidate, dict) or set(candidate) != {'data', 'harness', 'model'}
            or 'metarsi-' + digest({'profile': PROFILE, 'base': base_digest,
                'schema_version': 2, 'components': candidate}) != key):
        raise ValueError('Wide AIME candidate content does not match the verified identity')
    return candidate


def _runtime_components(candidate):
    return {'data_rows': len(candidate['data']), 'data_sha256': digest(candidate['data']),
            'prompt_instruction': candidate['harness']['instruction'],
            'prompt_exemplar_count': len(candidate['harness']['exemplars']),
            'prompt_sha256': digest(candidate['harness']), 'adapter': candidate['model']['checkpoint'],
            'adapter_sha256': candidate['model']['checkpoint_sha256']}


def _synthesis_evidence(lab_root, row, transition, train, old_gaps, base_digest):
    audit = transition['synthesis']
    admitted = audit['admitted']
    counts = ('fresh_blind_originals', 'fresh_original_failures', 'verified_original_recoveries',
              'nominated', 'already_held', 'rejected_by_anchor_or_verifier')
    entry = {k: audit[k] for k in counts}
    entry.update(rule=audit['rule'], verified_synthetic_admissions=len(admitted))
    if (entry['rule'] != 'observed-train-gap-affine-v1'
            or any(type(entry[k]) is not int or entry[k] < 0 for k in counts)
            or entry['fresh_original_failures'] + entry['verified_original_recoveries'] > entry['fresh_blind_originals']
            or entry['nominated'] > entry['fresh_original_failures']
            or entry['already_held'] + entry['rejected_by_anchor_or_verifier'] + len(admitted) > entry['nominated']
            or entry['verified_original_recoveries'] + len(admitted) != transition['verified_added']):
        raise ValueError('Inconsistent wide AIME synthesis counts')
    if not transition['verified_added']:
        return entry

    before = _read_candidate(lab_root, row['parent'], base_digest)
    after = _read_candidate(lab_root, row['candidate'], base_digest)
    rows = after['data'][len(before['data']):]
    proof = transition.get('refresh_proof') or {}
    if (before['data'] != after['data'][:len(before['data'])] or len(rows) != transition['verified_added']
            or before['model'] != after['model'] or before['harness'] != after['harness']
            or proof.get('added_sha256') != digest(rows) or proof.get('cumulative_sha256') != digest(after['data'])
            or proof.get('source_candidate_id') != row['parent'] or proof.get('candidate_id') != row['candidate']):
        raise ValueError('Wide AIME appended data does not match its committed refresh proof')

    width = 8 if row['profile'] == 'compact' else 16
    start = (row['round'] * 7) % len(old_gaps)
    cohort = [old_gaps[(start + i) % len(old_gaps)] for i in range(width)]
    created = {}
    for ident in cohort:
        made = aime_synthesis.propose(train[ident], 23, row['round'])
        if made is not None:
            created[made[0].id] = made
    verified_receipts = []
    recovered = 0
    for item in rows:
        ident = item['id']
        if ident in cohort:
            problem = train[ident]
            recovered += 1
        elif ident in created:
            problem, provenance = created[ident]
            verified_receipts.append({'source_sha256': digest(provenance['source_id']), 'rule': provenance['rule'],
                'item_sha256': digest(ident), 'row_sha256': digest(item)})
        else:
            raise ValueError('Wide AIME admitted data not linked to the selected training gaps')
        if item['prompt'] != problem.prompt or not problem.verifier.check(item['completion'], problem.answer).correct:
            raise ValueError('Wide AIME committed row fails independent integer regrading')
    if recovered != entry['verified_original_recoveries'] or verified_receipts != admitted:
        raise ValueError('Wide AIME admitted row identities do not match the synthesis receipts')
    entry['independently_regraded_admitted_rows'] = len(rows)
    entry['synthetic_admission_digests'] = verified_receipts
    return entry


def _nested_execution(run, train, old_gaps, base_digest):
    root = run / 'nested_runs'
    if root.is_symlink():
        raise ValueError('Nested run audit cannot use symlinked roots')
    output = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        child = path / 'child'
        if path.is_symlink() or child.is_symlink():
            raise ValueError('Nested run audit cannot follow symlinks')
        if not child.is_dir():
            continue
        request = MetaRSILab._read_owned(path, 'request.json')
        descriptor = MetaRSILab._read_owned(path, 'descriptor.json')
        manifest = request['manifest']
        child_roles = manifest['selection']['labs']['math']['config']['split_roles']
        if set(child_roles) != {'train', 'search', 'validation'}:
            raise ValueError('Unexpected nested trial cohorts')
        version = decode(manifest['version'])
        policy = version.config['config']
        entry = {'run_id': path.name, 'parent_call_id': request['parent_call_id'],
                 'execution_phase': descriptor['phase'], 'version_id': version.version_id,
                 'policy': {k: policy[k] for k in ('order', 'profiles', 'evidence', 'steps', 'round_offset') if k in policy}}
        entry.update(_lab_artifacts(child / 'meta-rsi', train, old_gaps, base_digest))
        output.append(entry)
    return output


def _lab_artifacts(lab, train, old_gaps, base_digest):
    entry = {'development_counts': [], 'proposals': []}
    for file in sorted(lab.glob('diagnose-*.json')):
        point = MetaRSILab._read_owned(lab, file.name)
        if point['split'] not in ('validation', 'search'):
            raise ValueError('Unexpected nested diagnostic cohort')
        aime.check_metrics(point, point['candidate_id'], point['split'], point['protocol_id'], point['count'])
        entry['development_counts'].append({k: point[k] for k in ('candidate_id', 'split', 'protocol_id', 'count',
                                           'correct', 'invalid_format', 'truncated')})
    for file in sorted(lab.glob('proposal-*.json')):
        result = MetaRSILab._read_owned(lab, file.name)
        row = {**result['request'], 'candidate': result['candidate_id'], 'reason': result['reason']}
        transition = result['transition']
        if 'synthesis' in transition:
            row['synthesis'] = _synthesis_evidence(lab, row, transition, train, old_gaps, base_digest)
        if 'training' in transition:
            row['training'] = _training_summary(lab, transition['training'])
        if row['candidate'] is not None:
            row['proposed_runtime'] = _runtime_components(_read_candidate(lab, row['candidate'], base_digest))
        entry['proposals'].append(row)
    selected = []
    for file in sorted(lab.glob('selection-*.json')):
        selected.append(MetaRSILab._read_owned(lab, file.name)['candidate_id'])
    entry['released_ids'] = selected
    return entry


def _repeat_measurements(parent, nested):
    """Surface score variation for the very same candidate and dev protocol."""
    groups = {}
    for group in ({'run_id': 'domain', **parent}, *nested):
        for point in group['development_counts']:
            key = tuple(point[k] for k in ('candidate_id', 'split', 'protocol_id', 'count'))
            groups.setdefault(key, []).append({'run_id': group['run_id'], 'correct': point['correct'],
                                                'truncated': point['truncated']})
    return [dict(candidate_id=k[0], split=k[1], protocol_id=k[2], count=k[3], observations=values)
            for k, values in sorted(groups.items()) if len(values) > 1]


def _training_summary(lab, training):
    fields = ('steps', 'microsteps', 'samples', 'completion_tokens_seen', 'coverage', 'world_size',
              'effective_batch_size', 'final_batch_size', 'learning_rate', 'rank_weights_identical')
    key = training['checkpoint']
    if not aime.sha256_text(key):
        raise ValueError('Invalid wide optimizer checkpoint ID')
    root = lab / 'models' / key
    path = root / 'training.json'
    if root.is_symlink() or path.is_symlink() or file_hash(path) != training['training_receipt_sha256']:
        raise ValueError('Wide optimizer receipt hash mismatch')
    stored = json.loads(path.read_bytes())
    if 'training_seed' in training:
        fields += ('training_seed',)
    if any(stored.get(k) != training.get(k) for k in fields):
        raise ValueError('Wide optimizer receipt does not match the recorded transition')
    return {k: training[k] for k in ('checkpoint', 'training_receipt_sha256', *fields)}


def _checked_config(run, config):
    launch = json.loads(config.read_bytes())
    if Path(launch['data_dir']).resolve() != run.resolve():
        raise ValueError('Wide audit config belongs to a different run')
    method = launch['framework']['method']['config']
    parent = method['labs']['math']
    child = method['controller'].get('trial_lab')
    stage = parent['wide_loop'].get('model_stage')
    if (child is None) != (stage is not None):
        raise ValueError('Expected either an ordinary MetaRSI campaign or an isolated Model stage')
    labs = (parent, child) if child is not None else (parent,)
    trainer = launch['resources']['pools']['trainer']['gpus']
    if (trainer != [0, 1, 2, 3] or any('test' in x['split_roles'] for x in labs)
            or any(x['model_options']['seed'] != 23 or x['model_options']['training_global_batch_size'] != 16
                   for x in labs)):
        raise ValueError('Unexpected runtime placement, seed, or measurement roles')
    return method, labs, trainer


def _training_cohort(lab):
    path = Path(lab['manifest'])
    splits = json.loads(path.read_bytes())['splits']
    if set(splits) != set(lab['split_roles'].values()):
        raise ValueError('Unexpected extra splits in wide AIME manifest')
    rows, _, _ = jsonl_problems(path.parent, splits[lab['split_roles']['train']], VERIFIERS,
        budgets=False, max_output_tokens=lab['model_options']['max_new_tokens'])
    train = {p.id: p for p in rows}
    bootstrap = load_bootstrap(lab['wide_loop'])
    if len(train) != 256 or set(train) != {r['id'] for r in bootstrap['rows']} | set(bootstrap['failed_problem_ids']):
        raise ValueError('Audit training provenance differs from the admitted 256-item pool')
    return train, bootstrap


def report(run, config):
    run, config = Path(run), Path(config)
    method, labs, trainer = _checked_config(run, config)
    result = json.loads((run / 'result.json').read_bytes())
    if result.get('sealed_evaluations') or result.get('final_evaluations'):
        raise ValueError('Development-only wide loop unexpectedly emitted final evaluation')
    controller = result['controller_report']
    summary = summarize_development(result)
    summary.update(host_adoptions=host_adoptions(run, controller), host_releases=host_releases(run, controller),
                   active_controller_version=controller['active_version'],
                   run_complete=(result.get('status') == 'finished'
                                 and result.get('stop_reason') == 'researcher_finished'
                                 and controller['status'] == 'done'
                                 and len(controller['terms']) == method['controller']['campaign']['terms']),
                   controller_error=controller.get('error'))
    summary['term_routing'] = [{k: t[k] for k in ('term', 'controller_version', 'order', 'profiles', 'operators')}
                               for t in controller['terms']]
    summary.update(config_sha256=file_hash(config), bootstrap_sha256=labs[0]['wide_loop']['bootstrap_sha256'],
                   seed_adapter_sha256=labs[0]['wide_loop']['checkpoint_sha256'],
                   physical_gpus=trainer, global_batch_size=16, optimizer_evidence=optimizer_evidence(result, run))
    train, bootstrap = _training_cohort(labs[0])
    base = bootstrap['source']['base_sha256']
    lab_root = run / 'meta-rsi'
    staged = labs[0]['wide_loop'].get('model_stage') is not None
    if controller['initial'] is not None:
        initial = _read_candidate(lab_root, controller['initial'], base)
        if (initial['data'][:len(bootstrap['rows'])] != bootstrap['rows']
                or (not staged and len(initial['data']) != len(bootstrap['rows']))
                or initial['model']['checkpoint_sha256'] != labs[0]['wide_loop']['checkpoint_sha256']):
            raise ValueError('Initial system does not match the inherited training and adapter')
        if staged:
            metadata = MetaRSILab._read_owned(lab_root, 'model-stage.json')
            if not isinstance(metadata, dict) or controller['initial'] != controller['selected']:
                raise ValueError('Isolated Model stage promoted a candidate or lost its source attestation')
            summary.update(staged_model_input=metadata, explicitly_non_promoting=True)
        summary['initial_runtime'] = _runtime_components(initial)
    if controller['selected'] is not None:
        deployed = _read_candidate(lab_root, controller['selected'], base)
        summary['incumbent_runtime'] = _runtime_components(deployed)
    evidence = []
    for row in controller['attempts']:
        transition = row.get('transition') or {}
        synthesis = transition.get('synthesis')
        training = transition.get('training')
        entry = {k: row.get(k) for k in ('term', 'round', 'operator', 'profile', 'parent', 'candidate', 'selected',
            'reason', 'search_score', 'validation_score', 'gain')}
        if row['candidate'] is not None:
            before = _read_candidate(lab_root, row['parent'], base)
            after = _read_candidate(lab_root, row['candidate'], base)
            if [slot for slot in before if before[slot] != after[slot]] != [row['operator']]:
                raise ValueError('Wide AIME proposal changed more than the declared component')
            entry['proposed_runtime'] = _runtime_components(after)
            if staged and 'gain' in row:
                entry['selection'] = 'operator_only_measurement_no_promotion'
            else:
                entry['selection'] = ('selected' if row['selected'] else 'unmeasured' if 'gain' not in row else
                                      'validation_regressed' if row['gain'] < 0 else 'no_gain_above_margin')
        if synthesis is not None:
            entry['synthesis'] = _synthesis_evidence(lab_root, row, transition, train,
                bootstrap['failed_problem_ids'], base)
        if training is not None:
            entry['training'] = _training_summary(lab_root, training)
        evidence.append(entry)
    summary['observer_evidence'] = evidence
    summary['nested_execution'] = _nested_execution(run, train, bootstrap['failed_problem_ids'], base)
    summary['repeated_development_measurements'] = _repeat_measurements(
        _lab_artifacts(lab_root, train, bootstrap['failed_problem_ids'], base), summary['nested_execution'])
    return summary


def _committed_policy_progress(run):
    recovery = run / 'recovery'
    checkpoint = recovery / 'checkpoint.json'
    if recovery.is_symlink() or checkpoint.is_symlink():
        raise ValueError('Wide audit cannot use a symlinked recovery checkpoint')
    if not checkpoint.is_file():
        return None
    snapshot = DurableRecovery.read(run)
    policy = snapshot['components']['decision']['state']['active']['policy']
    term = policy['term']
    return {'generation': snapshot['generation'], 'phase': policy['phase'],
            'active_version': policy['active_version'], 'incumbent': policy['current'],
            'term': term['index'] if term else None, 'stage': term['stage'] if term else None,
            'available_operators': list(policy['available']),
            'meta_decisions': [{k: item.get(k) for k in ('term', 'stage', 'target', 'decision', 'reason', 'version_id')}
                               for item in policy['meta']]}


def report_progress(run, config):
    """Read immutable evidence while the running campaign continues to write."""
    run, config = Path(run), Path(config)
    _, labs, trainer = _checked_config(run, config)
    train, bootstrap = _training_cohort(labs[0])
    old_gaps, base = bootstrap['failed_problem_ids'], bootstrap['source']['base_sha256']
    source = run / 'experience.sqlite3'
    if source.is_symlink() or not source.is_file():
        raise ValueError('Missing owned Host journal')
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        recent_operations = [dict(operation=op, status=status) for op, status in connection.execute(
            "SELECT json_extract(payload, '$.details.operation'), json_extract(payload, '$.details.status') "
            "FROM experience WHERE kind='runtime_feedback' ORDER BY seq DESC LIMIT 12")][::-1]
    parent = _lab_artifacts(run / 'meta-rsi', train, old_gaps, base)
    nested = _nested_execution(run, train, old_gaps, base)
    return {'development_only': True, 'seed': 23, 'physical_gpus': trainer, 'global_batch_size': 16,
            'config_sha256': file_hash(config), 'result_present': (run / 'result.json').is_file(),
            'recent_operations': recent_operations, 'committed_controller': _committed_policy_progress(run),
            'parent': parent, 'nested_execution': nested,
            'repeated_development_measurements': _repeat_measurements(parent, nested)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--live', action='store_true', help='Audit aggregate committed progress before the terminal result')
    args = parser.parse_args()
    export = report_progress if args.live else report
    print(json.dumps(export(args.run, args.config), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
