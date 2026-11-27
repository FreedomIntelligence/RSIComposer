"""Read aggregate AIME results or development-only run diagnostics."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from .aime import candidate_id
from .engine import digest, read_checkpoint
from .evidence import comparable


def _meta_comparison(item):
    trials = item.get('trials', [])
    row = {key: item[key] for key in ('term', 'stage', 'target', 'decision')}
    row['reason'] = item.get('reason')
    row['surface'] = (item.get('policy_diff') or {}).get('surface')
    row['arms'] = [{**{key: t[key] for key in ('start_id', 'baseline', 'selected_score', 'delta',
                    'count', 'attempts', 'allocated_budget', 'parent_observed_wall_seconds') if key in t},
                     'usable': 'error' not in t} for t in trials]
    matched = len(trials) == 2 and all(x['usable'] for x in row['arms']) and comparable(*trials)
    row['matched'] = matched
    row['challenger_advantage_pp'] = 100 * (trials[1]['delta'] - trials[0]['delta']) if matched else None
    return row


def _task_change(row):
    op = row['operator']
    transition = row.get('transition') or {}
    return {**{key: row[key] for key in ('term', 'round', 'operator', 'profile', 'parent', 'candidate',
             'data_count', 'training_performed', 'selected')},
            'validation_gain_pp': 100 * row['gain'] if 'gain' in row else None,
            'changed_slot': {'data': 'Experience', 'harness': 'System Prompt', 'model': 'LoRA adapter'}[op]
            if row.get('candidate') and row['candidate'] != row['parent'] else None,
            'edges': transition.get('edges', []), 'verified_added': transition.get('verified_added', 0),
            'rollout': transition.get('rollout')}


def _controller_evidence(report):
    return {'seed': report['campaign']['policy_seed'], 'track': report['campaign']['track'],
            'simulated': report['simulated'],
            'terms': [{k: t[k] for k in ('term', 'start_id', 'released_id', 'start_score', 'released_score',
                'operator_budget', 'operator_calls', 'trained_candidates', 'term_elapsed_seconds')}
                for t in report['terms']],
            'adopted': [{k: item[k] for k in ('term', 'stage', 'target', 'change')}
                        for item in report['meta'] if item['decision'] == 'adopted'],
            'task_changes': [_task_change(item) for item in report['attempts']],
            'meta_comparisons': [_meta_comparison(item) for item in report['meta']],
            'meta_trial_wall_seconds': report['meta_trial_wall_seconds']}


def summarize_development(run):
    """Summarize measured development evidence even if no campaign sealed successfully."""
    report = run.get('controller_report', {})
    if report.get('campaign', {}).get('protocol') != 'aime':
        raise ValueError('AIME development summary requires a campaign report')
    initial = report.get('initial')
    diagnosis = report['diagnoses'].get(initial)
    return {**_controller_evidence(report), 'development_only': True,
            'status': run.get('status'), 'stop_reason': run.get('stop_reason'),
            'controller_status': report['status'], 'initial_id': initial,
            'incumbent_id': report.get('selected'),
            'initial_validation': ({k: diagnosis[k] for k in
                ('protocol_id', 'count', 'accuracy', 'invalid_format', 'truncated') if k in diagnosis}
                if diagnosis is not None else None)}


def summarize(run):
    report = run.get('controller_report', {})
    if (run.get('status') != 'finished' or run.get('stop_reason') != 'researcher_finished'
            or report.get('status') != 'done' or report.get('campaign', {}).get('protocol') != 'aime'):
        raise ValueError('AIME summary requires a finished campaign')
    sealed = {}
    for event in run.get('events', []):
        if event.get('phase') != 'host_finish':
            continue
        operation, receipt = event.get('operation', {}), event.get('receipt', {})
        args, data = operation.get('arguments', {}), receipt.get('data', {})
        if operation.get('name') != 'math.evaluate' or args.get('split') != 'test':
            continue
        if receipt.get('status') != 'completed' or data.get('scored') is not True or type(args.get('baseline')) is not bool:
            raise ValueError('AIME Host sealing did not complete successfully')
        label = 'baseline' if args['baseline'] else 'selected'
        if label in sealed or args['candidate'] != report['initial' if args['baseline'] else 'selected']:
            raise ValueError('Duplicate or misattributed sealed result')
        q, execution = data['quality_metrics'], data['execution_metrics']
        cohorts = {name: {k: q[f'{name}_{k}'] for k in ('correct', 'count', 'accuracy')}
                   for key in q if key.endswith('_count') for name in (key[:-6],)}
        count, correct = data['count'], q['correct']
        if (count <= 0 or not 0 <= correct <= count or sum(x['count'] for x in cohorts.values()) != count
                or sum(x['correct'] for x in cohorts.values()) != correct or q['accuracy'] != correct / count):
            raise ValueError('Malformed AIME pooled sealed result')
        sealed[label] = {'candidate_id': args['candidate'],
                         'count': count, 'correct': correct, 'accuracy': q['accuracy'], 'exams': cohorts,
                         'invalid_format': execution['invalid_format'], 'truncated': execution['truncated']}
    if set(sealed) != {'baseline', 'selected'}:
        raise ValueError('AIME summary expects both Host-only sealed scores')
    base, final = sealed['baseline'], sealed['selected']
    if (base['count'] != final['count']
            or {k: x['count'] for k, x in base['exams'].items()} != {k: x['count'] for k, x in final['exams'].items()}):
        raise ValueError('Sealed cohort counts changed during this run')
    terms = report['terms']
    if (not terms or terms[0]['start_id'] != base['candidate_id'] or terms[-1]['released_id'] != final['candidate_id']
            or any(a['released_id'] != b['start_id'] for a, b in zip(terms, terms[1:]))):
        raise ValueError('AIME release chain does not match the sealed candidates')
    diagnoses = report['diagnoses']
    protocols = {diagnoses[t['start_id']]['protocol_id'] for t in terms}
    if len(protocols) != 1:
        raise ValueError('AIME development protocol changed across terms')
    return {**_controller_evidence(report), 'validation_protocol': protocols.pop(), 'sealed': sealed,
            'sealed_gain_pp': 100 * (final['accuracy'] - base['accuracy'])}


def require_full_aime(summary):
    """Reject any pilot, simulated, or incomplete four-exam aggregate."""
    expected = {f'aime-{year}-{exam}' for year in (2025, 2026) for exam in ('i', 'ii')}
    sealed = summary.get('sealed', {})
    if summary.get('simulated') is not False or set(sealed) != {'baseline', 'selected'}:
        raise ValueError('Full AIME needs two real Host sealed receipts')
    for row in sealed.values():
        exams = row['exams']
        if row['count'] != 60 or set(exams) != expected or any(x['count'] != 15 for x in exams.values()):
            raise ValueError('Full AIME must cover each of the four 15-question 2025/2026 exams')
        if (any(type(x['correct']) is not int or not 0 <= x['correct'] <= 15
                or x['accuracy'] != x['correct'] / 15 for x in exams.values())
                or any(type(row[k]) is not int or row[k] < 0 for k in ('correct', 'invalid_format', 'truncated'))
                or row['correct'] + row['invalid_format'] + row['truncated'] > 60):
            raise ValueError('Full AIME includes inconsistent exam or grading counts')


def optimizer_evidence(run, data_dir):
    """Verify parent-run LoRA checkpoints and report numeric optimizer receipts only."""
    report = run['controller_report']
    if report.get('simulated') is not False:
        return []
    root = Path(data_dir) / 'meta-rsi'
    if root.is_symlink():
        raise ValueError('AIME checkpoint evidence cannot use a symlinked Lab root')
    output = []
    for row in report['attempts']:
        if not row['training_performed']:
            continue
        transition = row.get('transition') or {}
        t = transition.get('training') or {}
        key = t.get('checkpoint')
        if (row['operator'] != 'model' or not candidate_id(row['candidate'])
                or t.get('label_scope') != 'assistant_only'):
            raise ValueError('AIME training receipt lacks an assistant-only model transition')
        payload = read_checkpoint(root / 'models', key, simulated=False)
        recipe = payload['recipe']
        path = root / f"{row['candidate']}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError('AIME model candidate must be an owned regular file')
        candidate = json.loads(path.read_bytes())
        if candidate.get('model') != {'checkpoint': key, 'checkpoint_sha256': digest(payload)}:
            raise ValueError('AIME model candidate does not match its checkpoint')
        if (payload['samples'] != t['cumulative_count'] or recipe.get('round') != row['round']
                or recipe.get('base_sha256') != t['base_sha256'] or recipe.get('label_scope') != t['label_scope']
                or recipe.get('refresh_sha256') != digest(transition.get('refresh_proof'))):
            raise ValueError('AIME checkpoint and training receipt do not agree')
        stats = json.loads((root / 'models' / key / 'training.json').read_bytes())
        counts = ('steps', 'microsteps', 'samples', 'completion_tokens_seen')
        reals = ('last_loss', 'token_weighted_loss', 'elapsed_seconds')
        if (any(type(stats.get(k)) is not int or stats[k] < 1 for k in counts)
                or any(type(stats.get(k)) not in (float, int) or not math.isfinite(stats[k]) or stats[k] < 0
                       for k in reals) or stats['microsteps'] < stats['steps']
                or any(stats[k] != payload[k] for k in ('steps', 'samples'))):
            raise ValueError('Invalid AIME optimizer statistics')
        output.append({**{k: row[k] for k in ('term', 'round', 'candidate', 'selected')},
                       'checkpoint': key, 'label_scope': 'assistant_only', **{k: stats[k] for k in (*counts, *reals)}})
    return output


def compare(adaptive, frozen):
    def index(rows, label):
        result = {r['seed']: r for r in rows}
        if len(result) != len(rows) or any(r['track'] != label or len(r['terms']) < 2 for r in rows):
            raise ValueError(f'Expected unique seeds from complete {label} campaigns with at least two terms')
        return result
    a, b = index(adaptive, 'adaptive'), index(frozen, 'frozen')
    if not a or set(a) != set(b):
        raise ValueError('Adaptive and frozen tracks require identical outer seeds')
    runs = []
    for seed in sorted(a):
        left, right = a[seed], b[seed]
        lb, rb = left['sealed']['baseline'], right['sealed']['baseline']
        if (left['simulated'] != right['simulated'] or lb != rb
                or left['validation_protocol'] != right['validation_protocol']
                or [t['operator_budget'] for t in left['terms']] != [t['operator_budget'] for t in right['terms']]
                or left['terms'][0]['released_id'] != right['terms'][0]['released_id']
                or left['terms'][1]['start_score'] != right['terms'][1]['start_score']):
            raise ValueError(f'Seed {seed} did not share a measured S1 and a sealed baseline')
        runs.append({'seed': seed, 's1': left['terms'][0]['released_id'],
                     'term2_dev_advantage_pp': 100 * (left['terms'][1]['released_score'] - right['terms'][1]['released_score']),
                     'final_sealed_advantage_pp': left['sealed_gain_pp'] - right['sealed_gain_pp']})
    return {'matched_seeds': runs, 'mean_term2_dev_advantage_pp': mean(x['term2_dev_advantage_pp'] for x in runs),
            'mean_final_sealed_advantage_pp': mean(x['final_sealed_advantage_pp'] for x in runs)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', nargs='+', type=Path, help='Run result files')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--frozen', nargs='+', type=Path, help='Matching frozen runs for the same-S1 comparison')
    mode.add_argument('--dev-only', action='store_true', help='Omit sealed scores; accept unfinished campaigns')
    p.add_argument('--require-full-aime', action='store_true',
                   help='Require real 60-question receipts with 15 from each 2025/2026 exam')
    p.add_argument('--optimizer-evidence', action='store_true',
                   help='Verify parent-run checkpoints and include numeric optimizer receipts beside result.json')
    args = p.parse_args(argv)
    if args.optimizer_evidence and args.frozen is not None:
        p.error('--optimizer-evidence cannot be combined with --frozen')
    if args.require_full_aime and args.dev_only:
        p.error('--require-full-aime needs sealed results, not --dev-only')
    read_run = summarize_development if args.dev_only else summarize
    try:
        def read_path(path):
            run = json.loads(path.read_text(encoding='utf-8'))
            row = read_run(run)
            if args.require_full_aime:
                require_full_aime(row)
            if args.optimizer_evidence:
                row['optimizer_evidence'] = optimizer_evidence(run, path.parent)
            return row

        rows = [read_path(path) for path in args.results]
        output = rows if args.frozen is None else compare(rows, [read_path(path) for path in args.frozen])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
