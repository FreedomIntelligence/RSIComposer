"""Host-only post-selection evaluation, with matched, resumable task tranches.

No meta-agent receives these results. The original host run must have completed its primary heldout receipts.
This runner never changes a candidate or chooses a new one from heldout scores.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
import time

from rsisys.api import Application, ExecutionStatus
from .lab import CodingSearchLab, atomic_json, digest


def aggregate(root, selected):
    paired = []
    for chunk in sorted(root.glob('chunk-*')):
        summaries = {}
        for ident in dict.fromkeys(('baseline', selected)):
            p = chunk / 'evaluations' / ident / 'heldout/summary.json'
            if p.exists():
                summaries[ident] = json.loads(p.read_text())
        if set(summaries) != set(('baseline', selected)):
            continue
        left = summaries['baseline']['tasks']
        right = summaries[selected]['tasks']
        if [(r['benchmark'], r['task_id']) for r in left] != [(r['benchmark'], r['task_id']) for r in right]:
            raise ValueError('Paired task identities differ')
        paired.extend({'benchmark': a['benchmark'], 'task': a['task_id'],
                       'baseline': a['score'], 'selected': b['score'],
                       'baseline_receipt': a['official_result'], 'selected_receipt': b['official_result']}
                      for a, b in zip(left, right))
    families = {}
    for family in sorted({r['benchmark'] for r in paired}):
        rows = [r for r in paired if r['benchmark'] == family]
        before = sum(r['baseline'] for r in rows) / len(rows)
        after = sum(r['selected'] for r in rows) / len(rows)
        families[family] = {'samples': len(rows), 'baseline': before, 'selected': after,
                            'delta': after - before,
                            'improved_tasks': [r['task'] for r in rows if r['selected'] > r['baseline']],
                            'regressed_tasks': [r['task'] for r in rows if r['selected'] < r['baseline']]}
    result = {'protocol': 'post-selection-matched-tranches', 'selected_candidate': selected,
              'completed_paired_tasks': len(paired), 'by_benchmark': families, 'pairs': paired,
              'selection_updated': False, 'updated_at': time.time(),
              'scope': 'Only fully completed preregistered tranches are aggregated; no full leaderboard claim.'}
    atomic_json(root / 'summary.json', result)
    return result


def primary_selection(app):
    """Require successful host receipts, rather than a method-private finalize file."""
    lab = app.assembly.lab_execution.labs['coding']
    if not isinstance(lab, CodingSearchLab) or lab.selected is None:
        raise ValueError('A coding_meta_agent selection is required')
    required = set(('baseline', lab.selected))
    completed = set()
    for row in app.experience.retrieve(kinds=('evaluation',)):
        details = row.details
        call, receipt = details.get('operation', {}), details.get('receipt', {})
        args = call.get('arguments', {})
        if (details.get('phase') == 'host_finish' and call.get('name') == 'coding.evaluate'
                and args.get('split') == 'heldout' and args.get('indices') is None
                and receipt.get('status') == ExecutionStatus.COMPLETED
                and receipt.get('data', {}).get('scored') is True):
            completed.add(args.get('candidate_id'))
    if not required <= completed:
        raise ValueError('Wait for completed host primary heldout evaluation')
    selection = json.loads((lab.root / 'selection.json').read_text())
    if selection['candidate_id'] != lab.selected:
        raise ValueError('Persisted selection differs from restored host state')
    return lab.config, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    with Application.resume(args.run_dir) as app:
        config, selection = primary_selection(app)
    source = Path(config['experiment_dir'])
    selected = selection['candidate_id']
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'selection_sha256': digest(selection), 'selected_candidate': selected,
                  'plan_sha256': digest(plan)}
    path = args.output / 'binding.json'
    if path.exists() and json.loads(path.read_text()) != provenance:
        raise ValueError('Post-selection binding changed')
    atomic_json(path, provenance)
    for index, heldout in enumerate(plan['tranches']):
        # Reserve time for the currently admitted tranche, without preempting it.
        if time.time() > args.deadline - 2700:
            break
        if shutil.disk_usage(args.output).free < 80 * 1024**3:
            raise RuntimeError('Less than 80 GiB free; stop new task admission')
        root = args.output / f'chunk-{index:03d}'
        root.mkdir(exist_ok=True)
        manifest_path = root / 'manifest.json'
        manifest = {'search': plan['search'], 'heldout': heldout, 'source': plan['source']}
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Frozen tranche changed')
        atomic_json(manifest_path, manifest)
        lab_config = dict(config, experiment_dir=str(root.resolve()), manifest=str(manifest_path.resolve()))
        lab = CodingSearchLab(SimpleNamespace(artifacts=None), lab_config)
        with lab._lock():
            for ident in dict.fromkeys(('baseline', selected)):
                src = source / 'candidates' / (ident + '.json')
                dest = lab.candidates / src.name
                if dest.exists() and dest.read_bytes() != src.read_bytes():
                    raise ValueError('Frozen candidate changed')
                if not dest.exists():
                    shutil.copyfile(src, dest)
            atomic_json(root / 'selection.json', {'candidate_id': selected, 'parent_selection': provenance,
                                                  'sealed_before_evaluation': True})
            for ident in dict.fromkeys(('baseline', selected)):
                candidate, candidate_path = lab._candidate(ident)
                lab._measure(candidate, candidate_path, 'heldout', None)
        result = aggregate(args.output, selected)
        print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}), flush=True)
    result = aggregate(args.output, selected)
    atomic_json(args.output / 'stop.json', {'at': time.time(), 'completed_paired_tasks': result['completed_paired_tasks']})


if __name__ == '__main__':
    main()
