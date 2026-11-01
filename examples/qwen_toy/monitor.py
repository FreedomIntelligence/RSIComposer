"""Public monitoring summaries; never copy model prompts or training examples."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

from rsisys.runtime.files import atomic_write, json_bytes


ACTIVE = {'proposed', 'submitted', 'accepted', 'running', 'waiting', 'cancel_requested'}
MODE_NAMES = {'train': '训练', 'sft': '训练', 'evaluate': '评测',
              'apply_candidate': '应用候选', 'execute': 'Actor 执行'}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def decision_summary(details, *, ident, round_number, at=None, source='model'):
    """Accept structured policy/history output and copy only public decision fields."""
    directive = details.get('directive') or {}
    review = details.get('review')
    execution = details.get('execution') or {}
    progress = execution.get('progress') or {}
    decision_model = details.get('decision_model') or {}
    actor_model = details.get('execution_model') or execution.get('model') or {}
    row = {'decision_id': ident, 'timestamp': at, 'source': source, 'round': round_number,
           'decision_revision': decision_model.get('revision'), 'actor_revision': actor_model.get('revision')}
    if review is not None:
        row.update(kind='review', action=review['action'], reason=str(review.get('reason', ''))[:2000],
                   directive_id=progress.get('directive_id'), mode=(progress.get('directive') or {}).get('mode'),
                   job_id=(progress.get('job') or {}).get('job_id'), status='reviewed',
                   next_review_seconds=review.get('next_review_seconds'))
    elif directive:
        payload = directive.get('payload') or {}
        row.update(kind='directive', directive_id=directive['directive_id'], mode=directive['mode'],
                   strategy=str(directive.get('strategy', ''))[:2000], status='proposed',
                   review_after_seconds=directive.get('review_after_seconds'))
        if isinstance(payload, dict):
            for key in ('recipe_ref', 'protocol_ref', 'method', 'candidate_id'):
                if isinstance(payload.get(key), str):
                    row[key] = payload[key][:1000]
            if isinstance(payload.get('candidate'), dict):
                row['candidate_id'] = payload['candidate'].get('candidate_id')
    else:
        row.update(kind='finish', reason=str(details.get('reason', ''))[:2000], status='completed')
    return row


class Monitor:
    def __init__(self, root, base, objective, *, rounds=1):
        self.root, self.jobs, self.candidates = root, {}, {}
        self.phase, self.started = 'starting', time.monotonic()
        self.data = {'schema_version': 1, 'run_id': root.name, 'objective': objective,
                     'status': 'running', 'loops': {}, 'decisions': [], 'current': {},
                     'workflow': {'phase': 'starting', 'round': 1, 'rounds': rounds, 'completed_directives': 0},
                     'models': {role: {'model_id': base.model_id, 'revision': base.revision}
                                for role in ('decision_maker', 'actor')}, 'events': []}
        self.write()

    def write(self):
        self.data['workflow']['elapsed_seconds'] = round(time.monotonic() - self.started, 3)
        self.data.update(updated_at=timestamp(), jobs=list(self.jobs.values()),
                         candidates=list(self.candidates.values()))
        atomic_write(self.root / 'monitor.json', json_bytes(self.data))

    def event(self, event, **details):
        current = self.data['current']
        if event == 'job':
            row = {**self.jobs.get(details['job_id'], {}), **deepcopy(details)}
            if current.get('status') in ACTIVE and current.get('mode') == details.get('kind'):
                row.update(directive_id=current['directive_id'], round=self.data['workflow']['round'])
                current['job_id'] = row['job_id']
                for key in ('step', 'total_steps', 'elapsed_seconds', 'metrics'):
                    if key in row:
                        current[key] = deepcopy(row[key])
                # A successful training job still awaits feedback/possible activation.
                current['phase'] = 'collecting_result' if row.get('status') == 'succeeded' else row.get('status')
            self.jobs[details['job_id']] = row
        if event in ('model_decision_requested', 'training_review_requested'):
            self.data['loops']['decision_making'] = {
                'status': 'running', 'summary': '模型正在复查当前任务' if event == 'training_review_requested'
                else f'模型正在决定下一步：{MODE_NAMES.get(self.phase, self.phase)}'}
        summaries = {'model_decision_requested': '请求模型决策', 'model_decision_validated': '模型决策通过校验',
                     'model_decision_rejected': '模型输出未通过校验，等待重试',
                     'training_review_requested': '模型开始复查训练', 'training_review_completed': '模型完成训练复查',
                     'training_review_rejected': '复查输出未通过校验', 'activation_verified': '候选推理验证通过，准备绑定',
                     'job': '后端任务状态更新', 'progress': '执行进度更新', 'phase': '运行阶段更新'}
        self.data['events'].append({'timestamp': timestamp(), 'kind': event,
                                   'summary': summaries.get(event, event), 'details': details})
        self.data['events'] = self.data['events'][-300:]
        self.write()

    def decision(self, decision, context, round_number):
        details = {**asdict(decision), 'decision_model': asdict(context.decision_model) if context.decision_model else {},
                   'execution_model': asdict(context.execution.model) if context.execution.model else {}}
        ident = decision.directive.directive_id if decision.directive else str(uuid4())
        row = decision_summary(details, ident=ident, round_number=round_number, at=timestamp())
        self.data['decisions'].append(row)
        self.data['decisions'] = self.data['decisions'][-300:]
        if row['kind'] == 'directive':
            self.data['current'] = {**deepcopy(row), 'phase': 'accepted'}
            self.data['loops']['decision_making'] = {'status': 'waiting', 'summary': '指令已选定，等待探索侧反馈'}
            self.data['loops']['exploration'] = {'status': 'running', 'summary': MODE_NAMES.get(row['mode'], row['mode'])}
        self.write()

    def review(self, reviewed, context, round_number):
        row = decision_summary({'review': asdict(reviewed), 'execution': asdict(context.execution),
                                'decision_model': asdict(context.decision_model) if context.decision_model else {}},
                               ident=str(uuid4()), round_number=round_number, at=timestamp())
        row['status'] = 'proposed'  # Runner applies the control after the harness records it.
        self.data['decisions'].append(row)
        self.data['decisions'] = self.data['decisions'][-300:]
        self.write()

    def advance(self, result):
        event, current = result.event, self.data['current']
        if result.directive_id and current.get('directive_id') == result.directive_id:
            progress = event.get('progress') or {}
            feedback = event.get('feedback') or {}
            current['status'] = feedback.get('status', 'running')
            current['phase'] = progress.get('phase', current['status'])
            if progress.get('elapsed_seconds') is not None:
                current['elapsed_seconds'] = progress['elapsed_seconds']
            job_id = result.job.job_id if result.job else None
            if not job_id and feedback.get('jobs'):
                job_id = feedback['jobs'][-1]['job_id']
            if job_id:
                current['job_id'] = job_id
            if feedback:
                current['metrics'] = {**feedback.get('training_metrics', {}),
                                      **feedback.get('quality_metrics', {}), **feedback.get('execution_metrics', {})}
            for row in self.data['decisions']:
                if row.get('directive_id') == result.directive_id and row['kind'] == 'directive':
                    row.update(status=current['status'], job_id=current.get('job_id'))
            if event.get('review'):
                review = event['review']
                current.update(review_status=review['action'], next_review_seconds=review['next_review_seconds'])
                for row in reversed(self.data['decisions']):
                    if row['kind'] == 'review' and row.get('directive_id') == result.directive_id:
                        row['status'] = 'cancel_requested' if review['action'] == 'cancel' else 'reviewed'
                        break
            if result.kind == 'completed':
                self.data['workflow']['completed_directives'] += 1
        if result.kind == 'waiting':
            self.data['loops']['decision_making'] = {'status': 'waiting', 'summary': '等待探索侧反馈；到期时复查'}
            self.data['loops']['exploration'] = {'status': 'running',
                'summary': MODE_NAMES.get(current.get('mode'), '执行指令')}
        self.write()

    def stage(self, phase, *, status='running'):
        self.phase = phase
        self.data['status'] = status
        self.data['workflow']['phase'] = phase
        active = self.data['current'].get('status') in ACTIVE
        self.data['loops'] = {
            'decision_making': {'status': 'waiting' if active and status == 'running' else status,
                                'summary': '等待指令反馈' if active and status == 'running' else f'当前阶段：{phase}'},
            'exploration': {'status': ('running' if active else 'idle') if status == 'running' else status,
                            'summary': MODE_NAMES.get(self.data['current'].get('mode'), '等待下一条指令')},
        }
        self.event('phase', status=status)


def enrich_completed_run(root):
    """Explicit offline export of existing evidence, never called by the web server."""
    root = Path(root)
    result = json.loads((root / 'result.json').read_text())
    if result.get('status') not in ('completed', 'failed', 'timed_out'):
        raise ValueError('Only a finished controller run may be exported')
    snapshot = json.loads((root / 'monitor.json').read_text())
    history = json.loads((root / 'decision_memory.json').read_text())['history']
    events = snapshot.get('events', [])
    validated = iter(e for e in events if e['kind'] == 'model_decision_validated')
    reviews = {e.get('details', {}).get('directive_id'): e for e in events
               if e['kind'] == 'training_review_completed'}
    rows, round_number = [], 0
    for record in history:
        details = record['details']
        if record['kind'] == 'decision':
            if (details.get('directive') or {}).get('mode') == 'train':
                round_number += 1
            if details.get('event') == 'review':
                event = reviews.get(record.get('task_id'), {})
            else:
                event = next(validated, {})
            row = decision_summary(details, ident=record['record_id'], round_number=round_number,
                                   at=event.get('timestamp'), source='history')
            rows.append(row)
        elif record['kind'] == 'feedback':
            feedback = details['feedback']
            for row in reversed(rows):
                if row['kind'] == 'directive' and row.get('directive_id') == feedback['directive_id']:
                    row['status'] = feedback['status']
                    row['job_id'] = (feedback.get('jobs') or [{}])[-1].get('job_id')
                    break
    snapshot['decisions'] = rows
    snapshot['workflow'] = {'phase': result['phase'], 'round': result.get('rounds_completed'),
                            'rounds': result.get('rounds_requested'),
                            'completed_directives': result.get('completed_directives'),
                            'elapsed_seconds': result.get('duration_seconds')}
    snapshot['round_results'] = result.get('round_results', [])
    snapshot['current'] = next((deepcopy(r) for r in reversed(rows) if r['kind'] == 'directive'), {})
    snapshot['current']['phase'] = snapshot['current'].get('status')
    # Preserve the run's last observation timestamp; this is not a new live update.
    atomic_write(root / 'monitor.json', json_bytes(snapshot))
    return snapshot


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export public decision summaries from a completed toy run')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    enriched = enrich_completed_run(args.run_dir)
    print(f"Exported {len(enriched['decisions'])} recorded decisions for {args.run_dir.name}")
