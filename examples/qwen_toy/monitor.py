"""Public summaries of submitted changes; exclude hidden reasoning and training examples."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.provenance import RunSource

if __package__:
    from .worker import metric_history, public_metrics
else:
    from worker import metric_history, public_metrics


ACTIVE = {'proposed', 'submitted', 'accepted', 'running', 'waiting', 'cancel_requested'}
MODE_NAMES = {'train': '训练', 'sft': '训练', 'evaluate': '评测',
              'apply_candidate': '应用候选', 'execute': 'Actor 执行'}
TEXT_LIMIT = 4000
RECIPE_FIELDS = ('steps', 'lr', 'seed', 'weight_decay', 'warmup_ratio')


def public_text(value):
    return value[:TEXT_LIMIT] if isinstance(value, str) else ''


def planned_changes(details):
    """Submitted intent only. Never infer a previous value or an applied change."""
    directive = details.get('directive') or {}
    mode, payload = directive.get('mode'), directive.get('payload') or {}
    changes = []

    def add(field, value):
        if isinstance(value, str):
            changes.append({'field': field, 'after': public_text(value)})
        elif type(value) in (bool, int, float) and (type(value) is not float or math.isfinite(value)):
            changes.append({'field': field, 'after': json.dumps(value)})

    if isinstance(payload, dict):
        fields = {'configure_prompt': ('instructions',),
                  'train': ('recipe_ref', 'method', 'apply_result'),
                  'sft': ('recipe_ref', 'method', 'apply_result'),
                  'rl': ('recipe_ref', 'method', 'apply_result'),
                  'evaluate': ('protocol_ref', 'candidate_id'),
                  'apply_candidate': ('candidate_id',)}.get(mode, ())
        for field in fields:
            add(field, payload.get(field))
        candidate = payload.get('candidate')
        if mode == 'evaluate' and isinstance(candidate, dict):
            if not any(c['field'] == 'candidate_id' for c in changes):
                add('candidate_id', candidate.get('candidate_id'))
            add('base_version', candidate.get('base_version'))
            components = candidate.get('changes')
            if isinstance(components, dict):
                add('components', ', '.join(sorted(k for k in components if isinstance(k, str))))
    operation = details.get('operation') or {}
    if details.get('kind') == 'invoke' and operation.get('name') == 'define_recipe':
        arguments = operation.get('arguments') or {}
        if isinstance(arguments, dict):
            for field in RECIPE_FIELDS:
                value = arguments.get(field)
                if type(value) in (int, float) and math.isfinite(value):
                    add(field, value)
    return changes[:20]


def result_summary(row, receipt, *, feedback=False):
    """Preserve observed text, including proposal/failed/cancelled semantics."""
    if isinstance(receipt.get('status'), str):
        row['status'] = receipt['status']
    for source, target in (('summary', 'feedback_summary'), ('error', 'error')):
        if isinstance(receipt.get(source), str):
            row[target] = public_text(receipt[source])
    if feedback:
        row['metrics'] = feedback_metrics(receipt)
        jobs = receipt.get('jobs') or []
        if jobs and isinstance(jobs[-1].get('job_id'), str):
            row['job_id'] = jobs[-1]['job_id']


def record_lineage(row, receipt, origins):
    """Project confirmed executor receipts, never requested or inferred adoption."""
    if receipt.get('status') != 'completed' or receipt.get('error'):
        return
    adopted = receipt.get('adopted_candidate_ids')
    if isinstance(adopted, (list, tuple)) and adopted:
        if all(isinstance(ident, str) and ident for ident in adopted):
            row['adopted_candidate_ids'] = list(dict.fromkeys(adopted))
            row['adopted_decision_ids'] = list(dict.fromkeys(origins[ident] for ident in adopted
                if ident in origins and origins[ident] != row.get('decision_id')))
    # Evaluation returns the same candidate; it is not its producer.
    if row.get('mode') not in ('evaluate', 'apply_candidate'):
        for candidate in receipt.get('candidates') or ():
            ident = candidate.get('candidate_id') if isinstance(candidate, dict) else None
            if (isinstance(ident, str) and ident and row.get('decision_id')
                    and ident not in (adopted if isinstance(adopted, (list, tuple)) else ())):
                origins.setdefault(ident, row['decision_id'])


def enrich_result(row, receipt, *, feedback=False):
    """Add recorded evidence without erasing earlier text/metrics with empty fields."""
    observed = {}
    result_summary(observed, receipt, feedback=feedback)
    for key, value in observed.items():
        if key == 'metrics':
            row[key] = {**row.get(key, {}), **value}
        elif value not in ('', None):
            row[key] = value


def operation_target(rows, event, job_id=None):
    """Core receipts have no call ID: only a unique pending call is attributable."""
    pending = [r for r in rows if r.get('kind') == 'operation' and r.get('status') in ACTIVE]
    ident = event.get('decision_id')
    if ident is not None:
        pending = [r for r in pending if r.get('decision_id') == ident]
    elif job_id is not None:
        pending = [r for r in pending if r.get('job_id') == job_id]
    return pending[0] if len(pending) == 1 else None


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def feedback_metrics(feedback):
    return {key: value for group in ('training_metrics', 'quality_metrics', 'execution_metrics')
            for key, value in public_metrics(feedback.get(group)).items()}


def decision_summary(details, *, ident, round_number, at=None, source='model'):
    """Accept structured policy/history output and copy only public decision fields."""
    directive = details.get('directive') or {}
    review = details.get('review')
    execution = details.get('execution') or {}
    progress = execution.get('progress') or {}
    decision_model = details.get('decision_model') or {}
    actor_model = details.get('execution_model') or execution.get('model') or {}
    row = {'decision_id': details.get('decision_id') or ident, 'timestamp': at, 'source': source, 'round': round_number,
           'reason': public_text(details.get('reason')), 'changes': planned_changes(details),
           'feedback_summary': '', 'error': '',
           'decision_revision': decision_model.get('revision'), 'actor_revision': actor_model.get('revision')}
    for key in ('depends_on_decision_ids', 'source_decision_ids'):
        if isinstance(details.get(key), (list, tuple)):
            row[key] = list(details[key])
    if review is not None:
        row.update(kind='review', action=review['action'], reason=public_text(review.get('reason')),
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
    elif details.get('kind') == 'invoke':
        operation = details.get('operation') or {}
        row.update(kind='operation', operation=operation.get('name'), status='proposed')
    else:
        row.update(kind=details.get('kind', 'finish'), status='completed')
    if details.get('validation_error'):
        row.update(status='failed', error=public_text(details['validation_error']))
    return row


def monitor_source(root: Path, source: RunSource | None = None) -> RunSource:
    """Use recorded identity; web addresses monitored runs by directory name."""
    metadata = root / 'run.json'
    if metadata.exists():
        recorded = RunSource.from_dict(json.loads(metadata.read_text()))
        if source is not None and source != recorded:
            raise ValueError('Monitor source conflicts with run.json')
        source = recorded
    source = source if source is not None else RunSource(root.name)
    if source.run_id != root.name:
        raise ValueError('Monitored run_id must match its directory name')
    return source


class Monitor:
    def __init__(self, root, base, objective, *, rounds=1, source: RunSource | None = None):
        source = monitor_source(root, source)
        self.root, self.jobs, self.candidates = root, {}, {}
        self.phase, self.started = 'starting', time.monotonic()
        self._completed = set()
        self._candidate_origins = {}
        self.data = {'schema_version': 1, **source.to_dict(), 'objective': objective,
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
            if 'metrics' in details:
                details['metrics'] = public_metrics(details['metrics'])
            if 'metric_history' in details:
                details['metric_history'] = metric_history(
                    self.jobs.get(details['job_id'], {}).get('metric_history'), details['metric_history'])
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
                                   'summary': summaries.get(event, event),
                                   # Store history once per job, not 300 copies in events.
                                   'details': {k: v for k, v in details.items() if k != 'metric_history'}})
        self.data['events'] = self.data['events'][-300:]
        self.write()

    def decision(self, decision, context, round_number):
        details = {**asdict(decision), 'decision_model': asdict(context.decision_model) if context.decision_model else {},
                   'execution_model': asdict(context.execution.model) if context.execution.model else {}}
        ident = decision.directive.directive_id if decision.directive else str(uuid4())
        row = decision_summary(details, ident=ident, round_number=round_number, at=timestamp())
        # A policy retry may return the same stable decision. Keep its original
        # timestamp and confirmed feedback instead of making an ambiguous node.
        if any(existing.get('decision_id') == row['decision_id'] for existing in self.data['decisions']):
            return
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
        operation_event = event.get('decision') if isinstance(event.get('decision'), dict) else event
        receipt = operation_event.get('operation_result')
        if isinstance(receipt, dict):
            # Parallel ticks may also contain unrelated exploration feedback.
            row = operation_target(self.data['decisions'], operation_event,
                                   result.job.job_id if result.job else None)
            if row is not None:
                result_summary(row, receipt)
                record_lineage(row, receipt, self._candidate_origins)
        elif result.job and not result.directive_id:
            row = operation_target(self.data['decisions'], operation_event)
            if row is not None:
                row.update(job_id=result.job.job_id, status='waiting')
        feedback = event.get('feedback') or {}
        directive_id = result.directive_id or feedback.get('directive_id')
        if feedback.get('directive_id', directive_id) != directive_id:
            feedback = {}  # Conflicting IDs are not evidence for either directive.
        targets = [r for r in self.data['decisions'] if r.get('kind') == 'directive'
                   and directive_id and r.get('directive_id') == directive_id]
        for row in targets:
            if feedback:
                result_summary(row, feedback, feedback=True)
                record_lineage(row, feedback, self._candidate_origins)
            elif result.kind == 'waiting':
                row['status'] = 'running'
        if feedback and targets and directive_id not in self._completed:
            self._completed.add(directive_id)
            self.data['workflow']['completed_directives'] += 1
        if directive_id and current.get('directive_id') == directive_id:
            progress = event.get('progress') or {}
            if feedback:
                result_summary(current, feedback, feedback=True)
            elif result.kind == 'waiting':
                current['status'] = 'running'
            current['phase'] = progress.get('phase', current['status'])
            if progress.get('elapsed_seconds') is not None:
                current['elapsed_seconds'] = progress['elapsed_seconds']
            job_id = result.job.job_id if result.job else None
            if not job_id and feedback.get('jobs'):
                job_id = feedback['jobs'][-1]['job_id']
            if job_id:
                current['job_id'] = job_id
            for row in targets:
                if job_id:
                    row['job_id'] = job_id
            if event.get('review'):
                review = event['review']
                current.update(review_status=review['action'], next_review_seconds=review['next_review_seconds'])
                for row in reversed(self.data['decisions']):
                    if row['kind'] == 'review' and row.get('directive_id') == directive_id:
                        row['status'] = 'cancel_requested' if review['action'] == 'cancel' else 'reviewed'
                        break
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


def row_signature(row):
    return tuple(row.get(k) for k in ('kind', 'directive_id', 'mode', 'operation', 'action'))


def existing_matches(rows, existing):
    """Reuse live identities only with explicit IDs or an unambiguous ordered window."""
    matches, used = {}, set()
    signatures = [row_signature(r) for r in rows]
    old_signatures = [row_signature(r) for r in existing]
    starts = [i for i in range(len(rows) - len(existing) + 1)
              if existing and signatures[i:i + len(existing)] == old_signatures]
    if len(starts) == 1:
        for j, old in enumerate(existing):
            matches[starts[0] + j] = old
            used.add(j)
    for i, row in enumerate(rows):
        if i in matches:
            continue
        candidates = [j for j, old in enumerate(existing) if j not in used and
                      (old.get('decision_id') == row['decision_id'] or
                       (row['kind'] == 'directive' and old.get('kind') == 'directive' and
                        old.get('directive_id') == row['directive_id']))]
        if len(candidates) == 1:
            j = candidates[0]
            matches[i], used = existing[j], used | {j}
    return matches, [r for j, r in enumerate(existing) if j not in used]


def merge_recorded_row(row, old):
    merged = deepcopy(old)
    for key, value in row.items():
        if key in ('decision_id', 'timestamp', 'source') and key in merged:
            continue
        if key not in merged or merged[key] in (None, '', []):
            merged[key] = deepcopy(value)
    # Regenerate only whitelisted submitted fields. Explicit recorded before values
    # survive only for the same field/after; no previous proposal supplies a before.
    changes = deepcopy(row['changes'])
    for change in changes:
        previous = [c for c in old.get('changes', []) if isinstance(c, dict)
                    and c.get('field') == change['field'] and c.get('after') == change['after']]
        if len(previous) == 1 and isinstance(previous[0].get('before'), str):
            change['before'] = public_text(previous[0]['before'])
    merged['changes'] = changes
    for key in ('reason', 'feedback_summary', 'error'):
        merged[key] = public_text(merged.get(key))
    return merged


def step_operation_receipts(history, steps):
    """Align the full dispatch sequence, never match repeated operations by name.

    A truncated/missing tick or unrecorded review tool makes alignment uncertain;
    in that case operation receipts remain absent (memory receipts still work).
    """
    expected, operation_ids, explicit = [], [], {}
    identities = {r["details"].get("decision_id", r["record_id"]): r["record_id"]
                  for r in history if r["kind"] == "decision" and r["details"].get("kind") == "invoke"}
    has_explicit = False
    for record in history:
        if record['kind'] != 'decision' or record['details'].get('review') is not None:
            continue
        details = record['details']
        if details.get('kind') == 'invoke':
            expected.append(('operation',))
            operation_ids.append(record['record_id'])
        elif details.get('directive'):
            expected.append(('directive', details['directive']['directive_id']))
        elif details.get('kind') == 'finish':
            expected.append(('finish',))
    observed, receipts, seen = [], [], set()
    for step in steps:
        event = step.get('event') or {}
        decision = event.get('decision') if isinstance(event.get('decision'), dict) else event
        receipt = decision.get('operation_result')
        if isinstance(receipt, dict):
            if decision.get('decision_id') is not None:
                has_explicit = True
                record_id = identities.get(decision['decision_id'])
                if record_id is not None:
                    explicit[record_id] = receipt
            observed.append(('operation',))
            receipts.append(receipt)
        directive = decision.get('directive') or {}
        ident = directive.get('directive_id')
        if ident and ident not in seen:
            observed.append(('directive', ident))
            seen.add(ident)
        if decision.get('kind') == 'finished':
            observed.append(('finish',))
    if has_explicit:
        return explicit
    return dict(zip(operation_ids, receipts)) if expected == observed else {}


def enrich_completed_run(root):
    """Explicit offline export of existing evidence, never called by the web server."""
    root = Path(root)
    result = json.loads((root / 'result.json').read_text())
    if result.get('status') not in ('completed', 'failed', 'timed_out', 'stopped'):
        raise ValueError('Only a finished controller run may be exported')
    snapshot = json.loads((root / 'monitor.json').read_text())
    source = monitor_source(root, None if (root / 'run.json').exists() else RunSource.from_dict(snapshot))
    if any(key in snapshot and snapshot[key] != value for key, value in source.to_dict().items()):
        raise ValueError('Export source conflicts with monitor.json')
    history = json.loads((root / 'decision_memory.json').read_text())['history']
    steps_path = root / 'steps.jsonl'
    steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()] if steps_path.exists() else []
    step_receipts = step_operation_receipts(history, steps)
    rows, records, round_number = [], [], 0
    for record in history:
        details = record['details']
        if record['kind'] == 'decision':
            if (details.get('directive') or {}).get('mode') == 'train':
                round_number += 1
            row = decision_summary(details, ident=record['record_id'], round_number=round_number,
                                   at=record.get('timestamp'), source='history')
            rows.append(row)
            records.append(record)
    matches, extras = existing_matches(rows, snapshot.get('decisions', []))
    rows = [merge_recorded_row(row, matches[i]) if i in matches else row for i, row in enumerate(rows)]
    # Preserve existing presentation identities, remapping explicit history
    # references and receipt lookups together when exporting an older snapshot.
    aliases = {}
    for row, record in zip(rows, records):
        stable = record['details'].get('decision_id')
        if stable:
            aliases[stable] = row['decision_id']
    for row in rows + extras:
        for key in ('depends_on_decision_ids', 'source_decision_ids', 'adopted_decision_ids'):
            if isinstance(row.get(key), list):
                row[key] = [aliases.get(ident, ident) for ident in row[key]]
        if row.get('parent_decision_id') in aliases:
            row['parent_decision_id'] = aliases[row['parent_decision_id']]
    by_record = {record['record_id']: row for record, row in zip(records, rows)}
    by_decision = {row['decision_id']: row for row in rows}
    by_decision.update({stable: by_decision[present] for stable, present in aliases.items()})
    previous = None
    origins = {}
    # Steps may be the only retained producer receipts. Establish origins before
    # enriching operation receipts, without inferring them from candidate names.
    for step in steps:
        feedback = (step.get('event') or {}).get('feedback') or {}
        ident = feedback.get('directive_id')
        if ident and step.get('directive_id') in (None, ident):
            for row in rows:
                if row.get('kind') == 'directive' and row.get('directive_id') == ident:
                    record_lineage(row, feedback, origins)
    for record in history:
        details = record['details']
        if record['kind'] == 'operation' and isinstance(details.get('result'), dict):
            row = by_decision.get(details.get('decision_id'))
            # Legacy history has no stable call identity; require adjacency and equality.
            if row is None and 'decision_id' not in details and previous is not None:
                decision = previous['details']
                if (previous['kind'] == 'decision' and decision.get('kind') == 'invoke'
                        and decision.get('operation') == details.get('operation')):
                    row = by_record[previous['record_id']]
            if row is not None:
                enrich_result(row, details['result'])
                record_lineage(row, details['result'], origins)
        elif record['kind'] == 'feedback':
            feedback = details.get('feedback') or {}
            for row in rows:
                if row['kind'] == 'directive' and row.get('directive_id') == feedback.get('directive_id'):
                    enrich_result(row, feedback, feedback=True)
                    record_lineage(row, feedback, origins)
        previous = record
    for ident, receipt in step_receipts.items():
        row = by_record[ident]
        # Memory receipts are the more direct source; ticks fill missing text/status.
        observed = {**row}
        result_summary(observed, receipt)
        record_lineage(row, receipt, origins)
        for key in ('feedback_summary', 'error', 'status'):
            if not row.get(key) or (key == 'status' and row[key] in ACTIVE):
                row[key] = observed[key]
    for step in steps:
        feedback = (step.get('event') or {}).get('feedback') or {}
        ident = feedback.get('directive_id')
        if ident and step.get('directive_id') in (None, ident):
            for row in rows:
                if row['kind'] == 'directive' and row.get('directive_id') == ident:
                    observed = {**row}
                    result_summary(observed, feedback, feedback=True)
                    record_lineage(row, feedback, origins)
                    for key, value in observed.items():
                        if not row.get(key) or (key == 'status' and row[key] in ACTIVE):
                            row[key] = value
    snapshot['decisions'] = rows + extras
    # Keep explicit provenance backfill compatibility; never infer a parent.
    for key, value in source.to_dict().items():
        snapshot.setdefault(key, value)
    # All other fields are the original observation, not a new run-state projection.
    atomic_write(root / 'monitor.json', json_bytes(snapshot))
    return snapshot


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export public decision summaries from a completed toy run')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    enriched = enrich_completed_run(args.run_dir)
    print(f"Exported {len(enriched['decisions'])} recorded decisions for {args.run_dir.name}")
