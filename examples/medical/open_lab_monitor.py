"""Public evidence projection for the open Lab; no inferred training or adoption."""
from datetime import datetime, timezone
from pathlib import Path
import time

from rsisys.runtime.files import atomic_write, json_bytes


class OpenLabMonitor:
    def __init__(self, root, run_id, model, revision):
        self.root, self.started, self.jobs = Path(root), time.monotonic(), {}
        self.data = {'schema_version': 1, 'run_id': run_id,
            'project': {'project_id': 'medical-open-lab', 'name': '医疗 Open Lab'},
            'experiment_name': run_id, 'status': 'initializing', 'decisions': [],
            'models': {'decision_maker': {'model_id': model.model_id, 'revision': model.revision}},
            'current': {'decision_revision': revision}, 'events': []}
        self.write()

    def write(self):
        self.data.update(updated_at=datetime.now(timezone.utc).isoformat(), jobs=list(self.jobs.values()))
        self.data['workflow'] = {'phase': self.data['status'],
            'elapsed_seconds': time.monotonic() - self.started}
        atomic_write(self.root / 'monitor.json', json_bytes(self.data))

    def update(self, harness, backend, compactor, step):
        rows = []
        results = {entry.details.get('decision_id'): entry.details.get('result', {})
                   for entry in harness.memory.history(kind='operation', limit=600)}
        for ident, record in list(harness.state.decision_records.items())[-300:]:
            proposal = record.get('proposal', {})
            operation = proposal.get('operation') or {}
            receipt = results.get(ident, {})
            row = {'decision_id': ident, 'kind': record.get('kind', proposal.get('kind')),
                   'action': operation.get('name', record.get('operation')),
                   'status': record.get('status'), 'reason': proposal.get('reason', ''),
                   'strategy': proposal.get('strategy', ''),
                   'feedback_summary': str(receipt.get('summary', ''))[:4000]}
            if receipt.get('error'):
                row['error'] = str(receipt['error'])[:4000]
            for key in ('research_attempt_id', 'research_attempt_title', 'depends_on_decision_ids',
                        'source_decision_ids', 'adopted_decision_ids', 'adopted_candidate_ids'):
                if key in record:
                    row[key] = record[key]
            rows.append(row)
        event = step.event.get('decision', step.event)
        result = event.get('operation_result') or {}
        data = result.get('data') or {}
        if data.get('job_id') and data.get('state'):
            ident = data['job_id']
            self.jobs[ident] = {'job_id': ident, 'kind': 'lab_program', 'status': data['state']}
        self.data.update(status='running', decisions=rows)
        self.data['current'].update(metrics={'model_calls': backend.calls,
            'compactions': compactor.compactions})
        if rows:
            self.data['current'].update(mode=rows[-1]['action'], status=rows[-1]['status'])
        self.write()

    def finish(self, result):
        self.data['status'] = 'failed' if result['status'] == 'failed' else 'completed'
        self.data['events'].append({'timestamp': datetime.now(timezone.utc).isoformat(),
            'kind': result['status'], 'summary': result.get('error', result.get('stop', result['status']))})
        self.write()
