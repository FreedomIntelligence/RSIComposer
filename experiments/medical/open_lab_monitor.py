"""Public evidence projection for the open Lab; no inferred training or adoption."""
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import time

from rsisys.runtime.files import atomic_write, json_bytes


HISTORY_LIMIT = 300
FEEDBACK_FIELDS = ('job_id', 'state', 'feedback_route', 'archive_status', 'memory_status')


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _feedback(request, data):
    fields = {key: deepcopy(data[key]) for key in ('job_id', 'state') if key in data}
    arguments = _mapping(request.get('arguments'))
    route = arguments.get('feedback_route', data.get('feedback_route'))
    if route is None and request.get('name') == 'lab_result':
        route = 'context'
    if route in ('context', 'memory'):
        fields['feedback_route'] = route
    for public, source in (('archive_status', 'archive_delivery'),
                           ('memory_status', 'memory_organization')):
        status = data.get(public, _mapping(data.get(source)).get('status'))
        if isinstance(status, str) and status:
            fields[public] = status
    return fields


def _fixed_metrics(value):
    """The host fixed protocol reports fractional accuracy and sample counts."""
    if not isinstance(value, dict) or not value:
        return None
    for row in value.values():
        if not isinstance(row, dict) or not row:
            return None
        for name, number in row.items():
            try:
                if type(number) not in (int, float) or not math.isfinite(number):
                    return None
            except OverflowError:
                return None
            if name == 'accuracy' and not 0 <= number <= 1:
                return None
    return deepcopy(value)


class OpenLabMonitor:
    def __init__(self, root, run_id, model, revision):
        self.root, self.started, self.jobs = Path(root), time.monotonic(), {}
        self.evaluations = {}
        self._evaluation_jobs = {}
        self._job_observations = {}
        self.data = {'schema_version': 1, 'run_id': run_id,
            'project': {'project_id': 'medical-open-lab', 'name': '医疗 Open Lab'},
            'experiment_name': run_id, 'status': 'initializing', 'decisions': [],
            'models': {'decision_maker': {'model_id': model.model_id, 'revision': model.revision}},
            'current': {'decision_revision': revision}, 'events': [], 'evaluations': [],
            'history_window': {'limit': HISTORY_LIMIT, 'total': 0, 'truncated': False}}
        self.write()

    def write(self):
        self.data.update(updated_at=datetime.now(timezone.utc).isoformat(), jobs=list(self.jobs.values()))
        self.data['workflow'] = {'phase': self.data['status'],
            'elapsed_seconds': time.monotonic() - self.started}
        atomic_write(self.root / 'monitor.json', json_bytes(self.data))

    def update(self, harness, backend, compactor, step):
        rows, results, observations = [], {}, {}
        observed_at = datetime.now(timezone.utc).isoformat()
        # Read original receipts, not the transient/compacted working view. A
        # decision window must not hide older verified evaluations or job states.
        for entry in harness.memory.history(kind='operation'):
            details = _mapping(entry.details)
            receipt = _mapping(details.get('result'))
            ident = details.get('decision_id')
            if ident is not None:
                results[ident] = details
            data = _mapping(receipt.get('data'))
            job_id, state = data.get('job_id'), data.get('state')
            if isinstance(job_id, str) and job_id and isinstance(state, str) and state:
                observations[job_id] = (entry.record_id, {
                    'job_id': job_id, 'kind': 'lab_program', 'status': state,
                    'state': state, 'decision_id': ident})
            self._evaluation(data, ident)
        for job_id, (record_id, job) in observations.items():
            observation = (record_id, job)
            if self._job_observations.get(job_id) != observation:
                self.jobs[job_id] = {**job, 'observed_at': observed_at}
                self._job_observations[job_id] = deepcopy(observation)
        records = harness.state.decision_records
        for ident, record in list(records.items())[-HISTORY_LIMIT:]:
            proposal = _mapping(record.get('proposal'))
            details = results.get(ident, {})
            operation = _mapping(details.get('operation')) or _mapping(proposal.get('operation'))
            receipt = _mapping(details.get('result'))
            row = {'decision_id': ident, 'kind': record.get('kind', proposal.get('kind')),
                   'action': operation.get('name', record.get('operation')),
                   'status': record.get('status'), 'reason': proposal.get('reason', record.get('reason', '')),
                   'strategy': proposal.get('strategy', ''),
                   'feedback_summary': str(receipt.get('summary', ''))[:4000]}
            row.update(_feedback(operation, _mapping(receipt.get('data'))))
            if receipt.get('error'):
                row['error'] = str(receipt['error'])[:4000]
            for key in ('research_attempt_id', 'research_attempt_title', 'depends_on_decision_ids',
                        'source_decision_ids', 'adopted_decision_ids', 'adopted_candidate_ids'):
                if key in record:
                    row[key] = record[key]
            rows.append(row)
        self.data.update(status='running', decisions=rows, evaluations=list(self.evaluations.values()),
            history_window={'limit': HISTORY_LIMIT, 'total': len(records),
                            'truncated': len(records) > HISTORY_LIMIT})
        self.data['current'].update(metrics={'model_calls': backend.calls,
            'compactions': compactor.compactions})
        if rows:
            self.data['current'].update(mode=rows[-1]['action'], status=rows[-1]['status'])
        for key in FEEDBACK_FIELDS:
            self.data['current'].pop(key, None)
            if rows and key in rows[-1]:
                self.data['current'][key] = deepcopy(rows[-1][key])
        self.write()

    def _evaluation(self, data, decision_id):
        # This field is produced only after host verification. Never inspect
        # stdout, summaries, candidate paths or arbitrary artifact files.
        fixed = _mapping(data.get('fixed_evaluation'))
        if fixed.get('verified') is False:
            return
        metrics = _fixed_metrics(fixed.get('metrics'))
        candidate = fixed.get('candidate_id')
        digest, job_id = fixed.get('record_sha256'), data.get('job_id')
        digest = digest if isinstance(digest, str) and digest else None
        job_id = job_id if isinstance(job_id, str) and job_id else None
        if metrics is None or not isinstance(candidate, str) or not candidate or not (digest or job_id):
            return
        key = 'record:' + digest.lower() if digest else 'job:' + job_id
        if not digest:
            key = self._evaluation_jobs.get(job_id, key)
        fallback = 'job:' + job_id if job_id else None
        previous = self.evaluations.get(key, {})
        if digest and fallback in self.evaluations:
            previous = self.evaluations.pop(fallback)
        row = {**previous, 'evaluation_id': key, 'candidate_id': candidate, 'metrics': metrics}
        if job_id:
            row['job_id'] = job_id
        if decision_id is not None:
            row.setdefault('decision_id', decision_id)
        for name in ('protocol', 'record_sha256', 'dataset_sha256', 'evaluator_sha256',
                     'adopted', 'weight_identity', 'adapter_identity', 'profile', 'profile_sha256',
                     'diagnostics', 'role', 'verified', 'split', 'evaluation_protocol',
                     'protocol_sha256', 'submission_sha256'):
            if name in fixed:
                row[name] = deepcopy(fixed[name])
        self.evaluations[key] = row
        if job_id:
            self._evaluation_jobs[job_id] = key
        definitions = self.data.setdefault('metric_definitions', {})
        for benchmark, values in metrics.items():
            for name in values:
                metric = f'{benchmark}.{name}'
                definitions[metric] = {'label': metric,
                    'unit': 'percent' if name == 'accuracy' else 'number',
                    'direction': 'higher' if name == 'accuracy' else 'neutral'}

    def finish(self, result):
        self.data['status'] = result['status']
        finished_at = result.get('finished_at', result.get('ended_at'))
        if type(finished_at) in (int, float):
            finished_at = datetime.fromtimestamp(finished_at, timezone.utc).isoformat()
        lifecycle = {'finished_at': finished_at or datetime.now(timezone.utc).isoformat()}
        if 'stop_reason' in result or 'stop' in result:
            lifecycle['stop_reason'] = deepcopy(result.get('stop_reason', result.get('stop')))
        cleanup = _mapping(result.get('cleanup'))
        if type(cleanup.get('confirmed')) is bool:
            lifecycle['cleanup'] = deepcopy(cleanup)
        self.data['lifecycle'] = lifecycle
        self.data['events'].append({'timestamp': datetime.now(timezone.utc).isoformat(),
            'kind': result['status'], 'summary': result.get('error', result.get('stop', result['status']))})
        self.write()
