"""Publish the reported facts of one RSIComposer run in the Monitor snapshot protocol."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import monotonic
import warnings
from uuid import uuid4

from rsicomposer.runtime.observation import Observations, activate, current_observations
from rsicomposer.runtime.storage.files import atomic_write, json_bytes


PUBLIC_ID = re.compile(r'[\w.-]{1,200}\Z')
SETTING_FIELDS = {'method', 'seed', 'rounds', 'plain_rounds', 'warmup_steps', 'max_seconds',
                  'max_decision_steps', 'dev_count', 'test_count', 'train_count', 'search_count',
                  'validation_count', 'model_id', 'condition', 'candidates_per_round', 'minibatch_size',
                  'learning_rate', 'global_batch', 'max_length'}
METRIC_GROUPS = ('quality_metrics', 'training_metrics', 'execution_metrics', 'collection')
CONTRACT = json.loads(Path(__file__).with_name('monitor_contract.json').read_text())
LIMITS = {'decisions': CONTRACT['max_rows'], 'evaluations': CONTRACT['max_rows'], 'samples': CONTRACT['max_samples'],
          'artifacts': CONTRACT['max_artifacts'], 'metric_points': CONTRACT['max_metric_points'],
          'logs': CONTRACT['max_logs']}
# Shed the bulkiest history first when a snapshot would exceed what the monitor accepts.
SHED_ORDER = ('metric_points', 'samples', 'logs', 'events')


def now():
    return datetime.now(timezone.utc).isoformat()


def numeric(value):
    return type(value) is int and abs(value) <= 2**53 or type(value) is float and math.isfinite(value)


def numeric_fields(values):
    if not isinstance(values, dict):
        return {}
    return {key: value for key, value in values.items()
            if isinstance(key, str) and re.fullmatch(r'[\w./-]{1,100}', key) and numeric(value)}


def public_settings(framework, config):
    method = framework.get('method') if isinstance(framework, dict) else None
    method = method if isinstance(method, dict) else {}
    settings = {'method': method.get('name'), 'max_decision_steps': config.max_decision_steps}
    model = getattr(config, 'roles', {}).get('actor')
    if model is not None:
        settings['model_id'] = model.model_id
    method_config = method.get('config', {})
    if isinstance(method_config, dict):
        controller = method_config.get('controller')
        labs = method_config.get('labs')
        lab = labs.get(method.get('name'), {}) if isinstance(labs, dict) else {}
        parameters = lab.get('parameters') if isinstance(lab, dict) else None
        data = lab.get('data') if isinstance(lab, dict) else None
        for source in (method_config, controller, lab, parameters, data):
            if isinstance(source, dict):
                for key in SETTING_FIELDS - {'method'}:
                    if key in source:
                        settings[key] = source[key]
                counts = source.get('counts')
                if isinstance(counts, dict):
                    for split in ('train', 'dev', 'test'):
                        if numeric(counts.get(split)):
                            settings[f'{split}_count'] = counts[split]
    return settings


class RunSnapshot:
    def __init__(self, root, engine_id, objective, *, settings=None, simulated=None, parent_id=None):
        self.root = Path(root).resolve()
        self.clock = monotonic()
        self.started_at = now()
        self.data = {key: [] for key in LIMITS}
        self.data.update(schema_version=1, run_id=self.root.name, engine_run_id=engine_id,
                         experiment_name=self.root.name, objective=objective[:1000],
                         started_at=self.started_at, status='starting', resources={},
                         config={key: val for key, val in (settings or {}).items()
                                 if key in SETTING_FIELDS and (numeric(val) or isinstance(val, str) and PUBLIC_ID.fullmatch(val))})
        if type(simulated) is bool:
            self.data['simulated'] = simulated
        if parent_id:
            self.data['parent_run_id'] = parent_id
        self.stages = {}
        self.current_id = None
        self.log_offsets = {}
        self.seen_evaluations = set()
        self.io_failed = False

    def continue_history(self):
        path = self.root / 'monitor.json'
        if path.is_symlink() or not path.is_file():
            return
        try:
            with path.open('rb') as file:
                payload = file.read(CONTRACT['max_bytes'] + 1)
            if len(payload) > CONTRACT['max_bytes']:
                return
            previous = json.loads(payload)
            json.dumps(previous, allow_nan=False)
        except (OSError, ValueError, RecursionError):
            return
        if (not isinstance(previous, dict) or type(previous.get('schema_version')) is not int
                or any(previous.get(key) != self.data[key]
                       for key in ('schema_version', 'run_id', 'engine_run_id'))):
            return

        # This is the previous public projection, independent of checkpoint state.
        for key, limit in LIMITS.items():
            rows = previous.get(key)
            if isinstance(rows, list):
                self.data[key] = [row for row in rows[-limit:] if isinstance(row, dict)]
        for key in ('started_at', 'experiment_name'):
            if isinstance(previous.get(key), str):
                self.data[key] = previous[key]
        for key in ('config', 'resources', 'metric_definitions', 'history_window'):
            if isinstance(previous.get(key), dict):
                self.data[key] = {**self.data.get(key, {}), **previous[key]}
        if type(previous.get('simulated')) is bool:
            self.data['simulated'] = previous['simulated']
        self.stages = {row.get('call_id') or row['decision_id']: row for row in self.data.pop('decisions')
                       if isinstance(row.get('decision_id'), str) and isinstance(row.get('operation'), str)}
        current = previous.get('current')
        if isinstance(current, dict) and isinstance(current.get('decision_id'), str):
            ident = current.get('call_id') or current['decision_id']
            if ident in self.stages:
                self.current_id = ident
        self.seen_evaluations = {row['evaluation_id'] for row in self.data['evaluations']
                                 if isinstance(row.get('evaluation_id'), str)}
        workflow = previous.get('workflow')
        elapsed = workflow.get('elapsed_seconds') if isinstance(workflow, dict) else None
        if numeric(elapsed) and elapsed >= 0:
            self.clock -= elapsed

    def publish(self, status=None):
        if status is not None:
            self.data['status'] = status
        self.data['updated_at'] = now()
        self.data['workflow'] = {'elapsed_seconds': round(monotonic() - self.clock, 3)}
        self.data['current'] = self.stages.get(self.current_id, {})
        payload = {**self.data, 'decisions': list(self.stages.values())}
        payload['jobs'] = [{'job_id': row['job_id'], **self.coordinates(row),
                            **{key: row[key] for key in ('status', 'step', 'total_steps', 'metrics') if key in row}}
                           for row in payload['decisions'][-LIMITS['decisions']:] if row.get('job_id')]
        payload['events'] = [{'timestamp': row['timestamp'], 'kind': row['source'], 'summary': row['message'],
                              'details': self.coordinates(row)} for row in self.data['logs'][-300:]]
        for key, limit in LIMITS.items():
            payload[key] = payload[key][-limit:]
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        # The monitor rejects larger files outright; keep the newest half of bulky lists until it fits.
        # observations.jsonl still holds every metric point.
        while len(encoded) > CONTRACT['max_bytes']:
            key = max(SHED_ORDER, key=lambda name: len(payload[name]))
            if len(payload[key]) < 2:
                break
            payload[key] = payload[key][len(payload[key]) // 2:]
            payload['history_window'] = {'truncated': True}
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        temporary = None
        try:
            with NamedTemporaryFile(mode='wb', dir=self.root, prefix='.monitor-', delete=False) as file:
                temporary = Path(file.name)
                file.write(encoded)
            os.replace(temporary, self.root / 'monitor.json')
            self.io_failed = False
        except OSError:
            if not self.io_failed:
                warnings.warn('RSIMonitor could not publish this run snapshot', RuntimeWarning, stacklevel=2)
            self.io_failed = True
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def coordinates(row):
        return {key: row[key] for key in ('decision_id', 'call_id') if row.get(key) is not None}

    def log(self, message, *, decision_id=None, call_id=None, level='INFO', source='runner'):
        logs = self.data['logs']
        logs.append({'timestamp': now(), 'level': level, 'source': source,
                     'message': message, 'decision_id': decision_id,
                     **({'call_id': call_id} if call_id is not None else {})})
        del logs[:-LIMITS['logs']]

    def start(self, decision_id, operation, arguments=None, details=None, *, call_id=None):
        args = arguments if isinstance(arguments, dict) else {}
        ident = call_id or decision_id
        row = self.stages.setdefault(ident, {'decision_id': decision_id, 'started_at': now(),
                                          **({'call_id': call_id} if call_id is not None else {})})
        row.update(operation=operation, status='running')
        # What the decision maker said and which earlier decisions it builds on: the decision graph.
        details = details if isinstance(details, dict) else {}
        if isinstance(details.get('reason'), str) and details['reason'].strip():
            row['reason'] = details['reason'][:4000]
        for key in ('depends_on_decision_ids', 'source_decision_ids'):
            # An empty list is left out: the board reads any declared relation as "this run reports lineage".
            idents = [ident for ident in details.get(key) or () if isinstance(ident, str) and 0 < len(ident) <= 200][:100]
            if idents:
                row[key] = idents
        attempt, title = details.get('research_attempt_id'), details.get('research_attempt_title')
        if isinstance(attempt, str) and attempt.strip() and isinstance(title, str) and title.strip():
            row.update(research_attempt_id=attempt[:200], research_attempt_title=title[:500])
        row.pop('finished_at', None)
        for key in ('split', 'model', 'label', 'round'):
            value = args.get(key)
            if numeric(value) or isinstance(value, str) and PUBLIC_ID.fullmatch(value):
                row[key] = value
        if row.get('model') == 'base' and row.get('label') in ('baseline-dev', 'baseline-test'):
            row['round'] = 0
        if 'round' not in row and isinstance(row.get('model'), str):
            match = re.fullmatch(r'round-([0-9]{1,6})', row['model'])
            if match:
                row['round'] = int(match[1])
        if type(args.get('steps')) is int and args['steps'] > 0:
            row['total_steps'] = args['steps']
        self.current_id = ident
        self.log(operation + ' · 开始', **self.coordinates(row))
        self.publish()

    def observe_training(self, row, stage_id, *, publish=True):
        stage = self.stages.get(stage_id)
        step = row.get('step')
        values = {key: row[key] for key in ('completion_loss', 'learning_rate', 'tokens_per_second', 'grad_norm')
                  if numeric(row.get(key))}
        if not stage or type(step) is not int or step < 0 or not values or step > stage.get('total_steps', step):
            return
        if step < stage.get('step', 0) or (step, values) == (stage.get('step'), stage.get('metrics')):
            return
        stage['step'] = step
        stage['metrics'] = values
        self.add_points(values, step=step, **self.coordinates(stage))
        message = 'step=' + str(step) + ' ' + ' '.join(f'{key}={value}' for key, value in values.items())
        self.log(message, source='trainer', **self.coordinates(stage))
        if publish:
            self.publish()

    def read_training_log(self, path, stage_id=None):
        stage_id = stage_id or self.current_id
        if self.stages.get(stage_id, {}).get('operation', '').rsplit('.', 1)[-1] != 'train':
            return
        try:
            before = self.log_offsets.get(path, 0)
            with Path(path).open('rb') as file:
                file.seek(self.log_offsets.get(path, 0))
                while True:
                    line = file.readline(256 * 1024)
                    if not line.endswith(b'\n'):
                        break
                    self.log_offsets[path] = file.tell()
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeError, RecursionError):
                        continue
                    if isinstance(row, dict):
                        self.observe_training(row, stage_id, publish=False)
            if self.log_offsets.get(path, 0) != before:
                self.publish()
        except OSError:
            pass

    def observe_metrics(self, values, *, step=None, definitions=None, decision_id=None, call_id=None, series=None):
        definitions = definitions or {}
        validated = {}
        for key, row in definitions.items():
            if (key not in values or not isinstance(row, dict)
                    or not isinstance(row.get('label'), str) or not 1 <= len(row['label']) <= 200
                    or row.get('unit') not in CONTRACT['metric_units']
                    or row.get('direction') not in CONTRACT['metric_directions']):
                raise ValueError('Invalid observation metric definition')
            validated[key] = {field: row[field] for field in ('label', 'unit', 'direction')}
        merged = {**self.data.get('metric_definitions', {}), **validated}
        for key, val in values.items():
            if merged.get(key, {}).get('unit') == 'percent' and not 0 <= val <= 1:
                raise ValueError('Percent observations must be ratios in [0, 1]')
        elapsed = round(monotonic() - self.clock, 3)
        observation_id = uuid4().hex
        scope = self.coordinates(self.stages.get(call_id or self.current_id, {})) if decision_id is None else {
            'decision_id': decision_id, **({'call_id': call_id} if call_id is not None else {})}
        records = [{'metric': key, 'value': val, 'timestamp': now(), 'elapsed_seconds': elapsed,
                    'observation_id': observation_id,
                    **({'step': step} if step is not None else {}),
                    **({'series': series} if series is not None else {}),
                    **scope}
                   for key, val in values.items()]
        # The snapshot is bounded; the append-only local journal retains all points.
        with (self.root / 'observations.jsonl').open('a', encoding='utf-8') as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        if validated:
            atomic_write(self.root / 'metric-definitions.json', json_bytes(merged))
            self.data['metric_definitions'] = merged
        self.data['metric_points'].extend(records)
        del self.data['metric_points'][:-LIMITS['metric_points']]

    def add_points(self, values, decision_id, *, call_id=None, **coordinates):
        if not values:
            return
        observations = current_observations()
        if observations is not None:
            observations.log(values, step=coordinates.get('step'), decision_id=decision_id, call_id=call_id)
            return
        points = self.data['metric_points']
        points.extend({'metric': key, 'value': val, 'decision_id': decision_id,
                       **({'call_id': call_id} if call_id is not None else {}),
                       'elapsed_seconds': round(monotonic() - self.clock, 3), **coordinates}
                      for key, val in values.items())
        del points[:-LIMITS['metric_points']]

    def add_artifact(self, artifact_id, kind, decision_id, *, call_id=None, **fields):
        if not isinstance(artifact_id, str) or not PUBLIC_ID.fullmatch(artifact_id):
            return
        artifacts = self.data['artifacts']
        if any(row['artifact_id'] == artifact_id and row['decision_id'] == decision_id
               and row.get('call_id') == call_id for row in artifacts):
            return
        artifacts.append({'artifact_id': artifact_id, 'kind': kind, 'decision_id': decision_id,
                          **({'call_id': call_id} if call_id is not None else {}), **fields})
        del artifacts[:-LIMITS['artifacts']]

    def report_samples(self, records, stage_id=None):
        row = self.stages.get(stage_id or self.current_id)
        if row is not None:
            self.add_samples(records, **self.coordinates(row))

    def add_samples(self, records, decision_id, *, call_id=None):
        samples = self.data['samples']
        for row in records:
            if not isinstance(row, dict):
                continue
            ident = row.get('id', row.get('index'))
            if type(ident) is int:
                ident = str(ident)
            if not isinstance(ident, str) or not PUBLIC_ID.fullmatch(ident):
                continue
            item = {'id': ident, 'decision_id': decision_id}
            for key in ('correct', 'parsed'):
                if type(row.get(key)) is bool:
                    item[key] = row[key]
            if 'correct' not in item and row.get('accuracy') in (0, 1) and numeric(row.get('accuracy')):
                item['correct'] = bool(row['accuracy'])
            for key in ('predicted', 'expected'):
                value = row.get(key)
                if isinstance(value, str) and re.fullmatch(r'-?[0-9./]{1,80}', value):
                    item[key] = value
            if len(item) > 2:
                if call_id is not None:
                    item['call_id'] = call_id
                samples.append(item)
        del samples[:-LIMITS['samples']]

    def evaluation_samples(self, evidence, **coordinates):
        if not isinstance(evidence, str):
            return
        try:
            candidate = Path(evidence).resolve(strict=True)
            if not candidate.is_relative_to(self.root) or candidate.name != 'scores.jsonl' or not candidate.is_file():
                return
            with candidate.open(encoding='utf-8') as file:
                records = []
                for _ in range(LIMITS['samples']):
                    line = file.readline(256 * 1024)
                    if not line or not line.endswith('\n'):
                        break
                    records.append(json.loads(line))
                self.add_samples(records, **coordinates)
            self.add_artifact(candidate.name, 'evaluation_scores', **coordinates)
        except (OSError, ValueError, RecursionError):
            pass

    def record(self, event):
        if not isinstance(event, dict):
            return
        decision = event.get('decision')
        if isinstance(decision, dict):
            self.record(decision)
        directive = event.get('directive') or {}
        receipt = event.get('operation_result') or event.get('feedback')
        progress = event.get('progress') or {}
        sealed = event.get('phase') == 'host_finish'
        operation = event.get('operation') if sealed else None
        if sealed:
            receipt = event.get('receipt')
            if not isinstance(operation, dict) or not isinstance(receipt, dict):
                return
        call_id = (receipt or {}).get('call_id') or progress.get('call_id')
        decision_id = event.get('decision_id') or directive.get('directive_id')
        if sealed:
            call_id = operation.get('call_id')
            decision_id = call_id
        ident = call_id or decision_id
        if not ident or not (receipt or progress or directive or event.get('job') or event.get('kind') == 'finished'):
            return
        if ident not in self.stages:
            if event.get('kind') == 'finished' and not (receipt or progress or directive or event.get('job')):
                return
            self.start(decision_id or ident, (operation or {}).get('name') or directive.get('strategy')
                       or event.get('kind') or '运行操作', (operation or {}).get('arguments'), call_id=call_id)
        row = self.stages[ident]
        if sealed:
            row['phase'] = 'host_finish'
        coordinates = self.coordinates(row)
        self.current_id = ident
        if isinstance(receipt, dict):
            already_closed = 'finished_at' in row
            status = receipt.get('status') or 'unknown'
            row.update(status=status, finished_at=row.get('finished_at') or now())
            for source, target in (('summary', 'feedback_summary'), ('error', 'error')):
                if isinstance(receipt.get(source), str) and receipt[source].strip():
                    row[target] = receipt[source][:4000]
            data = receipt.get('data') or {}
            grouped = {key: numeric_fields(data.get(key, receipt.get(key))) for key in METRIC_GROUPS}
            if sealed:
                grouped = {'quality_metrics': grouped['quality_metrics']}
            values = {key: value for group in grouped.values() for key, value in group.items()}
            row['metrics'] = {**row.get('metrics', {}), **values}
            if type(values.get('optimizer_steps')) is int:
                row['step'] = values['optimizer_steps']
            if not already_closed:
                self.add_points(values, **coordinates, **({'step': row['step']} if 'step' in row else {}))
                self.log(row['operation'] + ' · ' + status, **coordinates,
                         level='ERROR' if status == 'failed' else 'INFO')
            for artifact in (() if sealed else receipt.get('artifacts', receipt.get('artifact_refs', []))):
                if isinstance(artifact, dict):
                    kind = artifact.get('kind')
                    self.add_artifact(artifact.get('artifact_id'), kind if isinstance(kind, str) and PUBLIC_ID.fullmatch(kind) else 'artifact', **coordinates)
            model = data.get('model')
            if not sealed and data.get('checkpoint') and isinstance(model, str) and PUBLIC_ID.fullmatch(model):
                self.add_artifact(model, 'checkpoint', model=model, **coordinates)
            if not sealed and data.get('dataset') and grouped['collection']:
                self.add_artifact('accepted.jsonl', 'training_data', **coordinates)
            request = data.get('request') if isinstance(data.get('request'), dict) else {}
            split = row.get('split') or request.get('split') or data.get('split') or data.get('role')
            split = {'validation': 'dev', 'development': 'dev'}.get(split, split) if isinstance(split, str) else None
            if status == 'completed' and grouped['quality_metrics'] and ident not in self.seen_evaluations:
                self.seen_evaluations.add(ident)
                self.data['evaluations'].append({'evaluation_id': ident, **coordinates,
                    'split': split if isinstance(split, str) and PUBLIC_ID.fullmatch(split) else None,
                    'model': model if isinstance(model, str) and PUBLIC_ID.fullmatch(model) else row.get('model'),
                    'label': row.get('label'), 'round': row.get('round'), 'timestamp': now(),
                    'elapsed_seconds': round(monotonic() - self.clock, 3), 'metrics': grouped['quality_metrics']})
                if not sealed:
                    self.evaluation_samples(data.get('evidence'), **coordinates)
            if receipt.get('simulated') is True:
                self.data['simulated'] = True
            execution = grouped.get('execution_metrics')
            if execution:
                self.data['resources']['latest_execution_metrics'] = execution
                self.data['resources'].update(coordinates)
        elif progress or event.get('job'):
            job = progress.get('job') or event.get('job') or {}
            observed = progress.get('job_progress') or {}
            if job.get('job_id'):
                row['job_id'] = job['job_id']
            for key in ('step', 'total_steps', 'elapsed_seconds'):
                if numeric(observed.get(key)):
                    row[key] = observed[key]
            values = numeric_fields(observed.get('metrics', progress.get('metrics')))
            if values and (row.get('metrics') != values or row.get('reported_step') != row.get('step')):
                row['metrics'], row['reported_step'] = values, row.get('step')
                self.add_points(values, **coordinates, **({'step': row['step']} if 'step' in row else {}))
        elif event.get('kind') == 'finished':
            row.update(status='completed', finished_at=now())
        self.publish()

    def observe_model_calls(self, attempted, limit):
        self.data['resources'].update(model_calls_attempted=attempted, model_call_limit=limit)

    def observe_devices(self, devices):
        self.data['resources']['allocated_gpu_count'] = len(devices)
        self.publish()

    def fail(self, exception_name):
        row = self.stages.get(self.current_id)
        status = 'cancelled' if exception_name == 'KeyboardInterrupt' else 'failed'
        if row and row.get('status') == 'running':
            row.update(status=status, finished_at=now())
        self.log('运行中断 · ' + exception_name, level='ERROR', **self.coordinates(row or {}))
        self.publish(status)


def run_monitored(app, *, rounds, max_decision_steps, framework=None, experiment_name=None,
                  probe_paths=(), continue_history=False):
    config = app.config
    publisher = RunSnapshot(config.data_dir, config.run_id, config.objective,
                            settings=public_settings(framework, config), parent_id=config.parent_run_id)
    if continue_history:
        publisher.continue_history()
    if isinstance(experiment_name, str) and experiment_name.strip():
        publisher.data['experiment_name'] = experiment_name.strip()[:200]
    if max_decision_steps is not None:
        publisher.data['config']['max_decision_steps'] = max_decision_steps
    if rounds is not None:
        publisher.data['config']['runner_round_limit'] = rounds
    observations = Observations(publisher)
    with activate(observations, inbox=Path(config.data_dir) / 'log-inbox'):
        return _run(app, publisher, observations, rounds, max_decision_steps, probe_paths)


def _run(app, publisher, observations, rounds, max_decision_steps, probe_paths):
    config = app.config
    try:
        for index, path in enumerate(probe_paths):
            from importlib.util import module_from_spec, spec_from_file_location
            spec = spec_from_file_location(f'rsisys_probe_{index}', Path(path).resolve())
            if spec is None or spec.loader is None:
                raise ValueError('Probe plugins must be Python files exposing register(observations)')
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            module.register(observations)
        observations.trigger('start', {'run_id': config.run_id})
        with observations.lock:
            publisher.publish('running')
        result = app.run(rounds=rounds, max_decision_steps=max_decision_steps, wait_for_jobs=True)
        observations.trigger('finish', {'status': result.status})
        with observations.lock:
            publisher.publish(result.status)
        return result
    except BaseException as error:
        with observations.lock:
            publisher.fail(type(error).__name__)
        raise
