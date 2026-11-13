"""Publish the reported facts of one RSISys run in the Monitor snapshot protocol."""
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import monotonic
import warnings


active_monitor = ContextVar('active_monitor', default=None)
PUBLIC_ID = re.compile(r'[\w.-]{1,200}\Z')
SETTING_FIELDS = {'method', 'seed', 'rounds', 'plain_rounds', 'warmup_steps', 'max_seconds',
                  'max_decision_steps', 'dev_count', 'test_count', 'train_count', 'search_count',
                  'validation_count', 'model_id', 'condition', 'candidates_per_round', 'minibatch_size',
                  'learning_rate', 'global_batch', 'max_length'}
METRIC_GROUPS = ('quality_metrics', 'training_metrics', 'execution_metrics', 'collection')
LIMITS = {'decisions': 300, 'evaluations': 300, 'samples': 500, 'artifacts': 300,
          'metric_points': 10000, 'logs': 400}


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
        self.last_training = None
        self.log_offsets = {}
        self.seen_evaluations = set()
        self.io_failed = False

    def publish(self, status=None):
        if status is not None:
            self.data['status'] = status
        self.data['updated_at'] = now()
        self.data['workflow'] = {'elapsed_seconds': round(monotonic() - self.clock, 3)}
        self.data['current'] = self.stages.get(self.current_id, {})
        payload = {**self.data, 'decisions': list(self.stages.values())}
        payload['jobs'] = [{'job_id': row['job_id'], 'decision_id': row['decision_id'],
                            **{key: row[key] for key in ('status', 'step', 'total_steps', 'metrics') if key in row}}
                           for row in payload['decisions'][-LIMITS['decisions']:] if row.get('job_id')]
        payload['events'] = [{'timestamp': row['timestamp'], 'kind': row['source'], 'summary': row['message'],
                              'details': {'decision_id': row['decision_id']}} for row in self.data['logs'][-300:]]
        for key, limit in LIMITS.items():
            payload[key] = payload[key][-limit:]
        temporary = None
        try:
            with NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root, prefix='.monitor-', delete=False) as file:
                temporary = Path(file.name)
                json.dump(payload, file, ensure_ascii=False, allow_nan=False)
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

    def log(self, message, *, decision_id=None, level='INFO', source='runner'):
        logs = self.data['logs']
        logs.append({'timestamp': now(), 'level': level, 'source': source,
                     'message': message, 'decision_id': decision_id})
        del logs[:-LIMITS['logs']]

    def start(self, decision_id, operation, arguments=None):
        args = arguments if isinstance(arguments, dict) else {}
        row = self.stages.setdefault(decision_id, {'decision_id': decision_id, 'started_at': now()})
        row.update(operation=operation, status='running')
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
        self.current_id = decision_id
        self.last_training = None
        self.log(operation + ' · 开始', decision_id=decision_id)
        self.publish()

    def observe_training(self, row, *, publish=True):
        stage = self.stages.get(self.current_id)
        step = row.get('step')
        values = {key: row[key] for key in ('completion_loss', 'learning_rate', 'tokens_per_second', 'grad_norm')
                  if numeric(row.get(key))}
        if not stage or type(step) is not int or step < 0 or not values or step > stage.get('total_steps', step):
            return
        if step < stage.get('step', 0) or (step, values) == self.last_training:
            return
        self.last_training = (step, values)
        stage['step'] = step
        stage['metrics'] = values
        self.add_points(values, stage['decision_id'], step=step)
        message = 'step=' + str(step) + ' ' + ' '.join(f'{key}={value}' for key, value in values.items())
        self.log(message, source='trainer', decision_id=stage['decision_id'])
        if publish:
            self.publish()

    def read_training_log(self, path):
        if not self.current_id or self.stages[self.current_id].get('operation') != 'star.fit':
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
                        self.observe_training(row, publish=False)
            if self.log_offsets.get(path, 0) != before:
                self.publish()
        except OSError:
            pass

    def add_points(self, values, decision_id, **coordinates):
        points = self.data['metric_points']
        points.extend({'metric': key, 'value': val, 'decision_id': decision_id,
                       'elapsed_seconds': round(monotonic() - self.clock, 3), **coordinates}
                      for key, val in values.items())
        del points[:-LIMITS['metric_points']]

    def add_artifact(self, artifact_id, kind, decision_id, **fields):
        if not isinstance(artifact_id, str) or not PUBLIC_ID.fullmatch(artifact_id):
            return
        artifacts = self.data['artifacts']
        if any(row['artifact_id'] == artifact_id and row['decision_id'] == decision_id for row in artifacts):
            return
        artifacts.append({'artifact_id': artifact_id, 'kind': kind, 'decision_id': decision_id, **fields})
        del artifacts[:-LIMITS['artifacts']]

    def add_samples(self, records, decision_id):
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
                samples.append(item)
        del samples[:-LIMITS['samples']]

    def evaluation_samples(self, evidence, decision_id):
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
                self.add_samples(records, decision_id)
            self.add_artifact(candidate.name, 'evaluation_scores', decision_id)
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
        ident = event.get('decision_id') or directive.get('directive_id')
        if not ident or not (receipt or progress or directive or event.get('job') or event.get('kind') == 'finished'):
            return
        if ident not in self.stages:
            self.start(ident, directive.get('strategy') or event.get('kind') or '运行操作')
        row = self.stages[ident]
        self.current_id = ident
        if isinstance(receipt, dict):
            already_closed = 'finished_at' in row
            status = receipt.get('status') or 'unknown'
            row.update(status=status, finished_at=row.get('finished_at') or now())
            data = receipt.get('data') or {}
            grouped = {key: numeric_fields(data.get(key, receipt.get(key))) for key in METRIC_GROUPS}
            values = {key: value for group in grouped.values() for key, value in group.items()}
            row['metrics'] = {**row.get('metrics', {}), **values}
            if type(values.get('optimizer_steps')) is int:
                row['step'] = values['optimizer_steps']
            if not already_closed:
                self.add_points(values, ident, **({'step': row['step']} if 'step' in row else {}))
                self.log(row['operation'] + ' · ' + status, decision_id=ident,
                         level='ERROR' if status == 'failed' else 'INFO')
            for artifact in receipt.get('artifacts', receipt.get('artifact_refs', [])):
                if isinstance(artifact, dict):
                    kind = artifact.get('kind')
                    self.add_artifact(artifact.get('artifact_id'), kind if isinstance(kind, str) and PUBLIC_ID.fullmatch(kind) else 'artifact', ident)
            model = data.get('model')
            if data.get('checkpoint') and isinstance(model, str) and PUBLIC_ID.fullmatch(model):
                self.add_artifact(model, 'checkpoint', ident, model=model)
            if data.get('dataset') and grouped['collection']:
                self.add_artifact('accepted.jsonl', 'training_data', ident)
            request = data.get('request') if isinstance(data.get('request'), dict) else {}
            split = row.get('split') or request.get('split') or data.get('split') or data.get('role')
            split = {'validation': 'dev', 'development': 'dev'}.get(split, split) if isinstance(split, str) else None
            if status == 'completed' and grouped['quality_metrics'] and ident not in self.seen_evaluations:
                self.seen_evaluations.add(ident)
                self.data['evaluations'].append({'evaluation_id': ident, 'decision_id': ident,
                    'split': split if split in ('dev', 'test', 'train', 'search') else None,
                    'model': model if isinstance(model, str) and PUBLIC_ID.fullmatch(model) else row.get('model'),
                    'label': row.get('label'), 'round': row.get('round'), 'timestamp': now(),
                    'elapsed_seconds': round(monotonic() - self.clock, 3), 'metrics': grouped['quality_metrics']})
                self.evaluation_samples(data.get('evidence'), ident)
            if receipt.get('simulated') is True:
                self.data['simulated'] = True
            execution = grouped['execution_metrics']
            if execution:
                self.data['resources']['latest_execution_metrics'] = execution
                self.data['resources']['decision_id'] = ident
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
                self.add_points(values, ident, **({'step': row['step']} if 'step' in row else {}))
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
        self.log('运行中断 · ' + exception_name, decision_id=self.current_id, level='ERROR')
        self.publish(status)


def run_monitored(app, *, rounds, max_decision_steps, framework=None, experiment_name=None):
    config = app.config
    publisher = RunSnapshot(config.data_dir, config.run_id, config.objective,
                            settings=public_settings(framework, config), parent_id=config.parent_run_id)
    if isinstance(experiment_name, str) and experiment_name.strip():
        publisher.data['experiment_name'] = experiment_name.strip()[:200]
    if max_decision_steps is not None:
        publisher.data['config']['max_decision_steps'] = max_decision_steps
    if rounds is not None:
        publisher.data['config']['runner_round_limit'] = rounds
    token = active_monitor.set(publisher)
    try:
        publisher.publish('running')
        result = app.run(rounds=rounds, max_decision_steps=max_decision_steps)
        publisher.publish(result.status)
        return result
    except BaseException as error:
        publisher.fail(type(error).__name__)
        raise
    finally:
        active_monitor.reset(token)
