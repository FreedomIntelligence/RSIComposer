"""Lab prototype: copy this file to build your own Lab.

A Lab is an environment that owns its data, model calls, and scoring. The host
calls it; the Controller only sees the operations it publishes and the receipts
it returns. This arithmetic Lab scores a model's answers on a small dataset and
shows every part of the Lab contract:

  Required
    describe()          Publish operations with JSON input schemas.
    execute(call)       Run one operation. Return an OperationResult (synchronous)
                        or a JobRef (asynchronous, see below).

  Optional, recognized by the host when present
    split_roles()       What the Controller may see of each data split.
    primary_metric()    The score Controllers compare, reported in
                        receipt.data['quality_metrics'].
    data_bindings()     Fixed inputs, recorded in the run identity.
    poll_job(job)       Advance or check an asynchronous job.
    cancel_job(job)     Stop an asynchronous job.
    reattach_job(job)   Accept a job again after the run resumes.
    state(), restore()  Save and reload Lab state for durable runs.

Synchronous and asynchronous operations use the same interface. The only
difference is what execute() returns:

  - An OperationResult: the work is done; the host delivers the receipt at once.
  - A JobRef: the work continues. The host calls poll_job(job) until it returns
    a terminal JobOutcome, then delivers that outcome's OperationResult as the
    receipt. Meanwhile the Controller sees the call in window.pending and can
    return Decision('wait').

Here, evaluate(background=true) returns a JobRef and each poll_job() call scores
one batch of rows, so the example needs no threads. A Lab with real long-running
work, such as a training process or a cluster job, usually starts that work in
execute() and only checks it in poll_job().

To adapt this Lab: replace read_dataset() and the scoring in _score() with your
domain's data and scorer, change the operations in describe(), and keep the
receipt fields that your Controller reads.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from rsisys.api import (
    DEV, TRAIN, DatasetBinding, GenerationRequest, JobOutcome, JobProgress, JobRef, JobStatus,
    MetricSpec, OperationResult, OperationSpec,
)


SPLITS = {'train': TRAIN, 'dev': DEV}
BATCH_SIZE = 2  # Rows scored per poll_job() call for background evaluations.


def validate_config(config):
    """Reject unknown options at startup; the registry calls this before the factory."""
    datasets = config.get('datasets')
    if (set(config) != {'datasets'} or not isinstance(datasets, dict)
            or set(datasets) != set(SPLITS)
            or any(not isinstance(path, str) or not path.strip() for path in datasets.values())):
        raise ValueError('datasets must map train and dev to JSONL paths')


def read_dataset(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict)
                       or any(not isinstance(row.get(key), str) or not row[key].strip()
                              for key in ('id', 'question'))
                       or type(row.get('answer')) is not int for row in rows):
        raise ValueError(f'{path}: expected JSONL rows with id, question and an integer answer')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'{path}: duplicate row IDs')
    return rows, hashlib.sha256(raw).hexdigest()


class ArithmeticLab:
    def __init__(self, context, config):
        # The factory receives (context, config). context gives host services:
        # models, roles, artifacts. config is this Lab's section of the run config.
        validate_config(config)
        self.models = context.models
        self.actor = context.roles.get('actor')
        if self.actor is None or context.backends[self.actor.backend_id].binding.kind != 'chat_completions':
            raise ValueError('The template Lab requires an actor on a GPU-backed chat_completions endpoint')
        self.rows, self.hashes = {}, {}
        self.paths = {split: str(Path(path).resolve()) for split, path in config['datasets'].items()}
        for split, path in config['datasets'].items():
            self.rows[split], self.hashes[split] = read_dataset(path)
        for field in ('id', 'question'):
            train_values = {row[field] for row in self.rows['train']}
            if any(row[field] in train_values for row in self.rows['dev']):
                raise ValueError(f'Training and benchmark rows overlap on {field}')
        self.jobs = {}  # Background evaluations by job ID; saved by state().

    # -- Data access and scoring rules ------------------------------------------

    def split_roles(self):
        return dict(SPLITS)

    def data_bindings(self):
        # The host records these in the Lab identity and rejects a resume with changed inputs.
        return tuple(DatasetBinding(
            name='arithmetic/' + split, split=split, role=split, source=self.paths[split],
            schema_version='arithmetic-jsonl/v1', content_digest=self.hashes[split],
            row_ids=tuple(row['id'] for row in self.rows[split]),
        ) for split in sorted(SPLITS))

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', 'integer-text-exact-match/v1')

    # -- Operations --------------------------------------------------------------

    def describe(self):
        return (OperationSpec('evaluate', 'Evaluate a prompt on the dev split', {
            'type': 'object', 'properties': {
                'split': {'type': 'string', 'enum': ['dev']},
                'instruction': {'type': 'string', 'minLength': 1},
                'few_shot': {'type': 'integer', 'minimum': 0, 'maximum': len(self.rows['train'])},
                'background': {'type': 'boolean', 'description': 'Run as a job and return its result later'},
            }, 'required': ['split', 'instruction', 'few_shot'], 'additionalProperties': False,
        # read_only=False: the operation calls the model, so the host records its
        # receipt as evidence and delivers it to Controller Memory.
        }, read_only=False),)

    def execute(self, call):
        args = call.arguments
        count, instruction = args.get('few_shot'), args.get('instruction')
        if (call.name != 'evaluate' or not {'split', 'instruction', 'few_shot'} <= set(args)
                or set(args) - {'split', 'instruction', 'few_shot', 'background'}
                or args['split'] != 'dev' or type(count) is not int or not 0 <= count <= len(self.rows['train'])
                or not isinstance(instruction, str) or not instruction.strip()
                or type(args.get('background', False)) is not bool):
            raise ValueError('Expected evaluate(split=dev, instruction, few_shot[, background])')
        if not args.get('background', False):
            # Synchronous: do all the work now and return the receipt.
            return self._receipt(args, self._score(args, self.rows['dev']))
        # Asynchronous: record the work and return a JobRef. The host polls it.
        job = JobRef('evaluate-' + call.call_id, 'template-math')
        self.jobs[job.job_id] = {'arguments': dict(args), 'records': [], 'status': JobStatus.RUNNING.value}
        return job

    # -- Asynchronous jobs -------------------------------------------------------

    def poll_job(self, job):
        work = self.jobs[job.job_id]
        if work['status'] == JobStatus.RUNNING.value:
            done = len(work['records'])
            work['records'] += self._score(work['arguments'], self.rows['dev'][done:done + BATCH_SIZE])
            if len(work['records']) == len(self.rows['dev']):
                work['status'] = JobStatus.SUCCEEDED.value
        status = JobStatus(work['status'])
        if status == JobStatus.SUCCEEDED:
            return JobOutcome(job, status, result=self._receipt(work['arguments'], work['records']))
        return JobOutcome(job, status, progress=JobProgress(
            step=len(work['records']), total_steps=len(self.rows['dev'])))

    def cancel_job(self, job):
        work = self.jobs[job.job_id]
        if work['status'] == JobStatus.RUNNING.value:
            work['status'] = JobStatus.CANCELLED.value
        return self.poll_job(job)

    def reattach_job(self, job):
        # Called after a durable run resumes. The job's progress came back with
        # restore(), so it can simply continue.
        if job.job_id not in self.jobs:
            raise ValueError('Unknown evaluation job')

    # -- Shared work -------------------------------------------------------------

    def _score(self, args, rows):
        examples = ''.join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
                           for row in self.rows['train'][:args['few_shot']])
        records = []
        for row in rows:
            response = self.models.generate(GenerationRequest(
                'actor', self.actor, f"{examples}Question: {row['question']}\nAnswer:",
                system=f"{args['instruction']}\nReturn only the final integer.",
                temperature=0.0, max_output_tokens=64,
            ))
            answer = response.text.strip()
            records.append({'id': row['id'], 'answer': answer, 'correct': answer == str(row['answer'])})
        return records

    def _receipt(self, args, records):
        # The receipt the Controller reads. quality_metrics carries the primary metric.
        return OperationResult('Development benchmark evaluated', (), False, data={
            'quality_metrics': {'accuracy': sum(row['correct'] for row in records) / len(records)},
            'few_shot': args['few_shot'],
            'dataset_sha256': dict(self.hashes),
            'records': records,
        })

    # -- Durable runs ------------------------------------------------------------

    def state(self):
        return {'dataset_sha256': dict(self.hashes), 'jobs': deepcopy(self.jobs)}

    def restore(self, state):
        if state['dataset_sha256'] != self.hashes:
            raise ValueError('Lab dataset contents changed since the checkpoint')
        self.jobs = deepcopy(state['jobs'])
