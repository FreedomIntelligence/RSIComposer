"""Example-only HTTP adapters for worker.py; no GPU dependencies in the controller.

toy:// refs name files already provisioned in the worker's run directory. They
are not framework artifact-store URIs and do not upload or convert datasets.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from http.client import HTTPException
import json
import math
from pathlib import PurePosixPath
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.exploration.harness.evaluation import EvaluationResult


TRAIN_REF = ArtifactRef('toy-train', 'training_data', 'toy://train.jsonl')
EVAL_REF = ArtifactRef('toy-eval', 'evaluation_data', 'toy://eval.jsonl')
RECIPE = 'toy-lora'
PROTOCOL = 'heldout-nll'


class Deadline:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Timeout must be finite and positive')
        self.end = time.monotonic() + seconds

    def remaining(self):
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Controller deadline exceeded')
        return remaining


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class HTTPClient:
    def __init__(self, url, deadline, *, timeout=120):
        parts = urlsplit(url)
        if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username
                or parts.password or parts.query or parts.fragment or parts.path not in ('', '/', '/v1', '/v1/')):
            raise ValueError('Supply a service origin (optionally /v1), without credentials or query')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('HTTP timeout must be finite and positive')
        self.url = f'{parts.scheme}://{parts.netloc}'
        self.deadline, self.timeout = deadline, timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def call(self, path, payload=None):
        request = Request(self.url + path,
                          data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=min(self.timeout, self.deadline.remaining())) as response:
                raw = response.read(1024 * 1024 + 1)
            self.deadline.remaining()
            if len(raw) > 1024 * 1024:
                raise ValueError('Service response exceeds size limit')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('Service response must be an object')
            return value
        except HTTPError as exc:
            # Never include URLs, remote error bodies, or credentials in records.
            raise RuntimeError(f'Service HTTP status {exc.code}') from None
        except URLError:
            raise ConnectionError('Service transport failed') from None


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('Invalid service identifier')
    return value


def numeric_metrics(raw):
    if not isinstance(raw, dict):
        raise ValueError('Metrics must be an object')
    allowed = {'train_loss', 'train_runtime', 'train_samples_per_second', 'train_steps_per_second',
               'total_flos', 'epoch', 'loss', 'base_nll', 'candidate_nll', 'delta_nll', 'eval_samples'}
    return {key: float(value) for key, value in raw.items() if key in allowed
            and type(value) in (int, float) and math.isfinite(value)}


class RevisionBackend:
    """One backend ID; the declared base routes to baseline, registered revisions to worker."""
    def __init__(self, baseline, worker, base, *, notify=lambda *args, **kwargs: None):
        self.baseline, self.worker, self.base = baseline, worker, base
        self.candidates = {}
        self.active = base
        self.notify = notify

    def register(self, result):
        if result.base != self.active or result.model.backend_id != self.base.backend_id:
            raise ValueError('Candidate has a different backend or training base')
        if result.model.model_id != self.base.model_id or result.model == self.base:
            raise ValueError('Candidate model identity is invalid')
        previous = self.candidates.get(result.model.revision)
        if previous is not None and previous != result:
            raise ValueError('Revision reused with different training result')
        self.candidates[result.model.revision] = deepcopy(result)

    def generate(self, request):
        if request.model == self.base:
            client, verify = self.baseline, False
        elif (request.model.revision in self.candidates
              and self.candidates[request.model.revision].model == request.model):
            client, verify = self.worker, True
        else:
            raise ValueError('Model revision has no configured route')
        self.notify('inference', role=request.role, revision=request.model.revision)
        response = client.call('/v1/chat/completions', {
            'model': request.model.model_id,
            'messages': [{'role': 'system', 'content': request.system},
                         {'role': 'user', 'content': request.prompt}],
            'max_tokens': min(request.max_output_tokens, 200), 'temperature': 0,
            'stream': False, 'chat_template_kwargs': {'enable_thinking': False},
            **({'model_revision': request.model.revision} if verify else {}),
        })
        if response.get('model') != request.model.model_id:
            raise ValueError('Inference model identity mismatch')
        revision = response.get('model_revision')
        if (verify and revision != request.model.revision) or (revision is not None and revision != request.model.revision):
            raise ValueError('Inference revision mismatch')
        choice = response['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Inference was truncated or incomplete')
        content = choice['message']['content']
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Inference returned no text')
        usage = response.get('usage', {})
        return GenerationResponse(content, request.model, False,
            TokenUsage(usage.get('prompt_tokens'), usage.get('completion_tokens')),
            provider_model=response['model'], revision_verified=revision == request.model.revision)


@dataclass(frozen=True)
class _PendingSubmission:
    request: object
    idempotency_key: str
    payload: dict


class WorkerJobs:
    def __init__(self, worker, backend, *, kind, notify=lambda *args, **kwargs: None):
        self.worker, self.models, self.kind, self.notify = worker, backend, kind, notify
        self.backend_id = 'qwen-toy-' + kind
        self.requests, self.keys, self.outcomes = {}, {}, {}
        self.pending_submission = None

    @property
    def uncertain_submission(self):
        return self.pending_submission is not None

    def submit(self, request, *, idempotency_key):
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError('A stable request key is required')
        if idempotency_key in self.keys:
            job = self.keys[idempotency_key]
            if self.requests[job] != request:
                raise ValueError('Request key reused with different content')
            return job
        self.validate(request)
        pending = self.pending_submission
        if pending is not None:
            if pending.idempotency_key != idempotency_key or pending.request != request:
                raise ValueError('Resolve the pending submission before changing its request or key')
        else:
            # Do not create a pending submission after the run budget has expired.
            self.worker.deadline.remaining()
            frozen = deepcopy(request)
            self.pending_submission = _PendingSubmission(
                frozen, idempotency_key, {'idempotency_key': idempotency_key, 'request': asdict(frozen)},
            )
        return self.recover_submission(self.worker)

    def recover_submission(self, client, *, attempts=2):
        """Resolve an uncertain POST with its original body/key, never a new key.

        Runtime uses its existing deadline; cleanup supplies an independent
        short-budget client. Keep the frozen submission until job registration.
        """
        pending = self.pending_submission
        if pending is None:
            return None
        if type(attempts) is not int or attempts < 1:
            raise ValueError('Submission attempts must be a positive integer')
        for attempt in range(attempts):
            client.deadline.remaining()
            try:
                raw = client.call('/' + self.kind, deepcopy(pending.payload))
                break
            except (OSError, HTTPException):
                if attempt + 1 == attempts:
                    raise
                self.notify('submission_retry', kind=self.kind, status='uncertain')
        job = JobRef(identifier(raw['job_id']), self.backend_id)
        if job in self.requests and self.requests[job] != pending.request:
            raise ValueError('Worker reused a job ID')
        self.requests[job] = deepcopy(pending.request)
        self.keys[pending.idempotency_key] = job
        self.pending_submission = None
        self.notify('job', job_id=job.job_id, kind=self.kind, status='queued')
        return job

    def poll(self, job):
        if job not in self.requests:
            raise ValueError('Job is not owned by this adapter')
        if job in self.outcomes:
            return deepcopy(self.outcomes[job])
        raw = self.worker.call('/jobs/' + job.job_id)
        if raw.get('job_id') != job.job_id:
            raise ValueError('Worker job identity mismatch')
        status = JobStatus(raw['status'])
        result = self.convert(self.requests[job], raw['result']) if status == JobStatus.SUCCEEDED else None
        outcome = JobOutcome(job, status, result,
                             error='Worker job failed; inspect worker logs' if status == JobStatus.FAILED else None)
        if outcome.terminal:
            self.outcomes[job] = deepcopy(outcome)
        # Progress belongs to this training job, not a later evaluation/application.
        # Missing fields remain missing, including on successful or queued jobs.
        progress = {}
        if self.kind == 'train' and status != JobStatus.QUEUED:
            for field, minimum in (('step', 0), ('total_steps', 1)):
                value = raw.get(field)
                if type(value) is int and value >= minimum:
                    progress[field] = value
            elapsed = raw.get('elapsed_seconds')
            if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
                progress['elapsed_seconds'] = elapsed
            if isinstance(raw.get('metrics'), dict):
                metrics = {key: value for key, value in raw['metrics'].items()
                           if key in ('loss', 'learning_rate')
                           and type(value) in (int, float) and math.isfinite(value)}
                if metrics:
                    progress['metrics'] = metrics
        self.notify('job', job_id=job.job_id, kind=self.kind, status=status.value, **progress)
        return outcome

    def cancel(self, job):
        if job not in self.requests:
            raise ValueError('Job is not owned by this adapter')
        self.worker.call('/cancel', {'job_id': job.job_id})
        self.notify('job', job_id=job.job_id, kind=self.kind, status='cancel_requested')

    def result(self, job):
        outcome = self.outcomes.get(job)
        if outcome is None or outcome.status != JobStatus.SUCCEEDED:
            raise RuntimeError('Job has no successful result; poll first')
        return deepcopy(outcome.result)


class WorkerTrainer(WorkerJobs):
    def __init__(self, worker, backend, **kwargs):
        super().__init__(worker, backend, kind='train', **kwargs)

    def validate(self, request):
        if (request.base != self.models.active or request.method != 'sft' or request.recipe_ref != RECIPE
                or request.data_refs not in ((), (TRAIN_REF,)) or request.trajectory_refs):
            raise ValueError('Toy worker requires the configured base, toy-lora/sft and train.jsonl')

    def convert(self, request, raw):
        revision = identifier(raw['revision'])
        if raw.get('base_revision', request.base.revision) != request.base.revision:
            raise ValueError('Worker training parent revision mismatch')
        path = raw['adapter_path']
        if not isinstance(path, str) or not path.startswith('/') or '..' in PurePosixPath(path).parts:
            raise ValueError('Worker adapter path must be absolute')
        model = ModelRef(request.base.backend_id, request.base.model_id, revision)
        ref = ArtifactRef(f'{revision}-adapter', 'lora_adapter',
                          'toy-worker://' + path, version=revision)
        candidate = Candidate(revision, request.base.revision, {'model': ref}, ('actor',))
        result = TrainingResult(request.base, model, candidate, numeric_metrics(raw['metrics']))
        self.models.register(result)
        return result


class WorkerEvaluator(WorkerJobs):
    def __init__(self, worker, backend, **kwargs):
        super().__init__(worker, backend, kind='evaluate', **kwargs)

    def validate(self, request):
        result = self.models.candidates.get(request.candidate.candidate_id)
        if (result is None or result.candidate != request.candidate or request.protocol_ref != PROTOCOL
                or request.data_refs not in ((), (EVAL_REF,))):
            raise ValueError('Toy evaluation requires a known candidate and heldout-nll on eval.jsonl')

    def convert(self, request, raw):
        if (raw['revision'] != request.candidate.candidate_id
                or raw.get('base_revision', request.candidate.base_version) != request.candidate.base_version):
            raise ValueError('Evaluation revision mismatch')
        metrics = numeric_metrics(raw['metrics'])
        # Missing/nonfinite measurements fail closed. Preserve finite observed
        # measurements, omit unusable delta and let the model reject the candidate.
        required = ('base_nll', 'candidate_nll', 'delta_nll', 'eval_samples')
        if (any(key not in metrics for key in required) or metrics['eval_samples'] <= 0
                or min(metrics['base_nll'], metrics['candidate_nll']) < 0
                or not math.isclose(metrics['delta_nll'], metrics['base_nll'] - metrics['candidate_nll'],
                                    rel_tol=1e-6, abs_tol=1e-6)):
            metrics.pop('delta_nll', None)
            metrics['measurement_valid'] = 0.0
        return EvaluationResult(request.candidate.candidate_id, request.protocol_ref, metrics, ())


class WorkerActivator:
    def __init__(self, backend):
        self.backend, self.activated = backend, {}

    def activate(self, result, *, idempotency_key):
        if idempotency_key in self.activated:
            if self.activated[idempotency_key] != result:
                raise ValueError('Activation key reused with another result')
            return result.model
        known = self.backend.candidates.get(result.model.revision)
        if known != result:
            raise ValueError('Activation requires a registered training result')
        health = self.backend.worker.call('/health')
        if (health.get('ready') is not True or health.get('trained') is not True
                or result.model.revision not in health.get('revisions', [health.get('revision')])):
            raise ValueError('Candidate service is not ready at the expected revision')
        response = self.backend.generate(GenerationRequest('actor', result.model,
            'Return only JSON with key "parity" and value "even" or "odd" for integer 114.',
            system='Return only the requested JSON.', max_output_tokens=80))
        if not response.revision_verified:
            raise ValueError('Activation inference did not verify the candidate revision')
        value = json.loads(response.text)
        if not isinstance(value, dict) or set(value) != {'parity'} or value['parity'] not in ('even', 'odd'):
            raise ValueError('Candidate inference failed the JSON smoke test')
        self.activated[idempotency_key] = deepcopy(result)
        self.backend.active = result.model
        self.backend.notify('activation_verified', revision=result.model.revision)
        return result.model
