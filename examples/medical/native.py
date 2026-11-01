"""Explicit medical-to-native training and candidate serving; no tokenizer imports.

MedicalTrainingBackend(root, artifacts, *, tokenize, conversion_id,
                       recipe_template, process_backend=None)
uses tokenize(tuple[Sample, ...]) -> iterable[dict] and a trusted native recipe.
Pass learner.bind_inputs to build(bind_training_inputs=...). MedicalServing is
both build(inference=..., system_activator=..., prepare_inference=...).
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from rsisys.adapters.training import ProcessTrainingBackend, _json, _lock, _write, _result
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.backend import DeploymentStatus, validate_training_capabilities
from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.training import TrainingOrigin, ResumeMode, ModelActivationPending
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.runtime.files import atomic_write
from .task import samples, encoded


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _request(data):
    fields = dict(data)
    fields['base'] = ModelRef(**fields['base'])
    fields['data_refs'] = tuple(ArtifactRef(**ref) for ref in fields['data_refs'])
    fields['trajectory_refs'] = tuple(fields.get('trajectory_refs', ()))
    if fields.get('origin'):
        origin = dict(fields['origin'])
        origin['initial_model'] = ModelRef(**origin['initial_model'])
        origin['resume_mode'] = ResumeMode(origin['resume_mode'])
        if origin.get('checkpoint'):
            origin['checkpoint'] = ArtifactRef(**origin['checkpoint'])
        fields['origin'] = TrainingOrigin(**origin)
    return ActorTrainingRequest(**fields)


class MedicalTrainingBackend:
    """Durable domain wrapper around the local ProcessTrainingBackend.

    Caller requests and origins stay unchanged. Only the internal native request
    uses tokenized file references. The conversion identity is a trusted operator
    label for tokenizer/version/template semantics, not inferred from Python code.
    Heavy frameworks remain in the ProcessTrainingBackend's optional worker.
    """

    def __init__(self, root, artifacts, *, tokenize, conversion_id, recipe_template,
                 process_backend=None):
        if not callable(tokenize) or not isinstance(conversion_id, str) or not conversion_id.strip():
            raise ValueError('Explicit tokenize callback and conversion_id are required')
        if not isinstance(recipe_template, dict):
            raise TypeError('recipe_template must be a trusted native config dictionary')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts, self.tokenize, self.conversion_id = artifacts, tokenize, conversion_id
        self.template = json.loads(_json(recipe_template))
        self.template_digest = _hash(encoded(self.template))
        self.recipe_ref = 'medical-native:' + self.template_digest
        self.process = process_backend or ProcessTrainingBackend(self.root / 'process')
        if not isinstance(self.process, ProcessTrainingBackend):
            raise TypeError('process_backend must be a ProcessTrainingBackend')
        self.backend = 'medical:' + _hash(str(self.root).encode())

    def capabilities(self):
        return self.process.capabilities()

    def bind_inputs(self, request, artifacts):
        """Build hook: validate source bytes; conversion happens once at submit."""
        if not isinstance(request, ActorTrainingRequest) or not request.data_refs:
            raise ValueError('Medical training requires ActorTrainingRequest with source refs')
        if request.recipe_ref != self.recipe_ref:
            raise ValueError(f'Medical training requires the configured recipe_ref: {self.recipe_ref}')
        validate_training_capabilities(self.capabilities(), method=request.method, origin=request.origin)
        for ref in request.data_refs:
            samples(artifacts.get(ref))
        if self.template.get('initial_model', asdict(request.base)) != asdict(request.base):
            raise ValueError('Template initial_model differs from request.base')
        return request

    def _load(self, job):
        if (job.backend != self.backend or len(job.job_id) != 64
                or any(c not in '0123456789abcdef' for c in job.job_id)):
            raise ValueError('Foreign medical job reference')
        directory = self.root / job.job_id
        receipt = json.loads((directory / 'prepared.json').read_text())
        if receipt['conversion_id'] != self.conversion_id or receipt['template_sha256'] != self.template_digest:
            raise ValueError('Reattach requires the original conversion identity and recipe template')
        return directory, receipt

    def submit(self, request, *, idempotency_key):
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError('An idempotency key is required')
        self.bind_inputs(request, self.artifacts)
        original = json.loads(_json(asdict(request)))
        job = JobRef(_hash(idempotency_key.encode()), self.backend)
        directory = self.root / job.job_id
        with _lock(self.root / '.submit.lock'):
            directory.mkdir(exist_ok=True)
            prepared = directory / 'prepared.json'
            if prepared.exists():
                _, receipt = self._load(job)
                if receipt['request'] != original:
                    raise ValueError('Medical idempotency key reused with a different request')
                native_request = _request(receipt['native_request'])
            else:
                rows, sources = [], []
                for ref in request.data_refs:
                    content = self.artifacts.get(ref)
                    parsed = samples(content)
                    rows.extend(parsed)
                    sources.append({'ref': asdict(ref), 'sha256': _hash(content), 'rows': len(parsed)})
                if len({row.id for row in rows}) != len(rows):
                    raise ValueError('Medical source IDs must be unique across training refs')
                features = list(self.tokenize(tuple(rows)))
                if not features or any(not isinstance(row, dict) or 'input_ids' not in row
                                       or 'labels' not in row for row in features):
                    raise ValueError('tokenize must return native feature dictionaries with input_ids and labels')
                tokenized = b''.join(encoded(row) + b'\n' for row in features)
                native_file = directory / 'tokenized.jsonl'
                atomic_write(native_file, tokenized)
                native_ref = ArtifactRef(job.job_id, 'tokenized_rows', native_file.as_uri(), digest=_hash(tokenized))
                config = json.loads(_json(self.template))
                config.update(initial_model=asdict(request.base), train_dataset=str(native_file))
                recipe = directory / 'recipe.json'
                _write(recipe, config)
                native_request = replace(request, data_refs=(native_ref,), recipe_ref=str(recipe))
                receipt = {'request': original, 'native_request': asdict(native_request),
                    'conversion_id': self.conversion_id, 'template_sha256': self.template_digest,
                    'sources': sources, 'source_rows': len(rows), 'source_ids': [row.id for row in rows],
                    'native_rows': len(features), 'native_data_sha256': _hash(tokenized),
                    'native_recipe_sha256': _hash(recipe.read_bytes())}
                _write(prepared, receipt)
            # A lost response is retried with the same native request/key. No
            # re-tokenization and no second native process on a repeated submit.
            process_job = self.process.submit(native_request, idempotency_key=idempotency_key)
            receipt['process_job'] = asdict(process_job)
            _write(prepared, receipt)
        return job

    def _import_result(self, directory, receipt, result):
        if result.base != ModelRef(**receipt['request']['base']):
            raise ValueError('Native result changed the caller training base')
        origin = receipt['request'].get('origin')
        if origin and result.candidate.parent_system != origin['parent_system']:
            raise ValueError('Native result changed the caller experimental parent')
        changes = {}
        for name, ref in result.candidate.changes.items():
            if ref.kind == 'training_evidence':
                from urllib.parse import urlsplit, unquote
                parsed = urlsplit(ref.uri)
                if parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost'):
                    raise ValueError('Native training evidence must be a local file')
                content = Path(unquote(parsed.path)).read_bytes()
                if ref.digest and ref.digest.removeprefix('sha256:') != _hash(content):
                    raise ValueError('Native training evidence digest mismatch')
                json.loads(content)  # Published evidence is readable JSON, not weights.
                changes[name] = self.artifacts.put(content, kind=ref.kind)
            else:
                changes[name] = self.artifacts.put(encoded({'native_ref': asdict(ref)}), kind='native_model_manifest')
        native_directory = self.process.root / receipt['process_job']['job_id']
        job_evidence = {name: json.loads((native_directory / name).read_text())
            for name in ('receipt.json', 'state.json', 'request.json', 'progress.json', 'native-inputs.json')
            if (native_directory / name).exists()}
        provenance = {**receipt, 'native_result': asdict(result), 'native_job_evidence': job_evidence,
            'native_job_directory': str(native_directory),
            'published_native_evidence': {k: asdict(v) for k, v in changes.items()}}
        changes['medical_training'] = self.artifacts.put(encoded(provenance), kind='medical_training_evidence')
        return replace(result, candidate=replace(result.candidate, changes=changes))

    def poll(self, job):
        directory, receipt = self._load(job)
        if 'process_job' not in receipt:
            return JobOutcome(job, JobStatus.UNKNOWN, error='Native submission unconfirmed; repeat the same submit')
        outcome = self.process.poll(JobRef(**receipt['process_job']))
        if outcome.status != JobStatus.SUCCEEDED:
            return replace(outcome, job=job)
        with _lock(directory / 'result.lock'):
            path, failure = directory / 'result.json', directory / 'failure.json'
            if failure.exists():
                return JobOutcome(job, JobStatus.FAILED, error=json.loads(failure.read_text())['error'])
            try:
                if not path.exists():
                    _write(path, asdict(self._import_result(directory, receipt, outcome.result)))
                result = _result(json.loads(path.read_text()))
            except Exception as exc:
                error = f'Medical native result publication failed: {type(exc).__name__}: {exc}'
                _write(failure, {'error': error})
                return JobOutcome(job, JobStatus.FAILED, error=error, simulated=outcome.simulated)
        return replace(outcome, job=job, result=result)

    def reattach(self, job):
        return self.poll(job)

    def cancel(self, job):
        _, receipt = self._load(job)
        if 'process_job' not in receipt:
            raise RuntimeError('Reconcile native submission before cancellation')
        self.process.cancel(JobRef(**receipt['process_job']))

    def result(self, job):
        directory, _ = self._load(job)
        path = directory / 'result.json'
        if not path.exists():
            raise RuntimeError('Medical job has no validated result; poll first')
        return _result(json.loads(path.read_text()))


class MedicalServing:
    """Pre-evaluation preparation and SystemActivator over explicit deployments.

    Supply this object as inference and system_activator, and its
    prepare_inference method as the evaluator's prepare_inference hook. Static
    models remain read-only. Adapter paths are server-native and operator supplied;
    runtime LoRA loading requires explicit opt-in on the deployment backend.
    """

    def __init__(self, deployments, *, adapter_paths=None, adapter_path=None):
        if adapter_path is not None and not callable(adapter_path):
            raise TypeError('adapter_path must be callable or None')
        self.deployments = deployments
        self.adapter_paths = adapter_paths if adapter_paths is not None else {}
        self.adapter_path = adapter_path
        for model, path in self.adapter_paths.items():
            self._bind_path(model, path)

    def _bind_path(self, model, path):
        if not isinstance(path, str) or not path.strip():
            raise ValueError('Adapter path must be a nonempty server-native path')
        with _lock(self.deployments.root / '.medical-paths.lock'):
            existing = self.deployments.adapter_paths.get(model)
            if existing is not None and existing != path:
                raise ValueError('Model already has a different native adapter path')
            self.deployments.adapter_paths[model] = path

    def prepare_inference(self, system):
        if system.model is None:
            raise ValueError('Medical inference requires a model')
        path = self.adapter_paths.get(system.model)
        resolved = self.adapter_path(system) if self.adapter_path is not None else None
        if path is not None and resolved is not None and path != resolved:
            raise ValueError('Adapter callback conflicts with the explicit path mapping')
        if path is not None or resolved is not None:
            self._bind_path(system.model, path if path is not None else resolved)
        key = 'medical-inference:' + _hash(encoded(asdict(system.model)))
        state = self.deployments.prepare(system.model, idempotency_key=key)
        if state.status in (DeploymentStatus.FAILED, DeploymentStatus.RELEASED):
            raise RuntimeError(f'Medical deployment is {state.status}: {state.error or ""}')
        if not state.ready:
            raise ModelActivationPending(f'Medical deployment {state.deployment.deployment_id}: {state.status}')
        if state.model != system.model:
            raise ValueError('Deployment returned a different candidate model')
        return system

    def activate(self, system, *, idempotency_key):
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError('A system activation idempotency key is required')
        root = self.deployments.root
        with _lock(root / '.medical-activation.lock'):
            path = root / ('medical-activation-' + _hash(idempotency_key.encode()) + '.json')
            manifest = json.loads(_json(asdict(system)))
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError('Activation key reused with a different system')
            if not path.exists():
                _write(path, manifest)
        return self.prepare_inference(system)

    def generate(self, request):
        return self.deployments.backend.generate(request)

    def __call__(self, request):
        return self.generate(request)
