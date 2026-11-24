"""vLLM OpenAI transport and opt-in, isolated LoRA deployments."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from rsicomposer.backends.serving.native_chat import NativeChatBackend
from rsicomposer.loops.contracts.backend import (
    DeploymentRef,
    DeploymentState,
    DeploymentStatus,
)
from rsicomposer.loops.contracts.model_service import (
    ModelRef,
    ModelResponseError,
    ModelTransportError,
)
from rsicomposer.loops.contracts.training import ModelActivationPending
from rsicomposer.runtime.storage.process_store import _lock, _write


class VLLMBackend(NativeChatBackend):
    """vLLM generation and tokenization."""

    provider = 'vLLM'
    plain_routes = {'load_lora_adapter', 'unload_lora_adapter'}


class VLLMDeploymentBackend:
    """Observe static deployments, or explicitly load owned names on a LoRA engine.

    Does not start/restart the server or replace base/shared adapters. Paths are
    server-native paths. Reattach/query never perform load or unload operations.
    """

    def __init__(self, backend, root, *, adapter_paths=None, allow_runtime_lora=False):
        self.backend = backend
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.adapter_paths = dict(adapter_paths or {})
        self.allow_runtime_lora = allow_runtime_lora
        self.identity = backend.binding.backend_id
        with _lock(self.root / '.lock'):
            path = self.root / 'namespace.json'
            if not path.exists():
                _write(path, {'namespace': 'rsicomposer-' + uuid4().hex,
                    'endpoint': backend.binding.endpoint, 'backend': self.identity})
            namespace = json.loads(path.read_text())
            if namespace['endpoint'] != backend.binding.endpoint or namespace['backend'] != self.identity:
                raise ValueError('Deployment root belongs to another server')
            self.namespace = namespace['namespace']

    def _load(self, deployment):
        if (deployment.backend != self.identity or len(deployment.deployment_id) != 64
                or any(c not in '0123456789abcdef' for c in deployment.deployment_id)):
            raise ValueError('Foreign deployment reference')
        path = self.root / (deployment.deployment_id + '.json')
        return path, json.loads(path.read_text())

    def _state(self, deployment, record, status, error=None):
        return DeploymentState(deployment, ModelRef(**record['model']), status, error=error)

    def prepare(self, model, *, idempotency_key):
        if not idempotency_key or model.backend_id != self.identity:
            raise ValueError('A matching model and idempotency key are required')
        deployment = DeploymentRef(hashlib.sha256(idempotency_key.encode()).hexdigest(), self.identity)
        path = self.root / (deployment.deployment_id + '.json')
        with _lock(self.root / '.lock'):
            native_path = self.adapter_paths.get(model)
            if path.exists():
                record = json.loads(path.read_text())
                if record['model'] != asdict(model) or (native_path is not None and record['path'] != native_path):
                    raise ValueError('Deployment key reused with different model/path')
            else:
                if native_path and not self.allow_runtime_lora:
                    raise NotImplementedError('Runtime LoRA loading requires explicit opt-in')
                record = {'model': asdict(model), 'path': native_path, 'released': False,
                          'alias': self.namespace + '-' + deployment.deployment_id if native_path else
                          self.backend.models.get(model, model.model_id), 'load_attempted': False}
                _write(path, record)
            native_path = record['path']
            if record['released']:
                return self._state(deployment, record, DeploymentStatus.RELEASED)
            if native_path and not record['load_attempted']:
                if not self.allow_runtime_lora:
                    raise NotImplementedError('Runtime LoRA loading requires explicit opt-in')
                if record['alias'] in self.backend.list_models():
                    raise ValueError('Refusing to replace an existing serving alias')
                record['load_attempted'] = True
                _write(path, record)  # Intent survives an ambiguous network response.
                try:
                    self.backend._request('load_lora_adapter',
                        {'lora_name': record['alias'], 'lora_path': native_path})
                except ModelTransportError:
                    return self._state(deployment, record, DeploymentStatus.UNKNOWN,
                                       'LoRA load outcome uncertain; query the same deployment')
        return self.query(deployment)

    def query(self, deployment):
        _, record = self._load(deployment)
        if record['released']:
            return self._state(deployment, record, DeploymentStatus.RELEASED)
        model = ModelRef(**record['model'])
        try:
            rows = self.backend.list_models()
            row = rows.get(record['alias'])
            if row is None:
                return self._state(deployment, record, DeploymentStatus.PREPARING)
            # vLLM exposes a LoRA's root as its native path. Never adopt a name
            # now pointing elsewhere after a server restart/operator replacement.
            if record['path'] and row.get('root') != record['path']:
                return self._state(deployment, record, DeploymentStatus.UNKNOWN, 'LoRA path identity mismatch')
            self.backend.probe(model, record['alias'])
            self.backend.bind(model, record['alias'])
            return self._state(deployment, record, DeploymentStatus.READY)
        except (ModelTransportError, ModelResponseError) as exc:
            return self._state(deployment, record, DeploymentStatus.UNKNOWN, str(exc))

    def reattach(self, deployment):
        return self.query(deployment)

    def release(self, deployment, *, idempotency_key):
        if not idempotency_key:
            raise ValueError('An idempotency key is required')
        with _lock(self.root / '.lock'):
            path, record = self._load(deployment)
            if record['released']:
                return self._state(deployment, record, DeploymentStatus.RELEASED)
            if not record['path']:
                raise NotImplementedError('Static/shared deployments cannot be unloaded')
            if not self.allow_runtime_lora:
                raise NotImplementedError('Runtime LoRA unloading requires explicit opt-in')
            try:
                row = self.backend.list_models().get(record['alias'])
                if row is not None:
                    if row.get('root') != record['path']:
                        return self._state(deployment, record, DeploymentStatus.UNKNOWN, 'LoRA path identity mismatch')
                    self.backend._request('unload_lora_adapter', {'lora_name': record['alias']})
                if record['alias'] in self.backend.list_models():
                    return self._state(deployment, record, DeploymentStatus.RELEASING)
            except (ModelTransportError, ModelResponseError) as exc:
                return self._state(deployment, record, DeploymentStatus.UNKNOWN, str(exc))
            record['released'] = True
            _write(path, record)
            # Keep the route pinned after unload. A future inference must target
            # this removed alias (normally 404), never fall back to base weights.
            return self._state(deployment, record, DeploymentStatus.RELEASED)


class VLLMLoRAActivator:
    def __init__(self, deployments, adapter_path=None):
        self.deployments = deployments
        self.adapter_path = adapter_path

    def activate(self, result, *, idempotency_key):
        if result.simulated:
            raise ValueError('vLLM activation requires a real training result')
        if self.adapter_path is not None:
            path = self.adapter_path(result)
            existing = self.deployments.adapter_paths.get(result.model)
            if existing is not None and existing != path:
                raise ValueError('Model already has another native adapter path')
            self.deployments.adapter_paths[result.model] = path
        state = self.deployments.prepare(result.model, idempotency_key=idempotency_key)
        if state.status == DeploymentStatus.FAILED:
            raise RuntimeError(state.error)
        if not state.ready:
            raise ModelActivationPending(f'Deployment {state.deployment.deployment_id}: {state.status}')
        return state.model
