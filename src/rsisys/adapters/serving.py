"""vLLM OpenAI transport and opt-in, isolated LoRA deployments."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler
from uuid import uuid4

from rsisys.loops.contracts.backend import DeploymentRef, DeploymentState, DeploymentStatus
from rsisys.loops.contracts.model_service import ModelRef, GenerationResponse, TokenUsage
from rsisys.loops.contracts.training import ModelActivationPending
from rsisys.runtime.model import RemoteModelBackend, ModelResponseError, ModelTransportError
from .training import _json, _write, _lock


class VLLMBackend(RemoteModelBackend):
    """Read-only inference by default. Revision labels are never weight attestations.

    binding.endpoint is the API root including /v1. models optionally maps a
    ModelRef to a server alias; aliases may not represent two different revisions.
    """

    supports_response_schema = True  # Transport support; server may reject a schema.

    def __init__(self, binding, models=None, *, generation_options=None):
        super().__init__(replace(binding, kind='chat_completions'))
        self.models = {}
        self._aliases = {}  # Historical pins survive release/removal of active bindings.
        self._bindings_lock = threading.RLock()
        self.generation_options = json.loads(json.dumps(generation_options or {}, allow_nan=False))
        if set(self.generation_options) & {'model', 'messages', 'stream'}:
            raise ValueError('Native options cannot override request identity/messages/stream')
        if 'max_completion_tokens' in self.generation_options:
            raise ValueError('Use max_tokens for this Chat Completions adapter')
        limit = self.generation_options.get('max_tokens')
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('Native max_tokens must be a positive integer')
        for model, alias in (models or {}).items():
            self.bind(model, alias)

    def bind(self, model, alias):
        if model.backend_id != self.binding.backend_id or not isinstance(alias, str) or not alias:
            raise ValueError('Invalid model/alias binding')
        with self._bindings_lock:
            if model in self.models and self.models[model] != alias:
                raise ValueError('Model revision already bound to another alias')
            if alias in self._aliases and self._aliases[alias] != model:
                raise ValueError('Distinct model revisions require isolated serving aliases')
            self.models[model] = alias
            self._aliases[alias] = model

    def _alias(self, model, alias=None):
        """Compatibility: first use pins a static alias; later revisions must register."""
        with self._bindings_lock:
            alias = alias or self.models.get(model, model.model_id)
            self.bind(model, alias)
            return alias

    def generate(self, request):
        if request.model.backend_id != self.binding.backend_id:
            raise ValueError('Generation request does not match this backend')
        alias = self._alias(request.model)
        messages = ([{'role': 'system', 'content': request.system}] if request.system else [])
        messages.append({'role': 'user', 'content': request.prompt})
        payload = {'max_tokens': request.max_output_tokens}
        if self.binding.json_mode:
            payload['response_format'] = {'type': 'json_object'}
        payload.update(self.generation_options)
        if request.response_schema is not None:
            conflicts = {'response_format', 'structured_outputs', 'guided_json', 'guided_regex',
                         'guided_grammar', 'tools', 'tool_choice'} & self.generation_options.keys()
            if conflicts:
                raise ValueError('Request schema conflicts with native generation options: ' + ', '.join(sorted(conflicts)))
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'rsisys_decision', 'strict': True, 'schema': request.response_schema}}
        payload['max_tokens'] = min(request.max_output_tokens,
                                   self.generation_options.get('max_tokens', request.max_output_tokens))
        payload.update(model=alias, messages=messages, stream=False)
        data = self._request('chat/completions', payload)
        try:
            message = data['choices'][0]['message']
            if (data['model'] != alias or data.get('model_revision', request.model.revision) != request.model.revision
                    or data['choices'][0].get('finish_reason') not in ('stop', 'length')
                    or not isinstance(message['content'], str) or not message['content'].strip()
                    or message.get('refusal') or message.get('tool_calls')):
                raise ValueError()
            if request.response_schema is not None:
                if data['choices'][0].get('finish_reason') != 'stop':
                    raise ValueError('Structured response did not finish')
                def unique_object(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError('Duplicate JSON response key')
                        result[key] = value
                    return result
                if not isinstance(json.loads(message['content'], object_pairs_hook=unique_object), dict):
                    raise ValueError('Structured response must be an object')
            usage = data.get('usage')
            usage = TokenUsage(usage.get('prompt_tokens'), usage.get('completion_tokens')) if usage else None
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise ModelResponseError('Malformed or mismatched vLLM generation response') from None
        return GenerationResponse(message['content'], request.model, False, usage, alias,
                                  revision_verified=data.get('model_revision') is not None)

    token_count_mode = 'provider_tokenizer'

    def count_tokens(self, request):
        """Price the actual chat template at vLLM's root /tokenize endpoint."""
        if request.model.backend_id != self.binding.backend_id:
            raise ValueError('Token request does not match backend')
        messages = ([{'role': 'system', 'content': request.system}] if request.system else [])
        messages.append({'role': 'user', 'content': request.prompt})
        payload = {'model': self._alias(request.model), 'messages': messages,
                   'add_generation_prompt': True, 'return_tokens': False,
                   'chat_template_kwargs': self.generation_options.get('chat_template_kwargs', {})}
        result = self._request('tokenize', payload)
        count = result.get('count')
        if type(count) is not int or count < 0:
            raise ModelResponseError('Malformed vLLM token count; configure a compatible token counter')
        return count

    def _request(self, route, payload=None):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        if self.binding.credential_env:
            key = os.environ.get(self.binding.credential_env)
            if not key:
                raise ValueError('Missing model credential environment variable')
            headers['Authorization'] = 'Bearer ' + key
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        # Schema property order affects constrained decoding. Canonical sorting is
        # for artifact identities, never for the inference wire representation.
        endpoint = self.binding.endpoint.rstrip('/')
        if route == 'tokenize' and endpoint.endswith('/v1'):
            endpoint = endpoint[:-3]
        request = Request(endpoint + '/' + route,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(), headers=headers,
            method='GET' if payload is None else 'POST')
        try:
            with build_opener(NoRedirect).open(request, timeout=self.binding.timeout_seconds) as response:
                body = response.read(self.binding.max_response_bytes + 1)
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise ModelTransportError(f'vLLM returned HTTP {code}') from None
        except (OSError, URLError, TimeoutError):
            raise ModelTransportError('vLLM request failed or timed out') from None
        if len(body) > self.binding.max_response_bytes:
            raise ModelResponseError('vLLM response exceeded size limit')
        # vLLM load/unload handlers may return plain text instead of JSON.
        if route in ('load_lora_adapter', 'unload_lora_adapter'):
            return None
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError()
            return data
        except (ValueError, TypeError):
            raise ModelResponseError('Malformed vLLM response') from None

    def list_models(self):
        try:
            rows = self._request('models')['data']
            if not isinstance(rows, list) or any(not isinstance(row.get('id'), str) for row in rows):
                raise ValueError()
            return {row['id']: row for row in rows}
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ModelResponseError('Malformed vLLM model listing') from None

    def probe(self, model, alias=None):
        alias = self._alias(model, alias)
        payload = dict(self.generation_options)
        payload.update(model=alias, messages=[{'role': 'user', 'content': 'Reply OK.'}], stream=False)
        payload['max_tokens'] = 1
        data = self._request('chat/completions', payload)
        try:
            choice = data['choices'][0]
            message = choice['message']
            if (data['model'] != alias or choice.get('finish_reason') not in ('stop', 'length')
                    or not isinstance(message['content'], str) or message.get('refusal')
                    or message.get('tool_calls')):
                raise ValueError()
            if data.get('model_revision', model.revision) != model.revision:
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelResponseError('Readiness probe did not confirm the requested model') from None


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
                _write(path, {'namespace': 'rsisys-' + uuid4().hex,
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
