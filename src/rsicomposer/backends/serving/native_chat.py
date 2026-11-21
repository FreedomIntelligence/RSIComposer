"""Shared native Chat Completions wire protocol."""

import json
import os
import threading
from copy import deepcopy
from dataclasses import replace
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .chat_completions import RemoteModelBackend
from .responses import parse_tool_response
from rsicomposer.loops.contracts.model_service import (
    GenerationResponse, ModelResponseError, ModelTransportError, TokenUsage,
)


class NativeChatBackend(RemoteModelBackend):
    """Read-only inference by default. Revision labels are never weight attestations.

    binding.endpoint is the API root including /v1. models optionally maps a
    ModelRef to a server alias; aliases may not represent two different revisions.
    """

    supports_response_schema = True  # Transport support; server may reject a schema.
    provider = 'model server'
    root_routes = {'tokenize'}
    plain_routes = frozenset()

    def __init__(self, binding, models=None, *, generation_options=None, native_schema_mode='exact',
                 max_tokenize_response_bytes=64 * 1024 * 1024):
        super().__init__(replace(binding, kind='chat_completions'))
        self.binding = binding
        if native_schema_mode not in ('exact', 'portable'):
            raise ValueError('native_schema_mode must be exact or portable')
        if type(max_tokenize_response_bytes) is not int or max_tokenize_response_bytes < 1:
            raise ValueError('max_tokenize_response_bytes must be positive')
        self.max_tokenize_response_bytes = max_tokenize_response_bytes
        self.native_schema_mode = native_schema_mode
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
        with self._bindings_lock:
            if alias is not None:
                self.bind(model, alias)
            try:
                return self.models[model]
            except KeyError:
                raise ValueError('Model revision requires an explicit serving binding') from None

    def native_response_schema(self, schema):
        """Explicit provider dialect; canonical schemas/validators stay unchanged.

        Portable decoding omits string regex/length, item uniqueness and
        conditional refinements on objects with their own properties. The
        caller must still validate the canonical schema/Action Space before
        effects. No failed request is retried with a weaker grammar.
        """
        def convert(node):
            if not isinstance(node, dict):
                return deepcopy(node)
            result = deepcopy(node)
            if self.native_schema_mode == 'exact':
                return result
            for key in ('pattern', 'minLength', 'maxLength', 'uniqueItems'):
                result.pop(key, None)
            if isinstance(result.get('properties'), dict):
                # This compiler treats partial object branches as closed objects,
                # losing sibling fields. Keep the base object shape and defer
                # cross-field refinements to canonical validation. Discriminated
                # envelope unions (no sibling properties) are preserved.
                for key in ('allOf', 'anyOf', 'oneOf'):
                    result.pop(key, None)
            # Walk schema positions only, never literal const/enum/examples data.
            for key in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
                if isinstance(result.get(key), dict):
                    result[key] = {name: convert(value) for name, value in result[key].items()}
            for key in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
                if isinstance(result.get(key), list):
                    result[key] = [convert(value) for value in result[key]]
            for key in ('items', 'additionalProperties', 'contains', 'not', 'if', 'then', 'else',
                        'unevaluatedItems', 'unevaluatedProperties', 'propertyNames'):
                if isinstance(result.get(key), dict):
                    result[key] = convert(result[key])
            return result
        return convert(schema)

    def generate(self, request):
        return self.generate_with_response_observer(request, None)

    def _generation_payload(self, request):
        if request.model.backend_id != self.binding.backend_id:
            raise ValueError('Generation request does not match this backend')
        alias = self._alias(request.model)
        payload = {'max_tokens': request.max_output_tokens}
        if self.binding.json_mode and not request.tools:
            payload['response_format'] = {'type': 'json_object'}
        payload.update(self.generation_options)
        payload.update(request.sampling_options())
        if request.tools:
            conflicts = {'response_format', 'structured_outputs', 'guided_json', 'guided_regex',
                         'guided_grammar', 'tools', 'tool_choice', 'parallel_tool_calls'} & self.generation_options.keys()
            if conflicts:
                raise ValueError('Request tools conflict with native generation options: ' + ', '.join(sorted(conflicts)))
            payload.update(request.chat_tools())
        if request.response_schema is not None:
            conflicts = {'response_format', 'structured_outputs', 'guided_json', 'guided_regex',
                         'guided_grammar', 'tools', 'tool_choice'} & self.generation_options.keys()
            if conflicts:
                raise ValueError('Request schema conflicts with native generation options: ' + ', '.join(sorted(conflicts)))
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'rsisys_decision', 'strict': True,
                'schema': self.native_response_schema(request.response_schema)}}
        payload['max_tokens'] = min(request.max_output_tokens,
                                   self.generation_options.get('max_tokens', request.max_output_tokens))
        payload.update(model=alias, messages=request.chat_messages(), stream=False)
        return payload

    def generate_with_response_observer(self, request, observer):
        """Observer persists bounded wire evidence before parsing or validation.

        The observer is scoped to this call (never shared mutable adapter state).
        A failed evidence write aborts the call; no result reaches the executor.
        """
        payload = self._generation_payload(request)
        alias = payload['model']
        data = (self._request('chat/completions', payload, response_observer=observer)
                if observer is not None else self._request('chat/completions', payload))
        finish_reason = None
        provider_model = data.get('model') if isinstance(data, dict) else None
        try:
            finish_reason = data['choices'][0].get('finish_reason')
        except (KeyError, IndexError, TypeError, AttributeError):
            pass

        def reject(reason, *, retryable=False):
            raise ModelResponseError(f'Rejected {self.provider} generation response: ' + reason,
                reason=reason, retryable=retryable, finish_reason=finish_reason,
                provider_model=provider_model)

        # Identity checks precede recoverable format checks, even on truncated replies.
        if provider_model != alias:
            reject('model_identity_mismatch')
        if data.get('model_revision', request.model.revision) != request.model.revision:
            reject('model_revision_mismatch')
        try:
            message = data['choices'][0]['message']
            if not isinstance(message, dict):
                reject('malformed_message', retryable=True)
            if message.get('refusal'):
                reject('refusal')
            tool_calls = ()
            if request.tools:
                text, tool_calls = parse_tool_response(request, message, finish_reason, provider_model)
            elif message.get('tool_calls'):
                reject('unexpected_tool_calls')
            else:
                if finish_reason not in ('stop', 'length'):
                    reject('unexpected_finish_reason')
                if request.response_schema is not None and finish_reason == 'length':
                    reject('output_truncated', retryable=True)
                if not isinstance(message.get('content'), str) or not message['content'].strip():
                    reject('empty_or_invalid_content', retryable=True)
                text = message['content']
            if request.response_schema is not None:
                def unique_object(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            reject('duplicate_json_key', retryable=True)
                        result[key] = value
                    return result
                try:
                    decoded = json.loads(message['content'], object_pairs_hook=unique_object)
                except json.JSONDecodeError:
                    reject('invalid_json', retryable=True)
                if not isinstance(decoded, dict):
                    reject('nonobject_json', retryable=True)
            usage = data.get('usage')
            usage = TokenUsage(usage.get('prompt_tokens'), usage.get('completion_tokens')) if usage else None
        except ModelResponseError:
            raise
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            reject('malformed_response_fields', retryable=True)
        return GenerationResponse(text, request.model, False, usage, alias,
                                  revision_verified=data.get('model_revision') is not None,
                                  tool_calls=tool_calls, finish_reason=finish_reason)

    token_count_mode = 'provider_tokenizer'

    def count_tokens(self, request):
        """Count using the server's actual chat template."""
        payload = self._tokenize_payload(request)
        result = self._request('tokenize', payload)
        count = result.get('count')
        if type(count) is not int or count < 0:
            raise ModelResponseError(f'Malformed {self.provider} token count')
        return count

    def _tokenize_payload(self, request):
        generation = self._generation_payload(request)
        payload = {'model': generation['model'], 'messages': generation['messages'],
                   'add_generation_prompt': True, 'return_token_strs': False,
                   'chat_template_kwargs': self.generation_options.get('chat_template_kwargs', {})}
        for key in ('tools', 'chat_template'):
            if key in generation:
                payload[key] = generation[key]
        return payload

    def _request(self, route, payload=None, *, response_observer=None):
        # The tokenizers return token IDs. Counting the uncompressed context may
        # exceed the generation response bound before compaction can run.
        response_limit = (self.max_tokenize_response_bytes if route == 'tokenize'
                          else self.binding.max_response_bytes)
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
        if route in self.root_routes and endpoint.endswith('/v1'):
            endpoint = endpoint[:-3]
        request = Request(endpoint + '/' + route,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(), headers=headers,
            method='GET' if payload is None else 'POST')
        handlers = [NoRedirect]
        if (self.binding.credential_env or '').startswith('RSISYS_NESTED_MODEL_TOKEN_'):
            handlers.append(ProxyHandler({}))
        try:
            with build_opener(*handlers).open(request, timeout=self.binding.timeout_seconds) as response:
                body = response.read(response_limit + 1)
        except HTTPError as exc:
            code = exc.code
            delegated_error = None
            try:
                body = exc.read(response_limit + 1)
                if response_observer is not None:
                    response_observer({'body': body.decode('utf-8', errors='replace'),
                        'http_status': code, 'bytes_captured': len(body),
                        'capture_truncated': len(body) > response_limit})
                if (code == 502 and (self.binding.credential_env or '').startswith('RSISYS_NESTED_MODEL_TOKEN_')
                        and len(body) <= response_limit):
                    try:
                        details = json.loads(body)['error']
                        if (details.get('type') == 'ModelResponseError' and isinstance(details.get('reason'), str)
                                and type(details.get('retryable')) is bool):
                            delegated_error = ModelResponseError('Nested model provider rejected its response',
                                reason=details['reason'], retryable=details['retryable'])
                    except (ValueError, TypeError, KeyError, AttributeError):
                        pass
            finally:
                exc.close()
            if delegated_error is not None:
                raise delegated_error from None
            raise ModelTransportError(f'{self.provider} returned HTTP {code}') from None
        except (OSError, URLError, TimeoutError):
            raise ModelTransportError(f'{self.provider} request failed or timed out') from None
        if response_observer is not None:
            response_observer({'body': body.decode('utf-8', errors='replace'),
                               'bytes_captured': len(body),
                               'capture_truncated': len(body) > response_limit})
        if len(body) > response_limit:
            raise ModelResponseError(f'{self.provider} response exceeded size limit', reason='response_size_limit')
        # vLLM load/unload handlers may return plain text instead of JSON.
        if route in self.plain_routes:
            return None
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError()
            return data
        except (ValueError, TypeError):
            raise ModelResponseError(f'Malformed {self.provider} response', reason='invalid_response_envelope',
                                     retryable=True) from None

    def list_models(self):
        try:
            rows = self._request('models')['data']
            if not isinstance(rows, list) or any(not isinstance(row.get('id'), str) for row in rows):
                raise ValueError()
            return {row['id']: row for row in rows}
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ModelResponseError(f'Malformed {self.provider} model listing') from None

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
