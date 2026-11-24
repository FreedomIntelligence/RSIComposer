"""Configured Chat Completions transport and deterministic local fixture backend."""
import json

from rsicomposer.loops.contracts.model_service import (
    BackendBinding,
    GenerationRequest,
    GenerationResponse,
    ModelResponseError,
    ModelTransportError,
    TokenUsage,
)

from .responses import parse_tool_response


class MockModelBackend:
    """Deterministic JSON fixture. Does not execute, learn, or contact a service."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = {'kind': 'finish', 'reason': 'deterministic mock response'}
        return GenerationResponse(json.dumps(payload), request.model, simulated=True)


class RemoteModelBackend:
    """Non-streaming Chat Completions transport for a configured HTTP model server.

    The caller's revision is a deployment label unless the server supplies a matching
    model_revision. Provider model identity is always checked; immutable weights are not inferred.
    """

    supports_tool_calling = True

    def __init__(self, binding: BackendBinding):
        from urllib.parse import urlsplit
        if binding.kind not in self.kinds:
            raise ValueError("Invalid backend kind for this transport")
        parsed = urlsplit(binding.endpoint or "")
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("A plain HTTP(S) model API base URL is required")
        self.binding = binding

    kinds = ('http', 'chat_completions', 'openai_chat_completions')

    @property
    def supports_response_schema(self):
        return self.binding.kind == 'openai_chat_completions'

    def _validate_request(self, request):
        if not isinstance(request, GenerationRequest) or request.model.backend_id != self.binding.backend_id:
            raise ValueError('Generation request does not match this backend')

    def _auth_headers(self, key):
        return {'Authorization': f'Bearer {key}'} if key else {}

    def _post(self, route, payload):
        import os
        from urllib.error import HTTPError, URLError
        from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = None
        if self.binding.credential_env:
            key = os.environ.get(self.binding.credential_env)
            if not key:
                raise ValueError(f"Missing model credential environment variable: {self.binding.credential_env}")
        headers.update(self._auth_headers(key))
        url = self.binding.endpoint.rstrip('/') + '/' + route
        http_request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        handlers = [NoRedirect]
        if (self.binding.credential_env or '').startswith('RSICOMPOSER_NESTED_MODEL_TOKEN_'):
            handlers.append(ProxyHandler({}))
        try:
            with build_opener(*handlers).open(http_request, timeout=self.binding.timeout_seconds) as response:
                body = response.read(self.binding.max_response_bytes + 1)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise ModelTransportError(f"Model service returned HTTP {status}") from None
        except (URLError, TimeoutError, OSError):
            raise ModelTransportError("Model service could not be reached or timed out") from None
        if len(body) > self.binding.max_response_bytes:
            raise ModelResponseError("Model response exceeded the configured size limit")
        try:
            return json.loads(body)
        except (UnicodeError, ValueError):
            raise ModelResponseError('Malformed model service response') from None

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self._validate_request(request)
        if request.response_schema is not None and not self.supports_response_schema:
            raise NotImplementedError('This backend does not implement native response_schema; use a supporting adapter')
        limit = 'max_completion_tokens' if self.supports_response_schema else 'max_tokens'
        payload = {'model': request.model.model_id, 'messages': request.chat_messages(),
                   'stream': False, limit: request.max_output_tokens, **request.sampling_options(), **request.chat_tools()}
        if request.response_schema is not None:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'rsicomposer_response', 'strict': True, 'schema': request.response_schema}}
        elif self.binding.json_mode and not request.tools:
            payload['response_format'] = {'type': 'json_object'}
        data = self._post('chat/completions', payload)
        try:
            if data["model"] != request.model.model_id:
                raise ModelResponseError("Provider model identity differs from the requested model")
            revision = data.get("model_revision")
            if revision is not None and revision != request.model.revision:
                raise ModelResponseError("Provider model revision differs from the requested revision")
            choice = data["choices"][0]
            finish_reason = choice.get('finish_reason')
            message = choice["message"]
            usage_data = data.get("usage")
            usage = (TokenUsage(usage_data.get("prompt_tokens"), usage_data.get("completion_tokens"))
                     if usage_data is not None else None)
            tool_calls = ()
            if request.tools:
                text, tool_calls = parse_tool_response(request, message, finish_reason, data['model'])
            else:
                if finish_reason not in (None, 'stop'):
                    raise ModelResponseError('Model did not finish a complete text response',
                                             reason='output_truncated' if finish_reason == 'length' else 'unspecified',
                                             finish_reason=finish_reason,
                                             provider_model=data['model'], usage=usage)
                if message.get('refusal') or message.get('tool_calls'):
                    raise ModelResponseError('This policy transport requires a JSON text response')
                text = message['content']
                if text is not None and not isinstance(text, str):
                    raise ModelResponseError('Model response content is invalid')
                if text is None or not text.strip():
                    raise ModelResponseError('Model response content is empty or invalid',
                                             reason='empty_response', finish_reason=finish_reason,
                                             provider_model=data['model'], usage=usage)
        except ModelResponseError:
            raise
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise ModelResponseError("Malformed model service response") from None
        return GenerationResponse(text, request.model, simulated=False, usage=usage,
                                  provider_model=data["model"], revision_verified=revision is not None,
                                  tool_calls=tool_calls, finish_reason=finish_reason)
