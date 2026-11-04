"""Model invocation backends only; policies own prompt and decision semantics."""

import json

from rsisys.loops.contracts.model_service import BackendBinding, GenerationRequest, GenerationResponse, TokenUsage, ModelRef


class MockModelBackend:
    """Deterministic JSON fixture. Does not execute, learn, or contact a service."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        context = json.loads(request.prompt)
        if request.role == "decision_maker":
            if not context["operation_results"]:
                payload = {"kind": "invoke", "operation": "prepare_data",
                           "arguments": {"text": context["objective"]}}
            else:
                payload = {"kind": "explore", "strategy": "execute the prepared task"}
        else:
            payload = {"name": "echo", "arguments": {"text": context["observation"]}}
        return GenerationResponse(json.dumps(payload), request.model, simulated=True)


class ModelTransportError(RuntimeError):
    pass


class ModelResponseError(ValueError):
    """A rejected response; recovery is opt-in for explicit format failures only."""

    def __init__(self, message, *, reason='unspecified', retryable=False,
                 finish_reason=None, provider_model=None):
        super().__init__(message)
        self.reason = reason
        self.retryable = retryable
        self.finish_reason = finish_reason
        self.provider_model = provider_model



def parse_tool_response(request, message, finish_reason, provider_model):
    """Validate a native function-call envelope without executing its contents.

    Argument schemas and tool effects belong to the consuming library/harness.
    Keep argument strings verbatim for native SDKs and request/response evidence.
    """
    def reject(reason, *, retryable=False):
        raise ModelResponseError('Rejected native tool response: ' + reason,
            reason=reason, retryable=retryable, finish_reason=finish_reason,
            provider_model=provider_model)

    if not isinstance(message, dict):
        reject('malformed_message', retryable=True)
    if message.get('refusal'):
        reject('refusal')
    if finish_reason == 'length':
        reject('output_truncated', retryable=True)
    if finish_reason not in ('stop', 'tool_calls'):
        reject('unexpected_finish_reason')
    calls = message.get('tool_calls')
    calls = [] if calls is None else calls
    if not isinstance(calls, list):
        reject('malformed_tool_calls', retryable=True)
    text = message.get('content')
    if text is not None and not isinstance(text, str):
        reject('empty_or_invalid_content', retryable=True)
    if not calls:
        if (finish_reason == 'tool_calls' or request.tool_choice == 'required'
                or isinstance(request.tool_choice, dict)):
            reject('missing_tool_calls', retryable=True)
        if not isinstance(text, str) or not text.strip():
            reject('empty_or_invalid_content', retryable=True)
        return text, ()
    if request.tool_choice == 'none':
        reject('unexpected_tool_calls')
    if request.parallel_tool_calls is False and len(calls) > 1:
        reject('unexpected_parallel_tool_calls')
    names = {tool['function']['name'] for tool in request.tools}
    selected = (request.tool_choice['function']['name']
                if isinstance(request.tool_choice, dict) else None)
    seen = set()

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate argument field')
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError('Nonfinite JSON argument')

    for call in calls:
        function = call.get('function') if isinstance(call, dict) else None
        if (not isinstance(function, dict) or call.get('type') != 'function'
                or not isinstance(call.get('id'), str) or not call['id'].strip()
                or call['id'] in seen or not isinstance(function.get('name'), str)
                or not isinstance(function.get('arguments'), str)):
            reject('malformed_tool_call', retryable=True)
        seen.add(call['id'])
        if function.get('name') not in names or (selected and function['name'] != selected):
            reject('unexpected_tool_name')
        try:
            arguments = json.loads(function['arguments'], object_pairs_hook=unique_object,
                                   parse_constant=invalid_constant)
            if not isinstance(arguments, dict):
                raise ValueError('Function arguments must be an object')
        except ValueError:
            reject('invalid_tool_arguments', retryable=True)
    return text or '', tuple(calls)


class RemoteModelBackend:
    """Non-streaming Chat Completions transport for a configured HTTP model server.

    The caller's revision is a deployment label unless the server supplies a matching
    model_revision. Provider model identity is always checked; immutable weights are not inferred.
    """

    supports_tool_calling = True

    def __init__(self, binding: BackendBinding):
        from urllib.parse import urlsplit
        if binding.kind not in ("http", "chat_completions"):
            raise ValueError("RemoteModelBackend requires chat_completions or http kind")
        parsed = urlsplit(binding.endpoint or "")
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("A plain HTTP(S) model API base URL is required")
        self.binding = binding

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        import os
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, build_opener, HTTPRedirectHandler
        if not isinstance(request, GenerationRequest) or request.model.backend_id != self.binding.backend_id:
            raise ValueError("Generation request does not match this backend")
        if request.response_schema is not None:
            raise NotImplementedError('This backend does not implement native response_schema; use a supporting adapter')
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.binding.credential_env:
            key = os.environ.get(self.binding.credential_env)
            if not key:
                raise ValueError(f"Missing model credential environment variable: {self.binding.credential_env}")
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": request.model.model_id, "messages": request.chat_messages(),
                   "stream": False, "max_tokens": request.max_output_tokens}
        if self.binding.json_mode and not request.tools:
            payload["response_format"] = {"type": "json_object"}
        payload.update(request.chat_tools())
        url = self.binding.endpoint.rstrip("/") + "/chat/completions"
        http_request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        try:
            with build_opener(NoRedirect).open(http_request, timeout=self.binding.timeout_seconds) as response:
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
            data = json.loads(body)
            if data["model"] != request.model.model_id:
                raise ModelResponseError("Provider model identity differs from the requested model")
            revision = data.get("model_revision")
            if revision is not None and revision != request.model.revision:
                raise ModelResponseError("Provider model revision differs from the requested revision")
            choice = data["choices"][0]
            finish_reason = choice.get('finish_reason')
            message = choice["message"]
            tool_calls = ()
            if request.tools:
                text, tool_calls = parse_tool_response(request, message, finish_reason, data['model'])
            else:
                if finish_reason not in (None, 'stop'):
                    raise ModelResponseError('Model did not finish a complete text response')
                if message.get('refusal') or message.get('tool_calls'):
                    raise ModelResponseError('This policy transport requires a JSON text response')
                text = message['content']
                if not isinstance(text, str) or not text.strip():
                    raise ModelResponseError('Model response content is empty or invalid')
            usage_data = data.get("usage")
            usage = (TokenUsage(usage_data.get("prompt_tokens"), usage_data.get("completion_tokens"))
                     if usage_data is not None else None)
        except ModelResponseError:
            raise
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise ModelResponseError("Malformed model service response") from None
        return GenerationResponse(text, request.model, simulated=False, usage=usage,
                                  provider_model=data["model"], revision_verified=revision is not None,
                                  tool_calls=tool_calls, finish_reason=finish_reason)


def prepare_model_activation(result, activator, *, idempotency_key: str) -> ModelRef:
    """Confirm availability independently of changing a policy's role binding."""
    from copy import deepcopy
    from rsisys.loops.contracts.model_service import ModelRef
    if activator is None:
        if result.simulated:
            return result.model  # Mock references do not represent deployed weights.
        raise RuntimeError('Real training results require a model_activator before binding')
    ready = activator.activate(deepcopy(result), idempotency_key=idempotency_key)
    if not isinstance(ready, ModelRef) or ready != result.model:
        raise ValueError('Activated model must match the requested trained version')
    return ready
