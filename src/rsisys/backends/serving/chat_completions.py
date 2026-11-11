"""Configured Chat Completions transport and deterministic local fixture backend."""
import json

from rsisys.loops.contracts.model_service import (
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
        from urllib.request import HTTPRedirectHandler, Request, build_opener
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
        payload.update(request.sampling_options())
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
