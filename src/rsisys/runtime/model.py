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
    pass


class RemoteModelBackend:
    """Non-streaming Chat Completions transport for a configured HTTP model server.

    The caller's revision is a deployment label unless the server supplies a matching
    model_revision. Provider model identity is always checked; immutable weights are not inferred.
    """

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
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        payload = {"model": request.model.model_id, "messages": messages,
                   "stream": False, "max_tokens": request.max_output_tokens}
        if self.binding.json_mode:
            payload["response_format"] = {"type": "json_object"}
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
            if choice.get("finish_reason") not in (None, "stop"):
                raise ModelResponseError("Model did not finish a complete text response")
            message = choice["message"]
            if message.get("refusal") or message.get("tool_calls"):
                raise ModelResponseError("This policy transport requires a JSON text response")
            text = message["content"]
            if not isinstance(text, str) or not text.strip():
                raise ModelResponseError("Model response content is empty or invalid")
            usage_data = data.get("usage")
            usage = (TokenUsage(usage_data.get("prompt_tokens"), usage_data.get("completion_tokens"))
                     if usage_data is not None else None)
        except ModelResponseError:
            raise
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise ModelResponseError("Malformed model service response") from None
        return GenerationResponse(text, request.model, simulated=False, usage=usage,
                                  provider_model=data["model"], revision_verified=revision is not None)


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
