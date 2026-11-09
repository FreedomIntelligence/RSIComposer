"""Round-robin OpenAI-compatible text generation for explicitly configured model endpoints."""
import itertools
import json
import threading
import urllib.request


class ChatPool:
    def __init__(
        self, endpoints, model, *, timeout=180, max_tokens=2048, temperature=0.7
    ):
        if not endpoints or not all(
            isinstance(e, str) and e.startswith(("http://", "https://"))
            for e in endpoints
        ):
            raise ValueError("endpoints must be explicit OpenAI-compatible URLs")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be nonempty")
        self.endpoints, self.model = list(endpoints), model
        self.timeout, self.max_tokens, self.temperature = (
            timeout,
            max_tokens,
            temperature,
        )
        self._counter, self._lock = itertools.count(), threading.Lock()

    def complete(
        self,
        messages,
        *,
        seed=73,
        max_tokens=None,
        temperature=None,
        response_schema=None,
    ):
        with self._lock:
            endpoint = self.endpoints[next(self._counter) % len(self.endpoints)]
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": 0.95,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "harness_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        content = result["choices"][0]["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned no text")
        usage = result["usage"]
        return {
            "content": content,
            "tokens": usage["total_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "endpoint": endpoint,
            "finish_reason": result["choices"][0]["finish_reason"],
        }
