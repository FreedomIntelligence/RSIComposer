"""Fixed-model access and isolated execution for Python harnesses.

The subprocess sees only harness source, the current problem and its own trial
history. Dataset files, verifiers, other trials, host paths and network are absent.
"""

import ast
import hashlib
import itertools
import json
import subprocess
import threading
import urllib.request
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


class HarnessExecutionError(ValueError):
    pass


def validate_source(source):
    if not isinstance(source, str) or len(source) > 64000:
        raise ValueError("Harness source must be text of at most 64000 characters")
    tree = ast.parse(source)
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "step"
    ]
    if len(functions) != 1 or len(functions[0].args.args) != 2:
        raise ValueError("Harness must define step(problem, history)")


def sandbox_step(harness, problem, history, *, timeout=5):
    runner = Path(__file__).with_name("sandbox.py")
    source = Path(harness).resolve(strict=True)
    command = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(source),
        "/harness.py",
        "--ro-bind",
        str(runner),
        "/runner.py",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin",
        "--",
        "/usr/bin/python3",
        "-I",
        "/runner.py",
    ]
    process = subprocess.run(
        command,
        input=json.dumps({"problem": problem, "history": history}),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if process.returncode:
        raise HarnessExecutionError(process.stderr[-1500:])
    try:
        result = json.loads(process.stdout)
        action = result["action"]
        if not isinstance(action, dict):
            raise ValueError("action is not an object")
        if set(action) == {"answer"} and isinstance(action["answer"], str):
            return result
        messages = action["messages"]
        if not isinstance(messages, list) or not messages or len(messages) > 20:
            raise ValueError("invalid messages")
        if any(
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or m["role"] not in ("system", "user", "assistant")
            or not isinstance(m["content"], str)
            for m in messages
        ):
            raise ValueError("invalid message shape")
        if len(json.dumps(messages)) > 18000:
            raise ValueError("message window too large")
        return result
    except (ValueError, KeyError, TypeError) as exc:
        raise HarnessExecutionError("Invalid harness action: " + str(exc)) from exc
