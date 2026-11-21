"""Isolated Python step(problem, history) program harness."""
import ast
import json
import subprocess
from pathlib import Path


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
