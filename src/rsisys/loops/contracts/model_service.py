"""Pure model contracts; backends may be remote, locally served, or in-process."""

from dataclasses import dataclass
import json
from typing import Literal, Protocol, runtime_checkable

Role = Literal["decision_maker", "actor"]


@dataclass(frozen=True)
class ModelRef:
    backend_id: str
    model_id: str
    revision: str


@dataclass(frozen=True)
class BackendBinding:
    backend_id: str
    kind: str
    endpoint: str | None = None
    credential_env: str | None = None  # Environment variable name, never a secret.
    timeout_seconds: float = 60
    json_mode: bool = True
    max_response_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        import math
        if not self.backend_id or not self.kind:
            raise ValueError("Backend identity and kind are required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Model timeout must be finite and positive")
        if type(self.max_response_bytes) is not int or self.max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")


@dataclass(frozen=True)
class GenerationRequest:
    role: Role
    model: ModelRef
    prompt: str
    system: str = ""
    max_output_tokens: int = 2048
    response_schema: dict | None = None  # Optional native JSON Schema constraint.
    messages: tuple[dict, ...] | None = None  # Explicit Chat Completions history.
    tools: tuple[dict, ...] = ()  # Native function declarations, not executable bindings.
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None

    def __post_init__(self):
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.response_schema is not None:
            if not isinstance(self.response_schema, dict):
                raise TypeError('response_schema must be a JSON Schema object or None')
            # Copy caller-owned data and reject non-JSON schema values early.
            object.__setattr__(self, 'response_schema', json.loads(
                json.dumps(self.response_schema, allow_nan=False)))
        if self.messages is not None:
            if self.prompt or self.system:
                raise ValueError('Use either messages or prompt/system, not both')
            if not isinstance(self.messages, (list, tuple)) or not self.messages:
                raise ValueError('messages must be a nonempty sequence')
            messages = json.loads(json.dumps(self.messages, allow_nan=False))
            for message in messages:
                if (not isinstance(message, dict) or message.get('role') not in
                        ('system', 'developer', 'user', 'assistant', 'tool')):
                    raise ValueError('Invalid chat message role')
                if not isinstance(message.get('content'), str):
                    if not (message['role'] == 'assistant' and message.get('content') is None
                            and message.get('tool_calls')):
                        raise ValueError('Only text messages and assistant function calls are supported')
                if message['role'] == 'tool' and (not isinstance(message.get('tool_call_id'), str)
                                                or not message['tool_call_id'].strip()):
                    raise ValueError('Tool results require tool_call_id')
            object.__setattr__(self, 'messages', tuple(messages))
        if not isinstance(self.tools, (list, tuple)):
            raise TypeError('tools must be a sequence of function declarations')
        tools = json.loads(json.dumps(self.tools, allow_nan=False))
        names = set()
        for tool in tools:
            function = tool.get('function') if isinstance(tool, dict) else None
            if (not isinstance(function, dict) or tool.get('type') != 'function'
                    or not isinstance(function.get('name'), str) or not function['name'].strip()
                    or not isinstance(function.get('parameters', {}), dict)):
                raise ValueError('Invalid function declaration')
            if function['name'] in names:
                raise ValueError('Duplicate function declaration')
            names.add(function['name'])
        object.__setattr__(self, 'tools', tuple(tools))
        if tools and self.response_schema is not None:
            raise ValueError('Native tools and response_schema cannot be combined')
        choice = self.tool_choice
        if choice is not None:
            if not tools:
                raise ValueError('tool_choice requires tools')
            if isinstance(choice, str):
                if choice not in ('auto', 'none', 'required'):
                    raise ValueError('Invalid tool_choice')
            elif isinstance(choice, dict):
                choice = json.loads(json.dumps(choice, allow_nan=False))
                if (choice.get('type') != 'function' or not isinstance(choice.get('function'), dict)
                        or choice['function'].get('name') not in names):
                    raise ValueError('Named tool_choice must reference a declared function')
                object.__setattr__(self, 'tool_choice', choice)
            else:
                raise TypeError('Invalid tool_choice')
        if self.parallel_tool_calls is not None:
            if type(self.parallel_tool_calls) is not bool or not tools:
                raise ValueError('parallel_tool_calls requires tools and a boolean')

    def chat_messages(self) -> list[dict]:
        """Detached standard history; legacy prompt/system keep their wire shape."""
        if self.messages is not None:
            return json.loads(json.dumps(self.messages, allow_nan=False))
        messages = [{'role': 'system', 'content': self.system}] if self.system else []
        return messages + [{'role': 'user', 'content': self.prompt}]

    def chat_tools(self) -> dict:
        if not self.tools:
            return {}
        options = {'tools': json.loads(json.dumps(self.tools, allow_nan=False))}
        if self.tool_choice is not None:
            options['tool_choice'] = json.loads(json.dumps(self.tool_choice))
        if self.parallel_tool_calls is not None:
            options['parallel_tool_calls'] = self.parallel_tool_calls
        return options


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self):
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token counts must be nonnegative integers or unknown")


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model: ModelRef
    simulated: bool
    usage: TokenUsage | None = None
    provider_model: str | None = None
    revision_verified: bool = False
    tool_calls: tuple[dict, ...] = ()  # Standard wire calls; caller owns tool execution.
    finish_reason: str | None = None

    def __post_init__(self):
        if not isinstance(self.tool_calls, (list, tuple)):
            raise TypeError('tool_calls must be a sequence')
        object.__setattr__(self, 'tool_calls', tuple(json.loads(
            json.dumps(self.tool_calls, allow_nan=False))))


class ModelBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


@runtime_checkable
class ModelUpdatable(Protocol):
    """Optional policy capability, required only when applying learned versions."""
    model: ModelRef

    def update_model(self, model: ModelRef) -> None: ...
