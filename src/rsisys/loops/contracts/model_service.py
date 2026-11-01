"""Pure model contracts; backends may be remote, locally served, or in-process."""

from dataclasses import dataclass
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

    def __post_init__(self):
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.response_schema is not None:
            import json
            if not isinstance(self.response_schema, dict):
                raise TypeError('response_schema must be a JSON Schema object or None')
            # Copy caller-owned data and reject non-JSON schema values early.
            object.__setattr__(self, 'response_schema', json.loads(
                json.dumps(self.response_schema, allow_nan=False)))


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


class ModelBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


@runtime_checkable
class ModelUpdatable(Protocol):
    """Optional policy capability, required only when applying learned versions."""
    model: ModelRef

    def update_model(self, model: ModelRef) -> None: ...
