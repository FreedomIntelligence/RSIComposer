"""One answer to one problem, before verification."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Attempt:
    """One model answer to one problem, before verification.

    status is answered, truncated (the provider stopped at its output limit) or
    invalid_format (no usable text); only answered attempts are verified.
    extra holds a subclass's per-attempt evidence (trial, tool calls, errors),
    which the record keeps alongside the shared fields.
    """
    completion: str | None
    status: str = 'answered'
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    simulated: bool = False
    elapsed_seconds: float = 0.0
    extra: dict = field(default_factory=dict)
