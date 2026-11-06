"""Host-owned evaluation policy, independent of candidate inference methods."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProtocol:
    protocol_id: str = 'candidate-inference-evaluation-v1'
    score_protocol: str = 'exact-option-letter-v1'
    closed_interval_seconds: int = 1800

    def __post_init__(self):
        if not self.protocol_id or not self.score_protocol:
            raise ValueError('Evaluation protocol identifiers are required')
        if type(self.closed_interval_seconds) is not int or self.closed_interval_seconds < 1800:
            raise ValueError('Closed submissions must be at least 1800 seconds apart')
