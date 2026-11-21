"""Data contracts for versioned Controllers and parent-run evidence access."""
from copy import deepcopy
from dataclasses import dataclass
import json
import re

from .feedback_context import EVIDENCE_SOURCES
from .model_service import ModelRef


@dataclass(frozen=True)
class ControllerVersion:
    version_id: str
    config: dict
    model: ModelRef | None = None
    memory_state: object = None
    policy_state: object = None

    def __post_init__(self):
        if not isinstance(self.version_id, str) or not self.version_id.strip():
            raise ValueError('Controller version identity must be a nonempty string')
        if (not isinstance(self.config, dict) or set(self.config) - {'name', 'config'}
                or not isinstance(self.config.get('name'), str)
                or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_.-]*', self.config['name'])
                or not isinstance(self.config.get('config', {}), dict)):
            raise ValueError('Controller version config must select a registered name and config')
        config = {'name': self.config['name'], 'config': self.config.get('config', {})}
        object.__setattr__(self, 'config', json.loads(json.dumps(config, allow_nan=False)))
        if self.model is not None and (not isinstance(self.model, ModelRef)
                or any(not isinstance(value, str) or not value.strip()
                       for value in (self.model.backend_id, self.model.model_id, self.model.revision))):
            raise ValueError('Controller version model must be an explicit ModelRef')
        object.__setattr__(self, 'memory_state', deepcopy(self.memory_state))
        object.__setattr__(self, 'policy_state', deepcopy(self.policy_state))


@dataclass(frozen=True)
class EvidenceScope:
    sources: frozenset[str] = EVIDENCE_SOURCES
    kinds: frozenset[str] | None = None
    components: frozenset[str] | None = None
    since: str | None = None

    def __post_init__(self):
        for name in ('sources', 'kinds', 'components'):
            value = getattr(self, name)
            if value is None and name != 'sources':
                continue
            if (not isinstance(value, (set, frozenset, tuple, list))
                    or any(not isinstance(item, str) or not item.strip() for item in value)):
                raise ValueError('Evidence scope ' + name + ' must contain nonempty strings')
            object.__setattr__(self, name, frozenset(value))
        if not self.sources <= EVIDENCE_SOURCES:
            raise ValueError('Evidence scope contains an unknown source')
        if self.since is not None and (not isinstance(self.since, str) or not self.since.strip()):
            raise ValueError('Evidence scope since must be an event or checkpoint identity')
