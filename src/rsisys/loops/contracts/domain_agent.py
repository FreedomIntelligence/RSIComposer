"""The domain agent a Lab measures, and the runtime that runs it.

Two parts make up a research system. The RSI policy lives in the Controller
(ControllerDefinition: a policy, with its harness and memory). It decides what to
try next. The domain agent lives in the Lab: the thing that does the domain task
and is under study. It might be an instruction, a configuration, a program, an
agent source tree or a checkpoint. The RSI policy submits and revises domain
agents; the Lab pins each one by content hash, runs it under a fixed protocol and
reports the result. The Lab never proposes, edits or selects a domain agent.

This is unrelated to loops.contracts.candidate.Candidate, which is the
Controller's own proposed component change.

    DomainAgentSpec     What a runtime accepts: its kind, the fields a domain agent
                        may set (every other key is rejected) and the trust level
                        it needs.
    DomainAgentRuntime  name, version, model_access, accepts and run(domain_agent,
                        task, budget) -> Attempt. How one domain agent runs on a
                        benchmark's tasks.
    DomainAgent         A submitted domain agent, pinned by content hash.
    Runtime config      {"name": ..., "options": {...}} in a benchmark entry, next
                        to the budgets. Its identity (name, version, options,
                        model access, spec) is part of the benchmark's protocol_id,
                        so a benchmark is run the same way wherever it is mounted.
                        The domain agent's own content is not: it is the thing
                        under test, and is recorded on each receipt as
                        domain_agent_sha256.

Trust says where a domain agent's content may execute:

    data             Read as data only.
    sandboxed-code   Executed in a sandbox (bwrap, a task container) with no access
                     to the host, models or graders.
    host-code        Imported on a worker host, outside any task sandbox. A Lab runs
                     it only when its config lists "host-code" in `trust`.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Protocol

from .benchmark import canonical_json

KINDS = ('instruction', 'config', 'program', 'agent-tree', 'checkpoint')
TRUST = ('data', 'sandboxed-code', 'host-code')
# How the runtime reaches a model; declared and recorded, not unified.
MODEL_ACCESS = ('none', 'lab-mediated', 'endpoint-in-worker', 'engine')


class DomainAgentRejected(ValueError):
    """The submitted domain agent is outside what the benchmark's runtime accepts."""


@dataclass(frozen=True)
class DomainAgentSpec:
    kind: str
    fields: Mapping = field(default_factory=dict)   # {key: JSON-schema-like {"type": ..., bounds}}
    trust: str = 'data'

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'Domain agent kind must be one of {KINDS}')
        if self.trust not in TRUST:
            raise ValueError(f'Domain agent trust must be one of {TRUST}')
        if not isinstance(self.fields, Mapping):
            raise TypeError('Domain agent fields must be a mapping')
        try:
            fields = json.loads(canonical_json(dict(self.fields)))
        except (TypeError, ValueError) as error:
            raise ValueError('Domain agent fields must be JSON-compatible') from error
        object.__setattr__(self, 'fields', fields)

    def identity(self):
        return {'kind': self.kind, 'fields': self.fields, 'trust': self.trust}

    def check(self, data):
        """Reject keys the spec does not name and values outside their declared type or bounds."""
        if not isinstance(data, Mapping):
            raise DomainAgentRejected(f'A {self.kind} domain agent must be an object')
        extra = set(data) - set(self.fields)
        if extra:
            raise DomainAgentRejected(f'Domain agent sets fields this runtime does not accept: {sorted(extra)}')
        for key, rule in self.fields.items():
            if key in data:
                _check_value(key, data[key], rule)
            elif rule.get('required'):
                raise DomainAgentRejected(f'Domain agent requires {key}')


def _check_value(key, value, rule):
    kind = rule.get('type')
    ok = {'string': isinstance(value, str), 'boolean': type(value) is bool,
          'integer': type(value) is int, 'number': type(value) in (int, float),
          'object': isinstance(value, Mapping), 'array': isinstance(value, list)}.get(kind)
    if ok is None:
        raise ValueError(f'Unknown domain agent field type for {key}: {kind!r}')
    if not ok:
        raise DomainAgentRejected(f'{key} must be {kind}')
    if kind in ('integer', 'number') and (value < rule.get('minimum', value) or value > rule.get('maximum', value)):
        raise DomainAgentRejected(f"{key} must be between {rule.get('minimum')} and {rule.get('maximum')}")
    if kind == 'string' and not rule.get('empty', False) and not value.strip():
        raise DomainAgentRejected(f'{key} must be nonempty text')
    if kind == 'string' and not rule.get('min_length', 0) <= len(value) <= rule.get('max_length', 1 << 30):
        raise DomainAgentRejected(f"{key} must be {rule.get('min_length', 0)}–{rule.get('max_length', 'unbounded')} characters")
    if 'enum' in rule and value not in rule['enum']:
        raise DomainAgentRejected(f"{key} must be one of {rule['enum']}")


@dataclass(frozen=True)
class DomainAgent:
    """A submitted domain agent (the thing under measurement), pinned by content hash."""
    kind: str
    sha256: str
    path: str | None = None         # A file or directory, when the content lives on disk.
    data: Mapping | None = None     # The content itself, when it is plain data.

    @classmethod
    def from_data(cls, kind, data):
        content = json.loads(canonical_json(dict(data)))
        return cls(kind, sha256(canonical_json(content).encode()).hexdigest(), None, content)

    @classmethod
    def from_path(cls, kind, path):
        return cls(kind, path_digest(path), str(Path(path).absolute()), None)

    def identity(self):
        return {'kind': self.kind, 'domain_agent_sha256': self.sha256}

    def verify(self):
        if self.path is not None and path_digest(self.path) != self.sha256:
            raise ValueError('Domain agent changed after it was referenced')


def path_digest(path):
    """Content hash of a file, or of a directory's files, executable bits and names.

    Symlinks, special files, VCS directories and bytecode are rejected or skipped, so two
    trees with the same content have the same hash wherever they live.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Domain agents must not contain symlinks')
    if path.is_file():
        return sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise ValueError('A domain agent must be a file or a directory')
    entries = {}
    for entry in sorted(path.rglob('*')):
        relative = entry.relative_to(path)
        if any(part in {'.git', '__pycache__'} for part in relative.parts) or entry.suffix == '.pyc':
            continue
        if entry.is_symlink():
            raise ValueError(f'Domain agents must not contain symlinks: {relative}')
        if entry.is_file():
            entries[relative.as_posix()] = {'sha256': sha256(entry.read_bytes()).hexdigest(),
                                            'executable': bool(entry.stat().st_mode & 0o111)}
        elif not entry.is_dir():
            raise ValueError(f'Unsupported domain agent entry: {relative}')
    if not entries:
        raise ValueError('A domain agent directory is empty')
    # The same encoding CodingArtifact has always used, so existing pinned trees keep their hashes.
    return sha256(json.dumps(entries, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_domain_agent_data(domain_agent):
    """The domain agent's plain data: its inline content, or the JSON file it points at."""
    if domain_agent.data is not None:
        return domain_agent.data
    path = Path(domain_agent.path)
    if not path.is_file():
        raise DomainAgentRejected(f'A {domain_agent.kind} domain agent with plain data must be a JSON file')
    return json.loads(path.read_text())


class DomainAgentRuntime(Protocol):
    """How one domain agent runs on one task. Members are all a Lab relies on."""
    name: str
    version: str
    model_access: str               # One of MODEL_ACCESS.
    accepts: DomainAgentSpec

    def run(self, domain_agent, task, budget):
        """Run domain_agent on task under budget; return an Attempt (rsisys.labs.math.lab)."""


def runtime_identity(runtime, options):
    """What the benchmark's protocol records about its runtime."""
    return {'name': runtime.name, 'version': runtime.version, 'model_access': runtime.model_access,
            'options': json.loads(canonical_json(dict(options))), 'accepts': runtime.accepts.identity()}


def validate_runtime_config(config, *, known):
    """('name', options) of a benchmark's `runtime` entry, checked against the installed runtimes."""
    if not isinstance(config, Mapping) or not {'name'} <= set(config) <= {'name', 'options'}:
        raise ValueError('runtime must be {"name": ..., "options": {...}}')
    name, options = config['name'], config.get('options', {})
    if name not in known:
        raise ValueError(f'Unknown runtime {name!r}; installed: {sorted(known)}')
    if not isinstance(options, Mapping):
        raise ValueError('runtime options must be an object')
    return name, dict(options)


def allowed_trust(config):
    """The trust levels a Lab config permits. data and sandboxed-code are always allowed."""
    declared = config.get('trust', [])
    if (not isinstance(declared, list) or any(item not in TRUST for item in declared)
            or 'data' in declared or 'sandboxed-code' in declared):
        raise ValueError('trust lists only the extra levels to allow: ["host-code"]')
    return {'data', 'sandboxed-code', *declared}


def require_trust(spec, allowed, *, where):
    if spec.trust not in allowed:
        raise ValueError(f'{where} runs domain agents as {spec.trust}; allow it with trust = ["{spec.trust}"] in the Lab config')
