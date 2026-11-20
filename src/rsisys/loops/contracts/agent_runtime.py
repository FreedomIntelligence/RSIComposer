"""What a Lab measures (the subject) and the runtimes that run it.

Not to be confused with loops.contracts.candidate.Candidate, the Controller's own
proposed component changes. Here the subject is the thing under study: an instruction, a configuration, a program
or a whole agent source tree. A Controller submits it; the Lab fixes it by content
hash and never proposes, edits or selects it. A Runtime says how one candidate is
run on one task. It belongs to a benchmark (rsisys.loops.contracts.benchmark), next
to the budgets, so a benchmark is scored by the same runtime wherever it is mounted.

    CandidateSpec   What a runtime accepts: its kind, the fields a candidate may set
                    (every other key is rejected) and the trust level it needs.
    AgentRuntime    name, version, accepts and run(candidate, task, budget) -> Attempt.
    Runtime config  {"name": ..., "version": ..., "options": {...}} in a benchmark entry.
                    Its identity (name, version, options, spec) is part of the
                    benchmark's protocol_id; the candidate's content is not, and is
                    recorded on each receipt as candidate_sha256.

Trust says where candidate content may execute:

    data             Read as data only.
    sandboxed-code   Executed in a sandbox (bwrap, a task container) with no access to
                     the host, models or graders.
    host-code        Imported on a worker host, outside any task sandbox. A Lab runs it
                     only when its config lists "host-code" in `trust`.
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


class CandidateRejected(ValueError):
    """The submitted candidate is outside what the benchmark's runtime accepts."""


@dataclass(frozen=True)
class CandidateSpec:
    kind: str
    fields: Mapping = field(default_factory=dict)   # {key: JSON-schema-like {"type": ..., bounds}}
    trust: str = 'data'

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'Candidate kind must be one of {KINDS}')
        if self.trust not in TRUST:
            raise ValueError(f'Candidate trust must be one of {TRUST}')
        if not isinstance(self.fields, Mapping):
            raise TypeError('Candidate fields must be a mapping')
        try:
            fields = json.loads(canonical_json(dict(self.fields)))
        except (TypeError, ValueError) as error:
            raise ValueError('Candidate fields must be JSON-compatible') from error
        object.__setattr__(self, 'fields', fields)

    def identity(self):
        return {'kind': self.kind, 'fields': self.fields, 'trust': self.trust}

    def check(self, data):
        """Reject keys the spec does not name and values outside their declared type or bounds."""
        if not isinstance(data, Mapping):
            raise CandidateRejected(f'A {self.kind} candidate must be an object')
        extra = set(data) - set(self.fields)
        if extra:
            raise CandidateRejected(f'Candidate sets fields this runtime does not accept: {sorted(extra)}')
        for key, rule in self.fields.items():
            if key in data:
                _check_value(key, data[key], rule)
            elif rule.get('required'):
                raise CandidateRejected(f'Candidate requires {key}')


def _check_value(key, value, rule):
    kind = rule.get('type')
    ok = {'string': isinstance(value, str), 'boolean': type(value) is bool,
          'integer': type(value) is int, 'number': type(value) in (int, float),
          'object': isinstance(value, Mapping), 'array': isinstance(value, list)}.get(kind)
    if ok is None:
        raise ValueError(f'Unknown candidate field type for {key}: {kind!r}')
    if not ok:
        raise CandidateRejected(f'{key} must be {kind}')
    if kind in ('integer', 'number') and (value < rule.get('minimum', value) or value > rule.get('maximum', value)):
        raise CandidateRejected(f"{key} must be between {rule.get('minimum')} and {rule.get('maximum')}")
    if kind == 'string' and not rule.get('empty', False) and not value.strip():
        raise CandidateRejected(f'{key} must be nonempty text')
    if kind == 'string' and not rule.get('min_length', 0) <= len(value) <= rule.get('max_length', 1 << 30):
        raise CandidateRejected(f"{key} must be {rule.get('min_length', 0)}–{rule.get('max_length', 'unbounded')} characters")
    if 'enum' in rule and value not in rule['enum']:
        raise CandidateRejected(f"{key} must be one of {rule['enum']}")


@dataclass(frozen=True)
class Subject:
    """A submitted candidate (the thing under measurement) pinned by content hash."""
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
        return {'kind': self.kind, 'candidate_sha256': self.sha256}

    def verify(self):
        if self.path is not None and path_digest(self.path) != self.sha256:
            raise ValueError('Candidate changed after it was referenced')


def path_digest(path):
    """Content hash of a file, or of a directory's files, executable bits and names.

    Symlinks, special files, VCS directories and bytecode are rejected or skipped, so two
    trees with the same content have the same hash wherever they live.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Candidates must not contain symlinks')
    if path.is_file():
        return sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise ValueError('A candidate must be a file or a directory')
    entries = {}
    for entry in sorted(path.rglob('*')):
        relative = entry.relative_to(path)
        if any(part in {'.git', '__pycache__'} for part in relative.parts) or entry.suffix == '.pyc':
            continue
        if entry.is_symlink():
            raise ValueError(f'Candidates must not contain symlinks: {relative}')
        if entry.is_file():
            entries[relative.as_posix()] = {'sha256': sha256(entry.read_bytes()).hexdigest(),
                                            'executable': bool(entry.stat().st_mode & 0o111)}
        elif not entry.is_dir():
            raise ValueError(f'Unsupported candidate entry: {relative}')
    if not entries:
        raise ValueError('A candidate directory is empty')
    # The same encoding CodingArtifact has always used, so existing pinned trees keep their hashes.
    return sha256(json.dumps(entries, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_subject_data(subject):
    """The subject's plain data: its inline content, or the JSON file it points at."""
    if subject.data is not None:
        return subject.data
    path = Path(subject.path)
    if not path.is_file():
        raise CandidateRejected(f'A {subject.kind} candidate with plain data must be a JSON file')
    return json.loads(path.read_text())


class AgentRuntime(Protocol):
    """How one candidate runs on one task. Members are all a Lab relies on."""
    name: str
    version: str
    model_access: str               # One of MODEL_ACCESS.
    accepts: CandidateSpec

    def run(self, candidate, task, budget):
        """Run candidate on task under budget; return an Attempt (rsisys.labs.math.lab)."""


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
        raise ValueError(f'{where} runs candidates as {spec.trust}; allow it with trust = ["{spec.trust}"] in the Lab config')
