"""Benchmarks that a Lab mounts: each one owns its evaluation protocol.

A Benchmark is a named, frozen task set together with every setting that decides
how it is scored: the verifier, step and token budgets, timeouts. The protocol is
written once, in the benchmark's own entry, and is recorded with its content
hash (`protocol_id`). A Lab mounts benchmarks; it does not own, merge or
override their protocol. A setting a benchmark cannot take is rejected, not
ignored, and mounting order never matters, so the same benchmark has the same
protocol_id on any Lab, in any combination, on one Lab instance or several.

A benchmark also names its runtime (rsisys.loops.contracts.domain_agent): how a
submitted domain agent is run on its tasks. The runtime's name, version, options and
accepted domain agent shape are part of the protocol, so a benchmark is scored by the
same runtime wherever it is mounted.

Mounting several benchmarks reports each one's metrics separately. A Lab never
forms a cross-benchmark total on its own: when more than one benchmark is
mounted, the configuration declares the primary metric with `primary_metric`,
either one benchmark ({"benchmark": name}) or a named aggregate
({"aggregate": "benchmark-mean"}).
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import re


_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


@dataclass(frozen=True)
class Benchmark:
    name: str
    kind: str                      # The Lab family that can run it, such as coding-task or math-qa.
    protocol: Mapping = field(default_factory=dict)   # Complete: every defaulted setting is present.
    source: Mapping = field(default_factory=dict)     # Where its tasks are read from; not part of the protocol.
    runtime: object = field(default=None, compare=False, repr=False)  # The built DomainAgentRuntime, if the kind has one.

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(f'Invalid benchmark name: {self.name!r}')
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError('A benchmark kind is required')
        for label in ('protocol', 'source'):
            value = getattr(self, label)
            if not isinstance(value, Mapping):
                raise TypeError(f'Benchmark {label} must be a mapping')
            try:
                value = json.loads(canonical_json(deepcopy(dict(value))))
            except (TypeError, ValueError) as error:
                raise ValueError(f'Benchmark {label} must be JSON-compatible') from error
            object.__setattr__(self, label, value)

    @property
    def protocol_id(self):
        """Hash of exactly what decides how this benchmark is scored."""
        return sha256(canonical_json({'name': self.name, 'kind': self.kind,
                                      'protocol': self.protocol}).encode()).hexdigest()

    @property
    def identity(self):
        return {'name': self.name, 'kind': self.kind, 'protocol_id': self.protocol_id}


def mount_benchmarks(entries, *, kind, source_keys, protocol_defaults, runtimes=None, default_runtime=None):
    """Benchmarks from a Lab's `benchmarks` mapping, sorted by name.

    Each entry gives its source (source_keys) and may set any protocol field of
    protocol_defaults; unset fields take the default and are recorded. Anything
    else is an error, so a setting cannot silently mean something different on
    another Lab.

    runtimes maps the installed runtime names to factories (name -> callable(options)
    returning an DomainAgentRuntime). An entry may set `runtime` = {"name", "options"}, else
    default_runtime applies; the runtime's identity is recorded in the protocol.
    """
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError('benchmarks must be a nonempty mapping of names to benchmark entries')
    mounted = []
    for name, entry in sorted(entries.items()):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f'Invalid benchmark name: {name!r}')
        if not isinstance(entry, Mapping):
            raise ValueError(f'Benchmark {name!r} must be an object')
        unknown = set(entry) - set(source_keys) - set(protocol_defaults) - ({'runtime'} if runtimes else set())
        if unknown:
            raise ValueError(f'Benchmark {name!r} has unknown settings: {sorted(unknown)}')
        if set(source_keys) - set(entry):
            raise ValueError(f'Benchmark {name!r} requires {sorted(set(source_keys) - set(entry))}')
        for key in source_keys:
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f'Benchmark {name!r} {key} must be a path')
        protocol = {key: entry.get(key, default) for key, default in protocol_defaults.items()}
        if runtimes:
            from .domain_agent import runtime_identity, validate_runtime_config
            chosen = entry.get('runtime', default_runtime)
            rname, options = validate_runtime_config(chosen, known=runtimes)
            protocol['runtime'] = runtime_identity(runtimes[rname](options))
        mounted.append(Benchmark(name, kind, protocol, {key: entry[key] for key in source_keys},
                                 runtimes[rname](options) if runtimes else None))
    return tuple(mounted)


def choose_primary_metric(declared, benchmarks, *, aggregates):
    """('benchmark', name) or ('aggregate', kind): the primary metric a Lab declares.

    One mounted benchmark needs no declaration; several do.
    """
    names = [benchmark.name for benchmark in benchmarks]
    if declared is None:
        if len(names) == 1:
            return 'benchmark', names[0]
        raise ValueError('Mounting several benchmarks requires primary_metric: {"benchmark": name} '
                         f'or {{"aggregate": one of {sorted(aggregates)}}}')
    if not isinstance(declared, Mapping) or len(declared) != 1:
        raise ValueError('primary_metric must be {"benchmark": name} or {"aggregate": kind}')
    (key, value), = declared.items()
    if key == 'benchmark' and value in names:
        return 'benchmark', value
    if key == 'aggregate' and value in aggregates:
        return 'aggregate', value
    raise ValueError(f'primary_metric names an unmounted benchmark or unknown aggregate: {value!r}')
