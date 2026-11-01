"""Discoverable, read-only queries. Cursors page a live view, not a transaction."""

from dataclasses import dataclass, field, asdict
import json
import re
from typing import Protocol, runtime_checkable

from .artifacts import ArtifactRef

MAX_QUERY_LIMIT = 100
MAX_QUERY_OFFSET = 10000
# Cursors cover 0..10^4, without leading zeroes. Both schema and parser use it.
QUERY_CURSOR_PATTERN = rf'^(0|[1-9][0-9]{{0,{len(str(MAX_QUERY_OFFSET)) - 2}}}|{MAX_QUERY_OFFSET})$'
MAX_QUERY_BYTES = 128 * 1024
_NAME = re.compile(r'[a-z][a-z0-9_]{0,63}\Z')


@dataclass(frozen=True)
class QuerySpec:
    name: str
    description: str
    filters: tuple[str, ...] = ()
    max_limit: int = MAX_QUERY_LIMIT

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError('Query name must be a bounded lowercase identifier')
        if not isinstance(self.description, str) or not self.description or len(self.description) > 2000:
            raise ValueError('Query description must be bounded nonempty text')
        if (not isinstance(self.filters, tuple) or len(set(self.filters)) != len(self.filters)
                or any(not isinstance(k, str) or not _NAME.fullmatch(k) for k in self.filters)):
            raise ValueError('Query filters must be unique identifiers')
        if type(self.max_limit) is not int or not 1 <= self.max_limit <= MAX_QUERY_LIMIT:
            raise ValueError('Query max_limit must be between 1 and 100')


@dataclass(frozen=True)
class QueryRequest:
    name: str
    filters: dict[str, str] = field(default_factory=dict)
    limit: int = 20
    cursor: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError('Invalid query name')
        if (not isinstance(self.filters, dict) or len(self.filters) > 8
                or any(not isinstance(k, str) or not _NAME.fullmatch(k)
                       or not isinstance(v, str) or not v or len(v) > 256 for k, v in self.filters.items())):
            raise ValueError('Query filters must be bounded nonempty strings')
        if type(self.limit) is not int or not 1 <= self.limit <= MAX_QUERY_LIMIT:
            raise ValueError('Query limit must be between 1 and 100')
        if self.cursor is not None and (not isinstance(self.cursor, str)
                or not re.fullmatch(QUERY_CURSOR_PATTERN, self.cursor)
                or int(self.cursor) > MAX_QUERY_OFFSET):
            raise ValueError('Invalid or excessive query cursor')

    @property
    def offset(self) -> int:
        return int(self.cursor) if self.cursor is not None else 0

    def validate(self, spec: QuerySpec) -> None:
        self.__post_init__()
        if self.name != spec.name or set(self.filters) - set(spec.filters) or self.limit > spec.max_limit:
            raise ValueError('Query does not match its advertised capability')


@dataclass(frozen=True)
class QueryResult:
    name: str
    source: str
    observed_at: str
    version: str
    data: dict[str, object] = field(default_factory=dict)
    refs: tuple[ArtifactRef, ...] = ()
    next_cursor: str | None = None
    simulated: bool = False

    def __post_init__(self):
        QueryRequest(self.name, cursor=self.next_cursor)
        if any(not isinstance(v, str) or not v or len(v) > 256
               for v in (self.source, self.observed_at, self.version)):
            raise ValueError('Query source, time and version must be bounded nonempty strings')
        if not isinstance(self.data, dict) or type(self.simulated) is not bool:
            raise TypeError('Query data must be a mapping and simulated a bool')
        if (not isinstance(self.refs, tuple) or len(self.refs) > MAX_QUERY_LIMIT
                or any(not isinstance(r, ArtifactRef) for r in self.refs)):
            raise ValueError('Query refs must be a bounded ArtifactRef tuple')
        try:
            encoded = json.dumps(asdict(self), allow_nan=False).encode()
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError('Query results must contain finite JSON data') from exc
        if len(encoded) > MAX_QUERY_BYTES:
            raise ValueError('Query result exceeds the bounded response size')


@runtime_checkable
class QueryProvider(Protocol):
    def describe_queries(self) -> tuple[QuerySpec, ...]: ...
    def query(self, request: QueryRequest) -> QueryResult: ...
