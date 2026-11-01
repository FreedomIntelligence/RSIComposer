"""Explicit QueryProvider bindings; never inspect provider internals."""

from copy import deepcopy
from dataclasses import asdict
from functools import partial
from datetime import datetime, timezone
from hashlib import sha256

from rsisys.loops.contracts.queries import QueryProvider, QuerySpec, QueryRequest, QueryResult, MAX_QUERY_OFFSET
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.runtime.artifacts import FileArtifactStore
from ...action_space import OperationResult
from ...action_space.queries import query_definitions


class ArtifactReader:
    """Read only references registered with the injected store; never dereference a URI.

    Text response pages contain <=16 chunks of 1000 characters. File stores use
    verified resolve + bounded reads; generic get-only stores own input-size I/O.
    """
    max_input_bytes = 1024 * 1024
    spec = QuerySpec('artifact_read',
        'Read registered artifact text in pages of 1000-character chunks; binary returns only its reference. '
        'Supply artifact_id, kind, uri and any version/digest exactly as returned in a known ArtifactRef.',
        ('artifact_id', 'kind', 'uri', 'version', 'digest'), max_limit=16)

    def __init__(self, artifacts):
        self.artifacts = artifacts

    def describe_queries(self):
        return (self.spec,)

    def query(self, request):
        request.validate(self.spec)
        if not {'artifact_id', 'kind', 'uri'} <= set(request.filters):
            raise ValueError('artifact_read requires artifact_id, kind and uri from a known reference')
        ref = ArtifactRef(**request.filters)
        content = None
        if isinstance(self.artifacts, FileArtifactStore):
            path = self.artifacts.resolve(ref)  # Digest validation streams; it does not load the whole file.
            size = path.stat().st_size
            if size <= self.max_input_bytes:
                with path.open('rb') as stream:
                    content = stream.read(self.max_input_bytes + 1)
                size = len(content)
        else:
            content = self.artifacts.get(ref)
            size = len(content)
        if size > self.max_input_bytes:
            return QueryResult(request.name, 'local_artifacts', datetime.now(timezone.utc).isoformat(),
                ref.digest or ref.version or 'unknown',
                {'artifact_id': ref.artifact_id, 'available': False, 'items': [], 'bytes': size,
                 'unavailable_reason': 'Artifact exceeds the 1 MiB text-read limit; reference only'}, (ref,))
        try:
            decoded = content.decode('utf-8')
            binary = any(ord(c) < 32 and c not in '\t\r\n' for c in decoded)
        except UnicodeDecodeError:
            decoded, binary = '', True
        items, next_cursor = [], None
        if not binary:
            start = request.offset * 1000
            end = start + request.limit * 1000
            items = [{'text': decoded[pos:pos + 1000], 'chunk': pos // 1000}
                     for pos in range(start, min(end, len(decoded)), 1000)]
            if end < len(decoded) and request.offset + request.limit <= MAX_QUERY_OFFSET:
                next_cursor = str(request.offset + request.limit)
        data = {'artifact_id': ref.artifact_id, 'available': True, 'items': items,
                'binary': binary, 'bytes': len(content)}
        if binary:
            data['unavailable_reason'] = 'Binary content is reference-only; no text excerpt is available'
        elif not items:
            data['unavailable_reason'] = 'No text in this page'
        return QueryResult(request.name, 'local_artifacts', datetime.now(timezone.utc).isoformat(),
                           sha256(content).hexdigest(), data, (ref,), next_cursor)


class QueryTools:
    def __init__(self, provider: QueryProvider, *, artifacts=None):
        if not isinstance(provider, QueryProvider):
            raise TypeError('Expected an explicit QueryProvider')
        self.provider = provider
        self.artifacts = artifacts
        self.reader = ArtifactReader(artifacts) if artifacts is not None else None
        self.specs = deepcopy(tuple(provider.describe_queries()) + (self.reader.describe_queries() if self.reader else ()))
        self.definitions = query_definitions(self.specs)
        self.bindings = {definition.spec.name: partial(self.execute_query, definition=definition)
                         for definition in self.definitions}

    def describe_queries(self):
        return deepcopy(self.specs)

    def query(self, request: QueryRequest):
        if not isinstance(request, QueryRequest):
            raise TypeError('Expected a QueryRequest')
        spec = next((spec for spec in self.specs if spec.name == request.name), None)
        if spec is None:
            raise ValueError('Query was not advertised by this provider')
        request.validate(spec)
        if self.reader is not None and request.name == self.reader.spec.name:
            return self.reader.query(request)
        return self.provider.query(request)

    def execute_query(self, arguments, *, definition):
        request = definition.request(arguments)
        result = self.query(request)
        if not isinstance(result, QueryResult) or result.name != request.name:
            raise ValueError('Provider returned a result for another query')
        result.__post_init__()
        if isinstance(result.data.get('items'), list) and len(result.data['items']) > request.limit:
            raise ValueError('Provider exceeded the requested query page size')
        return OperationResult(f'Query {result.name} from {result.source}', result.refs, result.simulated,
                               data=deepcopy(asdict(result)))
