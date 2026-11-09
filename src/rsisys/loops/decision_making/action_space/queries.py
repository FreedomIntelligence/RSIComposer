"""Adapt advertised query capabilities to ordinary read-only decision operations."""

from copy import deepcopy

from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.queries import (
    MAX_QUERY_OFFSET,
    QUERY_CURSOR_PATTERN,
    QueryRequest,
    QuerySpec,
)


class QueryActionDefinition:
    def __init__(self, query_spec: QuerySpec):
        if not isinstance(query_spec, QuerySpec):
            raise TypeError('Expected a QuerySpec')
        query_spec.__post_init__()
        self.query_spec = deepcopy(query_spec)
        self.spec = OperationSpec('query_exploration_' + query_spec.name, query_spec.description, {
            'type': 'object', 'properties': {
                'filters': {'type': 'object', 'properties': {
                    key: {'type': 'string', 'minLength': 1, 'maxLength': 256} for key in query_spec.filters},
                    'additionalProperties': False, 'maxProperties': 8},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': query_spec.max_limit},
                'cursor': {'type': ['string', 'null'], 'pattern': QUERY_CURSOR_PATTERN, 'description':
                           f'Offset cursor returned by the previous page; maximum {MAX_QUERY_OFFSET}'},
            }, 'additionalProperties': False,
        }, read_only=True)

    def request(self, arguments: dict[str, object]) -> QueryRequest:
        if not isinstance(arguments, dict) or set(arguments) - {'filters', 'limit', 'cursor'}:
            raise ValueError('Queries accept only filters, limit and cursor')
        request = QueryRequest(self.query_spec.name, deepcopy(arguments.get('filters', {})),
                               arguments.get('limit', min(20, self.query_spec.max_limit)), arguments.get('cursor'))
        request.validate(self.query_spec)
        return request

    def validate(self, arguments: dict[str, object]) -> OperationCall:
        request = self.request(arguments)
        return OperationCall(self.spec.name, {'filters': request.filters, 'limit': request.limit, 'cursor': request.cursor})


def query_definitions(specs):
    definitions = tuple(QueryActionDefinition(spec) for spec in specs)
    if len({definition.spec.name for definition in definitions}) != len(definitions):
        raise ValueError('Query provider advertised duplicate names')
    return definitions
