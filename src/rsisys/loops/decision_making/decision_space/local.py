"""Basic local operations. Literature comes from explicitly imported artifacts."""

import json
from rsisys.runtime.artifacts import FileArtifactStore
from . import ModularDecisionSpace, OperationSpec, OperationResult


class PrepareData:
    spec = OperationSpec('prepare_data', 'Store text as a reusable input artifact', {
        'type': 'object', 'properties': {'text': {'type': 'string'}},
        'required': ['text'], 'additionalProperties': False,
    })

    def __init__(self, artifacts):
        self.artifacts = artifacts

    def execute(self, arguments):
        if set(arguments) != {'text'} or not isinstance(arguments['text'], str):
            raise ValueError('prepare_data requires exactly one string: text')
        ref = self.artifacts.put(arguments['text'].encode(), kind='input_data')
        return OperationResult('Prepared input artifact', (ref,), False)


class SearchLiterature:
    spec = OperationSpec('search_literature', 'Search imported text literature, not the internet', {
        'type': 'object', 'properties': {'query': {'type': 'string'}},
        'required': ['query'], 'additionalProperties': False,
    })

    def __init__(self, artifacts):
        self.artifacts = artifacts

    def execute(self, arguments):
        if set(arguments) != {'query'} or not isinstance(arguments['query'], str) or not arguments['query'].strip():
            raise ValueError('search_literature requires a nonempty query')
        matches = []
        for ref in self.artifacts.refs():
            if ref.kind != 'literature':
                continue
            text = self.artifacts.get(ref).decode('utf-8')
            position = text.casefold().find(arguments['query'].casefold())
            if position >= 0:
                matches.append({'artifact_id': ref.artifact_id, 'excerpt': text[max(0, position-100):position+900]})
            if len(matches) == 20:
                break
        result = self.artifacts.put(json.dumps(matches, ensure_ascii=False).encode(), kind='literature_results')
        return OperationResult(f'Found {len(matches)} local literature matches', (result,), False)


class StoreStrategy:
    spec = OperationSpec('store_strategy', 'Save an explicit strategy as an artifact', {
        'type': 'object', 'properties': {'strategy': {'type': 'string'}},
        'required': ['strategy'], 'additionalProperties': False,
    })

    def __init__(self, artifacts):
        self.artifacts = artifacts

    def execute(self, arguments):
        if set(arguments) != {'strategy'} or not isinstance(arguments['strategy'], str) or not arguments['strategy'].strip():
            raise ValueError('store_strategy requires a nonempty strategy')
        ref = self.artifacts.put(arguments['strategy'].encode(), kind='strategy')
        return OperationResult('Stored strategy', (ref,), False)


class LocalDecisionSpace(ModularDecisionSpace):
    def __init__(self, artifacts: FileArtifactStore):
        self.artifacts = artifacts
        super().__init__((PrepareData(artifacts), SearchLiterature(artifacts), StoreStrategy(artifacts)))
