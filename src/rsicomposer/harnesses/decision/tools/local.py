"""Local execution of decision operations over explicitly imported artifacts."""

import json

from rsicomposer.harnesses.decision.tools import BoundDecisionTools
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.runtime.storage.artifacts import FileArtifactStore


class LocalDecisionTools(BoundDecisionTools):
    def __init__(self, artifacts: FileArtifactStore, *, extra_bindings=None):
        self.artifacts = artifacts
        bindings = {
            "prepare_data": self._prepare_data,
            "search_literature": self._search_literature,
            "store_strategy": self._store_strategy,
        }
        if set(bindings).intersection(extra_bindings or {}):
            raise ValueError('Extra tool bindings must not replace built-in operations')
        super().__init__({**bindings, **(extra_bindings or {})})
        self.query_tools = None

    def _prepare_data(self, arguments: dict[str, object]) -> OperationResult:
        ref = self.artifacts.put(arguments["text"].encode(), kind="input_data")
        return OperationResult("Prepared input artifact", (ref,), False)

    def _search_literature(self, arguments: dict[str, object]) -> OperationResult:
        matches = []
        for ref in self.artifacts.refs():
            if ref.kind != "literature":
                continue
            text = self.artifacts.read(ref).decode("utf-8")
            position = text.casefold().find(arguments["query"].casefold())
            if position >= 0:
                matches.append({"artifact_id": ref.artifact_id,
                                "excerpt": text[max(0, position - 100):position + 900]})
            if len(matches) == 20:
                break
        result = self.artifacts.put(json.dumps(matches, ensure_ascii=False).encode(), kind="literature_results")
        return OperationResult(f"Found {len(matches)} local literature matches", (result,), False)

    def _store_strategy(self, arguments: dict[str, object]) -> OperationResult:
        ref = self.artifacts.put(arguments["strategy"].encode(), kind="strategy")
        return OperationResult("Stored strategy", (ref,), False)
