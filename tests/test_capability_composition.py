"""Custom providers compose with explicit actions through existing injection points."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.decision_making.action_space import (
    ModularDecisionActionSpace, OperationCall, OperationResult, OperationSpec,
)
from rsisys.loops.decision_making.action_space.capabilities import Capability, Capabilities
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.runtime.artifacts import InMemoryArtifactStore


class TextDefinition:
    def __init__(self, name, *, read_only=False):
        self.spec = OperationSpec(name, "Accept trimmed text", {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"], "additionalProperties": False,
        }, read_only=read_only)

    def validate(self, arguments):
        if (set(arguments) != {"text"} or not isinstance(arguments["text"], str)
                or not arguments["text"].strip()):
            raise ValueError("Expected nonempty text only")
        return OperationCall(self.spec.name, {"text": arguments["text"].strip()})


class CatalogProvider:
    """User-owned adapter for a preconfigured catalog, without a network client."""

    def __init__(self):
        self.catalog = {"paper": "An experiment in replaceable action spaces"}
        self.calls = []
        self.handler = Mock(side_effect=lambda args: OperationResult(
            self.catalog[args["text"]], (), False,
        ))

    def describe(self):
        self.calls.append("describe")
        yield Capability(TextDefinition("lookup", read_only=True), self.handler)

    def discover(self):
        self.calls.append("discover")
        yield Capability(TextDefinition("lookup", read_only=True), self.handler)

    def connect(self):
        raise AssertionError("Composition must never connect a provider")


class CapabilityCompositionTests(unittest.TestCase):
    def test_provider_and_explicit_action_run_through_existing_harness(self):
        for method in ("describe", "discover"):
            with self.subTest(method=method):
                provider, artifacts = CatalogProvider(), InMemoryArtifactStore()
                definition = TextDefinition("save_note")
                save = Mock(side_effect=lambda args: OperationResult("Saved note", (
                    artifacts.put(args["text"].encode(), kind="note"),
                ), False))
                explicit = Capability.from_spec(definition.spec, definition.validate, save)
                capabilities = Capabilities.from_provider(provider, method=method, explicit=(explicit,))
                space = ModularDecisionActionSpace(capabilities.definitions, directive_modes=())
                tools = BoundDecisionTools(capabilities.bindings)
                self.assertEqual(provider.calls, [method])
                self.assertEqual(tuple(spec.name for spec in space.describe()), ("lookup", "save_note"))
                self.assertEqual(tuple(spec.read_only for spec in space.describe()), (True, False))
                provider.handler.assert_not_called()
                save.assert_not_called()
                policy = Mock()
                harness = DefaultDecisionHarness(
                    policy, tools, BasicExperienceSystem(InMemoryExperienceStore()),
                )
                policy.decide.return_value = Decision("invoke", operation=OperationCall(
                    "lookup", {"text": "  paper  "},
                ))
                found = harness.step("Research and save a note", space).operation_result
                provider.handler.assert_called_once_with({"text": "paper"})
                policy.decide.return_value = Decision("invoke", operation=OperationCall(
                    "save_note", {"text": found.summary},
                ))
                saved = harness.step("Research and save a note", space).operation_result
                self.assertEqual(artifacts.get(saved.artifacts[0]), provider.catalog["paper"].encode())
                self.assertEqual(len(harness.memory.history(kind="operation")), 2)
                self.assertEqual(provider.calls, [method])

    def test_invalid_arguments_do_not_reach_provider_handler(self):
        provider = CatalogProvider()
        capabilities = Capabilities.from_provider(provider)
        space, tools = ModularDecisionActionSpace(capabilities.definitions), BoundDecisionTools(capabilities.bindings)
        for arguments in ({}, {"text": " "}, {"text": 1}, {"text": "paper", "extra": True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                tools.execute(space.validate(OperationCall("lookup", arguments)))
        provider.handler.assert_not_called()

    def test_duplicate_names_fail_across_all_sources_before_execution(self):
        provider = CatalogProvider()
        duplicate = Capability(TextDefinition("lookup"), Mock())
        with self.assertRaisesRegex(ValueError, "unique: lookup"):
            Capabilities.from_provider(provider, explicit=(duplicate,))
        with self.assertRaisesRegex(ValueError, "unique: lookup"):
            Capabilities((duplicate, duplicate))
        provider.handler.assert_not_called()
        duplicate.handler.assert_not_called()

    def test_malformed_capabilities_fail_at_construction(self):
        for definition, handler, error, message in (
            (object(), Mock(), TypeError, "OperationSpec"),
            (SimpleNamespace(spec="bad", validate=lambda args: None), Mock(), TypeError, "OperationSpec"),
            (SimpleNamespace(spec=OperationSpec("bad", "", {})), Mock(), TypeError, "validator"),
            (TextDefinition("valid"), None, TypeError, "handler"),
            (TextDefinition("valid"), object(), TypeError, "handler"),
            (TextDefinition(""), Mock(), ValueError, "nonempty"),
            (TextDefinition("  "), Mock(), ValueError, "nonempty"),
            (TextDefinition(42), Mock(), ValueError, "nonempty"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                Capability(definition, handler)
        with self.assertRaisesRegex(TypeError, "validator"):
            Capability.from_spec(OperationSpec("bad", "", {}), None, Mock())
        with self.assertRaises(TypeError):
            Capability(TextDefinition("missing_handler"))

    def test_provider_contract_errors_and_failures_are_not_silenced(self):
        for provider in (object(), SimpleNamespace(describe=None)):
            with self.subTest(provider=provider), self.assertRaisesRegex(TypeError, "describe"):
                Capabilities.from_provider(provider)
        with self.assertRaisesRegex(TypeError, "Capability instances"):
            Capabilities.from_provider(SimpleNamespace(describe=lambda: [TextDefinition("unbound")]))
        provider = CatalogProvider()
        with self.assertRaisesRegex(ValueError, "describe or discover"):
            Capabilities.from_provider(provider, method="connect")
        self.assertEqual(provider.calls, [])
        failure = Mock(side_effect=RuntimeError("provider unavailable"))
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            Capabilities.from_provider(SimpleNamespace(discover=failure), method="discover")
        failure.assert_called_once_with()

    def test_explicit_collections_are_independent_and_allow_async_handlers(self):
        definition = TextDefinition("custom_python")
        job = JobRef("user-job", "user-backend")
        handler = Mock(return_value=job)
        capability = Capability(definition, handler)
        source = [capability]
        first = Capabilities(source)
        source.clear()
        exported = first.bindings
        exported.clear()
        second = Capabilities()
        self.assertEqual(tuple(first), (capability,))
        self.assertEqual(second.definitions, ())
        self.assertEqual(second.bindings, {})
        space = ModularDecisionActionSpace(first.definitions)
        result = BoundDecisionTools(first.bindings).execute(space.validate(OperationCall(
            "custom_python", {"text": " user-owned job "},
        )))
        self.assertIs(result, job)
        handler.assert_called_once_with({"text": "user-owned job"})

    def test_definition_mutated_before_assembly_is_revalidated(self):
        definition = TextDefinition("original")
        capability = Capability(definition, Mock())
        definition.validate = None
        with self.assertRaisesRegex(TypeError, "validator"):
            Capabilities((capability,))


if __name__ == "__main__":
    unittest.main()
