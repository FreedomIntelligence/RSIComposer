"""Decision definitions validate independently from replaceable tool bindings."""

from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space import (
    DEFAULT_DIRECTIVE_MODES,
    DemoDecisionActionSpace,
    LocalDecisionActionSpace,
    ModularDecisionActionSpace,
    OperationCall,
    OperationResult,
    OperationSpec,
)
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools, DemoDecisionTools
from rsisys.loops.decision_making.harness.tools.local import LocalDecisionTools
from rsisys.runtime.artifacts import FileArtifactStore, InMemoryArtifactStore
from rsisys.runtime.sandbox import MockSandboxBackend


class DecisionActionTests(unittest.TestCase):
    def test_default_actions_and_modes(self):
        for space, names in (
            (DemoDecisionActionSpace(), ("prepare_data",)),
            (LocalDecisionActionSpace(), ("prepare_data", "search_literature", "store_strategy")),
            (ModularDecisionActionSpace(()), ()),
        ):
            with self.subTest(space=type(space).__name__):
                self.assertEqual(tuple(spec.name for spec in space.describe()), names)
                self.assertEqual(space.directive_modes(), DEFAULT_DIRECTIVE_MODES)
                self.assertFalse(hasattr(space, "execute"))
                self.assertFalse(hasattr(space, "artifacts"))
                self.assertFalse(hasattr(space, "sandbox"))

    def test_invalid_arguments_never_execute_a_handler(self):
        handler = Mock(return_value=OperationResult("unexpected", (), False))
        tools = BoundDecisionTools({name: handler for name in (
            "prepare_data", "search_literature", "store_strategy",
        )})
        for space in (DemoDecisionActionSpace(), LocalDecisionActionSpace()):
            for spec in space.describe():
                argument = spec.input_schema["required"][0]
                invalid = [{}, {argument: 3}, {argument: None}, {argument: True},
                           {argument: []}, {argument: "valid", "extra": "forbidden"}]
                if spec.name != "prepare_data":
                    invalid.extend(({argument: ""}, {argument: " \n\t"}))
                for arguments in invalid:
                    with self.subTest(operation=spec.name, arguments=arguments):
                        with self.assertRaises(ValueError):
                            tools.execute(space.validate(OperationCall(spec.name, arguments)))
        handler.assert_not_called()

    def test_validating_an_action_has_no_execution_side_effects(self):
        sandbox, artifacts = MockSandboxBackend(), InMemoryArtifactStore()
        tools = DemoDecisionTools(sandbox, artifacts)
        space = DemoDecisionActionSpace()
        call = OperationCall("prepare_data", {"text": ""})
        validated = space.validate(call)
        self.assertEqual(validated, call)
        self.assertIsNot(validated.arguments, call.arguments)
        self.assertEqual(sandbox.execution_count, 0)
        self.assertEqual(sandbox.active_count, 0)
        call.arguments["text"] = "changed"
        result = tools.execute(validated)
        self.assertEqual(artifacts.get(result.artifacts[0]), b"\n")
        self.assertEqual(sandbox.execution_count, 1)

    def test_one_space_can_use_independent_tool_bindings(self):
        space = DemoDecisionActionSpace()
        call = space.validate(OperationCall("prepare_data", {"text": "data"}))
        first_result = OperationResult("first", (), False)
        second_result = OperationResult("second", (), True)
        first, second = Mock(return_value=first_result), Mock(return_value=second_result)
        self.assertIs(BoundDecisionTools({"prepare_data": first}).execute(call), first_result)
        self.assertIs(BoundDecisionTools({"prepare_data": second}).execute(call), second_result)
        first.assert_called_once_with({"text": "data"})
        second.assert_called_once_with({"text": "data"})

    def test_custom_definition_normalizes_without_executing(self):
        class Definition:
            spec = OperationSpec("normalize", "Normalize text", {})

            def validate(self, arguments):
                arguments["text"] = arguments["text"].strip()
                return OperationCall(self.spec.name, arguments)

            def execute(self, arguments):
                raise AssertionError("Definitions must never execute")

        space = ModularDecisionActionSpace((Definition(),))
        original = OperationCall("normalize", {"text": "  data  "})
        validated = space.validate(original)
        self.assertEqual(original.arguments, {"text": "  data  "})
        self.assertEqual(validated.arguments, {"text": "data"})
        handler = Mock(return_value=OperationResult("normalized", (), False))
        BoundDecisionTools({"normalize": handler}).execute(validated)
        handler.assert_called_once_with({"text": "data"})

    def test_definition_names_are_unique(self):
        definition = Mock()
        definition.spec = OperationSpec("same", "", {})
        with self.assertRaisesRegex(ValueError, "unique"):
            ModularDecisionActionSpace((definition, definition))

    def test_harness_records_normalized_call_for_sync_async_and_tool_failure(self):
        class Definition:
            spec = OperationSpec("normalize", "Normalize text and supply a default", {})

            def validate(self, arguments):
                return OperationCall(self.spec.name, {"text": arguments["text"].strip(), "limit": 10})

        space = ModularDecisionActionSpace((Definition(),))
        expected = {"name": "normalize", "arguments": {"text": "data", "limit": 10}}
        result, job = OperationResult("done", (), False), JobRef("job", "test")
        for outcome in ("sync", "async", "failure"):
            with self.subTest(outcome=outcome):
                original = OperationCall("normalize", {"text": "  data  "})
                handler = Mock(return_value=job if outcome == "async" else result,
                               side_effect=RuntimeError("tool failed") if outcome == "failure" else None)
                harness = DefaultDecisionHarness(
                    Mock(decide=Mock(return_value=Decision("invoke", operation=original))),
                    BoundDecisionTools({"normalize": handler}),
                    BasicExperienceSystem(InMemoryExperienceStore()),
                )
                if outcome == "failure":
                    with self.assertRaisesRegex(RuntimeError, "tool failed"):
                        harness.step("task", space)
                else:
                    step = harness.step("task", space)
                    if outcome == "async":
                        self.assertEqual(step.job, job)
                        self.assertEqual(harness.state.pending_operation, OperationCall(**expected))
                        self.assertEqual(harness.memory.history(kind="operation"), ())
                        original.arguments["text"] = "changed after submission"
                        harness.resume(job, result)
                        self.assertIsNone(harness.state.pending_operation)
                    else:
                        self.assertEqual(step.operation_result, result)
                handler.assert_called_once_with(expected["arguments"])
                record, = harness.memory.history(kind="operation")
                self.assertEqual(record.details["operation"], expected)
                self.assertEqual(record.details["result"]["status"],
                                 ExecutionStatus.FAILED if outcome == "failure" else ExecutionStatus.COMPLETED)
                decision_record, = harness.memory.history(kind="decision")
                self.assertEqual(decision_record.details["operation"]["arguments"], {"text": "  data  "})

    def test_harness_validation_failure_records_original_call_without_execution(self):
        original = OperationCall("prepare_data", {"text": 42})
        tools = Mock()
        harness = DefaultDecisionHarness(
            Mock(decide=Mock(return_value=Decision("invoke", operation=original))), tools,
            BasicExperienceSystem(InMemoryExperienceStore()),
        )
        with self.assertRaises(ValueError):
            harness.step("task", DemoDecisionActionSpace())
        tools.execute.assert_not_called()
        record, = harness.memory.history(kind="operation")
        self.assertEqual(record.details["operation"], {"name": original.name, "arguments": original.arguments})
        self.assertEqual(record.details["result"]["status"], ExecutionStatus.FAILED)

    def test_definition_result_must_be_a_call_for_the_same_operation(self):
        for result, error in (
            (OperationResult("result", (), False), TypeError),
            (OperationCall("other", {}), ValueError),
            (OperationCall("custom", []), TypeError),
        ):
            with self.subTest(result=result):
                definition = Mock()
                definition.spec = OperationSpec("custom", "", {})
                definition.validate.return_value = result
                with self.assertRaises(error):
                    ModularDecisionActionSpace((definition,)).validate(OperationCall("custom", {}))

    def test_description_mutations_do_not_change_definitions(self):
        space = LocalDecisionActionSpace()
        schema = space.describe()[0].input_schema
        schema["properties"]["text"]["type"] = "integer"
        schema["required"].clear()
        self.assertEqual(space.describe()[0].input_schema["required"], ["text"])
        self.assertEqual(space.describe()[0].input_schema["properties"]["text"]["type"], "string")
        with self.assertRaises(ValueError):
            space.validate(OperationCall("prepare_data", {"text": 1}))

    def test_unknown_or_malformed_calls_are_rejected(self):
        handler = Mock()
        for boundary in (DemoDecisionActionSpace().validate,
                         BoundDecisionTools({"prepare_data": handler}).execute):
            for call, error in ((None, TypeError),
                                (OperationCall("prepare_data", []), TypeError),
                                (OperationCall("unknown", {}), ValueError),
                                (OperationCall([], {}), ValueError)):
                with self.subTest(boundary=boundary, call=call), self.assertRaises(error):
                    boundary(call)
        handler.assert_not_called()

    def test_directive_whitelist_checks_only_mode(self):
        for factory in (DemoDecisionActionSpace, LocalDecisionActionSpace,
                        lambda **kwargs: ModularDecisionActionSpace((), **kwargs)):
            space = factory(directive_modes=("execute", "sft"))
            self.assertEqual(space.directive_modes(), ("execute", "sft"))
            for mode in DEFAULT_DIRECTIVE_MODES + ("unknown",):
                # A malformed payload is deliberately left for exploration to reject.
                directive = Directive("id", "objective", "strategy", mode=mode, payload={"invalid_for_builtin_mode": True})
                with self.subTest(factory=factory, mode=mode):
                    if mode in ("execute", "sft"):
                        self.assertIsNone(space.validate_directive(directive))
                    else:
                        with self.assertRaisesRegex(ValueError, "Unsupported directive mode"):
                            space.validate_directive(directive)

    def test_default_directive_modes_allow_payload_validation_to_be_deferred(self):
        space = DemoDecisionActionSpace()
        for mode in DEFAULT_DIRECTIVE_MODES:
            with self.subTest(mode=mode):
                self.assertIsNone(space.validate_directive(Directive("id", "", "", mode=mode)))

    def test_directive_modes_can_be_empty_but_not_duplicated_or_extended(self):
        space = LocalDecisionActionSpace(directive_modes=())
        self.assertEqual(space.directive_modes(), ())
        with self.assertRaises(ValueError):
            space.validate_directive(Directive("id", "", ""))
        for modes in (("execute", "execute"), ("unknown",), ([],)):
            with self.subTest(modes=modes), self.assertRaises(ValueError):
                DemoDecisionActionSpace(directive_modes=modes)
        with self.assertRaises(TypeError):
            DemoDecisionActionSpace(directive_modes="execute")

    def test_tool_result_must_be_an_operation_result_or_job(self):
        call = OperationCall("custom", {})
        for result in (OperationResult("done", (), False), JobRef("job", "backend")):
            with self.subTest(result=result):
                tools = BoundDecisionTools({"custom": Mock(return_value=result)})
                self.assertIs(tools.execute(call), result)
        for result in (None, {}, "done"):
            with self.subTest(result=result), self.assertRaises(TypeError):
                BoundDecisionTools({"custom": Mock(return_value=result)}).execute(call)

    def test_bindings_require_unique_names_and_callable_handlers(self):
        class DuplicateBindings(dict):
            def items(self):
                return [("same", lambda arguments: None), ("same", lambda arguments: None)]

        with self.assertRaisesRegex(ValueError, "unique"):
            BoundDecisionTools(DuplicateBindings())
        with self.assertRaises(TypeError):
            BoundDecisionTools({"operation": object()})
        for name in ("", 1):
            with self.subTest(name=name), self.assertRaises(ValueError):
                BoundDecisionTools({name: lambda arguments: None})

    def test_handler_and_binding_mutations_do_not_change_validated_call(self):
        def handler(arguments):
            arguments["nested"].append("handler mutation")
            return OperationResult("done", (), False)

        bindings = {"custom": handler}
        tools = BoundDecisionTools(bindings)
        bindings.clear()
        call = OperationCall("custom", {"nested": ["original"]})
        self.assertEqual(tools.execute(call).summary, "done")
        self.assertEqual(call.arguments, {"nested": ["original"]})

    def test_demo_tools_preserve_artifact_and_sandbox_behavior(self):
        sandbox, artifacts = MockSandboxBackend(), InMemoryArtifactStore()
        tools = DemoDecisionTools(sandbox, artifacts)
        self.assertIs(tools.sandbox, sandbox)
        self.assertIs(tools.artifacts, artifacts)
        call = DemoDecisionActionSpace().validate(OperationCall("prepare_data", {"text": "hello"}))
        result = tools.execute(call)
        self.assertEqual(result.artifacts[0].kind, "demo_input")
        self.assertEqual(artifacts.get(result.artifacts[0]), b"hello\n")
        self.assertTrue(result.simulated)
        self.assertEqual(sandbox.active_count, 0)

    def test_demo_tools_close_sandbox_after_process_or_artifact_failure(self):
        class FailedSandbox(MockSandboxBackend):
            def poll(self, process):
                return replace(super().poll(process), status=JobStatus.FAILED, stderr="failed")

        call = DemoDecisionActionSpace().validate(OperationCall("prepare_data", {"text": "data"}))
        for sandbox, artifacts, error in (
            (FailedSandbox(), InMemoryArtifactStore(), RuntimeError),
            (MockSandboxBackend(), Mock(put=Mock(side_effect=OSError("write failed"))), OSError),
        ):
            with self.subTest(error=error):
                with self.assertRaises(error):
                    DemoDecisionTools(sandbox, artifacts).execute(call)
                self.assertEqual(sandbox.active_count, 0)

    def test_local_tools_store_data_strategy_and_search_imported_literature(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = FileArtifactStore(directory)
            tools, space = LocalDecisionTools(artifacts), LocalDecisionActionSpace()
            self.assertIs(tools.artifacts, artifacts)
            literature = artifacts.put("A paper about RSI and learning".encode(), kind="literature")
            artifacts.put(b"unrelated paper", kind="literature")
            artifacts.put(b"RSI outside the literature collection", kind="other")
            for name, arguments, content, kind in (
                ("prepare_data", {"text": "input"}, b"input", "input_data"),
                ("store_strategy", {"strategy": "Try new data"}, b"Try new data", "strategy"),
            ):
                result = tools.execute(space.validate(OperationCall(name, arguments)))
                self.assertEqual(artifacts.get(result.artifacts[0]), content)
                self.assertEqual(result.artifacts[0].kind, kind)
                self.assertFalse(result.simulated)
            for query, expected in (("rsi", [{"artifact_id": literature.artifact_id,
                                             "excerpt": "A paper about RSI and learning"}]),
                                    ("missing", [])):
                result = tools.execute(space.validate(OperationCall("search_literature", {"query": query})))
                self.assertEqual(json.loads(artifacts.get(result.artifacts[0])), expected)
                self.assertEqual(result.artifacts[0].kind, "literature_results")
                self.assertFalse(result.simulated)

    def test_local_literature_search_preserves_twenty_match_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = FileArtifactStore(directory)
            for index in range(21):
                artifacts.put(f"match {index}".encode(), kind="literature")
            call = LocalDecisionActionSpace().validate(OperationCall("search_literature", {"query": "match"}))
            result = LocalDecisionTools(artifacts).execute(call)
            self.assertEqual(len(json.loads(artifacts.get(result.artifacts[0]))), 20)

    def test_operation_result_retains_terminal_status_rules(self):
        for status, error in ((ExecutionStatus.FAILED, None),
                              (ExecutionStatus.COMPLETED, "unexpected")):
            with self.subTest(status=status), self.assertRaises(ValueError):
                OperationResult("result", (), False, status, error)
        self.assertEqual(OperationResult("cancelled", (), False, ExecutionStatus.CANCELLED).status,
                         ExecutionStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()
