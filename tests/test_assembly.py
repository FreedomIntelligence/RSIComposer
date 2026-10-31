"""Assembly injection and resource ownership without default harness internals."""

from contextlib import contextmanager
from dataclasses import replace
import fcntl
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from rsisys.application import Application, ApplicationConfig
from rsisys.assembly import build_default_decision, build_default_exploration
from rsisys.config import DemoConfig, build_demo
from rsisys.experience.records import ExperienceRecord
from rsisys.loops.contracts.coordination import ExplorationProgress
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, LocalDecisionActionSpace
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinition, DirectiveDefinitions, DirectiveSpec, builtin_directives,
)
from rsisys.loops.decision_making.harness.decision import LoopStepResult
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import ActionProposal, ActionSpec, EchoActionSpace
from rsisys.loops.exploration.harness.commands import CommandHandler
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.environment import SandboxEchoEnvironment
from rsisys.loops.exploration.harness.evaluation.mock import MockEvaluator
from rsisys.loops.exploration.harness.learner.backends.mock import MockActorLearner
from rsisys.loops.exploration.harness.policy import ModelActorPolicy
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.sandbox import SandboxSpec


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class ScheduledDecision:
    def __init__(self):
        self.directive = None
        self.done = False

    def step(self, objective, action_space):
        if self.done:
            return LoopStepResult("finished")
        self.directive = Directive("scheduled", objective, "execute once")
        action_space.validate_directive(self.directive)
        return LoopStepResult("dispatch", directive=self.directive)

    def pending(self):
        return self.directive if not self.done else None

    def receive_feedback(self, directive, feedback):
        if directive != self.directive or feedback.directive_id != directive.directive_id:
            raise ValueError("Wrong directive")
        self.done = True


class ExternalExecution:
    def __init__(self, context):
        self.context = context
        self.directive = None
        self.feedback = None

    def accept(self, directive):
        self.directive = directive
        return self.inspect(directive.directive_id)

    def advance(self, directive_id, action_space):
        ref = self.context.artifacts.put(b"external result", kind="external")
        self.context.experience.record(ExperienceRecord(
            "external-record", "runtime_feedback", "actor", directive_id, "external execution", False,
            source_refs=(ref.artifact_id,),
        ))
        self.feedback = Feedback(directive_id, ExecutionStatus.COMPLETED, "external execution", False,
                                 artifact_refs=(ref,), evidence_refs=("external-record",))
        return self.inspect(directive_id)

    def inspect(self, directive_id):
        return ExplorationProgress(directive_id, "completed" if self.feedback else "accepted",
                                   self.directive, feedback=self.feedback)

    def cancel(self, directive_id):
        self.feedback = Feedback(directive_id, ExecutionStatus.CANCELLED, "cancelled", False)
        return self.inspect(directive_id)

    def acknowledge(self, directive_id):
        self.acknowledged = directive_id


class ExternalActions(EchoActionSpace):
    def describe(self):
        return (ActionSpec("external", "External task", {}),)


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.demo = DemoConfig.load(CONFIGS / "demo.json")
        self.config = replace(ApplicationConfig.load(CONFIGS / "local.json"),
                              data_dir=self.root / "run", execution="mock",
                              experience_limit=7, memory_limit=9)

    @contextmanager
    def owner(self, kind, **factories):
        if kind == "application":
            with Application(self.config, **factories) as app:
                yield app, app.runner, app.assembly
        else:
            runner, _ = build_demo(self.demo, **factories)
            with runner:
                yield runner, runner, runner.assembly

    def test_both_factories_share_resources_and_run_without_policy_or_memory(self):
        for kind in ("application", "demo"):
            with self.subTest(kind=kind):
                seen = []

                def exploration_factory(context):
                    loop = ExplorationLoop(ExternalExecution(context), ExternalActions())
                    seen.append((context, loop))
                    return loop

                def decision_factory(context, exploration):
                    self.assertEqual(seen, [(context, exploration)])
                    return DecisionMakingLoop(ScheduledDecision(), DemoDecisionActionSpace())

                with self.owner(kind, decision_factory=decision_factory,
                                exploration_factory=exploration_factory) as (owner, runner, context):
                    self.assertEqual(runner.run("task", 1).status, "completed")
                    for loop in (runner.decision_making, runner.exploration):
                        self.assertFalse(hasattr(loop.harness, "policy"))
                        self.assertFalse(hasattr(loop.harness, "memory"))
                    self.assertEqual(context.artifacts.get(runner.exploration.harness.feedback.artifact_refs[0]),
                                     b"external result")
                    self.assertEqual(context.experience.retrieve()[0].record_id, "external-record")
                    if kind == "application":
                        self.assertIs(context.experience, owner.experience)
                        self.assertIs(context.artifacts, owner.artifacts)
                        self.assertIs(context.backends, owner.backends)
                        self.assertIs(context.sandbox, owner.sandbox)
                        self.assertEqual((context.experience_limit, context.memory_limit), (7, 9))
                        self.assertTrue(owner.custom_assembly)

    def test_only_decision_factory_can_replace_default_harness(self):
        for kind in ("application", "demo"):
            with self.subTest(kind=kind), self.owner(
                kind, decision_factory=lambda context, exploration:
                    DecisionMakingLoop(ScheduledDecision(), DemoDecisionActionSpace()),
            ) as (_, runner, context):
                self.assertIsInstance(runner.exploration.harness, DefaultExplorationHarness)
                self.assertIs(runner.exploration.harness.experience, context.experience)
                self.assertEqual(runner.run("task", 1).status, "completed")

    def test_only_exploration_factory_updates_default_decision_action_capabilities(self):
        def exploration_factory(context):
            context.exploration_directives = DirectiveDefinitions(
                definition for definition in builtin_directives() if definition.spec.mode == "execute"
            )
            return ExplorationLoop(ExternalExecution(context), ExternalActions())

        for kind in ("application", "demo"):
            with self.subTest(kind=kind), self.owner(
                kind, exploration_factory=exploration_factory,
            ) as (_, runner, context):
                decision = runner.decision_making.harness
                self.assertIsInstance(decision, DefaultDecisionHarness)
                self.assertIs(decision.experience, context.experience)
                self.assertEqual(decision.policy.allowed_actions, ("external",))
                self.assertEqual(runner.decision_making.action_space.directive_modes(), ("execute",))
                self.assertIs(decision.policy.directives, context.exploration_directives)
                self.assertIs(runner.decision_making.action_space.directives, context.exploration_directives)
                self.assertEqual(runner.run("task", 1).status, "completed")
                self.assertEqual(runner.exploration.harness.directive.allowed_actions, ("external",))

    def make_exploration(self, context, *, training=False, evaluation=False, policy=None, handlers=None):
        model = context.roles["actor"]
        policy = policy if policy is not None else ModelActorPolicy(context.backends[model.backend_id], model)
        environment = SandboxEchoEnvironment(context.sandbox, context.artifacts)
        context.add_cleanup(environment.close)
        return ExplorationLoop(DefaultExplorationHarness(
            policy, environment, context.experience,
            learner=MockActorLearner(context.artifacts) if training else None,
            evaluator=MockEvaluator(context.artifacts) if evaluation else None,
            command_handlers=handlers,
        ), EchoActionSpace())

    def test_modes_follow_actual_learner_and_evaluator_independently_of_config(self):
        for kind in ("application", "demo"):
            for configured in (False, True):
                for training, evaluation in ((False, False), (True, False), (False, True), (True, True)):
                    with self.subTest(kind=kind, configured=configured, training=training, evaluation=evaluation):
                        self.config = replace(self.config, learning="mock" if configured else "none",
                                              data_dir=self.root / f"{kind}-{configured}-{training}-{evaluation}")
                        kwargs = {"with_learning": configured} if kind == "demo" else {}
                        with self.owner(kind, exploration_factory=lambda context: self.make_exploration(
                            context, training=training, evaluation=evaluation,
                        ), **kwargs) as (_, runner, context):
                            expected = ("execute", "configure_prompt")
                            if training:
                                expected += ("sft", "rl")
                            if evaluation:
                                expected += ("evaluate",)
                            self.assertEqual(runner.decision_making.action_space.directive_modes(), expected)
                            self.assertIs(runner.decision_making.harness.policy.directives,
                                          runner.decision_making.action_space.directives)
                            self.assertTrue(runner.decision_making.action_space.directives.is_default)

    def test_training_modes_require_model_and_rl_requires_updatable_policy(self):
        class ReadOnlyActor:
            def act(self, context):
                return ActionProposal("echo", {"text": "read only"}, True)

        for has_model in (False, True):
            with self.subTest(has_model=has_model):
                def factory(context):
                    policy = ReadOnlyActor()
                    if has_model:
                        policy.model = context.roles["actor"]
                    return self.make_exploration(context, training=True, policy=policy)

                runner, _ = build_demo(self.demo, exploration_factory=factory)
                with runner:
                    self.assertEqual(runner.decision_making.action_space.directive_modes(),
                                     ("execute", "configure_prompt") + (("sft",) if has_model else ()))

    def test_unknown_harness_requires_declaration_or_decision_factory_and_cleans_up(self):
        for kind in ("application", "demo"):
            closed = []

            def factory(context):
                context.add_cleanup(lambda: closed.append(True))
                return ExplorationLoop(ExternalExecution(context), EchoActionSpace())

            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "decision_factory.*exploration_directives"):
                with self.owner(kind, exploration_factory=factory):
                    self.fail("Unknown capabilities must fail during assembly")
            self.assertEqual(closed, [True])

    def test_custom_handler_requires_explicit_catalog_to_be_advertised(self):
        for declared in (False, True):
            with self.subTest(declared=declared):
                def factory(context):
                    if declared:
                        context.exploration_directives = DirectiveDefinitions(
                            definition for definition in builtin_directives() if definition.spec.mode == "sft"
                        )
                    # A replacement handler may implement SFT externally with no local learner.
                    return self.make_exploration(context, handlers={"sft": CommandHandler()})

                runner, _ = build_demo(self.demo, exploration_factory=factory)
                with runner:
                    self.assertEqual(runner.decision_making.action_space.directive_modes(),
                                     ("sft",) if declared else ("execute", "configure_prompt"))

    def test_explicit_custom_catalog_is_used_for_policy_parsing_and_execution(self):
        class ExternalHandler(CommandHandler):
            def prepare(self, harness, state):
                if state.directive.payload != {"parsed": True}:
                    raise ValueError("Custom parser was not used")

            def advance(self, harness, state, action_space):
                return Feedback(state.directive.directive_id, ExecutionStatus.COMPLETED, "external completed", False)

        def factory(context):
            context.exploration_directives = DirectiveDefinitions((DirectiveDefinition(
                DirectiveSpec("external", "External work", {"type": "object"}),
                lambda arguments, candidates: {"parsed": True},
            ),))
            return self.make_exploration(context, handlers={"external": ExternalHandler()})

        runner, _ = build_demo(self.demo, exploration_factory=factory)
        with runner:
            policy = runner.decision_making.harness.policy
            with patch.object(policy.backend, "generate", return_value=GenerationResponse(
                '{"kind":"explore","mode":"external","strategy":"run","arguments":{}}', policy.model, False,
            )):
                self.assertEqual(runner.run("task", 1).status, "completed")
            self.assertEqual(runner.decision_making.harness.state.last_feedback.summary, "external completed")

    def test_explicit_catalog_cannot_advertise_missing_handlers_or_builtin_backends(self):
        for mode, message in (("external", "no exploration handler"),
                              ("sft", "lacks required exploration capabilities"),
                              ("evaluate", "lacks required exploration capabilities")):
            with self.subTest(mode=mode):
                def factory(context):
                    context.exploration_directives = DirectiveDefinitions((DirectiveDefinition(
                        DirectiveSpec(mode, "Declared capability", {}), lambda arguments, candidates: None,
                    ),))
                    return self.make_exploration(context)

                with self.assertRaisesRegex(ValueError, message):
                    build_demo(self.demo, exploration_factory=factory)

    def test_empty_explicit_catalog_is_not_replaced_with_defaults(self):
        def factory(context):
            context.exploration_directives = DirectiveDefinitions(())
            return ExplorationLoop(ExternalExecution(context), EchoActionSpace())

        runner, _ = build_demo(self.demo, with_learning=True, exploration_factory=factory)
        with runner:
            self.assertEqual(runner.decision_making.action_space.directive_modes(), ())
            self.assertEqual(runner.decision_making.harness.policy.directives.modes(), ())

    def test_default_factories_remain_usable_inside_custom_factories(self):
        events = []

        def exploration_factory(context):
            events.append("exploration")
            return build_default_exploration(context)

        def decision_factory(context, exploration):
            events.append("decision")
            return build_default_decision(context, exploration)

        runner, _ = build_demo(self.demo, decision_factory=decision_factory,
                               exploration_factory=exploration_factory)
        with runner:
            self.assertEqual(events, ["exploration", "decision"])
            self.assertEqual(runner.run("task", 1).status, "completed")

    def test_defaults_preserve_storage_limits_and_learning_modes(self):
        with Application(self.config) as app:
            self.assertFalse(app.custom_assembly)
            self.assertIsInstance(app.runner.decision_making.action_space, LocalDecisionActionSpace)
            self.assertEqual(app.runner.decision_making.harness.experience_limit, 7)
            self.assertEqual(app.runner.exploration.harness.memory_limit, 9)
            self.assertEqual(app.run().status, "completed")
            for name in ("decision_memory.json", "exploration_memory.json"):
                self.assertTrue((self.config.data_dir / name).is_file())
        for learning in (False, True):
            with self.subTest(learning=learning):
                runner, store = build_demo(self.demo, with_learning=learning)
                with runner:
                    self.assertIsInstance(runner.decision_making.action_space, DemoDecisionActionSpace)
                    self.assertEqual(runner.decision_making.action_space.directive_modes(),
                                     ("execute", "configure_prompt") + (("sft", "rl", "evaluate") if learning else ()))
                    self.assertEqual(runner.exploration.harness.learner is not None, learning)
                    self.assertEqual(runner.exploration.harness.evaluation.backend is not None, learning)
                    self.assertEqual(runner.run("task", 1).status, "completed")
                    self.assertTrue(store.records())

    def test_cleanup_is_lifo_once_and_shared_resources_remain_available_during_callbacks(self):
        for kind in ("application", "demo"):
            with self.subTest(kind=kind):
                events = []

                def exploration_factory(context):
                    context.add_cleanup(lambda: events.append(("exploration", len(context.experience.retrieve()))))
                    return build_default_exploration(context)

                def decision_factory(context, exploration):
                    context.add_cleanup(lambda: events.append(("decision", len(context.experience.retrieve()))))
                    return build_default_decision(context, exploration)

                with self.owner(kind, exploration_factory=exploration_factory,
                                decision_factory=decision_factory) as (owner, runner, context):
                    runner.run("task", 1)
                    count = len(context.experience.retrieve())
                owner.close()
                self.assertEqual(events, [("decision", count), ("exploration", count)])
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    context.add_cleanup(lambda: None)
                if kind == "application":
                    self.assertIsNone(owner.store)
                    self.assertTrue(owner._lock.closed)
                    with self.assertRaises(sqlite3.ProgrammingError):
                        context.experience.retrieve()

    def test_each_factory_failure_cleans_registered_callbacks_and_preserves_original_error(self):
        for kind in ("application", "demo"):
            for phase in ("exploration", "decision"):
                with self.subTest(kind=kind, phase=phase):
                    contexts, events = [], []

                    def register(context, name):
                        events.append(name)
                        context.experience.retrieve()  # Store is still open during cleanup.

                    def exploration_factory(context):
                        contexts.append(context)
                        context.add_cleanup(lambda: register(context, "exploration"))
                        if phase == "exploration":
                            raise ValueError("factory failed")
                        return build_default_exploration(context)

                    def decision_factory(context, exploration):
                        def failing_cleanup():
                            register(context, "decision")
                            raise OSError("cleanup failed")
                        context.add_cleanup(failing_cleanup)
                        raise ValueError("factory failed")

                    # Each failed Application leaves its initialized workspace intact.
                    self.config = replace(self.config, data_dir=self.root / f"failed-{kind}-{phase}")
                    with self.assertRaisesRegex(ValueError, "factory failed") as raised:
                        with self.owner(kind, exploration_factory=exploration_factory,
                                        decision_factory=decision_factory):
                            self.fail("Factory should not succeed")
                    self.assertEqual(events, ["exploration"] if phase == "exploration" else ["decision", "exploration"])
                    if phase == "decision":
                        self.assertTrue(any("cleanup failed" in note for note in raised.exception.__notes__))
                    contexts[0].close()
                    if kind == "application":
                        with self.assertRaises(sqlite3.ProgrammingError):
                            contexts[0].experience.retrieve()
                        with (self.config.data_dir / ".lock").open("a+b") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_default_environment_closes_if_later_factory_fails(self):
        contexts = []

        def decision_factory(context, exploration):
            contexts.append(context)
            exploration.harness.environment.reset(Directive("session", "task", "test"))
            self.assertEqual(context.sandbox.active_count, 1)
            raise RuntimeError("decision construction failed")

        with self.assertRaisesRegex(RuntimeError, "decision construction failed"):
            build_demo(self.demo, decision_factory=decision_factory)
        self.assertEqual(contexts[0].sandbox.active_count, 0)

    def test_application_closes_shared_local_sessions_after_factory_failure(self):
        contexts = []

        def exploration_factory(context):
            contexts.append(context)
            context.sandbox.create(SandboxSpec("local"))
            raise RuntimeError("factory failed")

        with self.assertRaisesRegex(RuntimeError, "factory failed"):
            Application(replace(self.config, execution="local_process"), exploration_factory=exploration_factory)
        self.assertEqual(contexts[0].sandbox.active_count, 0)

    def test_cleanup_failure_does_not_mask_body_error_or_skip_other_callbacks(self):
        for kind in ("application", "demo"):
            with self.subTest(kind=kind):
                events = []

                def exploration_factory(context):
                    context.add_cleanup(lambda: events.append("last"))

                    def fail():
                        events.append("first")
                        raise OSError("cleanup failed")

                    context.add_cleanup(fail)
                    return build_default_exploration(context)

                with self.assertRaisesRegex(ValueError, "body failed") as raised:
                    with self.owner(kind, exploration_factory=exploration_factory):
                        raise ValueError("body failed")
                self.assertEqual(events, ["first", "last"])
                self.assertTrue(any("cleanup failed" in note for note in raised.exception.__notes__))


if __name__ == "__main__":
    unittest.main()
