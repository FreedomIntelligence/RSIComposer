"""Feedback adapters work without default state, and preserve DSH sessions."""
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import time
import unittest
from unittest.mock import patch

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse, ModelRef
from rsisys.loops.decision_making.feedback import FeedbackAdapter, FeedbackEvidence
from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning, feedback_learning_decision_factory
from rsisys.loops.decision_making.harness.dsh import DeepSeekDecisionHarness, DSHFeedbackBackend, dsh_decision_factory
from rsisys.loops.decision_making.harness.context import IdentityContextCompactor
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.config import build_demo
from tests.test_feedback_learning_integration import Backend, ImmediateExecutor
from tests.test_dsh_harness import Runtime, CONFIG, DSH


class AlienHarness:
    """No default policy/state/memory/context attributes."""
    def __init__(self):
        self.summaries, self.receipts, self.service = [], [], None

    def receive_feedback(self, directive, feedback):
        if directive.directive_id != feedback.directive_id:
            raise ValueError('wrong directive')
        self.receipts.append(feedback)
        self.service.receive_feedback(directive, feedback)

    def step(self, objective, space):
        self.service.refresh()
        return tuple(self.summaries)


class AlienAdapter:
    def __init__(self, harness):
        self.harness = harness
        self.current_model = ModelRef('fixture', 'feedback', 'one')

    def model(self):
        return self.current_model

    def evidence(self, directive, feedback):
        return FeedbackEvidence()

    def publish(self, memory):
        self.harness.summaries.append(memory)

    def decision_ids(self):
        return ()

    def install(self, service):
        self.harness.service = service


class FeedbackDecouplingTests(unittest.TestCase):
    def test_custom_harness_needs_no_default_policy_state_or_compactor(self):
        harness = AlienHarness()
        loop = DecisionMakingLoop(harness, object())
        adapter, backend = AlienAdapter(harness), Backend()
        installed = enable_feedback_learning(loop, adapter=adapter, backend=backend,
                                            executor=ImmediateExecutor())
        self.assertIs(installed, harness)
        for ident in ('first', 'second'):
            adapter.current_model = ModelRef('fixture', 'feedback', ident)
            loop.receive_feedback(Directive(ident, 'objective', 'test'),
                Feedback(ident, ExecutionStatus.COMPLETED, 'Measured', True, quality_metrics={'accuracy': 0.0}))
            loop.step('objective')
        self.assertEqual([r.model.revision for r in backend.requests], ['first', 'second'])
        self.assertEqual(len(harness.summaries), 2)
        self.assertFalse(hasattr(harness, 'policy'))
        self.assertFalse(hasattr(harness, 'state'))
        self.assertEqual(json.loads(harness.summaries[0].content)['receipt_facts']['quality_metrics'], {'accuracy': 0.0})
        with self.assertRaisesRegex(ValueError, 'already installed'):
            enable_feedback_learning(loop, adapter=adapter, backend=backend)
        loop.feedback_service.close()

    def test_default_install_preserves_instances_and_exposes_raw_evidence(self):
        runner, _ = build_demo(CONFIG)
        loop = runner.decision_making
        harness, policy, compactor = loop.harness, loop.harness.policy, loop.harness.context_compactor
        try:
            enable_feedback_learning(loop, backend=Backend(), executor=ImmediateExecutor())
            self.assertIs(loop.harness, harness)
            self.assertIs(harness.policy, policy)
            self.assertIs(harness.context_compactor, compactor)
            learning = loop.feedback_service.learning
            ident = learning.capture(Directive('raw', 'objective', 'test'),
                Feedback('raw', ExecutionStatus.COMPLETED, 'Measured', True,
                         quality_metrics={'accuracy': 0.0}), ModelRef('fixture', 'decision', 'v1'))
            from rsisys.loops.decision_making.action_space import OperationCall
            call = loop.action_space.validate(OperationCall('feedback_learning_read', {
                'job_id': ident, 'evidence_path': '/receipt/feedback/quality_metrics', 'limit': 1}))
            receipt = harness.tools.execute(call)
            self.assertEqual(json.loads(receipt.data['text']), {'accuracy': 0.0})
            self.assertFalse(receipt.data['training_ready'])
        finally:
            runner.close()

    def test_snapshot_rejects_unintegrated_feedback_service(self):
        from dataclasses import replace
        from rsisys.application import Application, ApplicationConfig
        from rsisys.snapshot.local import LocalSnapshotCapture
        with TemporaryDirectory() as directory:
            config = replace(ApplicationConfig.load(Path('configs/local.json')), data_dir=Path(directory))
            with Application(config) as app:
                enable_feedback_learning(app.runner.decision_making, backend=Backend())
                try:
                    with self.assertRaisesRegex(TypeError, 'Feedback service'):
                        LocalSnapshotCapture(app).capture('feedback')
                    self.assertFalse((Path(directory) / 'snapshots').exists())
                finally:
                    app.runner.decision_making.feedback_service.close()

    def test_dsh_retains_harness_session_and_own_compactor(self):
        runtime = Runtime(['{"kind":"explore","strategy":"Collect one receipt"}', '{"kind":"finish"}'])
        backend = Backend()
        factory = feedback_learning_decision_factory(decision_factory=dsh_decision_factory(DSH),
            backend=backend, executor=ImmediateExecutor())
        with patch('rsisys.loops.decision_making.harness.dsh._sdk_runtime', return_value=runtime):
            runner, _ = build_demo(CONFIG, decision_factory=factory)
        try:
            loop = runner.decision_making
            self.assertIsInstance(loop.harness, DeepSeekDecisionHarness)
            compactor = loop.harness.context_compactor
            self.assertIsInstance(compactor, IdentityContextCompactor)
            def ready():
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    result = loop.step('objective')
                    if result.kind != 'waiting':
                        return result
                    time.sleep(.002)
                self.fail('DSH inference did not complete')
            first = ready()
            self.assertEqual(first.kind, 'dispatch')
            loop.receive_feedback(first.directive, Feedback(first.directive.directive_id,
                ExecutionStatus.COMPLETED, 'Negative observation', True, quality_metrics={'accuracy': 0.0}))
            self.assertEqual(ready().kind, 'finished')
            self.assertEqual(len(backend.requests), 1)
            self.assertEqual(runtime.opened, [('start', DSH.session_id)])
            self.assertEqual(len(runtime.prompts), 2)
            self.assertIn('research_state', runtime.prompts[-1])
            self.assertIn('Observed fixture result', runtime.prompts[-1])
            self.assertIs(loop.harness.context_compactor, compactor)
            self.assertIsInstance(loop.harness.policy.parser.context_compactor, IdentityContextCompactor)
        finally:
            runner.close()
        self.assertTrue(runtime.closed)
        self.assertTrue(loop.feedback_service.learning._closed)

    def test_same_dsh_implementation_uses_separate_feedback_sessions(self):
        created = []
        def factory(config):
            runtime = Runtime(['{"proposals":[]}'])
            created.append((config, runtime))
            return runtime
        backend = DSHFeedbackBackend(DSH, runtime_factory=factory)
        request = GenerationRequest('decision_maker', ModelRef('dsh', DSH.model, 'pinned'),
                                    '{"receipt":{}}', system='Return feedback proposals', max_output_tokens=2048)
        for _ in range(2):
            self.assertEqual(json.loads(backend.generate(request).text), {'proposals': []})
        self.assertEqual(len({cfg.session_id for cfg, _ in created}), 2)
        for config, runtime in created:
            self.assertNotEqual(config.session_id, DSH.session_id)
            self.assertFalse(config.resume)
            self.assertTrue(runtime.closed)
            self.assertIn('Return feedback proposals', runtime.prompts[0])
            self.assertNotIn('Propose one JSON action', runtime.prompts[0])
