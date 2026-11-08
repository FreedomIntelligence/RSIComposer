"""External capability access, including revocation between generation and execution."""
from tests import ROOT as REPO_ROOT
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from rsisys.application import Application, ApplicationConfig
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.model_service import BackendBinding, GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import OperationCall, OperationResult
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot
import tests.decision.test_researcher_review as fixtures
import tests.decision.test_interaction as interaction
class ActionPermissionsTests(unittest.TestCase):
    def setUp(self):
        # Reuse the fixture, not its test class, to avoid rerunning inherited tests.
        self.fixture = fixtures.ResearcherReviewTests()
        self.fixture.setUp()
        self.harness = self.fixture.harness
        self.loop = self.fixture.loop
        self.space = self.loop.action_space
        self.policy = self.fixture.policy

    def test_catalog_is_separate_from_current_allowed_actions(self):
        self.space.permissions = ActionPermissions(operations={'inspect'})
        self.assertEqual([s.name for s in self.space.describe()], ['inspect', 'modify'])
        self.assertEqual([s.name for s in authorized_space(self.space).describe()], ['inspect'])
        with self.assertRaises(PermissionError):
            authorized_space(self.space).validate(OperationCall('modify', {}))

    def test_open_review_can_execute_a_non_read_only_capability(self):
        self.space.permissions = ActionPermissions.unrestricted()
        self.fixture.responses({'kind': 'invoke', 'operation': 'modify', 'arguments': {}},
                               {'action': 'continue'})
        self.assertEqual(self.loop.review('task', self.fixture.progress).action, 'continue')
        self.fixture.write.assert_called_once_with({})
        entries = self.harness.memory.history()
        self.assertTrue(any(row.details.get('event') == 'review_operation' for row in entries))
        self.assertEqual(self.harness.pending(), self.fixture.directive)

    def test_named_review_allowlist_can_include_writes_without_other_operations(self):
        self.space.permissions = ActionPermissions(review_operations={'modify'})
        self.assertEqual([s.name for s in authorized_space(self.space, 'review').describe()], ['modify'])
        self.assertEqual([s.name for s in authorized_space(self.space).describe()], ['inspect', 'modify'])

    def test_revoke_after_model_generation_blocks_actual_execution(self):
        self.space.permissions = ActionPermissions.unrestricted()
        contexts = []
        def generate(request):
            contexts.append(json.loads(request.prompt))
            if len(contexts) == 1:
                self.assertIn('modify', [s['name'] for s in contexts[-1]['operations']])
                self.space.permissions = ActionPermissions(review_operations={'inspect'})
                payload = {'kind': 'invoke', 'operation': 'modify', 'arguments': {}}
            else:
                payload = {'action': 'continue'}
            return GenerationResponse(json.dumps(payload), self.fixture.model, False)
        self.fixture.backend.generate.side_effect = generate
        self.loop.review('task', self.fixture.progress)
        self.fixture.write.assert_not_called()
        self.assertEqual(contexts[-1]['operation_results'][-1]['status'], 'failed')
        self.assertIn('PermissionError', contexts[-1]['operation_results'][-1]['error'])
        self.assertNotIn('modify', [s['name'] for s in contexts[-1]['operations']])

    def test_execution_predicate_sees_validated_call_and_can_deny(self):
        seen = []
        def check(request):
            seen.append(request)
            return request.action is None or request.name != 'modify'
        self.space.permissions = ActionPermissions(review_operations='all', check=check)
        self.assertIn('modify', [s.name for s in authorized_space(self.space, 'review').describe()])
        self.fixture.responses({'kind': 'invoke', 'operation': 'modify', 'arguments': {}},
                               {'action': 'continue'})
        self.loop.review('task', self.fixture.progress)
        self.fixture.write.assert_not_called()
        actual = next(r for r in seen if isinstance(r.action, OperationCall))
        self.assertEqual(actual.action.arguments, {})
        self.assertEqual(actual.phase, 'review')

    def test_ordinary_decision_also_checks_execution_permission(self):
        self.harness.state.pending_directive_id = None
        self.harness.state.pending_directive = None
        def decide(context):
            self.assertIn('modify', [s.name for s in context.operations])
            self.space.permissions = ActionPermissions(operations={'inspect'})
            return Decision('invoke', operation=OperationCall('modify', {}))
        self.policy.decide = decide
        with self.assertRaises(PermissionError):
            self.loop.step('task')
        self.fixture.write.assert_not_called()

    def test_directive_permissions_cover_discovery_and_dispatch(self):
        self.harness.state.pending_directive_id = None
        self.harness.state.pending_directive = None
        self.space.permissions = ActionPermissions(directives={'execute'})
        view = authorized_space(self.space)
        self.assertEqual(view.directive_modes(), ('execute',))
        self.policy.decide = Mock(return_value=Decision('explore', directive=Directive(
            'new', 'task', 'train', mode='sft')))
        with self.assertRaises(PermissionError):
            self.loop.step('task')
        self.assertIsNone(self.harness.pending())

    def test_cancel_is_configurable_and_checked_after_generation(self):
        def review(context):
            self.assertIn('cancel', context.review_actions)
            self.space.permissions = ActionPermissions(allow_cancel=False)
            return ReviewDecision('cancel')
        self.policy.review = review
        with self.assertRaises(PermissionError):
            self.loop.review('task', self.fixture.progress)
        self.policy.review = lambda context: ReviewDecision('continue')
        self.assertEqual(self.loop.review('task', self.fixture.progress).action, 'continue')
        self.assertEqual(authorized_space(self.space, 'review').review_actions(), ('continue',))

    def test_cancel_revocation_during_recording_is_checked_before_return(self):
        record = self.harness.experience.record
        def revoke(entry):
            record(entry)
            self.space.permissions = ActionPermissions(allow_cancel=False)
        self.harness.experience.record = revoke
        self.policy.review = lambda context: ReviewDecision('cancel')
        with self.assertRaises(PermissionError):
            self.loop.review('task', self.fixture.progress)

    def test_uncertain_cancel_retry_rechecks_permissions_and_discards_old_intent(self):
        setup = interaction.InteractionTests()
        setup.setUp()
        setup.slow(ReviewDecision('cancel'), ReviewDecision('continue'))
        setup.runner.step('task')
        setup.now = 6
        cancel = setup.ex.cancel
        setup.ex.cancel = Mock(side_effect=OSError('not yet accepted'))
        with self.assertRaises(OSError):
            setup.runner.step('task')
        setup.dm.action_space.permissions = ActionPermissions(allow_cancel=False)
        with self.assertRaises(PermissionError):
            setup.runner.step('task')
        self.assertEqual(setup.ex.cancel.call_count, 1)
        self.assertEqual(setup.runner.step('task').kind, 'waiting')
        self.assertEqual(setup.ex.cancel.call_count, 1)
        # Explicit operator cancellation remains independent of model permissions.
        setup.ex.cancel = cancel
        setup.runner.cancel()
        self.assertEqual(setup.runner.step('task').kind, 'completed')

    def test_open_permissions_do_not_install_unknown_tools(self):
        self.space.permissions = ActionPermissions.unrestricted()
        with self.assertRaises(ValueError):
            authorized_space(self.space).validate(OperationCall('host_shell', {'command': 'anything'}))

    def test_async_review_submission_keeps_job_reference_without_replacing_experiment(self):
        self.space.permissions = ActionPermissions.unrestricted()
        self.fixture.write.return_value = JobRef('analysis-1', 'external')
        self.fixture.responses({'kind': 'invoke', 'operation': 'modify', 'arguments': {}},
                               {'action': 'continue'})
        self.loop.review('task', self.fixture.progress)
        context = json.loads(self.fixture.backend.generate.call_args.args[0].prompt)
        result = context['operation_results'][-1]
        self.assertEqual(result['data'], {'status': 'submitted', 'job': {
            'job_id': 'analysis-1', 'backend': 'external'}})
        self.assertIsNone(self.harness.state.pending_job)
        self.assertEqual(self.harness.pending(), self.fixture.directive)

    def test_config_validation_and_copy_roundtrip(self):
        for kwargs in ({'operations': 'inspect'}, {'directives': [1]}, {'review_operations': None},
                       {'review_operations': 'invalid'}, {'allow_cancel': 1}, {'check': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                ActionPermissions(**kwargs)
        policy = ActionPermissions(operations=['inspect'], review_operations='all', allow_cancel=False)
        self.assertEqual(ActionPermissions.from_dict(json.loads(json.dumps(policy.to_dict()))), policy)
        with self.assertRaises(TypeError):
            ActionPermissions(check=lambda request: True).to_dict()


class PermissionAssemblyTests(unittest.TestCase):
    def test_demo_exposes_external_permission_configuration(self):
        config = DemoConfig.load(REPO_ROOT / 'configs/demo.json')
        permissions = ActionPermissions.unrestricted()
        runner, _ = build_demo(config, action_permissions=permissions)
        with runner:
            self.assertIs(runner.decision_making.action_space.permissions, permissions)

    def test_snapshot_preserves_permission_scope_and_rejects_callback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = ModelRef('mock', 'model', 'v1')
            config = ApplicationConfig('task', root / 'run', {'mock': BackendBinding('mock', 'mock')},
                                       {'decision_maker': model, 'actor': model})
            permissions = ActionPermissions(review_operations={'prepare_data'}, allow_cancel=False)
            with Application(config, action_permissions=permissions) as app:
                manifest = LocalSnapshotCapture(app).capture('permissions')
                directory = config.data_dir / 'snapshots' / manifest.snapshot_id
                app.runner.decision_making.action_space.permissions = replace(permissions, check=lambda r: True)
                with self.assertRaisesRegex(TypeError, 'permission callbacks'):
                    LocalSnapshotCapture(app).capture('custom-permissions')
            with restore_snapshot(directory, root / 'restored') as restored:
                self.assertEqual(restored.runner.decision_making.action_space.permissions, permissions)


if __name__ == '__main__':
    unittest.main()
