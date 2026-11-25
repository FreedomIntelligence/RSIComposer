"""Keyless regressions for isolation and safe resource settlement."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rsicomposer.api import (ActionPermissions, Application, ApplicationConfig, ArtifactRef,
                             ComponentDefinition, ComponentGraph, ControllerDefinition, Decision, JobRef, JobStatus,
                             ModuleAssembly, OperationCall, OperationResult, OperationSpec, Registry, SEALED)
from rsicomposer.harnesses.defaults import build_harness_decision
from rsicomposer.loops.contracts.harness import NullHarness
from rsicomposer.loops.decision_making.action_space.access import (
    authorized_space, enforce_host_permissions,
)
from rsicomposer.runtime.execution.nested import NestedRun


class ResourceBoundariesTest(unittest.TestCase):
    def test_unknown_nested_worker_keeps_quota_until_original_is_settled(self):
        execut = Mock()
        reference = {'id': 'original'}
        execut.find_job.side_effect = (None, reference)
        execut.reattach.return_value.poll.return_value = None
        budget = Mock()
        with TemporaryDirectory() as data_dir:
            root = Path(data_dir)
            directory = root / ('a' * 32)
            directory.mkdir()
            descriptor = {'schema': 2, 'job_id': directory.name, 'request_digest': 'b' * 64,
                          'phase': 'uncertain', 'worker': None, 'outcome': None, 'gateway_owner': None}
            path = directory / 'descriptor.json'
            path.write_text(json.dumps(descriptor), encoding='utf-8')
            run = NestedRun(SimpleNamespace(resources=Mock(), executor=execut, budget=budget), root=root)
            ref = JobRef(directory.name, run.backend)

            with self.assertRaisesRegex(RuntimeError, 'not yet confirmed'):
                run.close()
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), descriptor)
            budget.settle_child.assert_not_called()
            execut.launch.assert_not_called()

            run.close()
            execut.launch.assert_not_called()
            execut.reattach.assert_called_once_with(reference)
            execut.reattach.return_value.terminate.assert_called_once()
            budget.settle_child.assert_called_once_with(ref.job_id, closed_tree=True)
            self.assertEqual(run.poll(ref).status, JobStatus.CANCELLED)

    def test_matching_operation_name_does_not_substitute_for_a_required_profile(self):
        class Policy:
            def required_operations(self):
                return ('math.evaluate',)

            def required_profiles(self):
                return {'math': 'audit-profile/v1'}

            def decide(self, window):
                raise AssertionError('the incompatible Controller should not sample')

        class Lab:
            def describe(self):
                return (OperationSpec('evaluate', 'Fixture scoring', {'type': 'object'}),)

            def execute(self, call):
                raise AssertionError('the incompatible Lab should not execute')

        registry = Registry()
        registry.register_controller('profile_fixture', ControllerDefinition(lambda ctx, cfg: Policy()), version='1')
        registry.register_lab('profile_fixture', lambda ctx, cfg: Lab(), version='1')
        modules = registry.build({'controller': {'name': 'profile_fixture'}, 'labs': {'math': {'name': 'profile_fixture'}}})
        with TemporaryDirectory() as temp:
            cfg = ApplicationConfig('Check profile compatibility', Path(temp), {}, {})
            with self.assertRaisesRegex(ValueError, 'audit-profile/v1'):
                with Application(cfg, modules=modules):
                    pass

    def test_explicit_endpoint_permissions_intersect_host_permissions(self):
        names = ('allowed', 'host_restricted', 'endpoint_restricted')

        class Lab:
            def describe(self):
                return tuple(OperationSpec(name, 'Fixture operation', {'type': 'object'}) for name in names)

            def execute(self, call):
                raise AssertionError('Permission inspection must not execute fixture tools')

        def endpoint(context, lab, config):
            loop = build_harness_decision(context, lab, policy=lambda _: Decision('finish'), harness=NullHarness())
            loop.action_space.permissions = ActionPermissions(['fixture.allowed', 'fixture.host_restricted'])
            return loop

        registry = Registry()
        registry.register_controller('permission_fixture', ControllerDefinition(endpoint, endpoint=True), version='1')
        registry.register_lab('permission_fixture', lambda context, config: Lab(), version='1')
        modules = registry.build({'controller': {'name': 'permission_fixture'},
                                  'labs': {'fixture': {'name': 'permission_fixture'}}})
        from rsicomposer.runtime.orchestration.lab_group import LabGroup

        def lab_factory(context):
            group = LabGroup()
            group.add('fixture', Lab())
            return group

        def decision_factory(context, lab):
            return endpoint(context, lab, {})

        graph = ComponentGraph({
            'controller': ComponentDefinition(lambda ctx, deps, cfg: decision_factory, '1', ('decision-factory/v1',)),
            'lab': ComponentDefinition(lambda ctx, deps, cfg: lab_factory(ctx), '1', ('lab-execution/v1',)),
        }, {'decision': {'component': 'controller'}, 'tools': {'component': 'lab'}})
        entries = {
            'registry': modules,
            'direct': ModuleAssembly(decision_factory=decision_factory, lab_factory=lab_factory),
            'graph': ModuleAssembly.from_components(graph, decision='decision', lab='tools'),
        }
        permissions = ActionPermissions(['fixture.allowed', 'fixture.endpoint_restricted'])
        with TemporaryDirectory() as temp:
            for entry, selection in entries.items():
                config = ApplicationConfig('Check both permission boundaries', Path(temp) / entry, {}, {})
                with self.subTest(entry=entry), Application(config, modules=selection,
                                                            action_permissions=permissions) as app:
                    space = authorized_space(app.runner.decision_making.action_space)
                    self.assertEqual([spec.name for spec in space.describe()], ['fixture.allowed'])
                    self.assertEqual(space.validate(OperationCall('fixture.allowed', {})).name, 'fixture.allowed')
                    for name in names[1:]:
                        with self.subTest(name=name), self.assertRaises(PermissionError):
                            space.validate(OperationCall('fixture.' + name, {}))

    def test_rebinding_same_permission_does_not_duplicate_checks(self):
        spec = OperationSpec('fixture.inspect', 'Fixture operation', {'type': 'object'}, read_only=True)
        check = Mock(return_value=True)
        permissions = ActionPermissions(check=check)
        source = SimpleNamespace(describe=lambda: (spec,), validate=lambda call: call, permissions=permissions)
        space = enforce_host_permissions(source, permissions)
        self.assertIs(enforce_host_permissions(authorized_space(space), permissions), space)
        self.assertEqual(space.describe(), (spec,))
        call = OperationCall(spec.name, {})
        self.assertEqual(space.validate(call), call)
        requests = [record.args[0] for record in check.call_args_list]
        self.assertEqual([request.action for request in requests], [None, call])
        self.assertTrue(all(request.read_only for request in requests))

    def test_sealed_final_runs_only_on_host_and_keeps_rows_private(self):
        effects, windows = [], []
        secret_row = 'SEALED_FIXTURE_ANSWERS_MUST_NOT_REACH_CONTROLLER'

        class Policy:
            def decide(self, window):
                windows.append(window)
                if len(windows) == 1:
                    return Decision('invoke', operations=(OperationCall('math.evaluate', {'split': 'sealed'}),))
                return Decision('finish')

        class Lab:
            def describe(self):
                return (OperationSpec('evaluate', 'Fixture scoring', {'type': 'object'}),)

            def split_roles(self):
                return {'sealed': SEALED}

            def final_evaluations(self):
                return (OperationCall('evaluate', {'split': 'sealed'}, call_id='fixture-final'),)

            def execute(self, call):
                effects.append(call.call_id)
                return OperationResult(secret_row, (), True, data={'metrics': {'accuracy': 1.0}, 'rows': [secret_row]})

        registry = Registry()
        registry.register_controller('sealed_fixture', ControllerDefinition(lambda context, config: Policy()), version='1')
        registry.register_lab('sealed_fixture', lambda context, config: Lab(), version='1')
        modules = registry.build({'controller': {'name': 'sealed_fixture'}, 'labs': {'math': {'name': 'sealed_fixture'}}})
        with TemporaryDirectory() as temp:
            cfg = ApplicationConfig('Verify sealed evidence boundaries', Path(temp), {}, {})
            with Application(cfg, modules=modules) as app:
                result = app.run(wait_for_jobs=True)
                self.assertEqual(result.status, 'finished')
                self.assertEqual(effects, ['fixture-final'])
                self.assertEqual(len(windows), 2)
                self.assertNotIn(secret_row, repr(windows))
                self.assertNotIn(secret_row, repr(result))
                self.assertTrue(any('sealed split is available only to the host' in repr(event) for event in result.events))
                final, = [event['receipt'] for event in result.events if event.get('phase') == 'host_finish']
                self.assertEqual(final['data'], {'metrics': {'accuracy': 1.0}})
                artifact = ArtifactRef(**final['artifacts'][0])
                self.assertIn(secret_row.encode(), app.assembly.artifacts.read(artifact))
                with self.assertRaises(PermissionError):
                    app.assembly.evidence_view().read(artifact)
