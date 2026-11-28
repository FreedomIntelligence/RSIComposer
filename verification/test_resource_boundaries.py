"""Keyless regressions for isolation and safe resource settlement."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import warnings

from rsicomposer.api import (ActionPermissions, Application, ApplicationConfig, ArtifactRef, BackendBinding,
                             ComponentDefinition, ComponentGraph, ControllerDefinition, ControllerPort,
                             ControllerVersion, Decision, JobRef, JobStatus, MemoryAssembly, ModelRef,
                             ModuleAssembly, OperationCall, OperationResult, OperationSpec, Registry,
                             ResearchAssembly, SEALED)
from rsicomposer.harnesses.defaults import build_default_decision, build_harness_decision
from rsicomposer.loops.contracts.harness import NullHarness
from rsicomposer.loops.decision_making.action_space.access import (
    authorized_space, enforce_host_permissions,
)
from rsicomposer.runtime.execution.nested import NestedRun


def manual_controller_assembly(endpoint, lab_factory, *, close=lambda: None):
    selection = {'name': 'fixture', 'config': {}}

    def controller(context, lab):
        instance = SimpleNamespace(endpoint=endpoint(context, lab),
            context=SimpleNamespace(controller_report=None, _memory_observer=None),
            capture=lambda: {}, restore=lambda state: None, close=close)
        return ControllerPort(instance, ControllerVersion('initial', selection))

    return ModuleAssembly(controller_factory=controller, lab_factory=lab_factory,
        controller_catalog={'fixture': ControllerDefinition(lambda ctx, cfg: lambda window: Decision('finish'))},
        controller_selection=selection)


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
        executed = []
        proposed = []

        class Lab:
            def describe(self):
                return tuple(OperationSpec(name, 'Fixture operation', {'type': 'object'}) for name in names)

            def execute(self, call):
                executed.append((call.name, call.arguments))
                return OperationResult('Fixture invoked', (), True)

        def endpoint(context, lab, config):
            def policy(_):
                if proposed:
                    name, arguments = proposed.pop(0)
                    return Decision('invoke', operations=(OperationCall('fixture.' + name, arguments),))
                return Decision('finish')

            loop = build_harness_decision(context, lab, policy=policy, harness=NullHarness())
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
            'manual-port': manual_controller_assembly(decision_factory, lab_factory),
        }
        permissions = ActionPermissions(['fixture.allowed', 'fixture.endpoint_restricted'],
                                        check=lambda request: request.action is None
                                        or request.action.arguments.get('blocked') is not True)
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
                    executed.clear()
                    proposed.append(('allowed', {}))
                    app.run(max_decision_steps=1, wait_for_jobs=True)
                    self.assertEqual(executed, [('allowed', {})])
                    for name, arguments in [('allowed', {'blocked': True}), *((name, {}) for name in names[1:])]:
                        with self.subTest(denied=name, arguments=arguments):
                            proposed.append((name, arguments))
                            with self.assertRaises(PermissionError):
                                app.run(max_decision_steps=1, wait_for_jobs=True)
                            self.assertEqual(executed, [('allowed', {})])

    def test_fixed_endpoint_authorizes_memory_tools_after_install(self):
        host_denied = OperationCall('research_task_read', {})
        local_denied = OperationCall('memory_read', {'memory_id': 'sample'})
        allowed = OperationCall('memory_search', {})

        def decision(context, lab):
            loop = build_default_decision(context, lab, policy=lambda _: Decision.invoke(host_denied))
            loop.action_space.permissions = ActionPermissions([allowed.name, host_denied.name])
            return loop

        modules = ModuleAssembly(decision_factory=decision, lab_factory=lambda _: SimpleNamespace(
            describe=lambda: (), execute=lambda call: None), memory=MemoryAssembly(ResearchAssembly.single_meta()))
        with TemporaryDirectory() as temp:
            config = ApplicationConfig('Check installed Memory tools', Path(temp),
                {'mock': BackendBinding('mock', 'mock')}, {'decision_maker': ModelRef('mock', 'fixture', 'v1')})
            with Application(config, modules=modules, action_permissions=ActionPermissions(
                    [allowed.name, local_denied.name])) as app:
                space = app.runner.decision_making.action_space
                self.assertEqual([spec.name for spec in space.describe()], [allowed.name])
                self.assertEqual(space.validate(allowed).name, allowed.name)
                for call in (host_denied, local_denied):
                    with self.subTest(operation=call.name), self.assertRaises(PermissionError):
                        space.validate(call)
                with self.assertRaises(PermissionError):
                    app.run(max_decision_steps=1, wait_for_jobs=True)

    def test_manual_port_rejects_miswiring_and_owns_rejected_instance(self):
        events = []

        class Lab:
            def describe(self):
                return ()

            def execute(self, call):
                raise AssertionError('Should fail before dispatch')

            def close(self):
                events.append('lab')

        class Policy:
            def decide(self, window):
                raise AssertionError('Should fail before sampling')

            def required_operations(self):
                return ('missing',) if fault == 'operation' else ()

            def required_profiles(self):
                return {'fixture': 'required/v1'} if fault == 'profile' else {}

        def endpoint(context, lab):
            loop = build_harness_decision(context, lab, policy=Policy(), harness=NullHarness())
            if fault == 'lab':
                loop.operation_router.lab = object()
            if fault == 'host':
                loop.host_execution = object()
            return loop

        modules = manual_controller_assembly(endpoint, lambda _: Lab(), close=lambda: events.append('controller'))
        with self.assertRaisesRegex(ValueError, 'not both'):
            replace(modules, decision_factory=endpoint)
        with TemporaryDirectory() as temp:
            for fault, message in (('lab', 'Lab routing'), ('host', 'another run execution'),
                                   ('operation', 'missing Lab operations'), ('profile', 'validates profiles')):
                events.clear()
                with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, message):
                    Application(ApplicationConfig('Check admission', Path(temp) / fault, {}, {}), modules=modules)
                self.assertEqual(events, ['controller', 'lab'])

    def test_manual_port_can_report_and_observe_on_the_assembly_context(self):
        from rsicomposer.api import save_controller_version

        class Next:
            def decide(self, window):
                return Decision('finish')

            def report(self):
                return {'owner': 'next'}

        selection = {'name': 'initial', 'config': {}}

        def controller(context, lab):
            if configured:
                context.controller_report = lambda: {'owner': 'initial'}
                context._memory_observer = observed.append
            endpoint = build_harness_decision(context, lab, policy=lambda _: Decision('finish'), harness=NullHarness())
            return ControllerPort(SimpleNamespace(endpoint=endpoint, context=context,
                capture=lambda: {}, restore=lambda _: None, close=lambda: None),
                ControllerVersion('manual', selection))

        modules = ModuleAssembly(controller_factory=controller,
            lab_factory=lambda _: SimpleNamespace(describe=lambda: (), execute=lambda call: None),
            controller_catalog={'initial': ControllerDefinition(lambda *_: lambda _: Decision('finish')),
                                'next': ControllerDefinition(lambda *_: Next())}, controller_selection=selection)
        with TemporaryDirectory() as temp:
            for configured in (False, True):
                observed = []
                config = ApplicationConfig('manual report', Path(temp) / str(configured), {}, {})
                with (self.subTest(configured=configured), warnings.catch_warnings(),
                      Application(config, modules=modules) as app):
                    warnings.simplefilter('error', RuntimeWarning)
                    self.assertEqual(app.controller_report(), {'owner': 'initial'} if configured else {})
                    initial = app.assembly.record('initial_report')
                    self.assertEqual(observed, [initial] if configured else [])
                    host = app.assembly.controller_host
                    version = ControllerVersion('v2', {'name': 'next'})
                    host.request_version(save_controller_version(app.artifacts, version))
                    host.advance()
                    self.assertEqual(app.controller_report(), {'owner': 'next'})
                    app.assembly.record('next_report')
                    self.assertEqual(observed, [initial] if configured else [])
                    self.assertEqual(app.run().status, 'finished')

    def test_manual_port_cleanup_retries_without_double_close(self):
        closes = []

        def close():
            closes.append('controller')
            if len(closes) == 1:
                raise RuntimeError('Controller close not yet acknowledged')

        def lab_factory(context):
            return SimpleNamespace(describe=lambda: (), execute=lambda call: None, close=lambda: closes.append('lab'))

        modules = manual_controller_assembly(lambda context, lab: build_harness_decision(context, lab,
            policy=lambda _: Decision('finish'), harness=NullHarness()), lab_factory, close=close)
        with TemporaryDirectory() as temp:
            app = Application(ApplicationConfig('Check close', Path(temp), {}, {}), modules=modules)
            with self.assertRaisesRegex(RuntimeError, 'not yet acknowledged'):
                app.close()
            self.assertEqual(closes, ['controller'])  # Dependencies are retained for retry.
            app.close()
            app.close()
            self.assertEqual(closes, ['controller', 'controller', 'lab'])

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

            def evaluation_operations(self):
                return ('evaluate',)

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
