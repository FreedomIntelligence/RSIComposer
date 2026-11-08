"""Replaceable modules round-trip without default harness internals or effect replay."""
from tests import ROOT as REPO_ROOT
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from rsisys.adapters.execution import OperationOnlyExploration
from rsisys.application import Application, ApplicationConfig
from rsisys.assembly import ModuleAssembly, AssemblyRunner
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import LoopStepResult
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.research import ResearchAssembly
from rsisys.runtime.component_state import AssemblyRecovery
from rsisys.runtime.recovery import RecoveryRequired, capture_runner, restore_runner
from rsisys.runtime.run_policy import TimeRunPolicy
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot


ROOT = REPO_ROOT


class Counter:
    """External harness: no builtin state, policy, notebook or memory attributes."""
    def __init__(self):
        self.count, self.closed, self.fail = 0, 0, False

    def step(self, objective, space):
        self.count += 1
        if self.fail:
            raise RuntimeError('uncertain fixture boundary')
        return LoopStepResult('progressed', reason=str(self.count))

    def pending(self):
        return None

    def receive_feedback(self, directive, feedback):
        pass

    def close(self):
        self.closed += 1


class NativeLab:
    def __init__(self):
        self.receipts, self.submissions, self.closed, self.active = [], 0, 0, False

    def describe(self):
        return ()

    def execute(self, call):
        self.submissions += 1
        raise AssertionError('Assembly/restore submitted new work')

    def close(self):
        self.closed += 1


class Port:
    def __init__(self, ident, capture=lambda: None, restore=lambda value: None, validate=lambda: None,
                 config=None):
        self.state_id = ident
        self._capture, self._restore, self._validate, self.config = capture, restore, validate, config

    def capture(self):
        return deepcopy(self._capture())

    def restore(self, state):
        return self._restore(deepcopy(state))

    def validate_snapshot(self):
        return self._validate()

    def configuration(self):
        return self.config


class ModuleAssemblyStateTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'),
            data_dir=self.root / 'run', roles={'decision_maker':
                ApplicationConfig.load(ROOT / 'configs/local.json').roles['decision_maker']})
        self.instances, self.ports, self.restores = [], [], []

    def modules(self, *, lab=True, port_id='fixture-counter/v1', port_config=None, portable=True,
                omit=(), state=True):
        def decision(context, exploration):
            harness = Counter()
            self.instances.append(harness)
            return DecisionMakingLoop(harness, ModularDecisionActionSpace(()))

        def states(context, decision, exploration):
            counter = decision.harness
            # A private file is exported by its owner; the host never guesses
            # files to copy from an arbitrary module's workspace.
            memory = context.data_dir / 'native-memory.txt' if context.data_dir else None
            if memory is not None and not memory.exists():
                memory.write_text('initial')

            def restore(value):
                self.restores.append('decision')
                counter.count = value['count']
                if memory is not None:
                    memory.write_text(value['memory'])

            ports = {'decision': Port(port_id,
                lambda: {'count': counter.count, 'memory': memory.read_text() if memory else ''},
                restore, config=port_config), 'exploration': Port('fixture-empty/v1')}
            if not portable:
                ports['decision'].validate_snapshot = None
            if lab:
                native = context.lab_execution
                def restore_lab(value):
                    self.restores.append('lab_execution')
                    native.receipts = value
                def validate_lab():
                    if native.active:
                        raise RuntimeError('Lab still owns work')
                ports['lab_execution'] = Port('fixture-lab/v1', lambda: native.receipts, restore_lab, validate_lab)
            for name in omit:
                ports.pop(name, None)
            self.ports.append(ports)
            return ports
        return ModuleAssembly(decision_factory=decision,
            lab_factory=(lambda context: NativeLab()) if lab else None,
            exploration_factory=None if lab else lambda ctx: OperationOnlyExploration(),
            state_factory=states if state else None)

    def start(self, modules=None, **kwargs):
        app = Application(self.config, modules=modules or self.modules(), **kwargs)
        self.addCleanup(app.close)
        return app

    def snapshot(self, app):
        manifest = LocalSnapshotCapture(app).capture('external modules')
        return app.config.data_dir / 'snapshots' / manifest.snapshot_id

    def test_demo_and_application_share_modules_and_close_without_calls(self):
        modules = self.modules(state=False)
        app = self.start(modules)
        runner, _ = build_demo(DemoConfig.load(ROOT / 'configs/demo.json'), modules=modules)
        self.addCleanup(runner.close)
        self.assertIsInstance(app.runner, AssemblyRunner)
        self.assertIsInstance(runner, AssemblyRunner)
        self.assertEqual([x.count for x in self.instances], [0, 0])
        self.assertEqual(app.runner.step('test').kind, 'progressed')
        self.assertEqual(runner.step('test').kind, 'progressed')
        app.close()
        runner.close()
        app.close()
        runner.close()
        self.assertEqual([x.closed for x in self.instances], [1, 1])
        self.assertEqual(app.assembly.lab_execution.submissions, 0)

    def test_conflicting_old_and_new_arguments_fail_before_creating_workspace(self):
        with self.assertRaisesRegex(ValueError, 'individual module arguments'):
            Application(self.config, modules=self.modules(), lab_factory=lambda ctx: NativeLab())
        self.assertFalse(self.config.data_dir.exists())

    def test_module_validation_rejects_ignored_feedback_bindings(self):
        for fields in ({'research_backend': object()}, {'feedback_options': {'max_calls': 2}},
                       {'research': ResearchAssembly(), 'feedback_options': {'owner': object()}}):
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                ModuleAssembly(**fields)

    def test_durable_external_harness_and_lab_restore_without_resubmission(self):
        modules = self.modules()
        with self.start(modules, durable=True) as app:
            self.assertIsInstance(app.recovery.contract, AssemblyRecovery)
            app.assembly.lab_execution.receipts.append({'job': 'existing', 'status': 'completed'})
            (app.config.data_dir / 'native-memory.txt').write_text('retained memory')
            app.runner.step('test')
            app.runner.step('test')
        with Application.resume(self.config.data_dir, modules=modules) as restored:
            self.assertEqual(restored.runner.decision_making.harness.count, 2)
            self.assertEqual(restored.assembly.lab_execution.receipts[0]['job'], 'existing')
            self.assertEqual((restored.config.data_dir / 'native-memory.txt').read_text(), 'retained memory')
            self.assertEqual(restored.assembly.lab_execution.submissions, 0)
            restored.runner.step('test')
            self.assertEqual(restored.runner.decision_making.harness.count, 3)
            self.assertFalse(hasattr(restored.runner.decision_making.harness, 'policy'))

    def test_full_legacy_recovery_contract_can_explicitly_restore_lab(self):
        class Contract:
            contract_id = 'fixture-full/v1'
            def capture(self, app):
                return {'count': app.runner.decision_making.harness.count}
            def restore(self, app, state):
                app.runner.decision_making.harness.count = state['count']
        modules = self.modules(state=False)
        with self.start(modules, durable=True, recovery_contract=Contract()) as app:
            app.runner.step('test')
        with Application.resume(self.config.data_dir, modules=modules, recovery_contract=Contract()) as app:
            self.assertEqual(app.runner.decision_making.harness.count, 1)
            self.assertEqual(app.assembly.lab_execution.submissions, 0)

    def test_all_identities_checked_before_any_restore(self):
        with self.start(durable=True) as app:
            app.runner.step('test')
        with self.assertRaisesRegex(TypeError, 'identity/configuration mismatch'):
            Application.resume(self.config.data_dir, modules=self.modules(port_id='fixture-counter/v2'))
        self.assertEqual(self.restores, [])
        with self.assertRaisesRegex(TypeError, 'identity/configuration mismatch'):
            Application.resume(self.config.data_dir, modules=self.modules(port_config={'changed': True}))
        self.assertEqual(self.restores, [])

    def test_missing_adapter_fails_and_closes_owned_resources(self):
        with self.assertRaisesRegex(TypeError, 'Missing component state adapters: lab_execution'):
            self.start(self.modules(omit=('lab_execution',)), durable=True)
        self.assertEqual(self.instances[-1].closed, 1)

    def test_feedback_cannot_be_silently_omitted_from_state_contract(self):
        config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'), data_dir=self.root / 'feedback')
        modules = ModuleAssembly(research=ResearchAssembly(), state_factory=lambda ctx, dm, lab: {
            'decision': Port('unused/v1'), 'exploration': Port('unused/v1')})
        with self.assertRaisesRegex(TypeError, 'Missing component state adapters: agent_memory, feedback'):
            Application(config, modules=modules, durable=True)

    def test_unregistered_component_swap_is_blocked_before_step(self):
        app = self.start(durable=True)
        replacement = Counter()
        app.runner.decision_making.harness = replacement
        with self.assertRaisesRegex(TypeError, 'components changed'):
            app.runner.step('test')
        self.assertEqual(replacement.count, 0)

    def test_unregistered_adapter_swap_is_blocked_before_step(self):
        app = self.start(durable=True)
        app.assembly.state_adapters['decision'] = Port('fixture-counter/v1')
        with self.assertRaisesRegex(TypeError, 'adapters changed'):
            app.runner.step('test')
        self.assertEqual(app.runner.decision_making.harness.count, 0)

    def test_interrupted_effect_remains_blocked_until_reconciled(self):
        modules = self.modules()
        with self.start(modules, durable=True) as app:
            app.runner.decision_making.harness.fail = True
            with self.assertRaisesRegex(RuntimeError, 'uncertain fixture'):
                app.runner.step('test')
        with Application.resume(self.config.data_dir, modules=modules) as app:
            self.assertEqual(app.runner.decision_making.harness.count, 1)
            with self.assertRaises(RecoveryRequired):
                app.runner.step('test')
            with self.assertRaisesRegex(RuntimeError, 'Reconcile'):
                self.snapshot(app)
            app.recovery.reconcile(lambda app, incident: None, evidence='fixture verified one local increment')
            app.runner.step('test')
            self.assertEqual(app.runner.decision_making.harness.count, 2)

    def test_snapshot_roundtrip_restores_private_memory_artifacts_and_counts(self):
        modules = self.modules()
        app = self.start(modules)
        artifact = app.artifacts.put(b'original evidence', kind='test')
        (app.config.data_dir / 'native-memory.txt').write_text('context from prior results')
        app.assembly.lab_execution.receipts = [{'job': 'observed', 'status': 'completed'}]
        app.runner.step('test')
        directory = self.snapshot(app)
        self.assertFalse((directory / 'native-memory.txt').exists())
        with restore_snapshot(directory, self.root / 'restored', modules=modules) as restored:
            self.assertEqual(restored.config.run_id, app.config.run_id)
            self.assertEqual(restored.runner.decision_making.harness.count, 1)
            self.assertEqual((restored.config.data_dir / 'native-memory.txt').read_text(),
                             'context from prior results')
            self.assertEqual(restored.artifacts.get(artifact), b'original evidence')
            self.assertEqual(restored.assembly.lab_execution.submissions, 0)
            restored.runner.step('test')
            self.assertEqual(restored.runner.decision_making.harness.count, 2)

    def test_snapshot_requires_portability_opt_in(self):
        app = self.start(self.modules(portable=False), durable=True)
        app.runner.step('test')
        with self.assertRaisesRegex(TypeError, 'portable snapshot integration'):
            self.snapshot(app)
        self.assertFalse((app.config.data_dir / 'snapshots').exists())

    def test_snapshot_rejects_active_external_work_without_side_effects(self):
        app = self.start()
        app.assembly.lab_execution.active = True
        with self.assertRaisesRegex(RuntimeError, 'Lab still owns work'):
            self.snapshot(app)
        self.assertEqual(app.assembly.lab_execution.submissions, 0)

    def test_snapshot_restore_requires_explicit_same_module_integration(self):
        directory = self.snapshot(self.start())
        target = self.root / 'no-integration'
        with self.assertRaisesRegex(TypeError, 'original modules'):
            restore_snapshot(directory, target)
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(TypeError, 'identity/configuration mismatch'):
            restore_snapshot(directory, target, modules=self.modules(port_id='wrong/v1'))
        self.assertFalse(target.exists())
        self.assertEqual(self.restores, [])

    def test_snapshot_corrupt_bytes_fail_before_factories_or_target(self):
        directory = self.snapshot(self.start())
        (directory / 'state.json').write_text('{}')
        factory = Mock()
        target = self.root / 'corrupt'
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            restore_snapshot(directory, target, modules=ModuleAssembly(state_factory=factory))
        factory.assert_not_called()
        self.assertFalse(target.exists())

    def test_portable_snapshot_cannot_reset_run_deadline_or_fork_implicitly(self):
        modules = self.modules()
        app = self.start(modules, run_policy=TimeRunPolicy(1))
        app.runner.step('test')
        with patch('rsisys.snapshot.components.time', return_value=0):
            directory = self.snapshot(app)
        with restore_snapshot(directory, self.root / 'expired', modules=modules) as restored:
            self.assertEqual(restored.runner.step('test').event['run_stop']['reason'], 'budget_exhausted')
            self.assertEqual(restored.runner.decision_making.harness.count, 1)
        with self.assertRaisesRegex(ValueError, 'preserve run identity'):
            restore_snapshot(directory, self.root / 'fork', modules=modules, run_id='new-run')

    def test_runner_shutdown_failure_and_old_field_defaults_roundtrip(self):
        app = self.start()
        app.runner._execution_shutdown_error = {'error_type': 'OSError', 'message': 'cleanup evidence'}
        state = capture_runner(app.runner)
        app.runner._execution_shutdown_error = None
        restore_runner(app.runner, state, 0)
        self.assertEqual(app.runner._execution_shutdown_error['message'], 'cleanup evidence')
        state.pop('_execution_shutdown_error')
        restore_runner(app.runner, state, 0)
        self.assertIsNone(app.runner._execution_shutdown_error)

    def test_portable_restore_failure_closes_owned_components_and_removes_target(self):
        modules = self.modules()
        directory = self.snapshot(self.start(modules))
        def broken_factory(ctx, dm, lab):
            ports = modules.state_factory(ctx, dm, lab)
            def fail(state):
                raise ValueError('invalid native state')
            ports['decision']._restore = fail
            return ports
        target = self.root / 'invalid-state'
        with self.assertRaisesRegex(ValueError, 'invalid native state'):
            restore_snapshot(directory, target, modules=replace(modules, state_factory=broken_factory))
        self.assertFalse(target.exists())
        self.assertEqual(self.instances[-1].closed, 1)

    def test_restore_cannot_drop_module_assembly_even_without_lab(self):
        modules = self.modules(lab=False)
        with self.start(modules, durable=True):
            pass
        with self.assertRaisesRegex(TypeError, 'original module assembly'):
            Application.resume(self.config.data_dir)

    def test_failed_restore_retains_workspace_while_owned_close_needs_retry(self):
        modules = self.modules()
        directory = self.snapshot(self.start(modules))
        captured = []
        original = Application.__init__

        def construct(app, *args, **kwargs):
            original(app, *args, **kwargs)
            captured.append(app)

        def broken_factory(ctx, dm, lab):
            ports = modules.state_factory(ctx, dm, lab)
            def fail_restore(state):
                raise ValueError('invalid state before cleanup')
            ports['decision']._restore = fail_restore
            dm.harness.close = Mock(side_effect=OSError('owned close needs retry'))
            return ports

        target = self.root / 'retained'
        with patch.object(Application, '__init__', construct), self.assertRaisesRegex(
                ValueError, 'invalid state before cleanup') as failure:
            restore_snapshot(directory, target, modules=replace(modules, state_factory=broken_factory))
        self.assertTrue(target.exists())
        self.assertTrue(any('owned resource cleanup is still pending' in note for note in failure.exception.__notes__))
        self.assertIsNotNone(captured[-1].store)
        captured[-1].runner.decision_making.harness.close = lambda: None
        captured[-1].close()


if __name__ == '__main__':
    unittest.main()
