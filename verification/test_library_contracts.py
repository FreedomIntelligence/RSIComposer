"""Public assembly and scientific evidence contracts, exercised without a model backend."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsicomposer.api import (
    Application, ApplicationConfig, ComponentDefinition, ComponentGraph, ControllerDefinition,
    Decision, GoalRunPolicy, JobRef, MetricGoal, ModuleAssembly, OperationCall, OperationResult, OperationSpec,
    Registry, RunDecision, RunResult, TimeRunPolicy, build_default_decision,
)
from rsicomposer.loops.contracts.profiles import SEALED, Invariant
from rsicomposer.runtime.orchestration.lab_group import LabGroup
from rsicomposer.runtime.orchestration.run_policy import EvaluationOutcome
from rsicomposer.runtime.orchestration.runner import Runner


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, _):
        call = next(self.actions, None)
        return Decision('invoke', operations=(call,)) if call else Decision('finish')


class LibraryContractTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def config(self, name):
        return ApplicationConfig('Check library contracts', self.root / name, {}, {})

    def test_direct_and_graph_decision_factories_are_activated(self):
        def decision(context, lab):
            return build_default_decision(context, lab, policy=ScriptedPolicy(()))

        graph = ComponentGraph({
            'decide': ComponentDefinition(lambda ctx, deps, cfg: decision, '1', ('decision-factory/v1',)),
            'lab': ComponentDefinition(lambda ctx, deps, cfg: LabGroup(), '1', ('lab-execution/v1',)),
        }, {'policy': {'component': 'decide'}, 'lab': {'component': 'lab'}})
        for name, modules in (
            ('direct', ModuleAssembly(decision_factory=decision, lab_factory=lambda _: LabGroup())),
            ('graph', ModuleAssembly.from_components(graph, decision='policy', lab='lab')),
        ):
            with self.subTest(name=name), Application(self.config(name), modules=modules) as app:
                result = app.run(wait_for_jobs=True)
                self.assertIsInstance(result, RunResult)
                self.assertEqual(result.stop_reason, 'researcher_finished')
                self.assertEqual(result.decision_steps, 1)

    def test_direct_lab_evaluation_declaration_supports_metric_goal(self):
        args = {'candidate_id': 'one', 'protocol_ref': 'fixture/v1', 'input_refs': ['dataset:a']}
        call = OperationCall('direct.measure', args, call_id='measure')

        class LabExecution:
            def describe(self):
                return (OperationSpec('direct.measure', 'Fixture', {'type': 'object', 'properties': {
                    'candidate_id': {'type': 'string'}, 'protocol_ref': {'type': 'string'},
                    'input_refs': {'type': 'array', 'items': {'type': 'string'}},
                }}),)

            def evaluation_operations(self):
                return ('direct.measure',)

            def execute(self, received):
                return OperationResult('evaluation', (), False, data={**received.arguments,
                                       'quality_metrics': {'accuracy': .91}})

        def decision(context, lab):
            return build_default_decision(context, lab, policy=ScriptedPolicy((call,)))

        modules = ModuleAssembly(decision_factory=decision, lab_factory=lambda _: LabExecution())
        goal = MetricGoal('accuracy', .9, 'fixture/v1', input_refs=('dataset:a',))
        with Application(self.config('direct-goal'), modules=modules, run_policy=GoalRunPolicy(goal)) as app:
            result = app.run(max_decision_steps=2, wait_for_jobs=True)
        self.assertEqual(result.stop_reason, 'goal_reached')
        self.assertEqual(result.decision_steps, 1)

    def test_direct_lab_evaluation_identity_is_checked_on_durable_resume(self):
        class Lab:
            def __init__(self, declared):
                self.declared = declared

            def describe(self):
                return (OperationSpec('example.measure', 'Measure', {'type': 'object'}),)

            def evaluation_operations(self):
                return self.declared

            def execute(self, _):
                raise AssertionError('restore should not execute Lab work')

        class Contract:
            contract_id = 'fixture-no-work/v1'

            def capture(self, app):
                return {}

            def restore(self, app, state):
                self.restored = state

        def decision(context, lab):
            return build_default_decision(context, lab, policy=ScriptedPolicy(()))

        def modules(declared):
            return ModuleAssembly(decision_factory=decision, lab_factory=lambda _: Lab(declared))

        cfg = self.config('direct-goal-durable')
        with Application(cfg, modules=modules(('example.measure',)), durable=True, recovery_contract=Contract()):
            pass
        with Application.resume(cfg.data_dir, modules=modules(('example.measure',)), recovery_contract=Contract()):
            pass
        with self.assertRaisesRegex(ValueError, 'evaluation operations differ'):
            Application.resume(cfg.data_dir, modules=modules(()), recovery_contract=Contract())

    def test_rejecting_a_foreign_host_does_not_close_its_harness(self):
        closed = []

        def decision(context, lab):
            endpoint = build_default_decision(context, lab, policy=ScriptedPolicy(()))
            endpoint.harness.close = lambda: closed.append('owner')
            return endpoint

        owner = ModuleAssembly(decision_factory=decision, lab_factory=lambda _: LabGroup())
        with Application(self.config('owner'), modules=owner) as app:
            foreign = app.assembly.host_execution.endpoint
            borrower = ModuleAssembly(decision_factory=lambda ctx, lab: foreign, lab_factory=lambda _: LabGroup())
            with self.assertRaisesRegex(ValueError, 'belongs to another run execution'):
                Application(self.config('borrower'), modules=borrower)
            self.assertEqual(closed, [])
            self.assertEqual(app.run().stop_reason, 'researcher_finished')
        self.assertEqual(closed, ['owner'])

    def test_role_declaration_failure_closes_new_lab(self):
        closed = []

        class Lab:
            def resource_roles(self):
                raise ValueError('invalid roles')

            def describe(self):
                return ()

            def execute(self, _):
                raise AssertionError('should never dispatch')

            def close(self):
                closed.append(True)

        with self.assertRaisesRegex(ValueError, 'invalid roles'):
            Application(self.config('roles'), modules=registry_modules(Lab, ()))
        self.assertEqual(closed, [True])

    def test_unsupported_dotted_local_names_fail_during_assembly(self):
        class Lab:
            def describe(self):
                return (OperationSpec('task.inspect', 'Fixture', {'type': 'object'}),)

            def execute(self, _):
                raise AssertionError('should never dispatch')

        with self.assertRaisesRegex(ValueError, 'local operation names cannot contain dots'):
            Application(self.config('names'), modules=registry_modules(Lab, ()))

    def test_standalone_lab_constraints_contribute_to_identity(self):
        class Lab:
            def describe(self):
                return ()

            def execute(self, _):
                raise AssertionError('should never dispatch')

            def split_roles(self):
                return {'reserved': SEALED}

            def invariants(self):
                return (Invariant.sealed_after_finish(),)

        group = LabGroup()
        group.add('test', Lab())
        identity = group.profile_identity()['test']
        self.assertIn('reserved', identity['split_roles'])
        self.assertEqual(identity['invariants'][0]['kind'], 'sealed_after_finish')

    def test_registry_build_does_not_call_controller_memory_factory(self):
        made = []

        def memory(config):
            made.append(config)
            return None

        class Lab:
            def describe(self):
                return ()

            def execute(self, _):
                raise AssertionError('should never dispatch')

        modules = registry_modules(Lab, (), memory=memory)
        self.assertEqual(made, [])
        with Application(self.config('memory'), modules=modules):
            self.assertEqual(made, [{}])

    def test_stop_timeout_does_not_forget_a_previously_settled_lab(self):
        elapsed = [0.0]
        shutdown = []

        class Waiting:
            def pending_research_job(self):
                return JobRef('fixture-job', 'fixture')

        def stop_lab(mode):
            shutdown.append(mode)
            return True

        runner = Runner(Waiting(), clock=lambda: elapsed[0], cleanup_timeout_seconds=2,
                        execution_shutdown=stop_lab)
        runner._begin_stop(RunDecision('stop', 'fixture', 'cancel'))
        self.assertEqual(runner.step('fixture').kind, 'waiting')
        elapsed[0] = 3.0
        stop = runner.step('fixture').event['run_stop']
        self.assertEqual(shutdown, ['cancel'])
        self.assertTrue(stop['cleanup_timed_out'])
        self.assertTrue(stop['unresolved_research_job'])
        self.assertFalse(stop['unresolved_execution'])

    def test_retry_evaluation_is_considered_the_latest_delivery(self):
        from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
        args = {'candidate_id': 'one', 'protocol_ref': 'fixture/v1', 'input_refs': ['dataset:a']}

        class EvaluationRunner(Runner):
            def _classify_outcome(self, receipt, decision_id):
                return EvaluationOutcome.from_call(OperationCall('test.measure', args), receipt)

        runner = EvaluationRunner(object(), clock=lambda: 0.0)
        runner._run_started = 0.0
        goal = MetricGoal('accuracy', .9, 'fixture/v1')
        runner._record_outcome(OperationResult('uncertain', (), False, ExecutionStatus.FAILED, 'unknown',
                               data={**args, 'effects_uncertain': True, 'retryable': True}, call_id='a'), None)
        runner._record_outcome(OperationResult('missed', (), False,
                               data={**args, 'quality_metrics': {'accuracy': .4}}, call_id='b'), None)
        self.assertFalse(goal(runner._run_context('fixture')))
        runner._record_outcome(OperationResult('succeeded on retry', (), False,
                               data={**args, 'quality_metrics': {'accuracy': .95}}, call_id='a'), None)
        self.assertTrue(goal(runner._run_context('fixture')))
        runner._record_outcome(OperationResult('not measured', (), False,
                               data={**args, 'quality_metrics': {'accuracy': 1.0}, 'scored': False}, call_id='c'), None)
        self.assertFalse(goal(runner._run_context('fixture')))

    def test_hard_stop_reports_receipt_recovered_from_pending_observation(self):
        attempts = []
        calls = []

        class Lab:
            def describe(self):
                return (OperationSpec('inspect', 'Fixture', {'type': 'object'}),)

            def execute(self, _):
                calls.append('inspect')
                return OperationResult('Recorded once', (), False)

        call = OperationCall('test.inspect', {}, call_id='inspection')
        with Application(self.config('terminal-receipt'), modules=registry_modules(Lab, (call,)),
                         run_policy=TimeRunPolicy(1)) as app:
            time = [10.0]
            app.runner._clock = lambda: time[0]
            harness = app.assembly.host_execution.harness
            original_observe = harness.observe

            def observe(event):
                if event.task_id == call.call_id:
                    attempts.append(event.event_id)
                    if len(attempts) == 1:
                        raise ValueError('temporary receipt observer failure')
                return original_observe(event)

            harness.observe = observe
            with self.assertRaisesRegex(ValueError, 'temporary receipt observer failure'):
                app.run(max_decision_steps=1, wait_for_jobs=True)
            time[0] = 12.0
            result = app.run(wait_for_jobs=True)
            self.assertTrue(result.shutdown_complete)
            delivered, = (event['operation_result'] for event in result.events if event.get('operation_result'))
            self.assertEqual(delivered['call_id'], call.call_id)
            self.assertIn(call.call_id, app.runner._outcomes)
            self.assertFalse(any(event.get('operation_result') for event in app.run().events))
        self.assertEqual(calls, ['inspect'])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(*attempts)

    def test_goal_ignores_training_unmatched_inputs_and_unverified_identity(self):
        calls = []

        class Lab:
            def describe(self):
                schema = {'type': 'object', 'properties': {
                    'candidate_id': {'type': 'string'}, 'protocol_ref': {'type': 'string'},
                    'input_refs': {'type': 'array', 'items': {'type': 'string'}},
                }}
                return (OperationSpec('train', 'Train', schema),
                        OperationSpec('measure', 'Measure', schema))

            def evaluation_operations(self):
                return ('measure',)

            def execute(self, call):
                calls.append(call.call_id)
                data = {**call.arguments, 'quality_metrics': {'accuracy': .99}}
                if call.call_id == 'mismatch':
                    data['candidate_id'] = 'not-the-requested-model'
                return OperationResult(call.name, (), call.call_id == 'simulated', data=data)

        matched = {'candidate_id': 'model', 'protocol_ref': 'evaluator/v2', 'input_refs': ['cohort:valid']}
        plan = tuple(OperationCall(f'test.{name}', args, call_id=ident) for ident, name, args in (
            ('train', 'train', matched), ('wrong_inputs', 'measure', {**matched, 'input_refs': ['cohort:train']}),
            ('mismatch', 'measure', matched), ('simulated', 'measure', matched), ('real', 'measure', matched)))
        modules = registry_modules(Lab, plan)
        goal = MetricGoal('accuracy', .95, 'evaluator/v2', input_refs=('cohort:valid',))
        with Application(self.config('goal'), modules=modules, run_policy=GoalRunPolicy(goal)) as app:
            result = app.run(max_decision_steps=6, wait_for_jobs=True)
        self.assertEqual(calls, [call.call_id for call in plan])
        self.assertEqual(result.stop_reason, 'goal_reached')
        self.assertEqual(result.decision_steps, 5)


def registry_modules(lab, actions, *, memory=None):
    registry = Registry()
    registry.register_controller('test', ControllerDefinition(lambda ctx, cfg: ScriptedPolicy(actions),
                                                              memory=memory), version='1')
    registry.register_lab('test', lambda ctx, cfg: lab(), version='1')
    return registry.build({'controller': {'name': 'test'}, 'labs': {'test': {'name': 'test'}}})
