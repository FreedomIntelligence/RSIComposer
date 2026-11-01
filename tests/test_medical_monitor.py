"""Medical monitor metadata follows model identity and delivered feedback."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.medical.experiment import JournalMonitor
from rsisys.loops.contracts.coordination import ExecutionView
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import TrainSpec
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.runtime.runner import RunnerStep
from rsisys.runtime.systems import SystemRegistry
from tests import test_medical_experiment as medical_fixture


class MedicalMonitorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.actor = ModelRef('local', 'actor', 'initial')
        self.researcher = ModelRef('remote', 'researcher', 'researcher-revision')
        self.monitor = JournalMonitor(self.root, self.actor, 'Research', decision_model=self.researcher)
        self.context = DecisionContext('Research', (), (), (),
            execution=ExecutionView(self.actor), decision_model=self.researcher)
        self.directive = Directive('train-1', 'Research', 'Train a candidate', mode='train',
                                   payload=TrainSpec('recipe', 'sft'))

    def read(self):
        return json.loads((self.root / 'monitor.json').read_text())

    def start_training(self):
        self.monitor.stage('train')
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 1)
        self.monitor.advance(RunnerStep('waiting', {}, directive_id=self.directive.directive_id))

    def feedback(self, status):
        self.monitor.advance(RunnerStep('completed', {'feedback': {
            'directive_id': self.directive.directive_id, 'status': status,
            'summary': 'Registration rejected' if status == 'failed' else 'Terminal feedback',
            'error': 'Candidate registration failed' if status == 'failed' else '',
        }}, directive_id=self.directive.directive_id))

    def test_independent_model_identity_uses_existing_public_schema(self):
        self.assertEqual(self.read()['models'], {
            'decision_maker': {'model_id': 'researcher', 'revision': 'researcher-revision'},
            'actor': {'model_id': 'actor', 'revision': 'initial'},
        })

    def test_active_training_keeps_exploration_running(self):
        self.start_training()
        self.assertEqual(self.read()['loops'], {
            'decision_making': {'status': 'waiting', 'summary': '等待探索侧反馈；到期时复查'},
            'exploration': {'status': 'running', 'summary': '训练'},
        })

    def test_terminal_feedback_releases_badge_and_preserves_directive_evidence(self):
        for status in ('completed', 'failed', 'cancelled', 'partial'):
            with self.subTest(status=status):
                self.start_training()
                self.feedback(status)
                snapshot = self.read()
                self.assertEqual(snapshot['current']['status'], status)
                self.assertEqual(snapshot['decisions'][-1]['status'], status)
                self.assertEqual(snapshot['workflow']['phase'], 'researching')
                self.assertEqual(snapshot['loops']['decision_making']['status'], 'running')
                self.assertEqual(snapshot['loops']['exploration'],
                                 {'status': 'idle', 'summary': '等待下一条指令'})
                if status == 'failed':
                    self.assertEqual(snapshot['current']['error'], 'Candidate registration failed')
                # Use a new directive ID: stable decisions intentionally deduplicate.
                self.directive = replace(self.directive, directive_id=self.directive.directive_id + '-next')

    def test_unrelated_feedback_does_not_finish_current_training(self):
        self.start_training()
        self.monitor.advance(RunnerStep('completed', {'feedback': {
            'directive_id': 'old-directive', 'status': 'failed',
        }}, directive_id='old-directive'))
        snapshot = self.read()
        self.assertEqual(snapshot['current']['status'], 'running')
        self.assertEqual(snapshot['loops']['exploration']['status'], 'running')

    def test_waiting_between_research_cycles_does_not_revive_failed_training(self):
        self.start_training()
        self.feedback('failed')
        self.monitor.advance(RunnerStep('waiting', {'run_control': 'restart_wait'}))
        self.assertEqual(self.read()['loops']['exploration'],
                         {'status': 'idle', 'summary': '等待下一条指令'})
        self.assertEqual(self.read()['current']['status'], 'failed')
        self.directive = replace(self.directive, directive_id='train-2')
        self.start_training()
        self.assertEqual(self.read()['loops']['exploration'],
                         {'status': 'running', 'summary': '训练'})

    def test_final_loops_follow_run_status_after_failed_directive(self):
        self.start_training()
        self.feedback('failed')
        for status in ('completed', 'stopped', 'failed'):
            with self.subTest(status=status):
                self.monitor.stage('controller-finished', status=status)
                snapshot = self.read()
                self.assertEqual(snapshot['current']['status'], 'failed')
                for loop in snapshot['loops'].values():
                    self.assertEqual(loop, {'status': status, 'summary': '当前阶段：controller-finished'})


class MedicalMonitorIntegrationTests(unittest.TestCase):
    setUp = medical_fixture.MedicalExperimentIntegrationTests.setUp
    execute = medical_fixture.MedicalExperimentIntegrationTests.execute
    read = medical_fixture.MedicalExperimentIntegrationTests.read

    def test_run_passes_independent_researcher_identity_to_monitor(self):
        researcher = ModelRef('remote', 'researcher', 'remote-revision')

        def finish(context):
            snapshot = self.read('monitor.json')
            self.assertEqual(snapshot['models']['decision_maker'],
                             {'model_id': researcher.model_id, 'revision': researcher.revision})
            self.assertEqual(context['decision_model']['revision'], researcher.revision)
            return medical_fixture.FINISH

        remote = medical_fixture.FixtureResearcher(researcher, self.process, [finish])
        result = self.execute([], decision_model=researcher, decision_backend=remote)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(self.read('monitor.json')['models']['decision_maker'],
                         {'model_id': researcher.model_id, 'revision': researcher.revision})
        self.assertEqual(self.read('monitor.json')['models']['actor']['model_id'], self.serving.base.model_id)

    def test_second_training_registration_failure_clears_live_and_final_badges(self):
        register = SystemRegistry.register_training

        def reject_second(registry, result, *, origin):
            if origin.parent_system != 'initial':
                raise ValueError('fixture second training registration failure')
            return register(registry, result, origin=origin)

        def second_train(context):
            recipe = next(receipt['data']['recipe_ref'] for receipt in context['operation_results']
                          if isinstance(receipt.get('data'), dict) and 'recipe_ref' in receipt['data'])
            return medical_fixture.train(context, recipe)

        def finish_after_failure(context):
            self.assertEqual(context['last_feedback']['status'], 'failed')
            snapshot = self.read('monitor.json')
            self.assertEqual(snapshot['current']['status'], 'failed')
            self.assertIn('fixture second training registration failure', snapshot['current']['error'])
            self.assertEqual(snapshot['loops']['exploration'],
                             {'status': 'idle', 'summary': '等待下一条指令'})
            self.assertEqual(snapshot['loops']['decision_making']['status'], 'running')
            return medical_fixture.FINISH

        define = {'kind': 'invoke', 'operation': 'define_recipe', 'arguments': {'seed': 29}}
        with patch.object(SystemRegistry, 'register_training', reject_second):
            result = self.execute([medical_fixture.train, medical_fixture.evaluate, medical_fixture.apply,
                                   define, second_train, finish_after_failure])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.process.requests), 2)
        snapshot = self.read('monitor.json')
        self.assertEqual(snapshot['current']['status'], 'failed')
        self.assertEqual(snapshot['loops']['exploration'],
                         {'status': 'completed', 'summary': '当前阶段：researcher_finished'})

    def test_interrupt_cleanup_leaves_no_running_loop_badge(self):
        self.process.interrupt = True
        result = self.execute([medical_fixture.train])
        self.assertEqual(result['status'], 'stopped', result)
        self.assertTrue(result['cleanup']['confirmed'])
        for loop in self.read('monitor.json')['loops'].values():
            self.assertEqual(loop, {'status': 'stopped', 'summary': '当前阶段：interrupted'})


if __name__ == '__main__':
    unittest.main()
