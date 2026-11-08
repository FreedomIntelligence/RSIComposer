"""New-identity handoff exercises real archives, memory CRUD and artifact files."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsisys.config import build_demo
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.continuation import freeze_continuation, install_continuation, read_continuation
from rsisys.runtime.recovery import RecoveryRequired
from tests.feedback.test_feedback_required_outputs import CONFIG
from tests.feedback.test_feedback_learning_integration import Backend, ImmediateExecutor


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.old, self.new = self.make_harness('old'), self.make_harness('new')
        self.failure = OperationResult('Real fixture failure', (), False, ExecutionStatus.FAILED,
                                       'tokenizer returned string keys', data={'stderr': 'RAW traceback', 'job_id': 'old-job'})
        self.old.memory.append(HistoryEntry('decision-1:operation', 'operation', 'old-job', 'Failure', False,
            details={'decision_id': 'decision-1', 'operation': {'name': 'custom_training_result', 'arguments': {}},
                     'result': self.failure.payload()}))
        self.mutate(self.old, 'memory_write', memory_id='tokenizer', content='Read input_ids, not dict keys.',
            source_refs=['history:decision_maker:decision-1:operation'], status='supported')
        self.mutate(self.old, 'research_task', task_id='debug', action='open', title='Repair tokenization')
        self.identity = {'campaign_id': 'campaign', 'attempt_id': 'old',
                         'axes': {'source': 'a'*64, 'data': 'b'*64, 'evaluation_account': 'f'*64}}
        self.target = {**self.identity, 'attempt_id': 'new', 'axes': {**self.identity['axes'], 'source': 'c'*64, 'data': 'd'*64}}
        self.weights = self.root / 'weights'
        self.weights.mkdir()
        (self.weights / 'adapter.safetensors').write_bytes(b'actual-test-bytes')
        self.bundle = self.root / 'handoff'
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()

    def make_harness(self, name):
        runner, _ = build_demo(CONFIG)
        self.addCleanup(runner.close)
        assemble_research(runner.decision_making, assembly=ResearchAssembly.single_meta(),
            path=self.root / name / 'feedback.json', backend=Backend(), executor=ImmediateExecutor())
        return runner.decision_making.harness

    @staticmethod
    def mutate(harness, name, **args):
        memory = harness.agent_memory
        return memory.execute(OperationCall(name, {'expected_revision': memory.export()['revision'], **args}))

    def freeze(self, **changes):
        args = dict(identity=self.identity, history=self.old.memory.history(), agent_state=self.old.agent_memory.export(),
            record_ids=['decision-1:operation'], memory_ids=['tokenizer'], task_ids=['debug'],
            artifacts={'trained-adapter': self.weights}, quiescent=True)
        destination = changes.pop('destination', self.bundle)
        return freeze_continuation(destination, **(args | changes))

    def install(self, **changes):
        return install_continuation(self.bundle, **(dict(identity=self.target, harness=self.new,
            workspace=self.workspace, receipt_path=self.root / 'receipt.json') | changes))

    def test_changed_identity_inherits_raw_failures_tasks_notes_and_weights_without_live_state(self):
        receipt = self.freeze()
        port = self.install()
        state = port.capture()
        self.assertFalse(state['source_comparison_axes_match'])
        self.assertEqual(state['changed_axes'], ['data', 'source'])
        self.assertEqual(state['actions_replayed'], 0)
        self.assertEqual(self.new.state.decision_records, {})
        self.assertEqual(self.new.state.operation_results, [])
        ref = state['history_record_ids'][0]
        raw = self.new.tools.execute(OperationCall('research_archive',
            {'collection': 'history', 'record_id': ref, 'limit': 8, 'cursor': None})).data
        self.assertIn('RAW traceback', ''.join(item['text'] for item in raw['items']))
        note = self.new.agent_memory.execute(OperationCall('memory_read', {'memory_id': 'tokenizer'}))['memory']
        self.assertEqual(note['content'], 'Read input_ids, not dict keys.')
        self.assertEqual(note['source_refs'], ['history:decision_maker:' + ref])
        self.assertIn('prose_unverified', note['authority'])
        task = self.new.agent_memory.export()['tasks']['debug']
        self.assertEqual(task['record_ids'], [ref])
        self.assertEqual(task['status'], 'open')
        context = self.new.agent_memory.project_context(self.new._context('Repair', None))
        self.assertEqual(context.research_agenda['active_feedback']['observations'][0]['result'],
                         json.loads(json.dumps(self.failure.payload())))
        copied = self.workspace / 'inherited/trained-adapter/adapter.safetensors'
        self.assertEqual(copied.read_bytes(), (self.weights / 'adapter.safetensors').read_bytes())
        copied.write_bytes(b'next-candidate')
        self.assertEqual((self.bundle / 'artifacts/trained-adapter/adapter.safetensors').read_bytes(), b'actual-test-bytes')
        self.mutate(self.new, 'research_task', task_id='debug', action='close',
                    resolution='inconclusive', outcome='No retest')
        self.install(resume=True).restore(state)
        self.assertEqual(self.new.agent_memory.export()['tasks']['debug']['status'], 'closed')
        self.assertEqual(read_continuation(self.bundle)[1], receipt['sha256'])

    def test_missing_cleared_provenance_and_unresolved_effects_block_export(self):
        for changes in ({'record_ids': []}, {'quiescent': False}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, RecoveryRequired)):
                self.freeze(**changes)
            self.assertFalse(self.bundle.exists())

    def test_task_progress_and_negative_outcome_survive_new_attempt_with_rebound_sources(self):
        self.mutate(self.old, 'research_task', task_id='debug', action='update',
                    next_step='Check adapter loading', acceptance_criteria='Fixed evaluation uses changed weights')
        self.mutate(self.old, 'research_task', task_id='debug', action='close', resolution='inconclusive',
                    outcome='Inconclusive: inference ignored the adapter')
        self.freeze()
        self.install()
        task = self.new.agent_memory.execute(OperationCall('research_task_read', {'task_id': 'debug'}))['task']
        self.assertEqual(task['next_step'], 'Check adapter loading')
        self.assertEqual(task['acceptance_criteria'], 'Fixed evaluation uses changed weights')
        self.assertEqual(task['outcome'], 'Inconclusive: inference ignored the adapter')
        self.assertEqual(task['status'], 'closed')
        self.assertTrue(task['record_ids'][0].startswith('prior-'))
        self.assertEqual(self.new.state.decision_records, {})

    def test_resolved_acceptance_rebinds_to_cleared_receipt_without_promoting_legacy_closure(self):
        self.old.memory.append(HistoryEntry('retest', 'operation', None, 'Verified fixture retest', False,
            details={'operation': {'name': 'lab_result', 'arguments': {}}, 'result':
                OperationResult('Retest', (), False, data={'state': 'succeeded', 'passed': True}).payload()}))
        self.mutate(self.old, 'research_task', task_id='debug', action='update', acceptance_checks=[
            {'id': 'retest', 'operation': 'lab_result', 'result_path': '/data/passed', 'expected': True}])
        self.mutate(self.old, 'research_task', task_id='debug', action='close', resolution='resolved',
                    outcome='Fixture retest passed', acceptance_evidence={'retest': 'retest'})
        with self.assertRaisesRegex(ValueError, 'cleared'):
            self.freeze()
        self.freeze(record_ids=['decision-1:operation', 'retest'])
        self.install()
        task = self.new.agent_memory.export()['tasks']['debug']
        self.assertEqual(task['resolution'], 'resolved')
        self.assertTrue(task['acceptance_evidence']['retest'].startswith('prior-'))
        self.assertNotEqual(task['acceptance_receipt'],
                            self.old.agent_memory.export()['tasks']['debug']['acceptance_receipt'])
        self.new.agent_memory._verify_acceptance(self.new.agent_memory.export())

    def test_inherited_open_task_reviews_only_new_attempt_results(self):
        from rsisys.loops.decision_making.harness.agent_memory import SPECS
        def card():
            context = replace(self.new._context('Continue inherited investigation', None), operations=SPECS)
            return self.new.agent_memory.project_context(context).research_agenda['agent_memory']
        self.old.memory.append(HistoryEntry('old-later', 'operation', 'old-job', 'Historical retest', False,
            details={'operation': {'name': 'custom_training_result', 'arguments': {}},
                     'result': self.failure.payload()}))
        receipt = self.freeze(record_ids=['decision-1:operation', 'old-later'])
        self.install()
        self.assertNotIn('task_reviews', card())
        self.assertNotIn('memory_review', card())
        self.new.memory.append(HistoryEntry('new-result', 'operation', 'new-job', 'Current retest', False,
            details={'operation': {'name': 'custom_training_result', 'arguments': {}},
                     'result': self.failure.payload()}))
        self.assertEqual(card()['task_reviews'][0]['task_id'], 'debug')
        self.assertEqual(card()['task_reviews'][0]['record_ids'], ['new-result'])
        self.assertEqual(card()['memory_review']['source_refs'], ['history:decision_maker:new-result'])
        self.assertEqual(read_continuation(self.bundle)[1], receipt['sha256'])

    def test_campaign_or_account_changes_and_incomplete_import_never_replay(self):
        self.freeze()
        for target in ({**self.target, 'campaign_id': 'other'},
                       {**self.target, 'axes': {**self.target['axes'], 'evaluation_account': '0'*64}}):
            with self.assertRaises(ValueError):
                self.install(identity=target)
            self.assertFalse(self.new.memory.history())
        (self.root / 'receipt.applying').write_text('{}')
        with self.assertRaisesRegex(RecoveryRequired, 'no automatic replay'):
            self.install()
        self.assertFalse(self.new.memory.history())

    def test_artifact_tamper_and_symlink_block_before_import(self):
        self.freeze()
        (self.bundle / 'artifacts/trained-adapter/adapter.safetensors').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.install()
        self.assertFalse(self.new.memory.history())
        (self.weights / 'escape').symlink_to(self.root)
        with self.assertRaises(ValueError):
            self.freeze(destination=self.root / 'second')

    def test_old_and_new_decision_ids_cannot_alias_and_new_memory_crud_remains_available(self):
        self.freeze()
        port = self.install()
        self.new.memory.append(HistoryEntry('decision-1:operation', 'operation', None, 'New result', False,
            details={'decision_id': 'decision-1', 'operation': {'name': 'custom_training_result', 'arguments': {}},
                     'result': OperationResult('Now succeeds', (), False).payload()}))
        self.mutate(self.new, 'memory_write', memory_id='tokenizer', content='Repair verified on a new fixture.',
            source_refs=['history:decision_maker:decision-1:operation'], status='supported')
        self.assertEqual(len(self.new.memory.history()), 2)
        self.assertEqual(port.capture()['memory_origins']['tokenizer']['revision'], 1)
        self.install(resume=True)

    def test_repeated_identical_native_results_do_not_rebind_existing_memory_provenance(self):
        from rsisys.runtime.json_tools import result_id
        self.mutate(self.old, 'memory_write', memory_id='tokenizer', content='Read input_ids.',
            source_refs=['result:' + result_id(self.failure.payload())], status='supported')
        self.old.memory.append(HistoryEntry('repeated', 'operation', None, 'Same failure again', False,
            details={'operation': {'name': 'custom_training_result', 'arguments': {}},
                     'result': self.failure.payload()}))
        self.freeze(record_ids=['decision-1:operation', 'repeated'])
        port = self.install()
        note = self.new.agent_memory.execute(OperationCall('memory_read', {'memory_id': 'tokenizer'}))['memory']
        self.assertEqual(note['source_refs'], ['history:decision_maker:' + port.capture()['history_record_ids'][0]])

    def test_a_shared_memory_file_without_its_raw_archive_is_rejected(self):
        from rsisys.loops.decision_making.harness.agent_memory import AgentMemory
        memory = AgentMemory(self.old.agent_memory.path)
        with self.assertRaisesRegex(ValueError, 'explicit continuation'):
            memory.bind(self.new)

    def test_expired_window_requires_bound_host_authorization_and_never_increases_calls(self):
        from experiments.medical.continuation import continuation_limits
        handoff = {'identity': self.identity, 'limits': {'deadline_epoch': 100, 'remaining_model_calls': 7}}
        options = dict(campaign_id='campaign', attempt_id='new', deadline_epoch=900, max_calls=100, now=200)
        self.assertEqual(continuation_limits(handoff, 'a'*64, **options), (100, 7))
        authorization = {'schema': 'rsisys-host-continuation-renewal/v1', 'handoff_sha256': 'a'*64,
            'campaign_id': 'campaign', 'target_attempt_id': 'new', 'previous_deadline_epoch': 100,
            'deadline_epoch': 800, 'authorized_at': 150, 'reason': 'User explicitly requested another window'}
        self.assertEqual(continuation_limits(handoff, 'a'*64, **options, authorization=authorization), (800, 7))
        self.assertEqual(continuation_limits(handoff, 'a'*64, **(options | {'max_calls': 3}),
            authorization=authorization), (800, 3))
        for changes in ({'handoff_sha256': 'b'*64}, {'target_attempt_id': 'other'},
                        {'campaign_id': 'other'}, {'previous_deadline_epoch': 99},
                        {'authorized_at': 50}, {'authorized_at': 201}, {'deadline_epoch': 199},
                        {'deadline_epoch': float('nan')}, {'max_calls': 1000}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                continuation_limits(handoff, 'a'*64, **options, authorization=authorization | changes)


if __name__ == '__main__':
    unittest.main()
