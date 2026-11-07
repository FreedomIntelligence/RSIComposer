"""New-identity handoff exercises real archives, memory CRUD and artifact files."""
from copy import deepcopy
from dataclasses import asdict
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
from tests.test_feedback_required_outputs import CONFIG
from tests.test_feedback_learning_integration import Backend, ImmediateExecutor


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
        self.mutate(self.new, 'research_task', task_id='debug', action='close')
        self.install(resume=True).restore(state)
        self.assertEqual(self.new.agent_memory.export()['tasks']['debug']['status'], 'closed')
        self.assertEqual(read_continuation(self.bundle)[1], receipt['sha256'])

    def test_missing_cleared_provenance_and_unresolved_effects_block_export(self):
        for changes in ({'record_ids': []}, {'quiescent': False}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, RecoveryRequired)):
                self.freeze(**changes)
            self.assertFalse(self.bundle.exists())

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


if __name__ == '__main__':
    unittest.main()
