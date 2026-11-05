"""Single-meta context, memory mutations and recovery through real boundaries."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.config import build_demo
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.decision import OperationArgumentError, OperationCall
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, TokenUsage
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.agent_memory import AgentMemory, AgentMemoryState, SPECS
from rsisys.loops.decision_making.harness.context import BasicContextCompactor
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.json_tools import encoded, result_id
from rsisys.runtime.recovery import RecoveryRequired
from tests.test_feedback_required_outputs import CONFIG, observation
from tests.test_feedback_learning_integration import Backend, ImmediateExecutor


class AgentMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner, _ = build_demo(CONFIG)
        self.addCleanup(self.runner.close)
        self.loop = self.runner.decision_making
        self.harness = self.loop.harness
        self.backend = Backend()
        self.service = assemble_research(self.loop, assembly=ResearchAssembly.single_meta(),
            path=self.root / 'feedback.json', backend=self.backend, executor=ImmediateExecutor())
        self.memory = self.harness.agent_memory

    def native(self, ident='failed-source', name='lab_result', text='Actual error: messages[0] omitted the user question', route=None):
        args = {'feedback_route': route} if route else {}
        result = OperationResult('Native fixture failed', (), False, ExecutionStatus.FAILED,
                                 'input consumption failed', data={'stderr': text, 'job_id': ident})
        self.harness.memory.append(HistoryEntry(ident, 'operation', None, result.summary, False,
            details={'operation': asdict(OperationCall(name, args)), 'result': result.payload()}))
        self.harness.state.operation_results.append(result)
        return result

    def mutate(self, name, **args):
        return self.memory.execute(OperationCall(name, {'expected_revision': self.memory.export()['revision'], **args}))

    def view(self):
        return self.harness._context('Repair candidate input handling', self.loop.action_space)

    def test_default_has_no_background_model_and_no_implicit_memory_write(self):
        ident = self.service.observe(observation('raw'))
        self.service.refresh()
        row = self.service.learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(self.service.learning.state['attempted'], 0)
        self.assertEqual(row['products']['context_memory']['charged_calls'], 0)
        self.assertEqual(self.memory.export()['memories'], {})
        self.assertEqual(self.harness.memory.recall(), ())
        self.assertEqual(self.backend.requests, [])
        self.assertEqual(self.loop.research_assembly['agent_topology'], 'single_acting_meta_agent')
        self.assertFalse(ResearchAssembly().requires_model)

    def test_agent_writes_updates_searches_and_deletes_with_exact_provenance(self):
        result = self.native()
        source = 'result:' + result_id(result.payload())
        self.mutate('memory_write', memory_id='input-contract', content='Use the canonical user message.',
                    source_refs=[source], status='supported', applicability='candidate adapter')
        row = self.memory.execute(OperationCall('memory_read', {'memory_id': 'input-contract'}))['memory']
        self.assertEqual(row['provenance'][source]['sha256'], result_id(result.payload()))
        self.assertIn('prose_unverified', row['authority'])
        self.mutate('memory_write', memory_id='input-contract', content='Render all canonical chat messages.',
                    source_refs=['history:decision_maker:failed-source'], status='unverified')
        self.assertEqual(self.memory.search(query='render')['total'], 1)
        self.mutate('memory_delete', memory_id='input-contract')
        self.assertEqual(self.memory.search()['total'], 0)
        self.assertIn('Render all canonical', encoded(self.memory.export()['revisions']))
        self.assertEqual(self.backend.requests, [])
        self.assertEqual(AgentMemoryState(self.memory).capture()['revision'], 3)

    def test_bad_sources_stale_revisions_and_invalid_assessments_do_not_mutate(self):
        before = self.memory.export()
        cases = [dict(source_refs=['history:decision_maker:invented'], status='supported'),
                 dict(source_refs=[], status='supported'), dict(source_refs=[], status='verified')]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(OperationArgumentError):
                self.mutate('memory_write', memory_id='bad', content='Unsupported finding', **case)
            self.assertEqual(self.memory.export(), before)
        self.mutate('memory_write', memory_id='hypothesis', content='A hypothesis', source_refs=[], status='unverified')
        with self.assertRaisesRegex(OperationArgumentError, 'Stale'):
            self.memory.execute(OperationCall('memory_delete', {'expected_revision': 0, 'memory_id': 'hypothesis'}))
        self.assertEqual(self.memory.search()['total'], 1)

    def test_search_revision_binds_pages_and_no_read_calls_inference(self):
        for i in range(3):
            self.mutate('memory_write', memory_id=str(i), content='Hypothesis ' + str(i), source_refs=[], status='unverified')
        page = self.memory.execute(OperationCall('memory_search', {'limit': 1}))
        self.assertEqual(page['next_offset'], 1)
        with self.assertRaisesRegex(OperationArgumentError, 'revision'):
            self.memory.execute(OperationCall('memory_search', {'limit': 1, 'offset': 1}))
        next_page = self.memory.execute(OperationCall('memory_search', {'limit': 1, 'offset': 1,
                                                        'expected_revision': page['revision']}))
        self.assertEqual(next_page['items'][0]['memory_id'], '1')
        self.mutate('memory_delete', memory_id='0')
        with self.assertRaisesRegex(OperationArgumentError, 'Stale'):
            self.memory.execute(OperationCall('memory_search', {'offset': 1, 'expected_revision': page['revision']}))
        self.assertEqual(self.backend.requests, [])

    def test_large_memory_read_is_paged_and_old_versions_survive_update_and_delete(self):
        self.mutate('memory_write', memory_id='large', content='中文\\\n' * 700,
                    source_refs=[], status='unverified')
        original = self.memory.export()['memories']['large']
        text, offset, digest = '', 0, None
        while True:
            args = {'memory_id': 'large', 'offset': offset}
            if digest:
                args['expected_digest'] = digest
            page = self.harness.tools.execute(OperationCall('memory_read', args)).data
            self.assertLess(len(encoded(page).encode('utf-8')), 8192)
            text += page['content']
            digest, offset = page['sha256'], page['next_offset']
            if offset is None:
                break
        self.assertEqual(json.loads(text), original)
        self.mutate('memory_write', memory_id='large', content='A revised hypothesis', source_refs=[], status='unverified')
        with self.assertRaisesRegex(OperationArgumentError, 'Stale'):
            self.memory.execute(OperationCall('memory_read', {'memory_id': 'large', 'offset': 4096,
                                                             'expected_digest': digest}))
        self.mutate('memory_delete', memory_id='large')
        page = self.memory.execute(OperationCall('memory_read', {'memory_id': 'large', 'revision': 1}))
        self.assertEqual(page['sha256'], result_id(original))
        with self.assertRaisesRegex(OperationArgumentError, 'Unknown memory_id'):
            self.memory.execute(OperationCall('memory_read', {'memory_id': 'large'}))

    def test_recency_and_relevance_are_stable_after_json_persistence(self):
        for ident in ('z-old', 'a-new'):
            self.mutate('memory_write', memory_id=ident, content='hypothesis', source_refs=[], status='unverified')
        restored = AgentMemory()
        restored.bind(self.harness, self.root / 'feedback.json')
        self.assertEqual([row['memory_id'] for row in restored.search()['items']], ['a-new', 'z-old'])
        self.assertEqual(restored.search(), self.memory.search())

    def test_recovery_does_not_accept_a_modified_index_or_revision(self):
        self.mutate('memory_write', memory_id='one', content='hypothesis', source_refs=[], status='unverified')
        original = self.memory.export()
        changed = deepcopy(original)
        changed['memories']['one']['content'] = 'edited outside the revision log'
        with self.assertRaisesRegex(ValueError, 'indexes differ'):
            self.memory.restore(changed)
        changed = deepcopy(original)
        changed['revisions'][0]['arguments']['content'] = 'edited write arguments'
        with self.assertRaisesRegex(ValueError, 'differs from its write'):
            self.memory.restore(changed)
        self.assertEqual(self.memory.export(), original)

    def test_assembly_is_reusable_configuration_without_sharing_mutable_run_state(self):
        assembly = ResearchAssembly()
        other, _ = build_demo(CONFIG)
        self.addCleanup(other.close)
        service = assemble_research(other.decision_making, assembly=assembly, executor=ImmediateExecutor())
        self.assertIsNone(assembly.context_memory.harness)
        self.assertIsNot(other.decision_making.harness.agent_memory, self.memory)
        self.assertEqual(service.context(), self.service.context())

    def test_recovery_rejects_a_deletion_that_keeps_a_record_even_with_matching_index(self):
        self.mutate('memory_write', memory_id='one', content='hypothesis', source_refs=[], status='unverified')
        record = self.memory.export()['memories']['one']
        self.mutate('memory_delete', memory_id='one')
        changed = self.memory.export()
        record['revision'] = 2
        changed['revisions'][-1]['memories']['one'] = record
        changed['memories']['one'] = record
        with self.assertRaisesRegex(ValueError, 'deletion revision must remove'):
            AgentMemory._validated(changed)

    def test_recovery_validates_task_transitions_against_arguments_and_prior_state(self):
        self.native()
        self.native('later')
        self.mutate('research_task', action='open', task_id='debug', title='Debug', record_ids=['failed-source'])
        self.mutate('research_task', action='pin', task_id='debug', record_ids=['later'])
        self.mutate('research_task', action='close', task_id='debug')
        original = self.memory.export()
        self.assertEqual(AgentMemory._validated(original), original)
        for field, value in [('status', 'open'), ('record_ids', []), ('title', 'silently replaced')]:
            changed = deepcopy(original)
            changed['tasks']['debug'][field] = value
            changed['revisions'][-1]['tasks']['debug'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Task mutation differs'):
                AgentMemory._validated(changed)
        changed = deepcopy(original)
        changed['revisions'][0]['arguments']['record_ids'] = ['later']
        with self.assertRaisesRegex(ValueError, 'Task opening differs'):
            AgentMemory._validated(changed)

    def test_raw_error_survives_reads_edits_and_retests_until_task_closes(self):
        original = self.native()
        self.mutate('research_task', action='open', task_id='adapter-debug', title='Repair message consumption')
        for i in range(12):
            self.native('read-' + str(i), name='lab_read_file', text='intermediate source page')
        retest = self.native('retest', text='Retest still failed')
        view = self.view().research_agenda['active_feedback']
        by_id = {row['record_id']: row for row in view['observations']}
        self.assertEqual(by_id['failed-source']['result'], original.payload())
        self.assertEqual(by_id['retest']['result'], retest.payload())
        self.assertEqual(len(view['observations']), 2)
        before = self.harness.memory.history()
        self.mutate('research_task', action='close', task_id='adapter-debug')
        self.assertEqual(self.view().research_agenda['active_feedback']['observations'], [])
        self.assertEqual(self.harness.memory.history(), before)
        self.assertEqual(self.memory.export()['memories'], {})

    def test_pin_can_retain_an_older_receipt_and_rejects_missing_ids(self):
        self.native()
        self.native('later')
        self.mutate('research_task', action='open', task_id='debug', title='Debug', record_ids=['failed-source'])
        self.mutate('research_task', action='pin', task_id='debug', record_ids=['later'])
        self.assertEqual(self.memory.export()['tasks']['debug']['record_ids'], ['failed-source', 'later'])
        before = self.memory.export()
        with self.assertRaisesRegex(OperationArgumentError, 'history record_ids'):
            self.mutate('research_task', action='pin', task_id='debug', record_ids=['invented'])
        self.assertEqual(self.memory.export(), before)

    def test_task_limits_are_explicit_not_silent_eviction(self):
        self.native()
        for i in range(2):
            self.mutate('research_task', action='open', task_id=str(i), title='Task')
        before = self.memory.export()
        with self.assertRaisesRegex(OperationArgumentError, 'third'):
            self.mutate('research_task', action='open', task_id='third', title='Task')
        self.assertEqual(self.memory.export(), before)

    def test_large_native_feedback_is_literal_paged_and_exact_archive_is_readable(self):
        original = self.native(text='中文 traceback\n' * 10000)
        self.mutate('research_task', action='open', task_id='debug', title='Debug')
        item = self.view().research_agenda['active_feedback']['observations'][0]
        self.assertFalse(item['complete'])
        page = item['raw_json_page']
        raw = encoded(original.payload()).encode('utf-8')
        self.assertEqual(page['content'].encode('utf-8'), raw[:page['end_offset']])
        self.assertLessEqual(page['end_offset'], 16384)
        path = dict(item['read_path'])
        path.pop('operation')
        text, cursor = '', None
        while True:
            output = self.harness.tools.execute(OperationCall('research_archive', {**path, 'limit': 8, 'cursor': cursor})).data
            text += ''.join(row['text'] for row in output['items'])
            cursor = output['next_cursor']
            if cursor is None:
                break
        self.assertEqual(json.loads(text), json.loads(encoded(original.payload())))

    def test_legacy_route_cannot_hide_feedback_from_the_single_meta(self):
        native = self.native(route='memory')
        self.assertEqual(self.view().research_agenda['previous_feedback']['result'], native.payload())
        self.assertEqual(self.backend.requests, [])

    def test_persistence_and_recovery_preserve_tasks_memories_and_versions(self):
        self.native()
        self.mutate('research_task', action='open', task_id='debug', title='Debug')
        self.mutate('memory_write', memory_id='note', content='Canonical messages',
                    source_refs=['history:decision_maker:failed-source'], status='unverified')
        saved = AgentMemoryState(self.memory).capture()
        restored = AgentMemory()
        restored.bind(self.harness, self.root / 'feedback.json')
        AgentMemoryState(restored).restore(saved)
        self.assertEqual(restored.export(), saved)
        self.assertEqual(restored.project_context(self.view()).research_agenda['active_feedback']['tasks'][0]['task_id'], 'debug')
        self.mutate('memory_delete', memory_id='note')
        before = (self.root / 'agent-memory.json').read_bytes()
        with self.assertRaisesRegex(RecoveryRequired, 'changed after checkpoint'):
            restored.restore(saved)
        self.assertEqual((self.root / 'agent-memory.json').read_bytes(), before)

    def test_failed_persistence_never_commits_an_in_memory_only_update(self):
        before = self.memory.export()
        with patch('rsisys.loops.decision_making.harness.agent_memory.atomic_write', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.mutate('memory_write', memory_id='note', content='Hypothesis', source_refs=[], status='unverified')
        self.assertEqual(self.memory.export(), before)

    def test_permissions_filter_memory_and_review_cannot_mutate(self):
        space = authorized_space(self.loop.action_space)
        # Permissions belong to the installed base ActionSpace, as for Lab tools.
        base = self.loop.action_space
        while hasattr(base, 'base'):
            base = base.base
        base.permissions = ActionPermissions(operations=frozenset({'memory_read', 'memory_search'}))
        self.assertEqual({s.name for s in space.describe()}, {'memory_read', 'memory_search'})
        with self.assertRaises(PermissionError):
            space.validate(OperationCall('memory_write', {'expected_revision': 0, 'memory_id': 'x',
                'content': 'x', 'source_refs': [], 'status': 'unverified'}))
        base.permissions = ActionPermissions()
        with self.assertRaises(PermissionError):
            authorized_space(self.loop.action_space, 'review').validate(OperationCall('memory_delete',
                {'expected_revision': 0, 'memory_id': 'x'}))

    def test_malformed_arguments_return_feedback_without_infrastructure_retry(self):
        for name, args in [('memory_write', {'expected_revision': True}),
                           ('memory_read', {'memory_id': 'missing'}),
                           ('research_task', {'expected_revision': 0, 'task_id': 'x', 'action': 'pin'})]:
            result = self.harness.tools.execute(OperationCall(name, args))
            self.assertEqual(result.status, ExecutionStatus.FAILED)
        self.assertEqual(self.memory.export()['revision'], 0)

    def test_real_decision_request_has_raw_task_feedback_and_only_one_model_call(self):
        native = self.native()
        self.mutate('research_task', action='open', task_id='debug', title='Debug')
        self.native('file', name='lab_read_file', text='candidate.py source')
        requests = []
        def generate(request):
            requests.append(request)
            return GenerationResponse('{"kind":"finish"}', request.model, True, TokenUsage(101, 9))
        with patch.object(self.harness.policy.backend, 'generate', side_effect=generate):
            self.harness._call_policy(self.harness.policy.decide, self.view())
        payload = json.loads(requests[0].prompt)
        observation = payload['research_agenda']['active_feedback']['observations'][0]
        self.assertEqual(observation['result'], json.loads(encoded(native.payload())))
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.harness.state.measured_input_tokens, 101)
        self.assertEqual(self.harness.state.measured_output_tokens, 9)

    def test_forced_compaction_uses_no_hidden_summary_model_request(self):
        for i in range(150):
            self.harness.memory.append(HistoryEntry('old-' + str(i), 'decision', None, 'old note ' * 120, True))
        self.harness.context_compactor = BasicContextCompactor(trigger_tokens=110000, target_tokens=65000,
                                                              summary_tokens=1024)
        requests = []
        def generate(request):
            requests.append(request)
            return GenerationResponse('{"kind":"finish"}', request.model, True, TokenUsage(100, 10))
        with patch.object(self.harness.policy.backend, 'generate', side_effect=generate), \
             patch.object(self.harness.policy, '_summarize_context', side_effect=AssertionError('hidden summary model')):
            self.harness._call_policy(self.harness.policy.decide, self.view())
        self.assertGreater(self.harness.context_compactor.compactions, 0)
        self.assertEqual(len(requests), 1)


if __name__ == '__main__':
    unittest.main()
