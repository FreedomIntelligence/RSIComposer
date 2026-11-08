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
from rsisys.loops.contracts.feedback_context import ContextQuery, FeedbackContextEvent
from rsisys.loops.contracts.model_service import GenerationResponse, TokenUsage
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.agent_memory import AgentMemory, AgentMemoryState, SPECS
from rsisys.loops.decision_making.harness.context import BasicContextCompactor
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.json_tools import encoded, result_id
from rsisys.runtime.recovery import RecoveryRequired
from tests.feedback.test_feedback_required_outputs import CONFIG, observation
from tests.feedback.test_feedback_learning_integration import Backend, ImmediateExecutor


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
        self.service = assemble_research(self.loop, assembly=ResearchAssembly.single_meta(require_memory_review=True),
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

    def test_task_acceptance_rejects_submission_training_and_incomplete_evaluation(self):
        self.native()
        checks = [{'id': 'score', 'operation': 'lab_result',
                   'result_path': '/data/fixed_evaluation/verified', 'expected': True},
                  {'id': 'count', 'operation': 'lab_result',
                   'result_path': '/data/fixed_evaluation/metrics/medqa/n', 'expected': 500}]
        self.mutate('research_task', action='open', task_id='evaluate', title='Evaluate new weights',
                    acceptance_criteria='A complete fixed MedQA evaluation', acceptance_checks=checks)
        def receipt(ident, name, data, status=ExecutionStatus.COMPLETED):
            result = OperationResult('Native fixture', (), False, status, data=data)
            self.harness.memory.append(HistoryEntry(ident, 'operation', None, result.summary, False,
                details={'operation': {'name': name, 'arguments': {}}, 'result': result.payload()}))
        receipt('submitted', 'medical_evaluate', {'state': 'submitted'})
        receipt('training', 'lab_result', {'state': 'succeeded', 'training_metrics': {'loss': .1}})
        receipt('partial', 'lab_result', {'state': 'succeeded', 'fixed_evaluation': {
            'verified': True, 'metrics': {'medqa': {'n': 1}}}})
        before = self.memory.export()
        for ident in ('submitted', 'training', 'partial', 'made-up'):
            with self.subTest(ident=ident), self.assertRaises(OperationArgumentError):
                self.mutate('research_task', action='close', task_id='evaluate', resolution='resolved',
                            outcome='Done', acceptance_evidence={'score': ident, 'count': ident})
            self.assertEqual(self.memory.export(), before)
        receipt('evaluation', 'lab_result', {'state': 'succeeded', 'fixed_evaluation': {
            'verified': True, 'metrics': {'medqa': {'n': 500}}}})
        self.mutate('research_task', action='close', task_id='evaluate', resolution='resolved', outcome='Measured',
                    acceptance_evidence={'score': 'evaluation', 'count': 'evaluation'})
        task = self.memory.export()['tasks']['evaluate']
        self.assertEqual(task['acceptance_receipt']['score']['record_id'], 'evaluation')
        restored = AgentMemory()
        restored.bind(self.harness, self.root / 'feedback.json')
        self.assertEqual(restored.export(), self.memory.export())
        self.assertFalse(self.backend.requests)
        # A forged digest passes JSON shape checks but cannot bind to history.
        forged = self.memory.export()
        for target in (forged['tasks']['evaluate'], forged['revisions'][-1]['tasks']['evaluate']):
            target['acceptance_receipt']['score']['sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'source identity'):
            self.memory.restore(forged)

    def test_task_cannot_silently_close_and_non_success_resolution_is_explicit(self):
        self.mutate('research_task', action='open', task_id='work', title='Unfinished work')
        for args in ({}, {'outcome': 'Done'}, {'outcome': 'Done', 'resolution': 'resolved'}):
            before = self.memory.export()
            with self.assertRaises(OperationArgumentError):
                self.mutate('research_task', action='close', task_id='work', **args)
            self.assertEqual(self.memory.export(), before)
        self.mutate('research_task', action='close', task_id='work', resolution='inconclusive',
                    outcome='Training succeeded but the fixed evaluation is missing')
        self.assertNotIn('acceptance_receipt', self.memory.export()['tasks']['work'])

    def test_repeated_failures_surface_operational_memory_without_fabricating_notes(self):
        self.native('missing-import-1', text='ModuleNotFoundError: candidate_adapter')
        self.native('missing-import-2', text='ModuleNotFoundError: candidate_adapter')
        group = self.memory.pending_review()['recurring_failures'][0]
        self.assertEqual(group['count'], 2)
        self.assertIn('operational', group['instruction'])
        self.assertEqual(len(group['source_refs']), 2)
        self.assertEqual(self.memory.export()['memories'], {})
        self.assertEqual(self.memory.export()['revision'], 0)

    def test_recurrence_counts_distinct_failed_calls_but_not_repeated_terminal_polls(self):
        failure = OperationResult('Missing path', (), False, ExecutionStatus.FAILED, 'FileNotFoundError')
        for ident in ('missing-path-1', 'missing-path-2'):
            self.harness.memory.append(HistoryEntry(ident, 'operation', None, failure.summary, False,
                details={'operation': asdict(OperationCall('lab_read_file', {'path': 'missing.py'})),
                         'result': failure.payload()}))
        for ident in ('poll-1', 'poll-2'):
            result = replace(failure, data={'job_id': 'same-failed-job', 'state': 'failed'})
            self.harness.memory.append(HistoryEntry(ident, 'operation', None, result.summary, False,
                details={'operation': asdict(OperationCall('lab_poll', {'job_id': 'same-failed-job'})),
                         'result': result.payload()}))
        inbox = self.memory.pending_review()
        self.assertEqual(inbox['pending_count'], 2)  # Identical receipts remain one review obligation.
        self.assertEqual([(g['operation'], g['count']) for g in inbox['recurring_failures']],
                         [('lab_read_file', 2)])
        self.assertEqual(self.memory.export()['memories'], {})

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

    def test_empty_memory_without_tasks_requires_a_source_disposition_before_new_work(self):
        from rsisys.loops.contracts.decision import OperationSpec
        self.native()
        original = replace(self.view(), operations=(*SPECS, OperationSpec('lab_start', 'Start', {}),
            OperationSpec('custom_inspect', 'Inspect', {}, True)))
        before = self.memory.export()
        view = self.memory.project_context(original)
        self.assertEqual(view.research_agenda['agent_memory']['memory_health']['pending_count'], 1)
        self.assertNotIn('lab_start', {s.name for s in view.operations})
        self.assertIn('custom_inspect', {s.name for s in view.operations})
        self.assertEqual(self.memory.export(), before)
        response = GenerationResponse('{"kind":"invoke","operation":"lab_start","arguments":{}}',
                                      self.harness.policy.model, True)
        with self.assertRaisesRegex(ValueError, 'pending Memory'):
            self.harness.policy._parse_decision(response, view)
        args = view.research_agenda['agent_memory']['memory_review']['review_arguments']
        result = self.harness.tools.execute(OperationCall('memory_review', {**args,
            'disposition': 'deferred', 'reason': 'Inspect the loader and rerun before claiming a reusable fix.'}))
        self.assertEqual(result.status, ExecutionStatus.COMPLETED)
        self.assertEqual(self.memory.export()['memories'], {})
        self.assertEqual(self.memory.pending_review()['pending_count'], 0)
        self.assertIn('lab_start', {s.name for s in self.memory.project_context(original).operations})
        self.assertNotIn('memory_consolidation', self.memory.project_context(original).research_agenda)
        self.native('new-failure')
        self.assertEqual(self.memory.pending_review()['pending_count'], 1)
        self.assertEqual(self.backend.requests, [])

    def test_dispositions_are_revisioned_recoverable_and_do_not_claim_usage_or_improvement(self):
        self.native()
        saved = self.memory.export()
        self.mutate('memory_review', source_refs=['history:decision_maker:failed-source'],
            disposition='not_useful', reason='This fixture failure has no transferable conclusion.')
        self.assertEqual(AgentMemory._validated(self.memory.export()), self.memory.export())
        with self.assertRaises(RecoveryRequired):
            self.memory.restore(saved)
        restored = AgentMemory()
        restored.bind(self.harness, self.root / 'feedback.json')
        self.assertEqual(restored.pending_review()['pending_count'], 0)
        self.assertEqual(restored.pending_review()['dispositions'], {'not_useful': 1})
        self.assertFalse(restored.pending_review()['usage_verified'])
        # Re-delivery alone does not manufacture another learning obligation.
        self.native()
        self.assertEqual(restored.pending_review()['pending_count'], 0)

    def test_review_rejects_blank_reasons_unknown_sources_and_unlinked_known_claims(self):
        self.native()
        before = self.memory.export()
        valid = dict(source_refs=['history:decision_maker:failed-source'], disposition='deferred', reason='Need a retest')
        for changes in ({'reason': ' '}, {'source_refs': []}, {'source_refs': ['invented']},
                        {'disposition': 'already_known'}, {'disposition': 'already_known', 'memory_ids': ['missing']}):
            with self.subTest(changes=changes), self.assertRaises(OperationArgumentError):
                self.mutate('memory_review', **(valid | changes))
            self.assertEqual(self.memory.export(), before)

    def test_existing_note_can_cover_new_evidence_only_by_explicit_link(self):
        self.native()
        self.mutate('memory_write', memory_id='input', content='Check canonical user-message consumption.',
            source_refs=['history:decision_maker:failed-source'], status='supported')
        self.native('retest')
        self.assertEqual(self.memory.pending_review()['pending_count'], 1)
        self.assertEqual(self.memory.search(query='canonical')['items'][0]['memory_id'], 'input')
        self.mutate('memory_review', source_refs=['history:decision_maker:retest'],
            disposition='already_known', memory_ids=['input'], reason='Same missing-input mechanism; existing procedure applies.')
        self.assertEqual(self.memory.pending_review()['pending_count'], 0)
        self.assertEqual(len(self.memory.export()['memories']), 1)
        self.assertEqual(AgentMemory._validated(self.memory.export()), self.memory.export())

    def test_memory_backlog_pages_without_dropping_unseen_receipts(self):
        for i in range(19):
            self.native(str(i))
        self.assertEqual(self.memory.pending_review()['pending_count'], 19)
        for expected in (11, 3, 0):
            page = self.memory.pending_review()['items']
            self.mutate('memory_review', source_refs=['history:decision_maker:' + row['record_id'] for row in page],
                disposition='deferred', reason='Need input-contract retests; these failures are not improvement evidence.')
            self.assertEqual(self.memory.pending_review()['pending_count'], expected)

    def test_plugin_named_feedback_and_recent_failures_survive_intermediate_successful_reads(self):
        from rsisys.adapters.local_lab_feedback import ImmediateFeedbackWindow
        failure = self.native(ident='training-error', name='custom_training_result')
        context = self.view()
        self.assertEqual(context.research_agenda['previous_feedback']['result'], failure.payload())
        self.harness.add_context_processor(ImmediateFeedbackWindow())
        for i in range(12):
            result = OperationResult('Source read', (), False, data={'content': 'source page'})
            self.harness.memory.append(HistoryEntry(f'page-{i}', 'operation', None, result.summary, False,
                details={'operation': {'name': 'lab_read_file', 'arguments': {}}, 'result': result.payload()}))
            self.harness.state.operation_results.append(result)
        context = self.view()
        self.assertIn('training-error', {row.record_id for row in context.history})
        health = context.research_agenda['feedback_health']['recent_failures']
        self.assertEqual(health[0]['record_id'], 'training-error')
        self.assertFalse(health[0]['tracked_by_open_task'])
        self.assertEqual(self.memory.export()['revision'], 0)
        self.assertFalse(self.backend.requests)

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

    def test_retrieval_follows_current_feedback_and_open_task_not_only_static_objective(self):
        self.mutate('memory_write', memory_id='adapter', content='Verify LoRA adapter loading before evaluation.',
                    source_refs=[], status='unverified')
        for i in range(8):
            self.mutate('memory_write', memory_id=str(i), content='Unrelated bookkeeping note.',
                        source_refs=[], status='unverified')
        query = ContextQuery(objective='Improve medical accuracy', latest_feedback=FeedbackContextEvent(
            'event', 'job', 'failed', {'error': 'LoRA adapter loading failed'}))
        self.assertEqual(json.loads(self.memory.context(query))['items'][0]['memory_id'], 'adapter')
        self.mutate('research_task', action='open', task_id='adapter', title='Verify LoRA adapter loading')
        self.assertEqual(json.loads(self.memory.context(ContextQuery(objective=query.objective)))
                         ['items'][0]['memory_id'], 'adapter')
        self.assertFalse(self.backend.requests)

    def test_task_progress_outcome_and_closed_lookup_survive_recovery(self):
        self.native()
        self.mutate('research_task', action='open', task_id='adapter', title='Evaluate trained adapter',
                    next_step='Check the inference loader', acceptance_criteria='Loaded weights and fixed evaluation')
        self.mutate('research_task', action='update', task_id='adapter', next_step='Run fixed evaluation')
        self.mutate('research_task', action='close', task_id='adapter', resolution='inconclusive', outcome='No benefit established')
        restored = AgentMemory()
        restored.bind(self.harness, self.root / 'feedback.json')
        task = restored.execute(OperationCall('research_task_read', {'task_id': 'adapter'}))['task']
        self.assertEqual(task['next_step'], 'Run fixed evaluation')
        self.assertEqual(task['acceptance_criteria'], 'Loaded weights and fixed evaluation')
        self.assertEqual(task['outcome'], 'No benefit established')
        self.assertEqual(task['record_ids'], ['failed-source'])
        self.assertEqual(task['status'], 'closed')
        self.assertEqual(restored.export(), self.memory.export())
        self.assertFalse(self.backend.requests)

    def test_untracked_failure_has_executable_task_guidance_without_implicit_writes(self):
        original = self.native()
        for i in range(10):
            result = OperationResult('Read source', (), False, data={'content': 'source'})
            self.harness.memory.append(HistoryEntry('read-' + str(i), 'operation', None, result.summary, False,
                details={'operation': {'name': 'lab_read_file', 'arguments': {}}, 'result': result.payload()}))
        context = self.view()
        guidance = context.research_agenda['agent_memory']['suggested_task_action']
        self.assertEqual(guidance['arguments']['record_ids'], ['failed-source'])
        self.assertEqual(guidance['arguments']['expected_revision'], 0)
        self.assertEqual(self.memory.export()['revision'], 0)
        self.harness.tools.execute(OperationCall(guidance['operation'], guidance['arguments']))
        self.assertEqual(self.view().research_agenda['active_feedback']['observations'][0]['result'],
                         original.payload())
        self.assertFalse(self.backend.requests)

    def test_task_read_pages_are_revision_bound_and_do_not_mutate(self):
        for i in range(3):
            self.mutate('research_task', action='open', task_id=str(i), title='Inspect evidence')
            self.mutate('research_task', action='close', task_id=str(i), resolution='inconclusive', outcome='Inconclusive')
        before = self.memory.export()
        page = self.memory.execute(OperationCall('research_task_read', {'status': 'closed', 'limit': 1}))
        self.assertEqual(page['total'], 3)
        self.assertEqual(page['next_offset'], 1)
        self.assertEqual(self.memory.export(), before)
        with self.assertRaisesRegex(OperationArgumentError, 'revision'):
            self.memory.execute(OperationCall('research_task_read', {'offset': 1}))
        self.mutate('research_task', action='open', task_id='new', title='Next investigation')
        with self.assertRaisesRegex(OperationArgumentError, 'Stale'):
            self.memory.execute(OperationCall('research_task_read',
                {'offset': 1, 'expected_revision': page['revision']}))

    def recorded_task_change(self, action, **fields):
        args = {'expected_revision': self.memory.export()['revision'], 'task_id': 'debug',
                'action': action, **fields}
        payload = self.memory.execute(OperationCall('research_task', args))
        result = OperationResult('Task changed', (), False, data=payload)
        self.harness.memory.append(HistoryEntry('task-change-' + str(payload['revision']), 'operation', None,
            result.summary, False, details={'operation': {'name': 'research_task', 'arguments': args},
                                           'result': result.payload()}))

    def test_new_result_guides_existing_task_review_without_duplicate_open_or_implicit_writes(self):
        self.native()
        self.recorded_task_change('open', title='Repair loader', next_step='Retest',
                                  acceptance_criteria='Real adapter load and fixed evaluation')
        later = self.native('later', name='custom_evaluation_result')
        before = self.memory.export()
        card = self.view().research_agenda['agent_memory']
        review = card['task_reviews'][0]
        self.assertEqual(review['task_id'], 'debug')
        self.assertEqual(review['record_ids'], ['later'])
        self.assertEqual({x['arguments']['action'] for x in review['action_templates']}, {'pin', 'update', 'close'})
        self.assertNotIn('suggested_task_action', card)
        self.assertEqual(self.memory.export(), before)
        self.assertEqual(self.view().research_agenda['active_feedback']['observations'][-1]['result'], later.payload())
        self.assertFalse(self.backend.requests)

    def test_pin_does_not_acknowledge_progress_but_explicit_update_does(self):
        self.native()
        self.recorded_task_change('open', title='Repair loader')
        self.native('later', name='custom_evaluation_result')
        self.recorded_task_change('pin', record_ids=['later'])
        self.assertEqual(self.view().research_agenda['agent_memory']['task_reviews'][0]['record_ids'], ['later'])
        self.recorded_task_change('update', next_step='Investigate the second error')
        card = self.view().research_agenda['agent_memory']
        self.assertNotIn('task_reviews', card)
        self.assertEqual(card['memory_review']['source_refs'],
                         ['history:decision_maker:failed-source', 'history:decision_maker:later'])
        self.mutate('memory_write', memory_id='loader', content='Inspect the bundle-relative path.',
                    source_refs=card['memory_review']['source_refs'], status='unverified')
        self.assertNotIn('memory_review', self.view().research_agenda['agent_memory'])

    def test_memory_inbox_covers_results_even_after_a_task_closes(self):
        self.native()
        self.recorded_task_change('open', title='Repair loader')
        self.native('decisive', name='custom_evaluation_result')
        self.recorded_task_change('close', resolution='abandoned', outcome='Abandoned after failed retest')
        self.native('unrelated', name='other_experiment')
        card = self.view().research_agenda['agent_memory']
        self.assertEqual(card['memory_review']['source_refs'], ['history:decision_maker:failed-source',
                         'history:decision_maker:decisive', 'history:decision_maker:unrelated'])
        self.assertNotIn('task_reviews', card)
        self.assertEqual(self.memory.export()['memories'], {})

    def test_imported_results_do_not_trigger_new_task_or_note_review(self):
        self.native('historical')
        entry = self.harness.memory.history()[-1]
        self.harness.memory._history[-1] = replace(entry, details={**entry.details,
            'continuation_origin': {'record_id': 'old'}})
        self.mutate('research_task', action='open', task_id='debug', title='Inherited investigation')
        self.assertNotIn('task_reviews', self.view().research_agenda['agent_memory'])
        self.assertNotIn('memory_review', self.view().research_agenda['agent_memory'])

    def test_imported_failures_do_not_implicitly_reopen_historical_work(self):
        result = OperationResult('Old failed experiment', (), False, ExecutionStatus.FAILED, 'old error')
        self.harness.memory.append(HistoryEntry('historical', 'operation', None, result.summary, False,
            details={'operation': {'name': 'lab_result', 'arguments': {}}, 'result': result.payload(),
                     'continuation_origin': {'record_id': 'old'}}))
        card = self.view().research_agenda['agent_memory']
        self.assertEqual(card['untracked_failure_count'], 0)
        self.assertNotIn('suggested_task_action', card)
        self.assertEqual(self.memory.export()['revision'], 0)

    def test_task_progress_tampering_is_detected_and_closure_does_not_fabricate_memory(self):
        self.mutate('research_task', action='open', task_id='work', title='Evaluate', next_step='Load weights')
        self.mutate('research_task', action='update', task_id='work', next_step='Measure')
        original = self.memory.export()
        changed = deepcopy(original)
        changed['tasks']['work']['next_step'] = 'Forged progress'
        changed['revisions'][-1]['tasks']['work']['next_step'] = 'Forged progress'
        with self.assertRaisesRegex(ValueError, 'Task mutation differs'):
            AgentMemory._validated(changed)
        self.mutate('research_task', action='close', task_id='work', resolution='interrupted', outcome='Interrupted; no score')
        self.assertEqual(self.memory.export()['memories'], {})
        self.assertEqual(self.view().research_agenda['agent_memory']['recent_closed_tasks'][0]
                         ['outcome_preview'], 'Interrupted; no score')

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
        self.mutate('research_task', action='close', task_id='debug', resolution='inconclusive', outcome='No retest')
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
            result = OperationResult('Source page', (), False, data={'content': 'intermediate source page'})
            self.harness.memory.append(HistoryEntry('read-' + str(i), 'operation', None, result.summary, False,
                details={'operation': {'name': 'lab_read_file', 'arguments': {}}, 'result': result.payload()}))
        retest = self.native('retest', text='Retest still failed')
        view = self.view().research_agenda['active_feedback']
        by_id = {row['record_id']: row for row in view['observations']}
        self.assertEqual(by_id['failed-source']['result'], original.payload())
        self.assertEqual(by_id['retest']['result'], retest.payload())
        self.assertEqual(len(view['observations']), 2)
        before = self.harness.memory.history()
        self.mutate('research_task', action='close', task_id='adapter-debug', resolution='abandoned', outcome='Investigation ended')
        self.assertEqual(self.view().research_agenda['active_feedback']['observations'], [])
        self.assertEqual(self.harness.memory.history(), before)
        self.assertEqual(self.memory.export()['memories'], {})

    def test_custom_producer_feedback_is_retained_across_reads_and_running_polls(self):
        from rsisys.loops.contracts.delivery import FeedbackDelivery
        self.native('original', name='custom_training')
        self.mutate('research_task', action='open', task_id='custom', title='Check custom producer')
        def record(ident, result):
            self.harness.memory.append(HistoryEntry(ident, 'operation', None, result.summary, False,
                details={'operation': {'name': 'plugin_' + ident, 'arguments': {}}, 'result': result.payload()}))
        opaque = OperationResult('Opaque producer output', (), False, data={'output': {'native': 'done'}},
                                 feedback_delivery=FeedbackDelivery('plugin:producer:1'))
        record('opaque', opaque)
        measured = OperationResult('Fixed evaluation', (), False, data={'quality_metrics': {'accuracy': 0.0}})
        record('measured', measured)
        for i in range(12):
            record('read-' + str(i), OperationResult('Source page', (), False, data={'content': 'page'}))
            record('poll-' + str(i), OperationResult('Still running', (), False,
                                                    data={'job_id': 'job', 'state': 'running'}))
        rows = {row['record_id']: row for row in self.view().research_agenda['active_feedback']['observations']}
        self.assertEqual(set(rows), {'original', 'opaque', 'measured'})
        self.assertEqual(rows['opaque']['result'], opaque.payload())
        self.assertEqual(rows['measured']['result'], measured.payload())
        record('terminal', OperationResult('Terminal job', (), False,
                                           data={'job_id': 'job', 'state': {'status': 'succeeded'}}))
        rows = self.view().research_agenda['active_feedback']['observations']
        self.assertEqual({row['record_id'] for row in rows}, {'original', 'measured', 'terminal'})
        self.assertEqual(self.memory.export()['revision'], 1)
        self.assertFalse(self.backend.requests)
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

    def test_state_port_does_not_claim_portable_file_relocation_support(self):
        with self.assertRaisesRegex(TypeError, 'same-directory recovery'):
            AgentMemoryState(self.memory).validate_snapshot()

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
            return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': 'memory_search',
                'arguments': {'query': 'input'}}), request.model, True, TokenUsage(101, 9))
        with patch.object(self.harness.policy.backend, 'generate', side_effect=generate):
            self.harness._call_policy(self.harness.policy.decide, self.view())
        payload = json.loads(requests[0].prompt)
        observation = payload['research_agenda']['active_feedback']['observations'][0]
        self.assertEqual(observation['result'], json.loads(encoded(native.payload())))
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.harness.state.measured_input_tokens, 101)
        self.assertEqual(self.harness.state.measured_output_tokens, 9)

    def test_same_agent_executes_task_progress_memory_and_closure_through_real_decision_loop(self):
        native = self.native()
        requests = []
        def generate(request):
            context = json.loads(request.prompt)
            requests.append(context)
            revision = context['research_agenda']['agent_memory']['revision']
            if len(requests) == 1:
                action = context['research_agenda']['agent_memory']['suggested_task_action']
                args = {**action['arguments'], 'task_id': 'input-debug',
                        'next_step': 'Inspect input consumption', 'acceptance_criteria': 'Retest canonical messages'}
                operation = 'research_task'
            elif len(requests) == 2:
                self.assertEqual(context['research_agenda']['active_feedback']['observations'][0]['result'],
                                 json.loads(encoded(native.payload())))
                operation, args = 'research_task', {'expected_revision': revision, 'task_id': 'input-debug',
                                                   'action': 'update', 'next_step': 'Retest the candidate'}
            elif len(requests) == 3:
                operation, args = 'memory_write', {'expected_revision': revision, 'memory_id': 'input-contract',
                    'content': 'Verify canonical user-message consumption before interpreting a score.',
                    'source_refs': ['history:decision_maker:failed-source'], 'status': 'supported'}
            elif len(requests) == 4:
                operation, args = 'research_task', {'expected_revision': revision, 'task_id': 'input-debug',
                    'action': 'close', 'resolution': 'interrupted', 'outcome': 'Interrupted; no successful retest yet'}
            else:
                return GenerationResponse('{"kind":"finish"}', request.model, True, TokenUsage(100, 10))
            return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': operation, 'arguments': args}),
                                      request.model, True, TokenUsage(100, 10))
        with patch.object(self.harness.policy.backend, 'generate', side_effect=generate):
            results = [self.harness.step('Repair input handling', self.loop.action_space) for _ in range(5)]
        self.assertEqual(results[-1].kind, 'finished')
        self.assertEqual([x['research_agenda']['agent_memory']['revision'] for x in requests], [0, 1, 2, 3, 4])
        self.assertEqual(self.memory.export()['tasks']['input-debug']['status'], 'closed')
        self.assertEqual(self.memory.export()['memories']['input-contract']['source_refs'],
                         ['history:decision_maker:failed-source'])
        self.assertEqual(self.harness.state.measured_input_tokens, 500)
        self.assertEqual(self.backend.requests, [])

    def test_forced_compaction_uses_no_hidden_summary_model_request(self):
        for i in range(150):
            self.harness.memory.append(HistoryEntry('old-' + str(i), 'decision', None, 'old note ' * 120, True))
        # Leave room for the installed memory/task schemas; history must still
        # compact without an extra summarization request.
        self.harness.context_compactor = BasicContextCompactor(trigger_tokens=130000, target_tokens=80000,
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
