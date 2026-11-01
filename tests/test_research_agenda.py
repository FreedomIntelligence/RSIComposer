"""Model-owned notebook semantics and ordinary capability execution."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.records import ExperienceRecord
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops._memory import HistoryEntry, Memory
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall, OperationResult
from rsisys.loops.decision_making.action_space.access import authorized_space
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.context import BasicContextCompactor
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.harness.tools.research import ResearchTools


def change(agenda, rows, **kwargs):
    return agenda.patch(expected_revision=agenda.revision, upserts=rows, remove_ids=[],
                        reason='Revise working hypotheses', evidence_refs=['history:decision_maker:receipt'], **kwargs)


class ResearchAgendaTests(unittest.TestCase):
    def setUp(self):
        self.agenda = ResearchAgenda()
        change(self.agenda, [
            {'id': 'question', 'kind': 'agenda', 'text': 'Compare alternatives'},
            {'id': 'next', 'kind': 'todo', 'text': 'Run a control', 'status': 'waiting_for_data'},
            {'id': 'stage-1', 'kind': 'stage_summary', 'text': 'Tentative signal; not a validated finding',
             'evidence_refs': ['artifact:trial-1']},
        ])

    def test_partial_patch_preserves_omitted_ids_and_fields(self):
        original = self.agenda.read()['items']
        result = change(self.agenda, [{'id': 'next', 'status': 'cancelled_by_model'}])
        rows = self.agenda.read()['items']
        self.assertEqual(result['revision'], 2)
        self.assertEqual(rows[0], original[0])
        self.assertEqual(rows[2], original[2])
        self.assertEqual(rows[1]['text'], 'Run a control')
        self.assertEqual(rows[1]['status'], 'cancelled_by_model')
        self.assertEqual(rows[1]['revision'], 2)
        self.assertEqual(self.agenda.read(view='revisions')['items'][-1]['evidence_refs'],
                         ['history:decision_maker:receipt'])

    def test_failed_late_validation_and_stale_patch_are_atomic(self):
        before = self.agenda.export()
        args = dict(expected_revision=1, upserts=[{'id': 'next', 'text': 'must not leak'}],
                    remove_ids=['missing'], reason='test', evidence_refs=[])
        with self.assertRaisesRegex(ValueError, 'unknown ID'):
            self.agenda.patch(**args)
        self.assertEqual(self.agenda.export(), before)
        args.update(remove_ids=[], expected_revision=0)
        with self.assertRaisesRegex(ValueError, 'Stale revision'):
            self.agenda.patch(**args)
        self.assertEqual(self.agenda.export(), before)
        with self.assertRaisesRegex(ValueError, 'New items require'):
            change(self.agenda, [{'id': 'next', 'text': 'must not leak'}, {'id': 'new'}])
        self.assertEqual(self.agenda.export(), before)

    def test_duplicates_unknown_fields_and_missing_explicit_evidence_rejected(self):
        args = dict(expected_revision=1, upserts=[{'id': 'next', 'text': 'new'}],
                    remove_ids=[], reason='test', evidence_refs=[])
        for edits in ({'upserts': [{'id': 'next'}, {'id': 'next'}]},
                      {'remove_ids': ['next']}, {'expected_revision': True},
                      {'upserts': [{'id': 'new', 'kind': 'fixed_workflow', 'text': 'bad'}]},
                      {'evidence_refs': None}, {'reason': ''},
                      {'upserts': [{'id': 'next', 'extra': True}]},
                      {'upserts': [], 'remove_ids': []}):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                self.agenda.patch(**{**args, **edits})
        incomplete = dict(args)
        del incomplete['evidence_refs']
        with self.assertRaises(ValueError):
            self.agenda.validate_patch(incomplete)
        self.assertEqual(self.agenda.revision, 1)

    def test_pagination_is_detached_revision_and_filter_pinned(self):
        first = self.agenda.read(limit=1)
        second = self.agenda.read(limit=1, cursor=first['next_cursor'])
        self.assertEqual(second['items'][0]['id'], 'next')
        first['items'][0]['text'] = 'client mutation'
        self.assertEqual(self.agenda.read()['items'][0]['text'], 'Compare alternatives')
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.agenda.read(kind='todo', cursor=first['next_cursor'])
        change(self.agenda, [{'id': 'next', 'status': 'done'}])
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.agenda.read(cursor=first['next_cursor'])
        self.assertEqual(self.agenda.read(kind='todo')['total'], 1)
        self.assertEqual(self.agenda.read(view='revisions', limit=1)['next_cursor'], '2:revisions:all:1')

    def test_removal_retains_original_revision_and_restore_is_validated_atomically(self):
        self.agenda.patch(expected_revision=1, upserts=[], remove_ids=['stage-1'],
                          reason='Superseded by new evidence', evidence_refs=['artifact:trial-2'])
        archive = self.agenda.export()
        self.assertIn('Tentative signal', json.dumps(archive['revisions']))
        restored = ResearchAgenda()
        restored.restore(archive)
        self.assertEqual(restored.export(), archive)
        bad = deepcopy(archive)
        bad['items'][0]['text'] = 'forged state'
        with self.assertRaisesRegex(ValueError, 'revision log'):
            restored.restore(bad)
        self.assertEqual(restored.export(), archive)

    def test_persistence_write_failure_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agenda.json'
            agenda = ResearchAgenda(path)
            change(agenda, [{'id': 'q', 'kind': 'agenda', 'text': 'question'}])
            before = path.read_bytes()
            with patch('rsisys.loops.decision_making.harness.research.atomic_write', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    change(agenda, [{'id': 'q', 'text': 'changed'}])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(ResearchAgenda(path).export(), agenda.export())
            change(agenda, [{'id': 'q', 'text': 'changed'}])
            self.assertEqual(ResearchAgenda(path).revision, 2)


class ResearchToolTests(unittest.TestCase):
    def setUp(self):
        self.agenda, self.memory = ResearchAgenda(), DecisionMemory()
        self.provider = ResearchTools(self.agenda, self.memory)
        self.tools = BoundDecisionTools(self.provider.bindings)
        self.space = ModularDecisionActionSpace(self.provider.definitions)

    def invoke(self, name, args):
        return self.tools.execute(self.space.validate(OperationCall(name, args)))

    def test_capabilities_permissions_and_complete_raw_pagination_during_appends(self):
        review = authorized_space(self.space, 'review')
        self.assertEqual({spec.name for spec in review.describe()}, {'research_read', 'research_archive'})
        raw = HistoryEntry('big', 'operation', None, 'summary', True,
                           source_refs=('artifact:raw',), details={'payload': '完整资料' * 1800})
        self.memory.append(raw)
        cursor, texts = None, []
        while True:
            result = self.invoke('research_archive', {'limit': 2, 'cursor': cursor})
            texts.extend(item['text'] for item in result.data['items'])
            self.memory.append(HistoryEntry(f'read-{len(texts)}', 'operation', None, 'read receipt', False))
            cursor = result.data['next_cursor']
            if cursor is None:
                break
        restored = json.loads(''.join(texts))
        self.assertEqual(restored['details'], raw.details)
        self.assertEqual(restored['source_refs'], ['artifact:raw'])
        self.assertEqual(restored['record_id'], 'big')

    def test_exact_memory_and_research_revision_can_be_read_in_small_chunks(self):
        self.memory.remember(Memory('memo', 'semantic', 'x' * 8000, ('history:source',)))
        first = self.invoke('research_archive', {'collection': 'memories', 'record_id': 'memo', 'limit': 1})
        self.memory.remember(Memory('memo', 'semantic', 'changed'))
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.invoke('research_archive', {'collection': 'memories', 'record_id': 'memo',
                                             'cursor': first.data['next_cursor']})
        change(self.agenda, [{'id': 's', 'kind': 'stage_summary', 'text': 'x' * 15000}])
        first = self.invoke('research_archive', {'collection': 'research_revisions', 'record_id': '1', 'limit': 1})
        self.assertIsNotNone(first.data['next_cursor'])
        self.assertLessEqual(len(first.data['items'][0]['text']), 1000)
        self.assertEqual(first.data['items'][0]['record_id'], '1')

    def test_compaction_sources_resolve_full_operation_and_experience_records(self):
        experience = BasicExperienceSystem(InMemoryExperienceStore())
        record = ExperienceRecord('exp', 'runtime_feedback', 'shared', None, 'Tentative observation',
                                  True, source_refs=('history:decision_maker:receipt',),
                                  details={'raw': 'x' * 6000})
        experience.record(record)
        result = OperationResult('Long tool response', (), True, data={'raw': 'y' * 6000})
        self.memory.append(HistoryEntry('receipt', 'operation', None, result.summary, True,
                                        details={'result': asdict(result)}))
        raw = DecisionContext('Investigate', (record,), (), (result, OperationResult('Latest', (), False)))
        projected = BasicContextCompactor(trigger_tokens=10000, target_tokens=5000, summary_tokens=500).compact(raw)
        provider = ResearchTools(self.agenda, self.memory, experience=experience)
        for collection in ('operation_results', 'experience'):
            with self.subTest(collection=collection):
                source = projected.context_summary['sources'][collection]
                self.assertEqual(source['read_path']['collection'], collection)
                ident = source['first_omitted'].split(':', 1)[1]
                cursor, chunks = None, []
                while True:
                    page = provider.bindings['research_archive']({
                        'collection': collection, 'record_id': ident, 'limit': 2, 'cursor': cursor})
                    self.assertTrue(page.data['found'])
                    chunks.extend(row['text'] for row in page.data['items'])
                    cursor = page.data['next_cursor']
                    if cursor is None:
                        break
                restored = json.loads(''.join(chunks))
                if collection == 'operation_results':
                    self.assertEqual(restored['result']['data'], result.data)
                    self.assertEqual(restored['history_refs'], ['history:decision_maker:receipt'])
                else:
                    self.assertEqual(restored['details'], record.details)
                    self.assertEqual(restored['simulated'], True)
        with self.assertRaisesRegex(ValueError, 'not bound'):
            self.invoke('research_archive', {'collection': 'experience'})

    def test_model_invokes_patch_through_existing_policy_and_receipts(self):
        model = ModelRef('backend', 'model', '1')
        args = {'expected_revision': 0,
                'upserts': [{'id': 'next', 'kind': 'todo', 'text': 'Try a different control'}],
                'remove_ids': [], 'reason': 'Choose next question', 'evidence_refs': []}
        seen = []
        class Backend:
            def generate(_, request):
                seen.append(json.loads(request.prompt))
                payload = ({'kind': 'invoke', 'operation': 'research_patch', 'arguments': args}
                           if len(seen) == 1 else {'kind': 'finish'})
                return GenerationResponse(json.dumps(payload), model, False)
        harness = DefaultDecisionHarness(ModelDecisionPolicy(Backend(), model), self.tools,
            BasicExperienceSystem(InMemoryExperienceStore()), memory=self.memory, research_agenda=self.agenda)
        self.assertEqual(harness.step('Investigate', self.space).kind, 'progressed')
        self.assertEqual(harness.step('Investigate', self.space).kind, 'finished')
        self.assertEqual(seen[1]['research_agenda']['revision'], 1)
        self.assertEqual(seen[1]['research_agenda']['items'][0]['id'], 'next')
        receipt = self.memory.history(kind='operation')[0]
        self.assertEqual(receipt.details['operation']['name'], 'research_patch')
        self.assertEqual(receipt.details['result']['data']['revision'], 1)

    def test_receipt_retry_does_not_repeat_the_successful_patch(self):
        args = {'expected_revision': 0, 'upserts': [{'id': 'q', 'kind': 'agenda', 'text': 'question'}],
                'remove_ids': [], 'reason': 'Choose question', 'evidence_refs': []}
        policy = Mock()
        policy.decide.return_value = Decision('invoke', operation=OperationCall('research_patch', args))
        harness = DefaultDecisionHarness(policy, self.tools, BasicExperienceSystem(InMemoryExperienceStore()),
                                         memory=self.memory, research_agenda=self.agenda)
        append = self.memory.append
        def fail_receipt(entry):
            if entry.kind == 'operation':
                raise OSError('receipt unavailable')
            append(entry)
        with patch.object(self.memory, 'append', side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, 'receipt unavailable'):
                harness.step('Investigate', self.space)
        self.assertEqual(self.agenda.revision, 1)
        result = harness.step('Investigate', self.space)
        self.assertEqual(result.kind, 'progressed')
        self.assertEqual(self.agenda.revision, 1)
        self.assertEqual(policy.decide.call_count, 1)
        self.assertEqual(len(self.memory.history(kind='operation')), 1)


if __name__ == '__main__':
    unittest.main()
