"""Bounded archive projections retain provenance without mutating raw stores."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import unittest
from unittest.mock import Mock

from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops._memory import HistoryEntry, Memory
from rsisys.loops.decision_making.action_space import OperationResult, OperationSpec
from rsisys.loops.decision_making.harness.context import BasicContextCompactor, IdentityContextCompactor, encoded
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools


def context(**kwargs):
    return DecisionContext('Investigate', (), (), (), **kwargs)


class ContextCompactionTests(unittest.TestCase):
    def test_full_history_provenance_and_exact_recent_tail(self):
        history = tuple(HistoryEntry(str(i), 'operation', None, f'observation {i}', i % 2 == 0,
                                     details={'raw': 'x' * 100}) for i in range(80))
        raw = context(history=history)
        projected = BasicContextCompactor(recent_tail=3).compact(raw)
        self.assertEqual(projected.history, history[-3:])
        source = projected.context_summary['sources']['history']
        self.assertEqual(source['input_count'], 80)
        self.assertEqual(source['omitted_count'], 77)
        self.assertEqual(source['first_omitted'], 'history:0')
        self.assertEqual(source['last_omitted'], 'history:76')
        digest = sha256()
        for row in history:
            digest.update(encoded(asdict(row)).encode())
            digest.update(b'\n')
        self.assertEqual(source['sha256'], digest.hexdigest())
        self.assertEqual(projected.context_summary['authority'], 'not_scientific_truth')
        self.assertTrue(all(item['incomplete'] for item in projected.context_summary['excerpts']))
        self.assertEqual(raw.history, history)
        projected.history[-1].details['raw'] = 'mutated projection'
        self.assertEqual(raw.history[-1].details['raw'], 'x' * 100)

    def test_each_archive_stream_and_summary_is_character_bounded(self):
        history = tuple(HistoryEntry(str(i), 'operation', None, 'observation' * 100, True,
                                     details={'payload': 'x' * 2000}) for i in range(30))
        memories = tuple(Memory(str(i), 'semantic', 'memory' * 200, ('evidence',)) for i in range(30))
        experiences = tuple(ExperienceRecord(str(i), 'runtime_feedback', 'decision_maker', None,
                                             'experience' * 100, True) for i in range(30))
        results = tuple(OperationResult('result' * 100, (), True, data={'payload': 'x' * 4000}) for _ in range(30))
        raw = replace(context(history=history, memories=memories), experience=experiences, operation_results=results)
        compactor = BasicContextCompactor(stream_chars=2048, summary_chars=4096)
        projected = compactor.compact(raw)
        for stream in compactor.streams:
            with self.subTest(stream=stream):
                self.assertLessEqual(len(encoded([asdict(row) for row in getattr(projected, stream)])), 2048)
                self.assertIn(stream, projected.context_summary['sources'])
        self.assertLessEqual(len(encoded(projected.context_summary)), 4096)
        self.assertEqual(len(raw.history), 30)
        self.assertEqual(projected.history, ())  # oversized newest record is explicitly summarized
        self.assertEqual(projected.memories[0].memory_id, '0')  # recall order is newest-first
        self.assertEqual(projected.experience[-1].record_id, '29')

    def test_metadata_bound_even_for_external_oversized_record_ids(self):
        raw = replace(context(history=(HistoryEntry('h' * 10000, 'operation', None, 'raw', False),)),
            memories=(Memory('m' * 10000, 'semantic', 'raw'),),
            experience=(ExperienceRecord('e' * 10000, 'runtime_feedback', 'decision_maker', None, 'raw', False),),
            operation_results=(OperationResult('x' * 10000, (), False),))
        projected = BasicContextCompactor(stream_chars=256, summary_chars=4096).compact(raw)
        self.assertLessEqual(len(encoded(projected.context_summary)), 4096)

    def test_default_harness_preserves_raw_memory_and_honors_memory_limit(self):
        memory = DecisionMemory()
        for i in range(20):
            memory.append(HistoryEntry(str(i), 'operation', None, f'observation {i}', False))
        memory.remember(Memory('idea', 'semantic', 'unverified idea', ('history:0',)))
        before = memory.export()
        harness = DefaultDecisionHarness(Mock(), BoundDecisionTools({}),
            BasicExperienceSystem(InMemoryExperienceStore()), memory=memory, memory_limit=2)
        projected = harness._context('Investigate')
        self.assertEqual([row.record_id for row in projected.history], ['18', '19'])
        self.assertEqual(projected.context_summary['sources']['history']['omitted_count'], 18)
        self.assertEqual(memory.export(), before)
        self.assertEqual(projected.research_agenda['revision'], 0)

    def test_identity_and_live_fields_are_not_subject_to_archive_budget(self):
        large = 'x' * 40000
        raw = replace(context(history=(HistoryEntry('one', 'operation', None, large, False),)),
                      operations=(OperationSpec('large_schema', large, {}),), objective=large,
                      run_control={'hint': large})
        projected = BasicContextCompactor(stream_chars=256).compact(raw)
        self.assertEqual(projected.operations, raw.operations)
        self.assertEqual(projected.objective, large)
        self.assertEqual(projected.run_control, raw.run_control)
        identity = IdentityContextCompactor().compact(raw, recent_limit=1)
        self.assertEqual(identity, raw)
        identity.run_control['hint'] = 'client mutation'
        self.assertEqual(raw.run_control['hint'], large)

    def test_review_policy_call_recompacts_new_tool_results_and_refreshes_agenda(self):
        agenda = ResearchAgenda()
        policy = Mock()
        policy.review.return_value = Decision('finish')
        harness = DefaultDecisionHarness(policy, BoundDecisionTools({}),
            BasicExperienceSystem(InMemoryExperienceStore()), research_agenda=agenda,
            context_compactor=BasicContextCompactor(stream_chars=1024))
        saved = harness._context('Investigate')
        agenda.patch(expected_revision=0, upserts=[{'id': 's', 'kind': 'stage_summary', 'text': 'Tentative'}],
                     remove_ids=[], reason='Record progress', evidence_refs=[])
        saved = replace(saved, operation_results=(OperationResult('oversized result', (), True,
                                                                  data={'payload': 'x' * 10000}),))
        harness._call_policy(policy.review, saved)
        seen = policy.review.call_args.args[0]
        self.assertEqual(seen.research_agenda['revision'], 1)
        self.assertEqual(seen.operation_results, ())
        self.assertEqual(seen.context_summary['sources']['operation_results']['omitted_count'], 1)
        self.assertEqual(len(saved.operation_results[0].data['payload']), 10000)

    def test_invalid_configuration_and_repeat_projection(self):
        for kwargs in ({'recent_tail': 0}, {'recent_tail': True}, {'stream_chars': 10}, {'summary_chars': 100}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BasicContextCompactor(**kwargs)
        raw = replace(context(), operation_results=(OperationResult('big', (), False, data={'x': 'x' * 10000}),))
        compactor = BasicContextCompactor(stream_chars=1024)
        first = compactor.compact(raw)
        second = compactor.compact(first)
        self.assertEqual(first.context_summary, second.context_summary)


if __name__ == '__main__':
    unittest.main()
