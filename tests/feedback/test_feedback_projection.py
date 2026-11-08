"""Replaceable evidence projections preserve source records and context shape."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import (
    BasicExperienceSystem, FeedbackSelector, RecentFeedbackSelector,
)
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.decision_making.harness.state import DecisionContext


class ObjectiveFeedbackSelector:
    """Test-only aggregation; the library prescribes no aggregation strategy."""

    def select(self, experience, objective, *, limit):
        sources = experience.retrieve(text=objective)
        if not sources:
            return ()
        return (ExperienceRecord(
            record_id='view:matching-evaluations', kind='insight', scope='shared',
            directive_id=None, summary=f'{len(sources)} matching observations',
            simulated=any(record.simulated for record in sources),
            source_refs=tuple(record.record_id for record in sources),
            details={'scores': [record.details['score'] for record in sources]},
        ),)[:limit]


class FeedbackProjectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        sqlite = SQLiteExperienceStore(Path(self.directory.name) / 'feedback.sqlite3')
        self.addCleanup(sqlite.close)
        self.records = tuple(ExperienceRecord(
            record_id=f'feedback:{index}', kind='evaluation', scope=scope,
            directive_id=None, summary=summary, simulated=index == 2,
            source_refs=(f'artifact:score:{index}',),
            details={'score': index, 'metrics': {'values': [index]}},
            run_id='run:original', created_at=f'2026-11-01T00:00:0{index}Z',
        ) for index, (scope, summary) in enumerate((
            ('actor', 'latency baseline'),
            ('shared', 'quality baseline'),
            ('decision_maker', 'latency follow-up'),
        )))
        self.systems = tuple(BasicExperienceSystem(store)
                             for store in (InMemoryExperienceStore(), sqlite))
        for system in self.systems:
            for record in self.records:
                system.record(record)

    def test_default_keeps_raw_recent_records_and_all_provenance(self):
        for system in self.systems:
            with self.subTest(store=type(system.store).__name__):
                selector: FeedbackSelector = RecentFeedbackSelector()
                with patch.object(system, 'record', side_effect=AssertionError('unexpected write')):
                    selected = selector.select(system, 'unrelated task', limit=2)
                self.assertIsInstance(selected, tuple)
                self.assertEqual(selected, self.records[-2:])
                self.assertEqual(selector.select(system, 'latency', limit=10), self.records)
                self.assertEqual(system.retrieve(), self.records)

    def test_custom_selector_replaces_raw_context_with_transient_sourced_view(self):
        for system in self.systems:
            with self.subTest(store=type(system.store).__name__):
                before = system.retrieve()
                contexts = []
                for selector in (RecentFeedbackSelector(), ObjectiveFeedbackSelector()):
                    with patch.object(system, 'record', side_effect=AssertionError('unexpected write')):
                        contexts.append(DecisionContext(
                            objective='latency',
                            experience=selector.select(system, 'latency', limit=1),
                            operations=(), operation_results=(),
                        ))
                raw, projected = contexts
                self.assertEqual(raw.experience, (self.records[-1],))
                self.assertEqual(len(projected.experience), 1)
                view = projected.experience[0]
                self.assertEqual(view.source_refs, ('feedback:0', 'feedback:2'))
                self.assertEqual(view.details, {'scores': [0, 2]})
                self.assertTrue(view.simulated)
                self.assertEqual(asdict(projected)['experience'][0]['source_refs'], view.source_refs)
                self.assertEqual(system.retrieve(), before)
                self.assertNotIn(view.record_id, {record.record_id for record in before})
                self.assertEqual(ObjectiveFeedbackSelector().select(system, 'absent', limit=1), ())

    def test_nested_edits_to_selected_records_do_not_rewrite_stored_evidence(self):
        for system in self.systems:
            with self.subTest(store=type(system.store).__name__):
                before = system.retrieve()
                selected = RecentFeedbackSelector().select(system, 'latency', limit=1)
                selected[0].details['metrics']['values'].append(99)
                self.assertEqual(system.retrieve(), before)
                self.assertEqual(self.records[-1].details['metrics']['values'], [2])

    def test_empty_store_and_positive_limit_contract(self):
        selector = RecentFeedbackSelector()
        system = BasicExperienceSystem(InMemoryExperienceStore())
        self.assertEqual(selector.select(system, 'task', limit=1), ())
        for limit in (0, -1, True, 1.5, None):
            with self.subTest(limit=limit):
                with patch.object(system, 'retrieve') as retrieve:
                    with self.assertRaisesRegex(ValueError, 'positive integer'):
                        selector.select(system, 'task', limit=limit)
                    retrieve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
