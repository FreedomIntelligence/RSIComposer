from copy import deepcopy
from dataclasses import asdict, replace
import unittest
from unittest.mock import Mock

from rsisys.loops._memory import HistoryEntry
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.loops.decision_making.harness.context import BasicContextCompactor, ContextBudgetError, encoded
from rsisys.loops.decision_making.harness.research_state import (
    bounded_evidence_view, full_history_ids, receipt_needs_protection, research_working_view,
)
from test_research_evidence_projection import fixture, measure, read_original


class RepeatedReceiptCompactionTests(unittest.TestCase):
    def repeated(self, count=45):
        context, tools, memory, experience = fixture(count=1)
        receipt = OperationResult('Training preview', (), False,
            data={'rows': [{'benchmark_id': 'fixture', 'text': 'sample ' * 1800}], 'offset': 0})
        rows = tuple(HistoryEntry(f'read-{i}', 'operation', None, receipt.summary, False,
            details={'decision_id': f'decision-read-{i}', 'result': receipt.payload()}) for i in range(count))
        for row in rows:
            memory.append(row)
        context = replace(context, history=(*context.history, *rows),
            operation_results=(*context.operation_results, *((receipt,) * count)))
        context = research_working_view(bounded_evidence_view(context), [], [])
        return context, tools, receipt

    def test_card_protection_tracks_content_not_representation(self):
        context, tools, _ = self.repeated(3)
        ordinary = context.history[-1]
        self.assertIn('evidence_card', ordinary.details)
        self.assertFalse(receipt_needs_protection(ordinary.details))
        for facts in ({'/data/quality_metrics': {'accuracy': 0.0}},
                      {'/details/result/status': 'failed'}, {'/error': 'failed to evaluate'}):
            card = deepcopy(ordinary.details)
            card['facts'].update(facts)
            self.assertTrue(receipt_needs_protection(card))
        self.assertTrue(receipt_needs_protection({'evidence_card': {'legacy': True}}))

    def test_repeated_archived_reads_fit_without_losing_measurements_or_latest(self):
        context, tools, receipt = self.repeated()
        saved = deepcopy(context)
        self.assertGreater(measure(context), 80000)
        writer = Mock(return_value='Repeated previews are archived; measured counterexamples remain visible.')
        compactor = BasicContextCompactor()
        view = compactor.compact(context, measure=measure, summarize=writer)
        self.assertLessEqual(measure(view), 24000)
        self.assertEqual(compactor.compactions, 1)
        writer.assert_called_once()
        self.assertEqual([h.record_id for h in view.history[-2:]], [h.record_id for h in context.history[-2:]])
        for visible, original in zip(view.history[-2:], context.history[-2:]):
            self.assertEqual(visible.details['facts'], original.details['facts'])
        self.assertEqual(view.operation_results[-1].data['facts'], context.operation_results[-1].data['facts'])
        self.assertEqual(view.last_feedback, context.last_feedback)
        self.assertTrue(any('quality_metrics' in encoded(asdict(h)) for h in view.history))
        self.assertEqual(context, saved)
        self.assertEqual(read_original(tools, next(h for h in context.history if h.record_id == 'read-0').details['evidence_card'])['record_id'], 'read-0')
        restored = BasicContextCompactor()
        restored.restore_state(compactor.export_state())
        again = restored.compact(context, measure=measure)
        self.assertEqual(again, view)

    def test_identical_raw_receipts_only_pin_latest_occurrence(self):
        context, _, _, _ = fixture(count=1)
        receipt = OperationResult('Repeated read', (), False, data={'content': 'x' * 1200})
        rows = tuple(HistoryEntry(f'read-{i}', 'operation', None, 'read', False,
                                 details={'result': receipt.payload()}) for i in range(100))
        context = replace(context, history=rows, operation_results=(receipt,) * 100,
                          experience=(), research_state={}, last_feedback=None)
        self.assertEqual(full_history_ids(context), {'read-98', 'read-99'})
        view = BasicContextCompactor().compact(context, measure=measure,
                                              summarize=lambda *args: 'Repeated reads archived.')
        self.assertLessEqual(measure(view), 24000)
        self.assertEqual(view.history[-2:], rows[-2:])
        self.assertEqual(view.operation_results[-1], receipt)

    def test_irreducible_failure_reports_minimum_and_field_sizes_without_committing(self):
        context, _, _ = self.repeated(3)
        context = replace(context, objective='fixed scope ' * 10000)
        compactor = BasicContextCompactor()
        before = compactor.export_state()
        with self.assertRaises(ContextBudgetError):
            compactor.compact(context, measure=measure)
        audit = compactor.last_measurement
        self.assertGreater(audit['minimum_input_tokens'], 24000)
        self.assertGreater(audit['minimum_field_bytes']['objective'], 80000)
        self.assertEqual(compactor.export_state(), before)

    def test_summary_allowance_accounts_for_facts_and_provenance(self):
        context, _, _ = self.repeated()
        probe = BasicContextCompactor()
        probe.compact(context, measure=measure, summarize=lambda *args: 'Archive navigation.')
        overhead = 24000 - probe.last_measurement['minimum_input_tokens'] - 2000
        writer = Mock(side_effect=lambda rows, previous, limit: 's' * limit)
        compactor = BasicContextCompactor()
        meter = lambda view: overhead + measure(view)
        view = compactor.compact(context, measure=meter, summarize=writer)
        allowance = writer.call_args.args[2]
        self.assertGreater(allowance, 0)
        self.assertLess(allowance, compactor.summary_tokens)
        self.assertLessEqual(meter(view), 24000)
