"""Archive-backed history must not make the protected floor grow every experiment."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.decision_making.harness.context import BasicContextCompactor, ContextBudgetError, encoded
from rsisys.loops.decision_making.harness.research_state import resident_evidence_indices, research_working_view
from test_research_evidence_projection import fixture, measured, measure, read_original


class ResidentEvidenceWindowTests(unittest.TestCase):
    def test_repeated_pressure_archives_old_negative_evidence_and_restores_consistently(self):
        compactor = BasicContextCompactor(trigger_tokens=30000, target_tokens=19000, summary_tokens=500)
        for n in (24, 48, 96):
            raw, tools, memory, experience = fixture(count=n)
            before = deepcopy(raw)
            view = compactor.compact(raw, measure=measure, summarize=lambda *a: 'Read original measurements; missing is not untested.')
            self.assertLessEqual(measure(view), 19000)
            self.assertLessEqual(len(view.history), 9)
            self.assertNotIn('evidence-0', {h.record_id for h in view.history})
            self.assertEqual(view.history[-2:], raw.history[-2:])
            self.assertEqual(view.last_feedback, raw.last_feedback)
            self.assertEqual(raw, before)
            receipt = tools._archive({'collection': 'history', 'record_id': 'evidence-0',
                                     'evidence_path': '/details/feedback/quality_metrics'})
            original = json.loads(''.join(x['text'] for x in receipt.data['items']))
            self.assertEqual(original['accuracy'], 0.0)
            self.assertEqual(original['delta'], -0.125)
            restored = BasicContextCompactor(trigger_tokens=30000, target_tokens=19000, summary_tokens=500)
            restored.restore_state(json.loads(json.dumps(compactor.export_state())))
            self.assertEqual(restored.compact(raw, measure=measure), view)
        self.assertEqual(compactor.compactions, 3)

    def test_experience_growth_has_a_bounded_resident_floor_and_exact_archive(self):
        c = BasicContextCompactor(trigger_tokens=30000, target_tokens=19000, summary_tokens=500)
        for n in (24, 48, 96):
            raw, tools, memory, experience = fixture(count=n)
            observations = tuple(replace(raw.experience[0], record_id=f'negative-{i}',
                directive_id=f'done-{i}', summary='Measured failure',
                details={'status': 'failed', 'error': 'evaluation rejected',
                         'quality_metrics': {'accuracy': 0.0}, 'prose': 'old qualification ' * 100})
                for i in range(n))
            for observation in observations:
                experience.record(observation)
            raw = replace(raw, experience=observations)
            view = c.compact(raw, measure=measure, summarize=lambda *a: 'Archived failures remain failures.')
            self.assertLessEqual(measure(view), 19000)
            self.assertLessEqual(c.last_measurement['protected_records']['experience'], 4)
            self.assertNotIn('negative-0', {e.record_id for e in view.experience})
            receipt = tools._archive({'collection': 'experience', 'record_id': 'negative-0',
                                     'evidence_path': '/details/quality_metrics'})
            self.assertEqual(json.loads(''.join(x['text'] for x in receipt.data['items'])), {'accuracy': 0.0})

    def test_recent_failures_remain_visible_after_new_successes(self):
        raw, _, _, _ = fixture(count=12)
        failed = replace(raw.history[0], details={'status': 'failed', 'error': 'candidate rejected'})
        raw = replace(raw, history=(failed, *raw.history[1:]))
        indices = resident_evidence_indices(raw, 'history')
        self.assertIn(0, indices)
        self.assertLessEqual(len(indices), 4)
        working = research_working_view(raw, [], [])
        self.assertIn(failed.record_id, working.research_state['protected_history_ids'])
        self.assertNotIn('evidence-1', working.research_state['protected_history_ids'])

    def test_live_old_measurement_is_not_archived_to_make_budget_fit(self):
        raw, _, _, _ = fixture(count=40)
        huge = replace(raw.history[0], details={'quality_metrics': {str(i): 0 for i in range(6000)}})
        raw = replace(raw, history=(huge, *raw.history[1:]),
                      outstanding=(Directive('directive-0', 'objective', 'Still used'),))
        c = BasicContextCompactor(trigger_tokens=30000, target_tokens=19000, summary_tokens=500)
        before = c.export_state()
        with self.assertRaises(ContextBudgetError):
            c.compact(raw, measure=measure)
        self.assertEqual(c.export_state(), before)

    def test_missing_archive_keeps_every_measured_receipt(self):
        raw, _, _, _ = fixture(count=24)
        raw = replace(raw, operations=())
        self.assertEqual(len(resident_evidence_indices(raw, 'history')), 26)
        with self.assertRaises(ContextBudgetError):
            BasicContextCompactor().compact(raw, measure=measure)

    def test_unrecorded_tool_metrics_cannot_be_folded_with_only_advertised_archive(self):
        raw, _, _, _ = fixture(count=4)
        results = tuple(replace(raw.operation_results[0], data={'quality_metrics': {'accuracy': i}})
                        for i in range(8))
        raw = replace(raw, operation_results=results)
        self.assertEqual(resident_evidence_indices(raw, 'operation_results'), set(range(8)))
