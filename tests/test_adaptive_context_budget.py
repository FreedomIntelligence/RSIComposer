"""Bounded adaptive targets; exact facts and call budgets remain authoritative."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rsisys.loops.decision_making.harness.context import BasicContextCompactor, ContextBudgetError
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.snapshot.local import research_configuration, restore_research
from test_context_compaction import context, measure, records


class AdaptiveContextBudgetTests(unittest.TestCase):
    def large(self, fixed=25000):
        return context(objective='s' * fixed, history=records(50))

    def compactor(self):
        return BasicContextCompactor(max_target_tokens=60000, target_increment_tokens=12000)

    def test_protected_minimum_uses_one_increment_and_one_summary_call(self):
        raw = self.large()
        saved = deepcopy(raw)
        with self.assertRaises(ContextBudgetError):
            BasicContextCompactor().compact(raw, measure=measure)
        compactor = self.compactor()
        writer = Mock(return_value='Archived repeated observations; current facts take precedence.')
        result = compactor.compact(raw, measure=measure, summarize=writer)
        self.assertEqual(compactor.target_tokens, 24000)
        self.assertEqual(compactor.last_measurement['target_tokens'], 36000)
        self.assertEqual([a['target_tokens'] for a in compactor.last_measurement['target_attempts']], [24000, 36000])
        self.assertLessEqual(measure(result), 36000)
        self.assertEqual(result.history[-2:], raw.history[-2:])
        self.assertEqual(result.objective, raw.objective)
        self.assertEqual(raw, saved)
        self.assertEqual(compactor.compactions, 1)
        writer.assert_called_once()

    def test_ceiling_failure_is_bounded_atomic_and_does_not_call_summary(self):
        compactor = self.compactor()
        before = compactor.export_state()
        writer = Mock()
        with self.assertRaises(ContextBudgetError):
            compactor.compact(self.large(65000), measure=measure, summarize=writer)
        self.assertEqual(compactor.export_state(), before)
        self.assertEqual([a['target_tokens'] for a in compactor.last_measurement['target_attempts']],
                         [24000, 36000, 48000, 60000])
        writer.assert_not_called()

    def test_provider_or_summary_validation_errors_are_not_retried(self):
        compactor = self.compactor()
        before = compactor.export_state()
        writer = Mock(side_effect=ValueError('Summary model identity mismatch'))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            compactor.compact(self.large(1000), measure=measure, summarize=writer)
        writer.assert_called_once()
        self.assertEqual(compactor.export_state(), before)

    def test_summary_overflow_retries_without_committing_failed_attempt(self):
        compactor = self.compactor()
        before = compactor.export_state()
        calls = []
        def summarize(*args):
            self.assertEqual(compactor.export_state(), before)
            calls.append(args[-1])
            return 's' * 30000 if len(calls) == 1 else 'Bounded summary'
        result = compactor.compact(self.large(1000), measure=measure, summarize=summarize)
        self.assertEqual(len(calls), 2)
        self.assertEqual(compactor.compactions, 1)
        self.assertEqual(compactor.last_measurement['target_tokens'], 36000)
        self.assertLessEqual(measure(result), 36000)

    def test_every_pressure_cycle_retries_base_target_not_a_sticky_larger_budget(self):
        compactor = self.compactor()
        compactor.compact(self.large(), measure=measure, summarize=lambda *a: 'First summary')
        small_fixed = context(history=tuple(replace(r, record_id='new-'+r.record_id) for r in records(60)))
        result = compactor.compact(small_fixed, measure=measure, summarize=lambda *a: 'Second summary')
        self.assertEqual(compactor.last_measurement['target_tokens'], 24000)
        self.assertLessEqual(measure(result), 24000)
        self.assertEqual(compactor.compactions, 2)

    def test_below_trigger_does_not_expand_or_call_summary(self):
        compactor = self.compactor()
        writer = Mock()
        raw = context(objective='s' * 40000)
        self.assertEqual(compactor.compact(raw, measure=measure, summarize=writer), raw)
        writer.assert_not_called()
        self.assertFalse(compactor.last_measurement['compacted'])
        self.assertEqual(compactor.last_measurement['target_tokens'], 24000)

    def test_snapshot_preserves_adaptive_config_and_folded_state(self):
        harness = SimpleNamespace(research_agenda=ResearchAgenda(), context_compactor=self.compactor())
        raw = self.large()
        result = harness.context_compactor.compact(raw, measure=measure, summarize=lambda *a: 'Stored summary')
        config = research_configuration(harness)
        restored = SimpleNamespace(research_agenda=ResearchAgenda())
        restore_research(restored, harness.research_agenda.export(), config)
        self.assertEqual(restored.context_compactor.max_target_tokens, 60000)
        self.assertEqual(restored.context_compactor.target_increment_tokens, 12000)
        self.assertEqual(restored.context_compactor.compact(raw, measure=measure), result)
        legacy = {k: v for k, v in config.items() if k not in ('max_target_tokens', 'target_increment_tokens')}
        restore_research(restored, harness.research_agenda.export(), legacy)
        self.assertEqual(restored.context_compactor.max_target_tokens, 24000)

    def test_invalid_targets_cannot_cross_trigger_or_escape_step_bounds(self):
        for kwargs in ({'max_target_tokens': 80000}, {'max_target_tokens': 20000},
                       {'max_target_tokens': 60001}, {'target_increment_tokens': 0},
                       {'max_target_tokens': True}, {'target_increment_tokens': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BasicContextCompactor(**kwargs)
