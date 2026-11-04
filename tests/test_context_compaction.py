"""Token pressure, complete recent feedback and source-qualified rolling state."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock
from rsisys.loops._memory import HistoryEntry, Memory
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.loops.decision_making.harness.context import BasicContextCompactor, IdentityContextCompactor, ContextBudgetError, encoded
from rsisys.loops.decision_making.harness.state import DecisionContext


def context(**kwargs):
    return replace(DecisionContext('Investigate', (), (), ()), **kwargs)


def measure(view):
    # Deterministic injected meter, not presented as a model tokenizer.
    return len(encoded(asdict(view)))


def records(n=12):
    return tuple(HistoryEntry(str(i), 'operation', None, f'Observation {i}', False,
        details={'data': 'x' * 1800}) for i in range(n))


class ContextCompactionTests(unittest.TestCase):
    def compactor(self):
        return BasicContextCompactor(trigger_tokens=22000, target_tokens=8500, summary_tokens=1000)

    def test_below_threshold_keeps_all_history_and_12000_character_tool_feedback(self):
        result = OperationResult('Visible evidence excerpt', (), False, data={'content': 'x'*12000})
        raw = context(operation_results=(result,), history=records(3))
        result = BasicContextCompactor().compact(raw, measure=measure)
        self.assertEqual(result, raw)
        result.operation_results[-1].data['content'] = 'mutated'
        self.assertEqual(len(raw.operation_results[-1].data['content']), 12000)

    def test_pressure_folds_old_records_and_protects_latest_complete_receipt(self):
        history = records()
        latest = OperationResult('Full latest feedback', (), False, data={'content': 'important'*150})
        raw = context(history=history, operation_results=(latest,))
        saved = deepcopy(raw)
        writer = Mock(return_value='Old observations are archived; retain uncertainty and evidence.')
        compactor = self.compactor()
        projected = compactor.compact(raw, measure=measure, summarize=writer)
        self.assertLessEqual(measure(projected), compactor.target_tokens)
        self.assertEqual(projected.history[-2:], history[-2:])
        self.assertEqual(projected.operation_results[-1], latest)
        self.assertEqual(raw, saved)
        self.assertEqual(compactor.compactions, 1)
        source = projected.context_summary['sources']['history']
        self.assertEqual(source['first_omitted'], 'history:0')
        self.assertEqual(source['read_path']['operation'], 'research_archive')
        writer.assert_called_once()
        self.assertEqual(compactor.last_measurement['output_tokens'], measure(projected))

    def test_source_manifest_is_priced_before_failing_or_calling_summarizer(self):
        raw = context(history=records(), memories=tuple(Memory(str(i), 'episodic', 'x'*200)
                                                       for i in range(40)))
        saved = deepcopy(raw)
        writer = Mock(return_value='Read the original sources before reusing a conclusion.')
        compactor = BasicContextCompactor(trigger_tokens=22000, target_tokens=8500, summary_tokens=100)
        projected = compactor.compact(raw, measure=measure, summarize=writer)
        self.assertLessEqual(measure(projected), 8500)
        self.assertEqual(projected.history[-2:], raw.history[-2:])
        self.assertEqual(set(projected.context_summary['sources']), {'history', 'memories'})
        writer.assert_called_once()
        self.assertEqual(raw, saved)

    def test_after_compaction_new_turns_grow_without_reattaching_old_records(self):
        raw = context(history=records())
        compactor = self.compactor()
        first = compactor.compact(raw, measure=measure, summarize=lambda *args: 'Evidence summary')
        fresh = HistoryEntry('next', 'operation', None, 'New action', False)
        second = compactor.compact(replace(raw, history=(*raw.history, fresh)), measure=measure,
            summarize=lambda *args: self.fail('Should not compact again below threshold'))
        self.assertEqual(second.history[-1], fresh)
        self.assertEqual(second.history[:-1], first.history)
        self.assertEqual(compactor.compactions, 1)
        self.assertEqual(compactor.compact(second, measure=measure), second)

    def test_second_pressure_cycle_summarizes_prior_summary_and_new_records(self):
        raw = context(history=records())
        compactor = self.compactor()
        compactor.compact(raw, measure=measure, summarize=lambda *args: 'First archived findings')
        later = tuple(replace(row, record_id='next-'+row.record_id) for row in records())
        writer = Mock(return_value='First and second observations remain qualified by original receipts')
        result = compactor.compact(replace(raw, history=(*raw.history, *later)), measure=measure, summarize=writer)
        self.assertEqual(compactor.compactions, 2)
        self.assertEqual(writer.call_args.args[1]['text'], 'First archived findings')
        self.assertEqual(result.history[-2:], later[-2:])
        self.assertLessEqual(measure(result), compactor.target_tokens)
        self.assertEqual(raw.history, records())

    def test_lightweight_refs_survive_folding_so_input_ids_stay_available(self):
        ref = ArtifactRef('data', 'training_data', 'artifact://data')
        old = OperationResult('Prepared data', (ref,), False, data={'text': 'x'*22000})
        new = OperationResult('Latest observation', (), False, data={'text': 'recent'})
        raw = context(operation_results=(old, new))
        projected = self.compactor().compact(raw, measure=measure, summarize=lambda *args: 'Prepared data is archived')
        self.assertEqual(projected.operation_results[-1], new)
        self.assertIn(ref, tuple(r for row in projected.operation_results for r in row.artifacts))

    def test_protected_oversize_fails_without_discarding_or_committing_state(self):
        raw = context(operation_results=(OperationResult('huge', (), False, data={'full': 'x'*30000}),))
        compactor = self.compactor()
        before = compactor.export_state()
        with self.assertRaisesRegex(ContextBudgetError, 'complete feedback'):
            compactor.compact(raw, measure=measure)
        self.assertEqual(compactor.export_state(), before)
        self.assertEqual(len(raw.operation_results[0].data['full']), 30000)

    def test_summary_failure_or_oversize_does_not_commit(self):
        for writer in (Mock(side_effect=RuntimeError('summary failed')), lambda *args: 'x'*20000):
            compactor = self.compactor()
            before = compactor.export_state()
            with self.assertRaises((RuntimeError, ContextBudgetError)):
                compactor.compact(context(history=records()), measure=measure, summarize=writer)
            self.assertEqual(compactor.export_state(), before)

    def test_portable_rolling_state_roundtrip(self):
        raw = context(history=records())
        compactor = self.compactor()
        projected = compactor.compact(raw, measure=measure, summarize=lambda *args: 'Retain evidence')
        restored = self.compactor()
        restored.restore_state(json.loads(json.dumps(compactor.export_state())))
        self.assertEqual(restored.compact(raw, measure=measure), projected)
        with self.assertRaises(ValueError): restored.restore_state({'invalid': True})

    def test_final_request_meter_includes_fixed_state_not_just_history(self):
        raw = context(history=records(2), objective='Long fixed instructions '+'x'*21000)
        with self.assertRaises(ContextBudgetError):
            self.compactor().compact(raw, measure=measure)

    def test_model_policy_counts_final_request_and_accounts_summary_calls(self):
        from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
        from rsisys.loops.contracts.model_service import ModelRef, GenerationResponse, TokenUsage
        from rsisys.loops.contracts.feedback import Feedback
        from rsisys.loops.contracts.lifecycle import ExecutionStatus
        model = ModelRef('fixture', 'fixture', 'r0')
        class Backend:
            token_count_mode = 'fixture_counter'
            def __init__(self): self.requests = []; self.priced = []
            def count_tokens(self, request):
                self.priced.append(request)
                return len(request.prompt) + len(request.system)
            def generate(self, request):
                self.requests.append(request)
                text = ('{"summary":"Archived observations retain original evidence and uncertainty."}'
                        if request.system.startswith('Summarize') else '{"kind":"finish","reason":"fixture"}')
                return GenerationResponse(text, model, True, TokenUsage(100, 10))
        backend = Backend()
        compactor = BasicContextCompactor(trigger_tokens=25000, target_tokens=16000, summary_tokens=1000)
        policy = ModelDecisionPolicy(backend, model, context_compactor=compactor)
        latest = OperationResult('Complete feedback', (), False, data={'content':'latest-body'*100})
        observed = []
        feedback = Feedback('apply', ExecutionStatus.COMPLETED, 'Candidate was applied', False,
                            adopted_candidate_ids=('measured-candidate',))
        with policy.track_usage(observed.append):
            decision = policy.decide(context(history=records(18), operation_results=(latest,),
                                             last_feedback=feedback))
        self.assertEqual(decision.kind, 'finish')  # Voluntary stopping remains available.
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(len(observed), 2)
        summary_request = backend.requests[0]
        summary_inputs = json.loads(summary_request.prompt)
        self.assertEqual(summary_inputs['current_state']['last_feedback']['adopted_candidate_ids'],
                         ['measured-candidate'])
        self.assertNotIn('summary', summary_inputs['current_state']['last_feedback'])
        self.assertIn('not the complete experiment history', summary_request.system)
        final = backend.requests[-1]
        self.assertLessEqual(len(final.prompt)+len(final.system), compactor.target_tokens)
        self.assertIn('latest-body'*100, final.prompt)
        self.assertEqual(compactor.last_measurement['count_method'], 'fixture_counter')
        self.assertTrue(all('directives' in json.loads(r.prompt) for r in backend.priced))

    def test_configuration_and_counter_contract(self):
        for params in ({'trigger_tokens':0},{'target_tokens':True}, {'trigger_tokens':20, 'target_tokens':30},
                       {'target_tokens':100, 'summary_tokens':100}):
            with self.subTest(params=params), self.assertRaises(ValueError): BasicContextCompactor(**params)
        with self.assertRaises(ValueError): BasicContextCompactor().compact(context(), measure=lambda c: True)
        self.assertEqual(IdentityContextCompactor().compact(context()), context())


if __name__ == '__main__': unittest.main()
