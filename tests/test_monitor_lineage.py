"""The monitor exports causal evidence, not a guessed chronological tree."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json

from examples.qwen_toy.monitor import Monitor, decision_summary, record_lineage, enrich_completed_run, step_operation_receipts
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.coordination import ExecutionView
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.runtime.runner import RunnerStep
from web.server import project_snapshot


class MonitorLineageTests(unittest.TestCase):
    def test_explicit_empty_dependencies_are_distinct_from_legacy(self):
        old = decision_summary({'kind': 'finish'}, ident='old', round_number=1)
        self.assertNotIn('depends_on_decision_ids', old)
        decision = Decision('finish')
        row = decision_summary(asdict(decision), ident='fallback', round_number=1)
        self.assertEqual(row['decision_id'], decision.decision_id)
        self.assertEqual(row['depends_on_decision_ids'], [])
        public = project_snapshot({'schema_version': 1, 'decisions': [old, row]}, 'run')['decisions']
        self.assertNotIn('depends_on_decision_ids', public[0])
        self.assertEqual(public[1]['depends_on_decision_ids'], [])

    def test_two_parallel_receipts_then_one_confirmed_adoption(self):
        with TemporaryDirectory() as tmp:
            model = ModelRef('test', 'model', 'v1')
            monitor = Monitor(Path(tmp), model, 'goal')
            context = DecisionContext('goal', (), (), (), execution=ExecutionView(model), decision_model=model)
            producers = [Decision('explore', directive=Directive(objective='goal', strategy='strategy', directive_id=f'd{i}')) for i in range(2)]
            for d in producers:
                monitor.decision(d, context, 1)
            for i in (1, 0):
                monitor.advance(RunnerStep('completed', {'feedback': {
                    'directive_id': f'd{i}', 'status': 'completed',
                    'candidates': [{'candidate_id': f'c{i}'}],
                }}, directive_id=f'd{i}'))
            join = Decision('invoke', operation=OperationCall('system_adopt', {}),
                            depends_on_decision_ids=tuple(d.decision_id for d in producers))
            monitor.decision(join, context, 1)
            receipt = {'status': 'completed', 'summary': 'Activated both components',
                       'adopted_candidate_ids': ['c0', 'c1']}
            monitor.advance(RunnerStep('progressed', {'decision': {
                'decision_id': join.decision_id, 'operation_result': receipt}}))
            rows = project_snapshot(monitor.data, 'run')['decisions']
            self.assertEqual(rows[-1]['adopted_decision_ids'], [d.decision_id for d in producers])
            self.assertEqual(rows[-1]['depends_on_decision_ids'], rows[-1]['adopted_decision_ids'])
            self.assertTrue(all(row['depends_on_decision_ids'] == [] for row in rows[:2]))

    def test_policy_retry_keeps_one_node_and_its_observed_feedback(self):
        with TemporaryDirectory() as tmp:
            model = ModelRef('test', 'model', 'v1')
            monitor = Monitor(Path(tmp), model, 'goal')
            context = DecisionContext('goal', (), (), (), execution=ExecutionView(model))
            decision = Decision('invoke', operation=OperationCall('query', {}))
            monitor.decision(decision, context, 1)
            monitor.advance(RunnerStep('progressed', {'decision_id': decision.decision_id,
                'operation_result': {'status': 'completed', 'summary': 'Observed result'}}))
            before = dict(monitor.data['decisions'][0])
            monitor.decision(decision, context, 1)
            self.assertEqual(monitor.data['decisions'], [before])

    def test_export_correlates_nonadjacent_operation_and_preserves_confirmed_join(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor = Monitor(root, ModelRef('test', 'model', 'v1'), 'goal')
            producers = [Decision('explore', directive=Directive(f'd{i}', 'goal', 'strategy')) for i in range(2)]
            join = Decision('invoke', operation=OperationCall('system_adopt', {}),
                            depends_on_decision_ids=tuple(d.decision_id for d in producers))
            history = []
            for d in producers:
                history.append({'record_id': d.decision_id, 'kind': 'decision', 'details': asdict(d)})
            for i in (1, 0):
                history.append({'record_id': f'f{i}', 'kind': 'feedback', 'details': {'feedback': {
                    'directive_id': f'd{i}', 'status': 'completed', 'candidates': [{'candidate_id': f'c{i}'}]}}})
            history.append({'record_id': join.decision_id, 'kind': 'decision', 'details': asdict(join)})
            history.append({'record_id': 'interleaved', 'kind': 'memory', 'details': {}})
            history.append({'record_id': 'receipt', 'kind': 'operation', 'details': {
                'decision_id': join.decision_id, 'operation': asdict(join.operation),
                'result': {'status': 'completed', 'adopted_candidate_ids': ['c0', 'c1']}}})
            (root / 'decision_memory.json').write_text(json.dumps({'history': history}))
            (root / 'result.json').write_text(json.dumps({'status': 'completed'}))
            for _ in range(2):
                exported = enrich_completed_run(root)
                row = exported['decisions'][-1]
                self.assertEqual(row['decision_id'], join.decision_id)
                self.assertEqual(row['adopted_decision_ids'], [d.decision_id for d in producers])
                self.assertEqual(row['status'], 'completed')

    def test_rejected_dependency_is_not_exported_as_completed_finish(self):
        row = decision_summary({'kind': 'finish', 'decision_id': 'bad',
            'depends_on_decision_ids': ['pending'], 'validation_error': 'Dependency is not complete'},
            ident='unused', round_number=1)
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['error'], 'Dependency is not complete')

    def test_explicit_step_ids_override_receipt_order(self):
        history = [{'record_id': ident, 'kind': 'decision', 'details': {
            'kind': 'invoke', 'decision_id': ident}} for ident in ('a', 'b')]
        steps = [{'event': {'decision_id': ident, 'operation_result': {
            'status': 'completed', 'summary': ident}}} for ident in ('b', 'a')]
        receipts = step_operation_receipts(history, steps)
        self.assertEqual({key: value['summary'] for key, value in receipts.items()}, {'a': 'a', 'b': 'b'})

    def test_step_only_producer_and_old_presentation_alias_stay_connected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor = Monitor(root, ModelRef('test', 'model', 'v1'), 'goal')
            producer = Decision('explore', directive=Directive('d', 'goal', 'strategy'))
            join = Decision('invoke', operation=OperationCall('system_adopt', {}),
                            depends_on_decision_ids=(producer.decision_id,))
            rows = [decision_summary(asdict(d), ident=d.decision_id, round_number=1) for d in (producer, join)]
            rows[0]['decision_id'] = 'old-live-id'
            monitor.data['decisions'] = rows
            monitor.write()
            history = [{'record_id': d.decision_id, 'kind': 'decision', 'details': asdict(d)} for d in (producer, join)]
            history.append({'record_id': 'receipt', 'kind': 'operation', 'details': {
                'decision_id': join.decision_id, 'result': {'status': 'completed', 'adopted_candidate_ids': ['c']}}})
            step = {'directive_id': 'd', 'event': {'feedback': {
                'directive_id': 'd', 'status': 'completed', 'candidates': [{'candidate_id': 'c'}]}}}
            (root / 'decision_memory.json').write_text(json.dumps({'history': history}))
            (root / 'result.json').write_text(json.dumps({'status': 'completed'}))
            (root / 'steps.jsonl').write_text(json.dumps(step) + '\n')
            exported = enrich_completed_run(root)['decisions']
            self.assertEqual(exported[0]['decision_id'], 'old-live-id')
            self.assertEqual(exported[1]['depends_on_decision_ids'], ['old-live-id'])
            self.assertEqual(exported[1]['adopted_decision_ids'], ['old-live-id'])

    def test_failures_unknown_sources_and_evaluation_do_not_invent_adoption(self):
        origins = {'c1': 'producer'}
        evaluation = {'decision_id': 'evaluation', 'mode': 'evaluate'}
        record_lineage(evaluation, {'status': 'completed', 'candidates': [{'candidate_id': 'c1'}]}, origins)
        self.assertEqual(origins, {'c1': 'producer'})
        for receipt in ({'status': 'failed', 'adopted_candidate_ids': ['c1']},
                        {'status': 'completed', 'data': {'adopted_candidate_ids': ['c1']}}):
            row = {'decision_id': 'join'}
            record_lineage(row, receipt, origins)
            self.assertNotIn('adopted_decision_ids', row)
        row = {'decision_id': 'join'}
        record_lineage(row, {'status': 'completed', 'adopted_candidate_ids': ['c1', 'unknown']}, origins)
        self.assertEqual(row['adopted_candidate_ids'], ['c1', 'unknown'])
        self.assertEqual(row['adopted_decision_ids'], ['producer'])


if __name__ == '__main__':
    unittest.main()
