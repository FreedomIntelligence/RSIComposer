"""Monitor projections use explicit decisions and observed progress, not raw prompts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.qwen_toy.monitor import Monitor, decision_summary, enrich_completed_run
from rsisys.loops.contracts.coordination import ExecutionView, ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import TrainSpec
from rsisys.loops.contracts.requests import PromptSpec
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.runtime.runner import RunnerStep


class MonitorProducerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = ModelRef('toy', 'model', 'v1')
        self.context = DecisionContext('PRIVATE_MODEL_PROMPT', (), (), (),
                                       execution=ExecutionView(self.model), decision_model=self.model)
        self.directive = Directive('d1', 'PRIVATE_MODEL_PROMPT', 'Improve formatting with SFT', mode='train',
                                   payload=TrainSpec('recipe', 'sft'), review_after_seconds=2)
        self.monitor = Monitor(self.root, self.model, 'Public run goal', rounds=2)

    def test_validated_decision_has_public_intent_and_version_but_no_prompt(self):
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 1)
        data = json.loads((self.root / 'monitor.json').read_text())
        row = data['decisions'][0]
        self.assertEqual((row['mode'], row['strategy'], row['recipe_ref'], row['method']),
                         ('train', 'Improve formatting with SFT', 'recipe', 'sft'))
        self.assertEqual((row['decision_revision'], row['actor_revision']), ('v1', 'v1'))
        self.assertNotIn('PRIVATE_MODEL_PROMPT', json.dumps(data))
        self.assertNotIn('payload', row)
        self.assertNotIn('step', data['current'])

    def test_job_progress_and_terminal_feedback_are_distinct(self):
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 1)
        self.monitor.event('job', job_id='j1', kind='train', status='running',
                           step=4, total_steps=24, elapsed_seconds=2, metrics={'loss': 0.03})
        self.assertEqual(self.monitor.data['current']['step'], 4)
        self.assertEqual(self.monitor.jobs['j1']['directive_id'], 'd1')
        self.monitor.event('job', job_id='j1', kind='train', status='succeeded', step=24, total_steps=24)
        self.assertEqual(self.monitor.data['current']['phase'], 'collecting_result')
        self.assertEqual(self.monitor.data['workflow']['completed_directives'], 0)
        self.monitor.advance(RunnerStep('completed', {'feedback': {'status': 'completed',
            'training_metrics': {'train_loss': .01}, 'jobs': [{'job_id': 'j1'}]}}, directive_id='d1'))
        self.assertEqual(self.monitor.data['decisions'][0]['status'], 'completed')
        self.assertEqual(self.monitor.data['workflow']['completed_directives'], 1)
        self.assertEqual(self.monitor.data['current']['metrics'], {'train_loss': .01})
        self.assertEqual(self.monitor.data['decisions'][0]['metrics'], {'train_loss': .01})

    def test_second_round_job_uses_its_decision_round_instead_of_legacy_workflow_counter(self):
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 1)
        self.monitor.event('job', job_id='first', kind='train', status='succeeded', step=32)
        second = replace(self.directive, directive_id='d2')
        self.monitor.decision(Decision('explore', directive=second), self.context, 2)
        self.monitor.event('job', job_id='second', kind='train', status='running', step=5)
        snapshot = json.loads((self.root / 'monitor.json').read_text())
        jobs = {row['job_id']: row for row in snapshot['jobs']}
        self.assertEqual((jobs['first']['directive_id'], jobs['first']['round']), ('d1', 1))
        self.assertEqual((jobs['second']['directive_id'], jobs['second']['round']), ('d2', 2))
        self.assertEqual(snapshot['current']['round'], 2)

    def test_late_first_job_progress_does_not_move_it_to_current_second_directive(self):
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 1)
        self.monitor.event('job', job_id='first', kind='train', status='running', step=1)
        self.monitor.decision(Decision('explore', directive=replace(self.directive, directive_id='d2')),
                              self.context, 2)
        self.monitor.event('job', job_id='second', kind='train', status='running', step=2)
        self.monitor.event('job', job_id='first', kind='train', status='succeeded', step=32)
        self.assertEqual((self.monitor.jobs['first']['directive_id'], self.monitor.jobs['first']['round']),
                         ('d1', 1))
        self.assertEqual(self.monitor.data['current']['job_id'], 'second')
        self.assertEqual(self.monitor.data['current']['step'], 2)

    def test_explicit_other_job_identity_and_round_do_not_attach_to_current_directive(self):
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 2)
        self.monitor.event('job', job_id='other', kind='train', status='running',
                           directive_id='foreign', round=7, step=11)
        self.assertEqual((self.monitor.jobs['other']['directive_id'], self.monitor.jobs['other']['round']),
                         ('foreign', 7))
        self.assertNotIn('job_id', self.monitor.data['current'])
        self.assertNotIn('step', self.monitor.data['current'])

    def test_feedback_metrics_merge_public_numeric_groups_and_project_with_operation(self):
        from rsisys.monitor import project_snapshot
        self.monitor.decision(Decision('invoke', operation=OperationCall('query_experiments', {})), self.context, 0)
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 1)
        self.monitor.advance(RunnerStep('completed', {'feedback': {'status': 'completed',
            'training_metrics': {'train_loss': .1, 'prompt_tokens': 123, 'nan': float('nan')},
            'quality_metrics': {'accuracy': .9, 'eval_samples': 20},
            'execution_metrics': {'duration': 3, 'secret': 321, 'config': {'loss': 1}},
            'jobs': [{'job_id': 'j1'}]}}, directive_id='d1'))
        public = project_snapshot(json.loads((self.root / 'monitor.json').read_text()), 'run')
        self.assertEqual(public['decisions'][0]['operation'], 'query_experiments')
        self.assertEqual(public['decisions'][1]['metrics'], {
            'train_loss': .1, 'accuracy': .9, 'eval_samples': 20, 'duration': 3})
        self.assertNotIn('prompt_tokens', json.dumps(public, allow_nan=False))

    def test_job_history_windows_merge_without_poll_duplicates_or_cross_job_leaks(self):
        points = [{'step': n, 'elapsed_seconds': n, 'metrics': {'loss': 1 / n}} for n in range(1, 1006)]
        self.monitor.event('job', job_id='j1', kind='train', status='running', metric_history=points[:900])
        self.monitor.event('job', job_id='j1', kind='train', status='running', metric_history=points[500:])
        self.monitor.event('job', job_id='j1', kind='train', status='running', metric_history=points[500:])
        self.monitor.event('job', job_id='j1', kind='train', status='cancel_requested')
        self.monitor.event('job', job_id='j2', kind='train', status='running', metric_history=points[:1])
        self.assertEqual(self.monitor.jobs['j1']['metric_history'], points[-1000:])
        self.assertEqual(self.monitor.jobs['j2']['metric_history'], points[:1])
        self.assertNotIn('metric_history', self.monitor.data['current'])

    def test_review_is_a_proposal_until_runner_applies_it(self):
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 1)
        progress = ExplorationProgress('d1', 'running', self.directive)
        context = DecisionContext('goal', (), (), (), execution=ExecutionView(self.model, progress),
                                  decision_model=self.model)
        review = ReviewDecision('cancel', reason='Observed error')
        self.monitor.review(review, context, 1)
        self.assertEqual(self.monitor.data['decisions'][-1]['status'], 'proposed')
        self.monitor.advance(RunnerStep('waiting', {'review': asdict(review), 'progress': asdict(progress)},
                                       directive_id='d1'))
        self.assertEqual(self.monitor.data['decisions'][-1]['status'], 'cancel_requested')
        self.assertEqual(self.monitor.data['current']['status'], 'running')
        self.assertEqual(self.monitor.data['decisions'][-1]['reason'], 'Observed error')

    def test_activation_readiness_does_not_claim_a_binding_change(self):
        self.monitor.event('activation_verified', revision='v2')
        self.assertEqual(self.monitor.data['models']['actor']['revision'], 'v1')

    def test_tool_observation_completes_operation_without_claiming_actor_execution(self):
        decision = Decision('invoke', operation=OperationCall('workspace_read',
            {'root': 'run', 'path': 'PRIVATE_PATH'}))
        self.monitor.decision(decision, self.context, 0)
        self.assertEqual(self.monitor.data['decisions'][0]['status'], 'proposed')
        self.monitor.advance(RunnerStep('progressed', {'kind': 'progressed',
            'operation_result': {'status': 'completed', 'data': {'text': 'PRIVATE_FILE_CONTENT'}}}))
        self.monitor.decision(Decision('finish', reason='Enough evidence'), self.context, 0)
        self.monitor.stage('finished', status='completed')
        snapshot = json.loads((self.root / 'monitor.json').read_text())
        self.assertEqual(snapshot['decisions'][0]['operation'], 'workspace_read')
        self.assertEqual(snapshot['decisions'][0]['status'], 'completed')
        self.assertEqual(snapshot['decisions'][1]['kind'], 'finish')
        self.assertEqual(snapshot['workflow']['completed_directives'], 0)
        self.assertEqual(snapshot['current'], {})
        for private in ('PRIVATE_MODEL_PROMPT', 'PRIVATE_PATH', 'PRIVATE_FILE_CONTENT'):
            self.assertNotIn(private, json.dumps(snapshot))

    def test_finish_and_measurement_delta_do_not_infer_candidate_rejection(self):
        self.monitor.candidates['v2'] = {'candidate_id': 'v2', 'status': 'proposed',
                                        'metrics': {'delta_nll': -0.5}}
        self.monitor.decision(Decision('finish', reason='Keep current model'), self.context, 1)
        self.monitor.stage('finished', status='completed')
        snapshot = json.loads((self.root / 'monitor.json').read_text())
        self.assertEqual(snapshot['candidates'][0]['status'], 'proposed')
        self.assertEqual(snapshot['models']['actor']['revision'], 'v1')
        self.assertEqual(snapshot['workflow']['completed_directives'], 0)

    def test_completed_run_export_uses_recorded_evidence_and_keeps_observation_time(self):
        before = json.loads((self.root / 'monitor.json').read_text())['updated_at']
        details = {**asdict(Decision('explore', directive=self.directive)),
                   'decision_model': asdict(self.model), 'execution_model': asdict(self.model)}
        history = [dict(record_id='record1', kind='decision', details=details),
                   dict(record_id='record2', kind='feedback', details={'feedback': {
                       'directive_id': 'd1', 'status': 'completed', 'jobs': [{'job_id': 'j1'}],
                       'training_metrics': {'train_loss': 0.25, 'api_key': 123}}})]
        (self.root / 'decision_memory.json').write_text(json.dumps({'history': history}))
        (self.root / 'result.json').write_text(json.dumps({'status': 'completed', 'phase': 'finished',
            'rounds_completed': 1, 'rounds_requested': 1, 'completed_directives': 1, 'duration_seconds': 3}))
        snapshot = enrich_completed_run(self.root)
        self.assertEqual(snapshot['decisions'][0]['source'], 'history')
        self.assertEqual(snapshot['decisions'][0]['job_id'], 'j1')
        self.assertEqual(snapshot['decisions'][0]['metrics'], {'train_loss': 0.25})
        self.assertEqual(snapshot['current'], {})  # Offline enrichment changes decisions only.
        self.assertEqual(snapshot['updated_at'], before)
        self.assertIsNone(snapshot['decisions'][0]['timestamp'])
        self.assertNotIn('PRIVATE_MODEL_PROMPT', json.dumps(snapshot))
        self.assertEqual(enrich_completed_run(self.root), snapshot)

    def test_export_refuses_live_run(self):
        (self.root / 'result.json').write_text('{"status":"running"}')
        with self.assertRaisesRegex(ValueError, 'finished'):
            enrich_completed_run(self.root)

    def test_reasons_are_explicit_and_plans_are_bounded_whitelisted_strings(self):
        details = {'kind': 'explore', 'reason': '模型给出的理由' * 1000,
                   'directive': {'directive_id': 'prompt', 'mode': 'configure_prompt',
                                 'payload': {'instructions': 'Actor instructions' * 1000,
                                             'request_prompt': 'PRIVATE_REQUEST',
                                             'reasoning_content': 'PRIVATE_REASONING'}}}
        row = decision_summary(details, ident='id', round_number=0)
        self.assertEqual(row['reason'], details['reason'][:4000])
        self.assertEqual(row['changes'], [{'field': 'instructions',
                                           'after': details['directive']['payload']['instructions'][:4000]}])
        self.assertEqual((row['feedback_summary'], row['error'], row['status']), ('', '', 'proposed'))
        details.pop('reason')
        self.assertEqual(decision_summary(details, ident='id', round_number=0)['reason'], '')
        self.assertNotIn('PRIVATE_', json.dumps(row))

    def test_evaluation_identifies_candidate_components_without_artifacts(self):
        row = decision_summary({'kind': 'explore', 'reason': 'Compare candidate against base',
            'directive': {'directive_id': 'evaluation', 'mode': 'evaluate', 'payload': {
                'protocol_ref': 'dev-accuracy', 'candidate': {'candidate_id': 'c1', 'base_version': 'initial',
                    'changes': {'model': {'uri': 'PRIVATE_WEIGHTS'},
                                'instructions': {'text': 'PRIVATE_ARTIFACT'}}}}}}, ident='id', round_number=0)
        self.assertEqual({c['field']: c['after'] for c in row['changes']}, {
            'protocol_ref': 'dev-accuracy', 'candidate_id': 'c1',
            'base_version': 'initial', 'components': 'instructions, model'})
        self.assertNotIn('PRIVATE_', json.dumps(row))
        self.assertEqual(row['status'], 'proposed')

    def test_training_apply_and_safe_recipe_fields_are_explicit(self):
        for mode in ('train', 'sft', 'rl', 'apply_candidate'):
            with self.subTest(mode=mode):
                payload = {'recipe_ref': 'r1', 'method': 'sft', 'apply_result': False,
                           'candidate_id': 'c1', 'origin': {'samples': 'PRIVATE_TRAINING'}}
                row = decision_summary({'kind': 'explore', 'directive': {
                    'directive_id': 'id', 'mode': mode, 'payload': payload}}, ident='id', round_number=0)
                expected = {'candidate_id': 'c1'} if mode == 'apply_candidate' else {
                    'recipe_ref': 'r1', 'method': 'sft', 'apply_result': 'false'}
                self.assertEqual({c['field']: c['after'] for c in row['changes']}, expected)
        row = decision_summary({'kind': 'invoke', 'operation': {'name': 'define_recipe', 'arguments': {
            'steps': 32, 'lr': 0.0001, 'seed': 0, 'weight_decay': .01, 'warmup_ratio': .1,
            'data': 'PRIVATE_DATA', 'prompt': 'PRIVATE_PROMPT'}}}, ident='id', round_number=0)
        self.assertEqual([c['field'] for c in row['changes']],
                         ['steps', 'lr', 'seed', 'weight_decay', 'warmup_ratio'])
        self.assertNotIn('PRIVATE_', json.dumps(row))

    def test_completed_prompt_proposals_never_supply_an_inferred_before(self):
        for ident, instructions, status in [('p1', 'First proposal', 'completed'),
                                            ('p2', 'Second proposal', 'failed'),
                                            ('p3', '', 'cancelled'), ('p4', 'Last proposal', 'completed')]:
            self.monitor.decision(Decision('explore', directive=Directive(
                ident, 'PRIVATE_MODEL_PROMPT', 'Try prompt', mode='configure_prompt',
                payload=PromptSpec(instructions))), self.context, 0)
            feedback = {'directive_id': ident, 'status': status,
                        'summary': 'Prompt proposed; evaluate before application'}
            if status != 'completed':
                feedback['error'] = 'Not applied'
            self.monitor.advance(RunnerStep('finished', {'feedback': feedback}))
            row = self.monitor.data['decisions'][-1]
            self.assertEqual(row['changes'], [{'field': 'instructions', 'after': instructions}])
            self.assertEqual(row['feedback_summary'], feedback['summary'])
            self.assertEqual(row['error'], feedback.get('error', ''))
        self.assertEqual(self.monitor.data['models']['actor']['revision'], 'v1')

    def test_live_receipts_require_evidence_and_a_unique_target(self):
        call = OperationCall('unknown_operation', {'data': 'PRIVATE_ARGUMENT'})
        self.monitor.decision(Decision('invoke', operation=call, reason='Read evidence'), self.context, 0)
        self.monitor.advance(RunnerStep('progressed', {'kind': 'progressed'}))
        first = self.monitor.data['decisions'][0]
        self.assertEqual(first['status'], 'proposed')
        self.monitor.decision(Decision('invoke', operation=call), self.context, 0)
        receipt = {'status': 'failed', 'summary': 'Query failed', 'error': 'No such resource',
                   'data': {'text': 'PRIVATE_RESULT'}}
        self.monitor.advance(RunnerStep('progressed', {'operation_result': receipt}))
        self.assertEqual([r['status'] for r in self.monitor.data['decisions']], ['proposed', 'proposed'])
        self.monitor.advance(RunnerStep('progressed', {'decision_id': first['decision_id'],
                                                     'operation_result': receipt}))
        self.assertEqual((first['status'], first['feedback_summary'], first['error']),
                         ('failed', 'Query failed', 'No such resource'))
        self.assertEqual(first['changes'], [])
        self.assertNotIn('PRIVATE_', json.dumps(self.monitor.data))

    def test_live_feedback_uses_directive_id_even_when_current_is_another_directive(self):
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 0)
        other = Directive('d2', 'private', 'Second training', mode='train', payload=TrainSpec('r2', 'sft'))
        self.monitor.decision(Decision('explore', directive=other), self.context, 0)
        feedback = {'directive_id': 'd1', 'status': 'failed', 'summary': 'Training failed', 'error': 'Out of memory'}
        self.monitor.advance(RunnerStep('finished', {'feedback': feedback}))
        self.monitor.advance(RunnerStep('finished', {'feedback': feedback}))
        self.assertEqual(self.monitor.data['decisions'][0]['error'], 'Out of memory')
        self.assertEqual(self.monitor.data['decisions'][1]['status'], 'proposed')
        self.assertEqual(self.monitor.data['current']['directive_id'], 'd2')
        self.assertEqual(self.monitor.data['workflow']['completed_directives'], 1)
        self.monitor.advance(RunnerStep('completed', {'feedback': feedback}, directive_id='d2'))
        self.assertEqual(self.monitor.data['decisions'][1]['status'], 'proposed')

    def export_fixture(self, history, steps=(), result=None):
        (self.root / 'decision_memory.json').write_text(json.dumps({'history': history}))
        (self.root / 'result.json').write_text(json.dumps(result or {'status': 'completed'}))
        if steps:
            (self.root / 'steps.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in steps))

    def test_medical_export_only_enriches_decisions_and_preserves_proposal_semantics(self):
        directive = Directive('prompt1', 'PRIVATE_MODEL_PROMPT', 'Try explicit answer format',
                              mode='configure_prompt', payload=PromptSpec('Return a short answer.'))
        self.monitor.decision(Decision('explore', directive=directive), self.context, 0)
        self.monitor.data['decisions'][0].pop('changes')  # Old producer snapshot.
        self.monitor.data['decisions'][0].pop('reason')
        self.monitor.stage('researcher_finished', status='completed')
        before = json.loads((self.root / 'monitor.json').read_text())
        old = before['decisions'][0]
        history = [dict(record_id='memory-prompt', kind='decision', details=asdict(Decision('explore', directive=directive))),
                   dict(record_id='memory-feedback', kind='feedback', details={'feedback': {
                       'directive_id': 'prompt1', 'status': 'completed',
                       'summary': 'Prompt proposed; evaluate before application'}})]
        self.export_fixture(history, result={'status': 'completed', 'reason': 'researcher_finished',
                                            'duration_seconds': 999})
        evidence = {p.name: p.read_bytes() for p in self.root.iterdir() if p.name != 'monitor.json'}
        snapshot = enrich_completed_run(self.root)
        self.assertEqual({k: v for k, v in snapshot.items() if k != 'decisions'},
                         {k: v for k, v in before.items() if k != 'decisions'})
        row = snapshot['decisions'][0]
        for key in ('decision_id', 'timestamp', 'source'):
            self.assertEqual(row[key], old[key])
        self.assertEqual(row['reason'], '')
        self.assertEqual(row['changes'], [{'field': 'instructions', 'after': 'Return a short answer.'}])
        self.assertEqual(row['feedback_summary'], 'Prompt proposed; evaluate before application')
        self.assertEqual(enrich_completed_run(self.root), snapshot)
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir() if p.name != 'monitor.json'}, evidence)

    def test_memory_operation_receipts_pair_only_with_adjacent_exact_calls(self):
        def decision(ident, steps):
            return dict(record_id=ident, kind='decision', details=asdict(Decision(
                'invoke', operation=OperationCall('define_recipe', {'steps': steps}))))
        def receipt(ident, steps, summary):
            return dict(record_id=ident, kind='operation', details={
                'operation': asdict(OperationCall('define_recipe', {'steps': steps})),
                'result': {'status': 'completed', 'summary': summary, 'data': {'text': 'PRIVATE_DATA'}}})
        history = [decision('a', 1), receipt('ar', 1, 'First recipe defined'),
                   decision('b', 2), receipt('br', 99, 'Mismatched recipe'),
                   decision('c', 2), receipt('cr', 2, 'Second recipe defined'),
                   decision('d', 2)]
        self.export_fixture(history)
        rows = enrich_completed_run(self.root)['decisions']
        self.assertEqual([r['feedback_summary'] for r in rows],
                         ['First recipe defined', '', 'Second recipe defined', ''])
        self.assertEqual([r['status'] for r in rows], ['completed', 'proposed', 'completed', 'proposed'])
        self.assertNotIn('PRIVATE_DATA', json.dumps(rows))

    def test_steps_pair_repeated_operations_by_complete_dispatch_order(self):
        history = [dict(record_id=f'op-{n}', kind='decision', details=asdict(Decision(
            'invoke', operation=OperationCall('read_artifact', {'artifact_id': f'a{n}'})))) for n in range(2)]
        steps = [dict(kind='progressed', event={'kind': 'progressed', 'operation_result': {
            'status': 'completed', 'summary': f'Read artifact {n}', 'data': {'text': 'PRIVATE_DATA'}}}) for n in range(2)]
        self.export_fixture(history, steps)
        rows = enrich_completed_run(self.root)['decisions']
        self.assertEqual([r['feedback_summary'] for r in rows], ['Read artifact 0', 'Read artifact 1'])
        self.assertNotIn('PRIVATE_DATA', json.dumps(rows))
        self.assertEqual(enrich_completed_run(self.root)['decisions'], rows)
        # A missing tick makes association ambiguous; do not assign the lone receipt.
        self.monitor.data['decisions'] = []
        self.monitor.write()
        self.export_fixture(history, steps[:1])
        self.assertEqual([r['feedback_summary'] for r in enrich_completed_run(self.root)['decisions']], ['', ''])

    def test_steps_feedback_can_fill_missing_memory_feedback_without_state_rewrite(self):
        history = [dict(record_id='d1-record', kind='decision', details=asdict(Decision('explore', directive=self.directive)))]
        steps = [dict(kind='finished', event={'kind': 'dispatch', 'directive': asdict(self.directive),
                    'feedback': {'directive_id': 'd1', 'status': 'failed', 'summary': 'Training stopped',
                                 'error': 'Worker unavailable', 'training_metrics': {'train_loss': .25}}})]
        self.export_fixture(history, steps, {'status': 'stopped'})
        row = enrich_completed_run(self.root)['decisions'][0]
        self.assertEqual((row['status'], row['feedback_summary'], row['error']),
                         ('failed', 'Training stopped', 'Worker unavailable'))
        self.assertEqual(row['metrics'], {'train_loss': .25})

    def test_export_retains_explicit_before_only_for_the_same_submitted_change(self):
        self.monitor.decision(Decision('explore', directive=self.directive), self.context, 0)
        self.monitor.data['decisions'][0]['changes'][0]['before'] = 'explicit previous recipe'
        self.monitor.write()
        self.export_fixture([dict(record_id='record', kind='decision',
                                  details=asdict(Decision('explore', directive=self.directive)))])
        row = enrich_completed_run(self.root)['decisions'][0]
        self.assertEqual(row['changes'][0]['before'], 'explicit previous recipe')


if __name__ == '__main__':
    unittest.main()
