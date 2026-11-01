"""Monitor projections use explicit decisions and observed progress, not raw prompts."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.qwen_toy.monitor import Monitor, enrich_completed_run
from rsisys.loops.contracts.coordination import ExecutionView, ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import TrainSpec
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

    def test_feedback_metrics_merge_public_numeric_groups_and_project_with_operation(self):
        from web.server import project_snapshot
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
        self.assertEqual(snapshot['current']['status'], 'completed')
        self.assertEqual(snapshot['updated_at'], before)
        self.assertIsNone(snapshot['decisions'][0]['timestamp'])
        self.assertNotIn('PRIVATE_MODEL_PROMPT', json.dumps(snapshot))
        self.assertEqual(enrich_completed_run(self.root), snapshot)

    def test_export_refuses_live_run(self):
        (self.root / 'result.json').write_text('{"status":"running"}')
        with self.assertRaisesRegex(ValueError, 'finished'):
            enrich_completed_run(self.root)


if __name__ == '__main__':
    unittest.main()
