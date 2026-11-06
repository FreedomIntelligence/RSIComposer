"""Public projection tests with fake receipts; no execution or model providers."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.medical.open_lab_monitor import OpenLabMonitor


class FakeMemory:
    def __init__(self):
        self.entries = []

    def history(self, *, kind, limit=None):
        assert kind == 'operation'
        return tuple(self.entries if limit is None else self.entries[-limit:])


class OpenLabMonitorTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.monitor = OpenLabMonitor(self.root, 'fake-run',
            SimpleNamespace(model_id='fake-model', revision='fake-revision'), 'decision-revision')
        self.harness = SimpleNamespace(memory=FakeMemory(),
            state=SimpleNamespace(decision_records={}))

    def receipt(self, ident, action='lab_poll', data=None, arguments=None, **result):
        operation = {'name': action, 'arguments': arguments or {}}
        self.harness.state.decision_records[ident] = {'kind': 'invoke', 'status': 'completed',
            'proposal': {'operation': deepcopy(operation), 'reason': '模型原始判断', 'strategy': 'free'}}
        self.harness.memory.entries.append(SimpleNamespace(record_id=ident + ':operation',
            details={'decision_id': ident, 'operation': operation,
                     'result': {'summary': '原始即时反馈', 'data': data or {}, **result}}))

    def update(self, step=None):
        before = deepcopy(self.harness)
        self.monitor.update(self.harness, SimpleNamespace(calls=19),
            SimpleNamespace(compactions=2), step or SimpleNamespace(event={}))
        self.assertEqual(self.harness.state.decision_records, before.state.decision_records)
        self.assertEqual(self.harness.memory.entries, before.memory.entries)
        return json.loads((self.root / 'monitor.json').read_text())

    @staticmethod
    def fixed(digest='a' * 64, **fields):
        result = {'candidate_id': 'candidate.py', 'protocol': 'exact-option-letter-v1',
            'record_sha256': digest, 'metrics': {'MedQA': {'accuracy': .703125, 'n': 64}},
            'dataset_sha256': {'MedQA': 'd' * 64}, 'evaluator_sha256': 'e' * 64,
            'adopted': False, 'weight_identity': 'not_verified_by_prediction_protocol', **fields}
        if digest is None:
            result.pop('record_sha256')
        return result

    def test_raw_action_reason_feedback_and_identity_are_preserved(self):
        self.receipt('d1', 'custom_unknown_operation', {'job_id': 'job', 'state': 'running'},
                     summary='program exited without score', error='producer error')
        record = self.harness.state.decision_records['d1']
        record.update(research_attempt_id='attempt', depends_on_decision_ids=['source'])
        # The executed receipt is authoritative if a proposal projection differs.
        record['proposal']['operation']['name'] = 'different-proposal'
        public = self.update()
        row = public['decisions'][0]
        self.assertEqual(row['action'], 'custom_unknown_operation')
        self.assertEqual(row['reason'], '模型原始判断')
        self.assertEqual(row['feedback_summary'], 'program exited without score')
        self.assertEqual(row['error'], 'producer error')
        self.assertEqual(row['research_attempt_id'], 'attempt')
        self.assertEqual(row['depends_on_decision_ids'], ['source'])
        self.assertEqual((row['job_id'], row['state']), ('job', 'running'))
        self.assertNotIn('metrics', row)
        self.assertEqual(public['current']['metrics'], {'model_calls': 19, 'compactions': 2})

    def test_route_default_only_applies_to_result_and_request_wins(self):
        self.receipt('poll')
        self.receipt('result', 'lab_result')
        self.receipt('memory', 'lab_result', {'feedback_route': 'context'},
                     {'feedback_route': 'memory'})
        public = self.update()
        poll, result, memory = public['decisions']
        self.assertNotIn('feedback_route', poll)
        self.assertEqual(result['feedback_route'], 'context')
        self.assertEqual(memory['feedback_route'], 'memory')
        for row in public['decisions']:
            self.assertNotIn('memory_status', row)
            self.assertNotIn('archive_status', row)
        self.assertNotIn('feedback', public)

    def test_reported_archive_and_memory_states_are_independent(self):
        self.receipt('reported', 'lab_result', {'archive_delivery': {'status': 'archived', 'count': 243},
            'memory_organization': {'status': 'queued'}}, {'feedback_route': 'memory'})
        public = self.update()
        self.assertEqual(public['decisions'][0]['archive_status'], 'archived')
        self.assertEqual(public['decisions'][0]['memory_status'], 'queued')
        self.assertEqual(public['current']['memory_status'], 'queued')
        self.receipt('later', 'lab_poll', {'archive_status': 'archived', 'memory_status': 'failed'})
        self.assertEqual(self.update()['decisions'][-1]['memory_status'], 'failed')
        self.receipt('sparse', 'lab_poll')
        current = self.update()['current']
        for key in ('archive_status', 'memory_status', 'feedback_route', 'job_id', 'state'):
            self.assertNotIn(key, current)

    def test_fixed_evaluation_poll_result_dedup_preserves_distinct_digests(self):
        fixed = self.fixed()
        self.receipt('poll', data={'job_id': 'job1', 'state': 'succeeded', 'fixed_evaluation': fixed})
        self.receipt('result', 'lab_result', {'job_id': 'job1', 'state': 'succeeded',
                                            'fixed_evaluation': fixed})
        self.receipt('next', data={'job_id': 'job2', 'state': 'succeeded',
                                  'fixed_evaluation': self.fixed('b' * 64)})
        public = self.update()
        self.assertEqual(len(public['evaluations']), 2)
        self.assertEqual({row['record_sha256'] for row in public['evaluations']}, {'a' * 64, 'b' * 64})
        for row in public['evaluations']:
            self.assertEqual(row['candidate_id'], 'candidate.py')
            self.assertNotIn('role', row)
            self.assertEqual(row['metrics']['MedQA']['accuracy'], .703125)
            self.assertFalse(row['adopted'])
        self.assertEqual(public['metric_definitions']['MedQA.accuracy']['unit'], 'percent')
        self.assertEqual(len(self.update()['evaluations']), 2)

    def test_job_identity_dedup_and_later_digest_do_not_add_an_evaluation(self):
        self.receipt('p', data={'job_id': 'j', 'fixed_evaluation': self.fixed(None)})
        self.receipt('r', data={'job_id': 'j', 'fixed_evaluation': self.fixed()})
        self.receipt('again', data={'job_id': 'j', 'fixed_evaluation': self.fixed(None)})
        self.assertEqual(len(self.update()['evaluations']), 1)
        self.assertEqual(len(self.update()['evaluations']), 1)
        self.assertEqual(self.monitor.data['evaluations'][0]['record_sha256'], 'a' * 64)

    def test_checkpoint_profile_and_raw_output_diagnostics_remain_public(self):
        fixed = self.fixed(profile={'id': 'medical-checkpoint-nonthinking-v1'},
            profile_sha256='c' * 64, adapter_identity={'sha256': 'd' * 64},
            diagnostics={'invalid_answers': 3, 'output_limit_hits': 4})
        self.receipt('checkpoint', data={'job_id': 'j', 'fixed_evaluation': fixed})
        row = self.update()['evaluations'][0]
        for name in ('profile', 'profile_sha256', 'adapter_identity', 'diagnostics'):
            self.assertEqual(row[name], fixed[name])

    def test_only_explicit_evaluation_role_marks_baseline(self):
        self.receipt('baseline-named', data={'job_id': 'baseline-job',
            'fixed_evaluation': self.fixed(candidate_id='baseline.py')})
        self.receipt('explicit', data={'job_id': 'other-job',
            'fixed_evaluation': self.fixed('b' * 64, role='baseline')})
        first, second = self.update()['evaluations']
        self.assertNotIn('role', first)
        self.assertEqual(second['role'], 'baseline')

    def test_no_metrics_from_text_submission_unverified_or_unidentified_data(self):
        fixed = self.fixed()
        self.receipt('submission', 'medical_evaluate', {'job_id': 'submitted', 'state': 'running'})
        self.receipt('text', data={'stdout': json.dumps(fixed), 'metrics': fixed['metrics'],
            'output': {'fixed_evaluation': fixed}}, summary=json.dumps(fixed))
        self.harness.state.decision_records['text']['proposal']['reason'] = json.dumps(fixed)
        self.receipt('unverified', data={'fixed_evaluation': self.fixed(verified=False)})
        self.receipt('no-identity', data={'fixed_evaluation': self.fixed(None)})
        self.receipt('bad-percent', data={'fixed_evaluation': self.fixed(
            metrics={'MedQA': {'accuracy': 70.3125}})})
        self.receipt('bad-bool', data={'fixed_evaluation': self.fixed(metrics={'MedQA': {'accuracy': True}})})
        self.receipt('bad-huge', data={'fixed_evaluation': self.fixed(metrics={'MedQA': {'accuracy': 10 ** 1000}})})
        public = self.update()
        self.assertEqual(public['evaluations'], [])
        self.assertNotIn('metric_definitions', public)
        self.assertTrue(all('metrics' not in row for row in public['decisions']))

    def test_history_window_does_not_discard_older_evaluation_or_job(self):
        self.receipt('old', data={'job_id': 'old-job', 'state': 'running',
                                  'fixed_evaluation': self.fixed()})
        for index in range(601):
            self.receipt('d' + str(index))
        public = self.update()
        self.assertEqual(public['history_window'], {'limit': 300, 'total': 602, 'truncated': True})
        self.assertEqual(len(public['decisions']), 300)
        self.assertEqual(public['decisions'][0]['decision_id'], 'd301')
        self.assertEqual(len(public['evaluations']), 1)
        self.assertEqual(public['jobs'][0]['job_id'], 'old-job')

    def test_step_payload_cannot_invent_a_job_or_evaluation(self):
        public = self.update(SimpleNamespace(event={'decision': {'operation_result': {'data': {
            'job_id': 'invented', 'state': 'succeeded', 'fixed_evaluation': self.fixed()}}}}))
        self.assertEqual(public['jobs'], [])
        self.assertEqual(public['evaluations'], [])

    def test_jobs_keep_last_original_observation_through_finish(self):
        self.receipt('poll', data={'job_id': 'j', 'state': 'running'})
        job = self.update()['jobs'][0]
        self.assertEqual(job['decision_id'], 'poll')
        self.assertTrue(job['observed_at'])
        self.receipt('unrelated', 'lab_read_file')
        self.assertEqual(self.update()['jobs'][0], job)
        self.monitor.finish({'status': 'completed', 'ended_at': 1793850000, 'stop': 'host_deadline'})
        self.assertEqual(self.monitor.data['jobs'][0], job)
        self.assertEqual(self.monitor.data['lifecycle']['stop_reason'], 'host_deadline')
        self.assertEqual(self.monitor.data['lifecycle']['finished_at'], '2026-11-05T03:40:00+00:00')
        self.assertNotIn('cleanup', self.monitor.data['lifecycle'])

    def test_cleanup_only_comes_from_explicit_boolean_confirmation(self):
        # A file reporting successful cleanup is not a monitor input.
        (self.root / 'controller-cleanup.json').write_text('{"confirmed":true}')
        for cleanup, expected in ((None, None), ({'owned_lab_closed': True}, None),
                ({'confirmed': 'true'}, None), ({'confirmed': False}, False),
                ({'confirmed': True}, True)):
            with self.subTest(cleanup=cleanup):
                result = {'status': 'completed'}
                if cleanup is not None:
                    result['cleanup'] = cleanup
                with patch.object(Path, 'read_text', side_effect=AssertionError('no file reads')), \
                     patch.object(Path, 'read_bytes', side_effect=AssertionError('no file reads')):
                    self.monitor.finish(result)
                lifecycle = self.monitor.data['lifecycle']
                self.assertNotIn('stop_reason', lifecycle)
                if expected is None:
                    self.assertNotIn('cleanup', lifecycle)
                else:
                    self.assertIs(lifecycle['cleanup']['confirmed'], expected)

    def test_finish_keeps_host_limit_status_instead_of_claiming_completion(self):
        self.monitor.finish({'status': 'host_limit_reached', 'stop': 'inference_admission_limit'})
        self.assertEqual(self.monitor.data['status'], 'host_limit_reached')

    def test_finish_preserves_explicit_failure_and_stop_reason(self):
        result = {'status': 'failed', 'error': 'explicit error', 'stop_reason': 'explicit-stop',
                  'finished_at': '2026-11-05T00:00:00Z'}
        self.monitor.finish(result)
        self.assertEqual(self.monitor.data['status'], 'failed')
        self.assertEqual(self.monitor.data['lifecycle'], {
            'finished_at': result['finished_at'], 'stop_reason': 'explicit-stop'})
        self.assertEqual(self.monitor.data['events'][-1]['summary'], 'explicit error')

    def test_producer_fields_survive_the_read_only_server_projection(self):
        from rsimonitor.server import project_snapshot

        self.receipt('result', 'lab_result', {'job_id': 'j', 'state': 'succeeded',
            'archive_status': 'archived', 'memory_status': 'completed',
            'fixed_evaluation': self.fixed(role='baseline')}, {'feedback_route': 'memory'})
        self.update()
        self.monitor.finish({'status': 'completed', 'stop': 'explicit-stop',
                             'cleanup': {'confirmed': True}})
        raw = self.monitor.data
        projected = project_snapshot(deepcopy(raw), 'fake-run')
        for key in ('evaluations', 'metric_definitions', 'history_window', 'lifecycle', 'jobs'):
            self.assertEqual(projected[key], raw[key])
        for key in ('feedback_route', 'archive_status', 'memory_status', 'job_id', 'state'):
            self.assertEqual(projected['decisions'][0][key], raw['decisions'][0][key])
            self.assertEqual(projected['current'][key], raw['current'][key])


if __name__ == '__main__':
    unittest.main()
