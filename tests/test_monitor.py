"""Standalone stdlib tests for the isolated read-only monitor service."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SPEC = importlib.util.spec_from_file_location("rsisys_monitor", Path(__file__).resolve().parents[1] / "web" / "server.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class MonitorProjectionTests(unittest.TestCase):
    """Projection and file-store regressions without opening a listening socket."""

    def project(self, **fields):
        return monitor.project_snapshot({'schema_version': 1, **fields}, 'run-1')

    def test_public_workflow_execution_decisions_and_round_results(self):
        workflow = {'phase': 'training', 'round': 2, 'rounds': 3, 'completed_directives': 5,
                    'elapsed_seconds': 12.5}
        current = {'directive_id': 'd2', 'mode': 'train', 'status': 'running', 'phase': 'training',
                   'strategy': 'Train another candidate', 'decision_revision': 'decision-v1',
                   'actor_revision': 'actor-v2', 'job_id': 'j2', 'elapsed_seconds': 3.5,
                   'review_after_seconds': 2, 'next_review_seconds': 4, 'review_status': 'continue',
                   'step': 3, 'total_steps': 10, 'metrics': {'loss': 0.4, 'eval_samples': 32}}
        decision = {'decision_id': 'review-2', 'timestamp': '2026-11-01T00:00:00Z', 'kind': 'review',
                    'source': 'model', 'round': 2, 'directive_id': 'd2', 'mode': 'train', 'action': 'continue',
                    'strategy': 'Train another candidate', 'reason': 'Loss is still improving', 'status': 'accepted',
                    'decision_revision': 'decision-v1', 'actor_revision': 'actor-v2', 'job_id': 'j2',
                    'candidate_id': 'c2', 'protocol_ref': 'heldout-nll', 'recipe_ref': 'toy-lora', 'method': 'sft',
                    'review_after_seconds': 2, 'next_review_seconds': 4}
        round_result = {'round': 1, 'base_revision': 'base', 'candidate_revision': 'actor-v2',
                        'selection': 'accepted', 'actor_revision': 'actor-v2', 'applied': True,
                        'answer_correct': False, 'training_review_count': 2,
                        'metrics': {'delta_nll': 0.1, 'candidate_nll': 1.5}}
        job = {'job_id': 'j2', 'kind': 'training', 'status': 'running', 'directive_id': 'd2', 'round': 2,
               'step': 3, 'total_steps': 10, 'elapsed_seconds': 3.5, 'metrics': {'loss': 0.4}}
        data = self.project(workflow=workflow, current=current, decisions=[decision],
                            round_results=[round_result], jobs=[job])
        self.assertEqual(data['workflow'], workflow)
        self.assertEqual(data['current'], current)
        self.assertEqual(data['decisions'], [decision])
        self.assertEqual(data['round_results'], [round_result])
        self.assertEqual(data['jobs'], [job])
        self.assertEqual(data['schema_version'], 1)

    def test_old_snapshots_and_malformed_optional_containers_have_empty_states(self):
        legacy = self.project(jobs=[{'job_id': 'j', 'kind': 'training', 'status': 'running'}])
        self.assertEqual(legacy['jobs'], [{'job_id': 'j', 'kind': 'training', 'status': 'running'}])
        for data in (legacy, *(self.project(workflow=value, current=value, decisions=value, round_results=value)
                              for value in (None, 'broken', 42, True))):
            self.assertEqual(data['workflow'], {})
            self.assertEqual(data['current'], {})
            self.assertEqual(data['decisions'], [])
            self.assertEqual(data['round_results'], [])
        data = self.project(workflow=[], current=[], decisions={}, round_results={})
        self.assertEqual((data['workflow'], data['current'], data['decisions'], data['round_results']),
                         ({}, {}, [], []))
        data = self.project(decisions=[None, [], 7, {'action': 'continue'}],
                            round_results=[False, 'broken', {'round': 1, 'metrics': []}],
                            current={'metrics': 'broken'}, jobs=[{'metrics': []}])
        self.assertEqual(data['decisions'], [{'action': 'continue'}])
        self.assertEqual(data['round_results'], [{'round': 1, 'metrics': {}}])
        self.assertEqual(data['current'], {'metrics': {}})
        self.assertEqual(data['jobs'], [{'metrics': {}}])

    def test_new_fields_never_forward_private_attachments_or_non_numeric_metrics(self):
        private = {key: {'contents': 'TOP_SECRET'} for key in
                   ('payload', 'raw_payload', 'prompt', 'response', 'config', 'credentials', 'reasoning')}
        metrics = {'loss': 0.4, 'eval_samples': 32, 'accuracy': {'baseline': 0.5, 'candidate': 0.8},
                   'text': 'TOP_SECRET', 'flag': True, 'values': ['TOP_SECRET'], 'prompt_tokens': 50,
                   'raw_payload': {'loss': 3}, 'response': {'loss': 4}, 'config': {'loss': 5},
                   'credentials': {'loss': 6}, 'NaN': float('nan'), 'inf': float('inf'), 'huge': 10 ** 1000}
        raw = {'schema_version': 1, 'workflow': {'phase': 'training', **private},
               'current': {'strategy': 'Public strategy', 'metrics': metrics, **private},
               'decisions': [{'reason': 'Public decision explanation', **private}],
               'round_results': [{'round': 1, 'metrics': metrics, **private}],
               'jobs': [{'job_id': 'j', 'metrics': metrics, **private}]}
        original = deepcopy(raw)
        data = monitor.project_snapshot(raw, 'run-1')
        safe_metrics = {'loss': 0.4, 'eval_samples': 32, 'accuracy': {'baseline': 0.5, 'candidate': 0.8}}
        self.assertEqual(data['workflow'], {'phase': 'training'})
        self.assertEqual(data['decisions'], [{'reason': 'Public decision explanation'}])
        self.assertEqual(data['current'], {'strategy': 'Public strategy', 'metrics': safe_metrics})
        self.assertEqual(data['round_results'], [{'round': 1, 'metrics': safe_metrics}])
        self.assertEqual(data['jobs'], [{'job_id': 'j', 'metrics': safe_metrics}])
        self.assertNotIn('TOP_SECRET', json.dumps(data, allow_nan=False))
        self.assertEqual(raw, original)

    def test_event_details_accept_public_decision_fields_but_not_arbitrary_objects(self):
        details = {'directive_id': 'd', 'decision_id': 'review', 'round': 2, 'source': 'model',
                   'action': 'cancel', 'mode': 'train', 'reason': 'Public cancellation reason',
                   'strategy': 'Train a candidate', 'decision_revision': 'd-v1', 'actor_revision': 'a-v2',
                   'candidate_revision': 'a-v3', 'review_after_seconds': 2, 'next_review_seconds': None,
                   'elapsed_seconds': 8, 'total_steps': 20, 'selection': 'rejected', 'applied': False,
                   'answer_correct': True, 'training_review_count': 3}
        data = self.project(events=[{'kind': 'review', 'details': {
            **details, 'payload': {'raw': 'TOP_SECRET'}, 'response': 'TOP_SECRET',
            'config': 'TOP_SECRET', 'credentials': 'TOP_SECRET', 'metrics': {'loss': 0.4}}}])
        self.assertEqual(data['events'][0]['details'], details)
        bad = self.project(events=[{'details': {'reason': {'prompt': 'TOP_SECRET'}}}],
                           decisions=[{'strategy': ['TOP_SECRET']}])
        self.assertIsNone(bad['events'][0]['details']['reason'])
        self.assertIsNone(bad['decisions'][0]['strategy'])
        self.assertNotIn('TOP_SECRET', json.dumps(bad))

    def test_job_history_is_bounded_and_decision_operation_metrics_are_projected(self):
        points = [{'step': n, 'elapsed_seconds': n / 2, 'metrics': {'loss': n / 1000}}
                  for n in range(1005)]
        data = self.project(jobs=[{'job_id': 'j', 'metric_history': points}],
            decisions=[{'kind': 'operation', 'operation': 'query_experiments',
                        'payload': 'TOP_SECRET', 'metrics': {'loss': 0.25, 'api_key': 123}}])
        self.assertEqual(data['jobs'][0]['metric_history'], points[-1000:])
        self.assertEqual(data['decisions'], [{'kind': 'operation', 'operation': 'query_experiments',
                                            'metrics': {'loss': 0.25}}])
        for value in (None, {}, False, 'broken'):
            self.assertEqual(self.project(jobs=[{'metric_history': value}])['jobs'], [{'metric_history': []}])
        self.assertNotIn('TOP_SECRET', json.dumps(data))

    def test_explanations_and_recent_collections_are_bounded(self):
        explanation = '公开说明' * 400
        rows = [{'decision_id': str(i), 'reason': explanation} for i in range(monitor.MAX_ROWS + 5)]
        data = self.project(current={'strategy': explanation}, decisions=rows,
                            round_results=[{'round': i} for i in range(monitor.MAX_ROWS + 5)],
                            events=[{'details': {'reason': explanation}}])
        self.assertEqual(len(data['decisions']), monitor.MAX_ROWS)
        self.assertEqual(data['decisions'][0]['decision_id'], '5')
        self.assertEqual(len(data['round_results']), monitor.MAX_ROWS)
        self.assertEqual(data['round_results'][-1]['round'], monitor.MAX_ROWS + 4)
        for text in (data['current']['strategy'], data['decisions'][0]['reason'],
                     data['events'][0]['details']['reason']):
            self.assertEqual(text, explanation[:1000])

    def test_new_metrics_keep_existing_depth_and_entry_limits(self):
        metrics = {'m' + str(i): i for i in range(120)}
        data = self.project(current={'metrics': metrics})
        self.assertEqual(len(data['current']['metrics']), 100)
        deep = {'a': {'b': {'c': {'d': {'e': {'hidden': 1}}}}}}
        data = self.project(round_results=[{'metrics': deep}])
        self.assertNotIn('hidden', json.dumps(data))

    def test_file_store_projects_extended_schema_without_serving_raw_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run-1'
            run.mkdir()
            (run / 'monitor.json').write_text(json.dumps({
                'schema_version': 1, 'workflow': {'round': 2}, 'current': {'mode': 'train'},
                'decisions': [{'reason': 'Public reason', 'response': 'TOP_SECRET'}],
                'round_results': [{'round': 1, 'applied': False, 'metrics': {'delta_nll': -0.1}}],
            }))
            store = monitor.RunStore(root)
            data = store.read('run-1')
            self.assertEqual(data['workflow'], {'round': 2})
            self.assertEqual(data['decisions'], [{'reason': 'Public reason'}])
            self.assertNotIn('TOP_SECRET', json.dumps(data))
            # Run-list summaries retain their existing compact contract.
            self.assertEqual(store.list()[0]['counts'], {'jobs': 0, 'candidates': 0, 'events': 0})


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runs"
        self.root.mkdir()
        self.server = monitor.MonitorServer(("127.0.0.1", 0), self.root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def write(self, run_id="run-1", **fields):
        directory = self.root / run_id
        directory.mkdir(exist_ok=True)
        data = dict(schema_version=1, run_id=run_id, status="running", objective="A real test run", **fields)
        (directory / "monitor.json").write_text(json.dumps(data), encoding="utf-8")
        return directory

    def get(self, path, method="GET", headers=None):
        request = Request(self.url + path, method=method, headers=headers or {})
        try:
            response = urlopen(request, timeout=2)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, response.read()

    def test_empty_and_missing_root(self):
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"], [])
        self.root.rmdir()
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"], [])
        self.assertFalse(self.root.exists())

    def test_unrelated_preview_directory_is_not_a_run(self):
        (self.root / "monitor-preview").mkdir()
        (self.root / "monitor-preview" / "wide.jpg").write_bytes(b"preview")
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"], [])

    def test_worker_flat_metrics_preserve_values_and_direction(self):
        metrics = {"base_nll": 1.942, "candidate_nll": 1.781,
                   "delta_nll": -0.161, "eval_samples": 128}
        self.write(candidates=[{"candidate_id": "c", "metrics": metrics}])
        data = json.loads(self.get("/api/runs/run-1")[2])
        self.assertEqual(data["candidates"][0]["metrics"], metrics)

    def test_huge_numeric_value_does_not_crash(self):
        self.write(candidates=[{"metrics": {"score": 10 ** 1000}}])
        self.assertEqual(self.get("/api/runs/run-1")[0], 200)

    def test_summary_and_snapshot(self):
        self.write(jobs=[{"job_id": "j-1", "kind": "training", "status": "running"}],
                   loops={"decision_making": {"status": "running", "summary": "Planning"}},
                   models={"actor": {"model_id": "actor", "revision": "r2"}})
        summary = json.loads(self.get("/api/runs")[2])["runs"][0]
        self.assertEqual(summary["counts"], {"jobs": 1, "candidates": 0, "events": 0})
        self.assertNotIn("jobs", summary)
        status, headers, body = self.get("/api/runs/run-1")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["loops"]["decision_making"]["summary"], "Planning")
        self.assertEqual(data["models"]["actor"]["revision"], "r2")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))

    def test_optional_fields_and_wrong_types(self):
        self.write(loops=None, models=[], jobs="wrong", candidates=[None, 2, {}], events=None)
        data = json.loads(self.get("/api/runs/run-1")[2])
        self.assertEqual(data["jobs"], [])
        self.assertEqual(data["loops"], {})
        self.assertEqual(data["candidates"], [{"metrics": {}}])

    def test_only_monitor_is_read_and_private_payloads_are_dropped(self):
        directory = self.write(config={"api_key": "TOP_SECRET"}, training_samples=["TOP_SECRET"],
            jobs=[{"job_id": "j", "config": {"password": "TOP_SECRET"}}],
            candidates=[{"candidate_id": "c", "metrics": {"score": {"baseline": .4, "candidate": .6},
                        "api_key": "TOP_SECRET", "samples": {"private": 42}, "config": {"private": 42}}}],
            events=[{"kind": "trained", "details": {"job_id": "j", "credentials": "TOP_SECRET", "samples": ["TOP_SECRET"], "config": "TOP_SECRET"}}])
        (directory / "config.json").write_text('"TOP_SECRET"')
        status, _, body = self.get("/api/runs/run-1")
        self.assertEqual(status, 200)
        self.assertNotIn(b"TOP_SECRET", body)
        self.assertNotIn(b"private", body)
        self.assertEqual(json.loads(body)["events"][0]["details"], {"job_id": "j"})
        self.assertEqual(self.get("/api/runs/run-1/config.json")[0], 404)
        self.assertEqual(self.get("/runs/run-1/config.json")[0], 404)

    def test_paths_cannot_escape(self):
        self.write()
        for path in ["/api/runs/..", "/api/runs/%2e%2e", "/api/runs/%2e%2e%2fsecret",
                     "/api/runs/run-1%2fmonitor.json", "/api/runs/run-1%5cmonitor.json",
                     "/api/runs/%252e%252e", "/api/runs/%00", "/../server.py", "/server.py", "/README.md"]:
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_symlink_run_and_snapshot_are_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "monitor.json").write_text('{"schema_version":1,"objective":"TOP_SECRET"}')
        (self.root / "linked-run").symlink_to(outside, target_is_directory=True)
        local = self.root / "linked-file"
        local.mkdir()
        (local / "monitor.json").symlink_to(outside / "monitor.json")
        self.assertEqual(self.get("/api/runs/linked-run")[0], 404)
        self.assertEqual(self.get("/api/runs/linked-file")[0], 404)
        data = json.loads(self.get("/api/runs")[2])
        self.assertNotIn("linked-run", [r["run_id"] for r in data["runs"]])
        self.assertNotIn("TOP_SECRET", json.dumps(data))

    def test_directory_swap_to_symlink_is_rejected(self):
        directory = self.write()
        directory.rename(self.root / "old-run")
        (self.root / "run-1").symlink_to(self.root / "old-run", target_is_directory=True)
        self.assertEqual(self.get("/api/runs/run-1")[0], 404)

    def test_non_regular_snapshot_rejected_without_blocking(self):
        import os
        directory = self.root / "fifo-run"
        directory.mkdir()
        os.mkfifo(directory / "monitor.json")
        self.assertEqual(self.get("/api/runs/fifo-run")[0], 404)

    def test_partial_json_is_retryable_and_recovers(self):
        directory = self.write()
        (directory / "monitor.json").write_text('{"schema_version":')
        status, headers, _ = self.get("/api/runs/run-1")
        self.assertEqual(status, 503)
        self.assertEqual(headers["Retry-After"], "2")
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"][0]["status"], "unavailable")
        self.write()
        self.assertEqual(self.get("/api/runs/run-1")[0], 200)

    def test_snapshot_missing_and_schema_mismatch(self):
        directory = self.root / "run-1"
        directory.mkdir()
        self.assertEqual(self.get("/api/runs/run-1")[0], 404)
        for raw in [[], {"schema_version": 2}, {"schema_version": True}, {}]:
            (directory / "monitor.json").write_text(json.dumps(raw))
            self.assertEqual(self.get("/api/runs/run-1")[0], 422)

    def test_large_snapshot_rejected(self):
        directory = self.write()
        (directory / "monitor.json").write_bytes(b" " * (monitor.MAX_BYTES + 1))
        self.assertEqual(self.get("/api/runs/run-1")[0], 413)

    def test_no_write_methods(self):
        directory = self.write()
        original = (directory / "monitor.json").read_bytes()
        for method in ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"]:
            self.assertEqual(self.get("/api/runs/run-1", method=method)[0], 405)
        self.assertEqual((directory / "monitor.json").read_bytes(), original)

    def test_host_and_cross_site_guard(self):
        self.assertEqual(self.get("/api/runs", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.get("/api/runs", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)

    def test_static_allowlist_head_and_demo(self):
        for path in ["/", "/style.css", "/i18n.js", "/lineage.js", "/app.js", "/demo.json"]:
            self.assertEqual(self.get(path)[0], 200)
        self.assertEqual(self.get("/", method="HEAD")[2], b"")
        self.assertEqual(json.loads(self.get("/demo.json")[2])["run_id"], "example-dual-loop-001")
        self.assertEqual(json.loads(self.get("/api/runs")[2])["runs"], [])

    def test_run_directory_name_is_authoritative(self):
        directory = self.write()
        (directory / "monitor.json").write_text('{"schema_version":1,"run_id":"../elsewhere"}')
        data = json.loads(self.get("/api/runs/run-1")[2])
        self.assertEqual(data["run_id"], "run-1")

    def test_non_finite_numbers_do_not_break_json(self):
        directory = self.write()
        (directory / "monitor.json").write_text('{"schema_version":1,"candidates":[{"metrics":{"score":NaN,"loss":Infinity}}]}')
        data = json.loads(self.get("/api/runs/run-1")[2])
        self.assertEqual(data["candidates"][0]["metrics"], {})


if __name__ == "__main__":
    unittest.main()
