"""Standalone stdlib tests for the isolated read-only monitor service."""
import importlib.util
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
        for path in ["/", "/style.css", "/app.js", "/demo.json"]:
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
