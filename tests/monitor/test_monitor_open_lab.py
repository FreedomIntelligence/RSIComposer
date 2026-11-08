"""Open Lab public protocol projections, independent of the execution system."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.request import urlopen

from rsimonitor import server as monitor


class OpenLabProjectionTests(unittest.TestCase):
    def project(self, **fields):
        return monitor.project_snapshot({"schema_version": 1, **fields}, "open-lab")

    def evaluation(self, **fields):
        return {"evaluation_id": "evaluation-1", "job_id": "job-1",
                "decision_id": "evaluate-1", "candidate_id": "candidate-1",
                "protocol": "exact-option-letter-v1", "record_sha256": "a" * 64,
                "metrics": {"medxpertqa": {"accuracy": .1875, "n": 64},
                            "medqa": {"accuracy": .703125, "n": 64}}, **fields}

    def test_legacy_public_snapshot_keeps_actions_metrics_and_last_job_state(self):
        # These are the fields actually available from the historical public
        # API. Reason/summary prose is deliberately not an evidence source.
        actions = ("lab_cancel", "lab_poll", "lab_read_file", "lab_read_log",
                   "lab_result", "lab_status", "lab_write_file", "medical_evaluate")
        data = self.project(
            status="completed", updated_at="2026-11-05T05:38:36.228754+00:00",
            workflow={"phase": "completed", "elapsed_seconds": 1716.2004459699383},
            current={"mode": "lab_poll", "status": "pending",
                     "metrics": {"model_calls": 252, "compactions": 4}},
            jobs=[{"job_id": "last-job", "kind": "lab_program", "status": "running"}],
            decisions=[{"decision_id": f"d-{i}", "kind": "invoke", "action": action,
                        "status": "completed", "reason": "Baseline accuracy 0.1875",
                        "feedback_summary": "Program succeeded"} for i, action in enumerate(actions)],
            events=[{"kind": "completed", "summary": "host_deadline_or_signal"}])
        self.assertEqual([row["action"] for row in data["decisions"]], list(actions))
        self.assertEqual(data["current"]["metrics"], {"model_calls": 252, "compactions": 4})
        self.assertEqual(data["current"]["status"], "pending")
        self.assertEqual(data["jobs"], [{"job_id": "last-job", "kind": "lab_program", "status": "running"}])
        for key in ("evaluations", "lifecycle", "history_window"):
            self.assertNotIn(key, data)
        for row in data["decisions"]:
            for key in ("feedback_route", "archive_status", "memory_status", "metrics"):
                self.assertNotIn(key, row)

    def test_feedback_archive_and_memory_states_are_independent(self):
        states = [{"feedback_route": "memory", "archive_status": "archived", "memory_status": "failed"},
                  {"feedback_route": "context", "archive_status": "pending"},
                  {"memory_status": "pending"}, {}]
        for state in states:
            with self.subTest(state=state):
                row = {"action": "lab_result", "status": "completed", "state": "succeeded",
                       "job_id": "job-1", "feedback_summary": "Terminal evidence", **state}
                data = self.project(current=row, decisions=[row])
                for projected in (data["current"], data["decisions"][0]):
                    for key in ("feedback_route", "archive_status", "memory_status"):
                        self.assertEqual({key: projected[key]} if key in projected else {},
                                         {key: state[key]} if key in state else {})
                    self.assertEqual(projected["state"], "succeeded")
                    self.assertEqual(projected["job_id"], "job-1")

    def test_feedback_does_not_infer_completion_from_route_or_summary(self):
        row = {"feedback_route": "memory", "status": "completed",
               "feedback_summary": "archived; memory updated",
               "archive_status": {"payload": "PRIVATE"}, "memory_status": True}
        data = self.project(current=row, decisions=[row, {"feedback_route": "other"}])
        for projected in (data["current"], data["decisions"][0]):
            self.assertEqual(projected["feedback_route"], "memory")
            self.assertNotIn("archive_status", projected)
            self.assertNotIn("memory_status", projected)
        self.assertNotIn("feedback_route", data["decisions"][1])
        self.assertNotIn("PRIVATE", json.dumps(data))

    def test_job_observations_do_not_follow_run_terminal_status(self):
        job = {"job_id": "job-1", "status": "running", "state": "running",
               "decision_id": "poll-1", "observed_at": "2026-11-05T05:38:00Z"}
        for status in ("completed", "failed", "stopped", "cancelled", "running"):
            with self.subTest(status=status):
                data = self.project(status=status, jobs=[job])
                self.assertEqual(data["jobs"], [job])
                self.assertNotIn("lifecycle", data)

    def test_weight_application_is_only_an_explicit_top_level_boolean(self):
        for applied in (True, False):
            data = self.project(weight_applied=applied)
            self.assertIs(data["weight_applied"], applied)
        for invalid in (None, 0, 1, "true", {}, []):
            self.assertNotIn("weight_applied", self.project(weight_applied=invalid))
        self.assertNotIn("weight_applied", self.project(
            status="completed", evaluations=[self.evaluation(adopted=True)],
            decisions=[{"action": "system_adopt", "status": "completed"}]))

    def test_lifecycle_only_preserves_explicit_reported_fields_including_false(self):
        for confirmed in (False, True):
            lifecycle = {"finished_at": "2026-11-05T05:38:36Z", "stop_reason": "host_deadline_or_signal",
                         "cleanup": {"confirmed": confirmed}}
            self.assertEqual(self.project(lifecycle=lifecycle)["lifecycle"], lifecycle)
        self.assertEqual(self.project(lifecycle={"finished_at": "reported"})["lifecycle"],
                         {"finished_at": "reported"})
        for value in (None, {}, {"cleanup": {}}, {"cleanup": {"confirmed": "true"}},
                      {"finished_at": False, "stop_reason": {"secret": "PRIVATE"}}):
            with self.subTest(value=value):
                self.assertNotIn("lifecycle", self.project(status="completed", lifecycle=value))

    def test_history_window_never_infers_missing_counts_or_truncation(self):
        window = {"limit": 300, "total": 245, "truncated": False}
        self.assertEqual(self.project(history_window=window)["history_window"], window)
        self.assertEqual(self.project(history_window={"total": 0})["history_window"], {"total": 0})
        rows = [{"decision_id": str(i)} for i in range(monitor.MAX_ROWS + 3)]
        data = self.project(decisions=rows)
        self.assertNotIn("history_window", data)
        self.assertEqual(len(data["decisions"]), monitor.MAX_ROWS)
        for invalid in (None, [], {}, {"limit": True, "total": -1, "truncated": 1},
                        {"limit": 2.5, "total": float("inf")}, {"total": 10 ** 1000}):
            with self.subTest(invalid=invalid):
                self.assertNotIn("history_window", self.project(history_window=invalid))

    def test_evaluation_roundtrip_preserves_formal_identity_and_explicit_adoption(self):
        row = self.evaluation(dataset_sha256="b" * 64, evaluator_sha256="c" * 64,
                              adopted=False, role="baseline",
                              weight_identity={"kind": "served", "backend_id": "modelmate",
                                               "model_id": "Qwen3.5-9B", "revision": "existing"})
        self.assertEqual(self.project(evaluations=[row])["evaluations"], [row])
        data = self.project(evaluations=[row])
        self.assertEqual(monitor.project_snapshot(json.loads(json.dumps(data)), "open-lab")["evaluations"],
                         data["evaluations"])
        self.assertEqual(self.project(evaluations=[])["evaluations"], [])
        self.assertNotIn("evaluations", self.project())
        self.assertNotIn("evaluations", self.project(evaluations={"payload": "PRIVATE"}))

    def test_explicit_unverified_records_are_rejected_without_inventing_verification(self):
        rejected = self.evaluation(verified=False)
        accepted = self.evaluation(verified=True, record_sha256="b" * 64)
        implicit = self.evaluation(record_sha256="c" * 64)
        data = self.project(evaluations=[rejected, accepted, implicit])["evaluations"]
        self.assertEqual(len(data), 2)
        self.assertEqual([row["record_sha256"] for row in data], ["b" * 64, "c" * 64])
        for row in data:
            self.assertNotIn("verified", row)

    def test_dataset_digests_support_bounded_benchmark_maps_without_attachments(self):
        datasets = {"MedQA": "d" * 64, "MedXpertQA": "E" * 64, "bad": "not-a-digest",
                    "nested": {"secret": "PRIVATE"}, "payload": "a" * 64,
                    "": "b" * 64, "x" * 101: "c" * 64}
        raw = self.evaluation(dataset_sha256=datasets)
        projected = self.project(evaluations=[raw])["evaluations"][0]
        self.assertEqual(projected["dataset_sha256"], {"MedQA": "d" * 64, "MedXpertQA": "E" * 64})
        projected["dataset_sha256"]["MedQA"] = "f" * 64
        self.assertEqual(datasets["MedQA"], "d" * 64)
        many = {f"b-{i}": "a" * 64 for i in range(105)}
        projected = self.project(evaluations=[self.evaluation(dataset_sha256=many)])["evaluations"][0]
        self.assertEqual(len(projected["dataset_sha256"]), 100)

    def test_evaluations_deduplicate_only_explicit_digest_or_identity(self):
        first = self.evaluation(adopted=False)
        latest = self.evaluation(adopted=True)
        other_record = self.evaluation(record_sha256="b" * 64)
        other_job = self.evaluation(evaluation_id="evaluation-2", job_id="job-2", record_sha256="c" * 64)
        rows = [first, other_record, latest, other_job]
        self.assertEqual(self.project(evaluations=rows)["evaluations"], [other_record, latest, other_job])
        # Hex case is not a distinct SHA-256 identity; its reported spelling is preserved.
        upper = self.evaluation(record_sha256="A" * 64)
        self.assertEqual(self.project(evaluations=[first, upper])["evaluations"], [upper])
        no_digest = {key: value for key, value in first.items() if key != "record_sha256"}
        self.assertEqual(self.project(evaluations=[no_digest, no_digest])["evaluations"], [no_digest])
        job_only = {key: value for key, value in no_digest.items() if key != "evaluation_id"}
        self.assertEqual(self.project(evaluations=[job_only, job_only])["evaluations"], [job_only])

    def test_matching_candidate_and_scores_do_not_create_evaluation_identity(self):
        row = {"candidate_id": "same", "protocol": "fixed", "metrics": {"accuracy": .5}}
        data = self.project(evaluations=[row, row])
        self.assertEqual(data["evaluations"], [row, row])
        for projected in data["evaluations"]:
            for field in ("evaluation_id", "job_id", "record_sha256", "adopted", "role", "weight_identity"):
                self.assertNotIn(field, projected)
        distinct = [self.evaluation(evaluation_id=f"e-{i}", job_id=f"j-{i}", record_sha256=str(i) * 64)
                    for i in (1, 2)]
        self.assertEqual(self.project(evaluations=distinct)["evaluations"], distinct)

    def test_evaluation_metrics_are_finite_numbers_without_raw_attachments(self):
        metrics = {"medqa": {"accuracy": .703125, "n": 64, "sample_count": 64,
                               "payload": "PRIVATE", "samples": ["PRIVATE"], "secret": 42,
                               "bad": float("nan"), "infinite": float("inf"), "bool": True},
                   "compactions": 0, "negative": -1.5, "api_key": 42, "huge": 10 ** 1000}
        row = self.evaluation(metrics=metrics, payload={"secret": "PRIVATE"})
        projected = self.project(evaluations=[row])["evaluations"][0]
        self.assertEqual(projected["metrics"], {
            "medqa": {"accuracy": .703125, "n": 64, "sample_count": 64},
            "compactions": 0, "negative": -1.5})
        self.assertNotIn("PRIVATE", json.dumps(projected, allow_nan=False))

    def test_new_fields_use_strict_whitelists_without_payload_leaks(self):
        private = {"payload": {"secret": "PRIVATE"}, "credentials": "PRIVATE"}
        data = self.project(
            evaluations=[self.evaluation(
                adopted="yes", dataset_sha256="invalid", evaluator_sha256={"secret": "PRIVATE"},
                weight_identity={"model_id": "public", "revision": "known", **private}, **private)],
            lifecycle={"cleanup": {"confirmed": False, **private}, **private},
            history_window={"limit": 300, **private},
            jobs=[{"status": "running", "observed_at": {"secret": "PRIVATE"}, **private}])
        self.assertEqual(data["evaluations"][0]["weight_identity"], {"model_id": "public", "revision": "known"})
        for key in ("adopted", "dataset_sha256", "evaluator_sha256"):
            self.assertNotIn(key, data["evaluations"][0])
        self.assertEqual(data["lifecycle"], {"cleanup": {"confirmed": False}})
        self.assertEqual(data["jobs"], [{"status": "running"}])
        self.assertNotIn("PRIVATE", json.dumps(data))
        text = self.evaluation(weight_identity="existing served model")
        self.assertEqual(self.project(evaluations=[text])["evaluations"], [text])

    def test_evaluation_projection_is_bounded_and_does_not_mutate_input(self):
        rows = [self.evaluation(evaluation_id=f"e-{i}", job_id=f"j-{i}", record_sha256=f"{i:064x}")
                for i in range(monitor.MAX_ROWS + 2)]
        raw = {"schema_version": 1, "evaluations": rows,
               "lifecycle": {"cleanup": {"confirmed": False}},
               "history_window": {"limit": 300, "total": 302, "truncated": True}}
        before = deepcopy(raw)
        data = monitor.project_snapshot(raw, "open-lab")
        self.assertEqual(data["evaluations"], rows[-monitor.MAX_ROWS:])
        data["evaluations"][0]["metrics"]["medqa"]["accuracy"] = 0
        data["lifecycle"]["cleanup"]["confirmed"] = True
        data["history_window"]["total"] = 0
        self.assertEqual(raw, before)
        oversized = self.evaluation(evaluation_id="x" * 1001, record_sha256="malformed")
        projected = self.project(evaluations=[oversized])["evaluations"][0]
        self.assertNotIn("evaluation_id", projected)
        self.assertNotIn("record_sha256", projected)

    def test_http_serves_only_public_snapshot_without_rewriting_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "open-lab"
            run.mkdir()
            raw = {"schema_version": 1, "status": "completed", "evaluations": [self.evaluation()],
                   "jobs": [{"job_id": "job-1", "status": "running", "observed_at": "reported"}],
                   "lifecycle": {"cleanup": {"confirmed": False}}}
            snapshot = run / "monitor.json"
            snapshot.write_text(json.dumps(raw), encoding="utf-8")
            # These deliberately contain no usable data and must never be consulted.
            (run / "result.json").write_text("not public evidence", encoding="utf-8")
            (run / "decision_memory.json").write_text("not public evidence", encoding="utf-8")
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            server = monitor.MonitorServer(("127.0.0.1", 0), root)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/runs/open-lab", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    data = json.load(response)
                self.assertEqual(data, monitor.project_snapshot(raw, "open-lab"))
                self.assertEqual(data["jobs"][0]["status"], "running")
                self.assertNotIn("finished_at", data["lifecycle"])
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
            self.assertEqual({path.name: path.read_bytes() for path in run.iterdir()}, before)

    def test_monitor_import_remains_independent_of_rsisys(self):
        source = Path(monitor.__file__).resolve().parents[1]
        code = (f"import sys; sys.path.insert(0, {str(source)!r}); "
                "from rsimonitor import server; "
                "assert not any(n == 'rsisys' or n.startswith('rsisys.') for n in sys.modules)")
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
