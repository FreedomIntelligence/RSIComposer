#!/usr/bin/env python3
"""Dependency-free, read-only RSISys monitor. Never serves the runs directory."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
MAX_BYTES = 4 * 1024 * 1024
MAX_ROWS = 300
MAX_METRIC_HISTORY = 1000
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/demo.json": ("demo.json", "application/json; charset=utf-8")}
DETAIL_KEYS = {"job_id", "candidate_id", "model_id", "revision", "base_revision",
               "role", "status", "kind", "step", "iteration", "duration_seconds",
               "metric", "value", "baseline", "candidate", "score",
               "decision_id", "directive_id", "decision_revision", "actor_revision", "candidate_revision",
               "source", "action", "mode", "reason", "strategy", "round", "rounds", "phase",
               "protocol_ref", "recipe_ref", "method", "review_after_seconds", "next_review_seconds",
               "review_status", "total_steps", "elapsed_seconds", "completed_directives",
               "selection", "applied", "answer_correct", "training_review_count"}
WORKFLOW_FIELDS = ("phase", "round", "rounds", "completed_directives", "elapsed_seconds")
CURRENT_FIELDS = ("directive_id", "mode", "status", "phase", "strategy", "decision_revision", "actor_revision",
                  "job_id", "elapsed_seconds", "review_after_seconds", "next_review_seconds", "review_status",
                  "step", "total_steps")
DECISION_FIELDS = ("decision_id", "timestamp", "kind", "source", "round", "directive_id", "mode", "action",
                   "strategy", "reason", "status", "decision_revision", "actor_revision", "job_id", "candidate_id",
                   "protocol_ref", "recipe_ref", "method", "review_after_seconds", "next_review_seconds", "operation")
ROUND_FIELDS = ("round", "base_revision", "candidate_revision", "selection", "actor_revision", "applied",
                "answer_correct", "training_review_count")
JOB_FIELDS = ("job_id", "kind", "status", "directive_id", "round", "step", "total_steps", "elapsed_seconds")
PRIVATE_KEY = re.compile(r"secret|password|credential|api.?key|authorization|token(?!_count)|config|prompt|response|payload|sample|dataset|training_data", re.I)


class SnapshotError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def scalar(value, limit=1000):
    if isinstance(value, str):
        return value[:limit]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return value if math.isfinite(value) else None
        except OverflowError:
            return None
    return None


def pick(source, keys):
    if not isinstance(source, dict):
        return {}
    return {key: scalar(source[key]) for key in keys if key in source}


def metrics_only(value, depth=0):
    """Bounded scalar metrics; never pass through arbitrary monitor attachments."""
    if depth > 4 or not isinstance(value, dict):
        return {}
    result = {}
    for key, item in list(value.items())[:100]:
        if not isinstance(key, str) or len(key) > 100:
            continue
        count_metric = key in {"eval_samples", "num_samples", "sample_count"} and isinstance(item, (int, float))
        if PRIVATE_KEY.search(key) and not count_metric:
            continue
        if isinstance(item, dict):
            result[key] = metrics_only(item, depth + 1)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            number = scalar(item)
            if number is not None:
                result[key] = number
    return result


def metric_history_only(value):
    """Public curve protocol: sparse observations, never poll-generated points."""
    if not isinstance(value, list):
        return []
    result, seen = [], set()
    for row in value[-MAX_METRIC_HISTORY:]:
        if not isinstance(row, dict):
            continue
        step, elapsed = row.get('step'), row.get('elapsed_seconds')
        if (type(step) is not int or scalar(step) is None or step < 0 or type(elapsed) not in (int, float)
                or scalar(elapsed) is None or elapsed < 0):
            continue
        metrics = {key: number for key, number in metrics_only(row.get('metrics')).items()
                   if key.strip() and type(number) in (int, float)}
        if not metrics:
            continue
        identity = (step, elapsed, tuple(sorted(metrics.items())))
        if identity in seen:
            continue
        seen.add(identity)
        result.append({'step': step, 'elapsed_seconds': elapsed, 'metrics': metrics})
    return result


def project_snapshot(raw, run_id):
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise SnapshotError(422, "Unsupported monitor schema; expected schema_version=1.")
    result = pick(raw, ("objective", "status", "updated_at"))
    result.update(schema_version=1, run_id=run_id)
    result["loops"] = {key: pick(raw.get("loops", {}).get(key), ("status", "summary"))
                       for key in ("decision_making", "exploration")} if isinstance(raw.get("loops"), dict) else {}
    result["models"] = {key: pick(raw.get("models", {}).get(key), ("model_id", "revision"))
                        for key in ("decision_maker", "actor")} if isinstance(raw.get("models"), dict) else {}
    result["workflow"] = pick(raw.get("workflow"), WORKFLOW_FIELDS)
    result["current"] = pick(raw.get("current"), CURRENT_FIELDS)
    if isinstance(raw.get("current"), dict) and "metrics" in raw["current"]:
        result["current"]["metrics"] = metrics_only(raw["current"]["metrics"])
    for key, fields in (("jobs", JOB_FIELDS),
                        ("candidates", ("candidate_id", "status", "base_revision", "revision")),
                        ("events", ("timestamp", "kind", "summary")),
                        ("decisions", DECISION_FIELDS), ("round_results", ROUND_FIELDS)):
        result[key] = []
        rows = raw.get(key)
        if not isinstance(rows, list):
            continue
        # Retain recent rows; the file byte limit also bounds parsing work.
        for row in rows[-MAX_ROWS:]:
            if not isinstance(row, dict):
                continue
            item = pick(row, fields)
            if key in ("candidates", "round_results") or (key in ("jobs", "decisions") and "metrics" in row):
                item["metrics"] = metrics_only(row.get("metrics"))
            elif key == "events":
                item["details"] = pick(row.get("details"), DETAIL_KEYS)
            if key == "jobs" and "metric_history" in row:
                item["metric_history"] = metric_history_only(row["metric_history"])
            result[key].append(item)
    return result


class RunStore:
    def __init__(self, root):
        # Resolve the explicitly configured root once; every child uses dir_fd +
        # O_NOFOLLOW, so replacing directories/files with links cannot escape it.
        self.root = Path(root).expanduser().resolve()

    def open_root(self):
        try:
            return os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise SnapshotError(404, "Runs directory is not available yet.") from None
        except OSError:
            raise SnapshotError(503, "Runs directory is unavailable.") from None

    def read(self, run_id, root_fd=None):
        if not RUN_ID.fullmatch(run_id):
            raise SnapshotError(404, "Run not found.")
        owned = root_fd is None
        if owned:
            root_fd = self.open_root()
        directory_fd = file_fd = None
        try:
            directory_fd = os.open(run_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
            file_fd = os.open("monitor.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise SnapshotError(404, "Monitor snapshot not found.")
            if info.st_size > MAX_BYTES:
                raise SnapshotError(413, "Monitor snapshot exceeds 4 MiB.")
            with os.fdopen(file_fd, "rb") as stream:
                file_fd = None
                data = stream.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise SnapshotError(413, "Monitor snapshot exceeds 4 MiB.")
            try:
                raw = json.loads(data, parse_constant=lambda _: None)
            except (ValueError, UnicodeError, RecursionError):
                raise SnapshotError(503, "Monitor snapshot is being written or is invalid; retry shortly.") from None
            return project_snapshot(raw, run_id)
        except FileNotFoundError:
            raise SnapshotError(404, "Monitor snapshot not found.") from None
        except OSError:
            raise SnapshotError(404, "Monitor snapshot is not accessible.") from None
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if directory_fd is not None:
                os.close(directory_fd)
            if owned:
                os.close(root_fd)

    def list(self):
        try:
            root_fd = self.open_root()
        except SnapshotError as error:
            if error.status == 404:
                return []
            raise
        try:
            rows = []
            for name in sorted(os.listdir(root_fd)):
                if not RUN_ID.fullmatch(name):
                    continue
                try:
                    info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISDIR(info.st_mode):
                    continue
                try:
                    snapshot = self.read(name, root_fd)
                    row = pick(snapshot, ("schema_version", "run_id", "objective", "status", "updated_at"))
                    row["counts"] = {key: len(snapshot[key]) for key in ("jobs", "candidates", "events")}
                except SnapshotError as error:
                    if error.status == 404:
                        continue  # Ignore unrelated directories and runs without a snapshot.
                    row = {"run_id": name, "status": "unavailable", "error": error.message}
                rows.append(row)
            return rows
        finally:
            os.close(root_fd)


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runs_dir):
        self.store = RunStore(runs_dir)
        super().__init__(address, MonitorHandler)


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "RSISysMonitor/1"

    def log_message(self, format, *args):
        # Do not log request paths, query strings, or monitoring contents.
        pass

    def respond(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; font-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        if status == 503:
            self.send_header("Retry-After", "2")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        try:
            host = urlsplit("//" + self.headers.get("Host", "")).hostname
            allowed = {"localhost", "127.0.0.1", "::1", self.server.server_address[0]}
            if host not in allowed or self.headers.get("Sec-Fetch-Site") == "cross-site":
                return self.respond(403, {"error": "Local same-origin requests only."})
            path = unquote(urlsplit(self.path).path, errors="strict")
            if path in STATIC:
                filename, mime = STATIC[path]
                return self.respond(200, (STATIC_DIR / filename).read_bytes(), mime)
            if path == "/api/runs":
                return self.respond(200, {"schema_version": 1, "runs": self.server.store.list()})
            if path.startswith("/api/runs/"):
                return self.respond(200, self.server.store.read(path[len("/api/runs/"):]))
            self.respond(404, {"error": "Not found."})
        except SnapshotError as error:
            self.respond(error.status, {"error": error.message})
        except (ValueError, UnicodeError):
            self.respond(400, {"error": "Invalid request."})
        except OSError:
            self.respond(503, {"error": "Monitor is temporarily unavailable."})

    def do_HEAD(self):
        self.do_GET()

    def reject_write(self):
        self.respond(405, {"error": "This monitor is read-only."})

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = reject_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Run directories containing monitor.json (default: ./runs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = MonitorServer((args.host, args.port), args.runs_dir)
    print(f"RSISys monitor: http://{args.host}:{server.server_port} (read-only)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
