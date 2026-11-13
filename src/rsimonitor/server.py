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
SHA256 = re.compile(r"[A-Fa-f0-9]{64}\Z")
MAX_BYTES = 4 * 1024 * 1024
MAX_ROWS = 300
MAX_METRIC_HISTORY = 1000
MAX_DASHBOARD_PANELS = 24
MAX_PANEL_METRICS = 12
MAX_METRIC_DEFINITIONS = 100
MAX_METRIC_POINTS = 10000
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC = {"/": ("board.html", "text/html; charset=utf-8"),
          "/legacy": ("index.html", "text/html; charset=utf-8"),
          "/board.js": ("board.js", "text/javascript; charset=utf-8"),
          "/live.js": ("live.js", "text/javascript; charset=utf-8"),
          "/lucide.min.js": ("lucide.min.js", "text/javascript; charset=utf-8"),
          "/board.css": ("board.css", "text/css; charset=utf-8"),
          "/i18n.js": ("i18n.js", "text/javascript; charset=utf-8"),
          "/decisions.js": ("decisions.js", "text/javascript; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}
DETAIL_KEYS = {"job_id", "candidate_id", "model_id", "revision", "base_revision",
               "role", "status", "kind", "step", "iteration", "duration_seconds",
               "metric", "value", "baseline", "candidate", "score",
               "decision_id", "directive_id", "decision_revision", "actor_revision", "candidate_revision",
               "source", "action", "mode", "reason", "strategy", "round", "rounds", "phase",
               "protocol_ref", "recipe_ref", "method", "review_after_seconds", "next_review_seconds",
               "review_status", "total_steps", "elapsed_seconds", "completed_directives",
               "selection", "applied", "answer_correct", "training_review_count"}
WORKFLOW_FIELDS = ("phase", "round", "rounds", "completed_directives", "elapsed_seconds")
CURRENT_FIELDS = ("directive_id", "mode", "status", "state", "phase", "strategy", "decision_revision", "actor_revision",
                  "job_id", "elapsed_seconds", "review_after_seconds", "next_review_seconds", "review_status",
                  "step", "total_steps", "decision_id", "operation", "started_at", "finished_at", "split", "model", "label", "round")
DECISION_FIELDS = ("decision_id", "timestamp", "kind", "source", "round", "directive_id", "mode", "action",
                   "strategy", "reason", "status", "state", "decision_revision", "actor_revision", "job_id", "candidate_id",
                   "protocol_ref", "recipe_ref", "method", "review_after_seconds", "next_review_seconds", "operation", "parent_decision_id",
                   "feedback_summary", "error", "started_at", "finished_at", "step", "total_steps", "split", "model", "label")
ROUND_FIELDS = ("round", "base_revision", "candidate_revision", "selection", "actor_revision", "applied",
                "answer_correct", "training_review_count")
JOB_FIELDS = ("job_id", "kind", "status", "state", "role", "candidate_id", "directive_id", "round", "step", "total_steps", "elapsed_seconds")
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


def metric_text(value, limit=200):
    """Validate public metric text without truncating names or identities."""
    return isinstance(value, str) and 0 < len(value) <= limit and bool(value.strip())


def reported_text(source, fields, limit=1000):
    """Optional public text; reject malformed identities instead of truncating them."""
    if not isinstance(source, dict):
        return {}
    return {key: source[key] for key in fields if metric_text(source.get(key), limit)}


def feedback_state_only(source):
    """Archiving raw feedback and updating memory are separate reported states."""
    result = reported_text(source, ("archive_status", "memory_status"))
    if isinstance(source, dict) and source.get("feedback_route") in ("context", "memory"):
        result["feedback_route"] = source["feedback_route"]
    return result


def lifecycle_only(source):
    if not isinstance(source, dict):
        return {}
    result = reported_text(source, ("finished_at", "stop_reason"))
    cleanup = source.get("cleanup")
    if isinstance(cleanup, dict) and type(cleanup.get("confirmed")) is bool:
        result["cleanup"] = {"confirmed": cleanup["confirmed"]}
    return result


def history_window_only(source):
    if not isinstance(source, dict):
        return {}
    result = {key: source[key] for key in ("limit", "total")
              if type(source.get(key)) is int and source[key] >= 0 and scalar(source[key]) is not None}
    if type(source.get("truncated")) is bool:
        result["truncated"] = source["truncated"]
    return result


def evaluations_only(source):
    """Reported evaluation records, never reconstructed from actions or summaries."""
    if not isinstance(source, list):
        return []
    result, seen = [], set()
    # Prefer the most recent report of an explicit immutable identity. Rows
    # without identity remain separate even when their scores happen to match.
    for row in reversed(source[-MAX_ROWS:]):
        if not isinstance(row, dict) or row.get("verified") is False:
            continue
        item = reported_text(row, ("evaluation_id", "job_id", "decision_id", "candidate_id", "protocol", "role"))
        item.update(reported_text(row, ("split", "model", "label", "timestamp"), 200))
        for key in ("round", "elapsed_seconds"):
            value = row.get(key)
            if type(value) in (int, float) and scalar(value) is not None and value >= 0:
                item[key] = value
        for key in ("record_sha256", "dataset_sha256", "evaluator_sha256"):
            if isinstance(row.get(key), str) and SHA256.fullmatch(row[key]):
                item[key] = row[key]
        datasets = row.get("dataset_sha256")
        if isinstance(datasets, dict):
            digests = {name: digest for name, digest in list(datasets.items())[:100]
                       if metric_text(name, 100) and not PRIVATE_KEY.search(name)
                       and isinstance(digest, str) and SHA256.fullmatch(digest)}
            if digests:
                item["dataset_sha256"] = digests
        if "metrics" in row:
            item["metrics"] = metrics_only(row["metrics"])
        if type(row.get("adopted")) is bool:
            item["adopted"] = row["adopted"]
        identity = row.get("weight_identity")
        if metric_text(identity, 1000):
            item["weight_identity"] = identity
        elif isinstance(identity, dict):
            weight = reported_text(identity, ("kind", "status", "backend_id", "model_id",
                                              "revision", "sha256", "digest"))
            if weight:
                item["weight_identity"] = weight
        key = (("record_sha256", item["record_sha256"].lower()) if "record_sha256" in item
               else ("evaluation_id", item["evaluation_id"]) if "evaluation_id" in item
               else ("job_id", item["job_id"]) if "job_id" in item else None)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        if item:
            result.append(item)
    return list(reversed(result))


def metric_definitions_only(value):
    """Project bounded display metadata; definitions never contain raw data."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, row in list(value.items())[:MAX_METRIC_DEFINITIONS]:
        if (not metric_text(name, 100) or not isinstance(row, dict)
                or not metric_text(row.get('label'))
                or row.get('unit') not in ('number', 'percent', 'seconds', 'tokens_per_second')
                or row.get('direction') not in ('higher', 'lower', 'neutral')):
            continue
        result[name] = {key: row[key] for key in ('label', 'unit', 'direction')}
    return result


def metric_point_only(row, metric_definitions=None):
    """Reject an invalid observation as a whole; never coerce failures to zero."""
    if not isinstance(row, dict) or not metric_text(row.get('metric'), 100):
        return None
    value = row.get('value')
    if type(value) not in (int, float) or scalar(value) is None:
        return None
    definition = (metric_definitions or {}).get(row['metric'], {})
    if definition.get('unit') == 'percent' and not 0 <= value <= 1:
        return None
    result = {'metric': row['metric'], 'value': value}
    for key in ('step', 'elapsed_seconds'):
        if key not in row:
            continue
        number = row[key]
        if (type(number) not in ((int,) if key == 'step' else (int, float))
                or scalar(number) is None or number < 0):
            return None
        result[key] = number
    for key in ('timestamp', 'task_id', 'decision_id', 'attempt_id'):
        if key in row:
            if not metric_text(row[key], 100 if key == 'timestamp' else 200):
                return None
            result[key] = row[key]
    return result


def metric_points_only(value, metric_definitions=None):
    """Retain recent reported points, without inferring coordinates or units."""
    if not isinstance(value, list):
        return []
    return [point for row in value[-MAX_METRIC_POINTS:]
            if (point := metric_point_only(row, metric_definitions)) is not None]


def dashboard_only(value):
    """Project a lightweight layout. Unsupported dashboard versions are omitted."""
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
            or value['schema_version'] != 1 or not isinstance(value.get('panels'), list)):
        return None
    panels, seen_ids = [], set()
    for row in value['panels'][:MAX_DASHBOARD_PANELS]:
        if (not isinstance(row, dict) or not metric_text(row.get('id'))
                or not metric_text(row.get('title'))
                or row.get('type') not in ('line', 'scalar', 'table')
                or row.get('x') not in ('step', 'elapsed_seconds', 'evaluation')
                or row.get('group_by') not in ('task', 'attempt', 'none')
                or not isinstance(row.get('metrics'), list)):
            continue
        metrics = row['metrics'][:MAX_PANEL_METRICS]
        if not all(metric_text(name, 100) for name in metrics):
            continue
        if row['id'] in seen_ids:
            continue
        seen_ids.add(row['id'])
        panel = {**{key: row[key] for key in ('id', 'title', 'type', 'x', 'group_by')},
                 'metrics': list(dict.fromkeys(metrics))}
        layout = row.get('layout')
        if (isinstance(layout, dict) and type(layout.get('w')) is int
                and type(layout.get('h')) is int
                and 3 <= layout['w'] <= 12 and 240 <= layout['h'] <= 720):
            panel['layout'] = {key: layout[key] for key in ('w', 'h')}
        y_range = row.get('y_range')
        if isinstance(y_range, dict):
            bounds = {key: y_range[key] for key in ('min', 'max') if key in y_range}
            if bounds and all(type(number) in (int, float) and scalar(number) is not None
                              for number in bounds.values()):
                span = bounds['max'] - bounds['min'] if len(bounds) == 2 else None
                if len(bounds) == 1 or (scalar(span) is not None and span > 0):
                    panel['y_range'] = bounds
        panels.append(panel)
    return {'schema_version': 1, 'panels': panels}


def run_project(raw, run_id):
    """Only a valid snapshot project ID establishes membership, never run identity."""
    project = raw.get("project")
    project = project if isinstance(project, dict) else {}
    project_id = project.get("project_id")
    if not isinstance(project_id, str) or not RUN_ID.fullmatch(project_id) or project_id == "unassigned":
        return {"project_id": "unassigned", "name": "未归属"}
    result = pick(project, ("description", "system_id"))
    name = next((value.strip()[:1000] for value in (project.get("name"), raw.get("project_name"))
                 if isinstance(value, str) and value.strip()), project_id)
    result.update(project_id=project_id, name=name)
    return result


def experiment_name(raw, run_id):
    project = raw.get("project")
    legacy_name = project.get("name") if isinstance(project, dict) and project.get("project_id") == run_id else None
    return next((value.strip()[:1000] for value in (raw.get("experiment_name"), raw.get("display_name"), raw.get("name"), legacy_name)
                 if isinstance(value, str) and value.strip()), run_id)


def attempt_interpretations(raw):
    """Bounded public projection; interpretation is distinct from execution truth."""
    if not isinstance(raw, dict) or raw.get('schema_version') != 1 or raw.get('authority') != 'model_interpretation':
        return None

    def strings(values):
        return [v[:1000] for v in values[:10000] if isinstance(v, str)] if isinstance(values, list) else []

    def items(values):
        projected = []
        for row in values[:MAX_ROWS] if isinstance(values, list) else []:
            if not isinstance(row, dict) or not isinstance(row.get('attempt_id'), str) or not row['attempt_id'].strip():
                continue
            item = {key: row[key][:limit] for key, limit in (('attempt_id', 200), ('title', 500), ('summary', 4000)) if isinstance(row.get(key), str)}
            item['member_decision_ids'] = strings(row.get('member_decision_ids'))
            item['source_refs'] = strings(row.get('source_refs'))
            if type(row.get('revision')) is int:
                item['revision'] = row['revision']
            if type(row.get('simulated')) is bool:
                item['simulated'] = row['simulated']
            provenance = row.get('provenance')
            if isinstance(provenance, dict):
                item['provenance'] = pick(provenance, ('job_id', 'input_digest'))
                item['provenance']['model'] = pick(provenance.get('model'), ('backend_id', 'model_id', 'revision'))
            projected.append(item)
        return projected

    visible = items(raw.get('items'))
    visible_ids = {row['attempt_id'] for row in visible}
    latest = {(row['attempt_id'], row.get('revision')) for row in visible}
    revisions = raw.get('revisions')
    eligible = [(index, row) for index, row in enumerate(revisions if isinstance(revisions, list) else [])
                if isinstance(row, dict) and isinstance(row.get('attempt_id'), str)
                and row['attempt_id'] in visible_ids and type(row.get('revision')) is int]
    # The journal is grouped by attempt ID, not by time. Preserve the exact
    # current revisions of visible attempts before filling their history window;
    # taking the first 300 entries could hide every recent revision of one idea.
    selected = {index for index, row in eligible if (row['attempt_id'], row['revision']) in latest}
    if len(selected) > MAX_ROWS:
        selected = set(sorted(selected)[-MAX_ROWS:])
    for index, _ in reversed(eligible):
        if len(selected) >= MAX_ROWS:
            break
        selected.add(index)
    history = items([row for index, row in eligible if index in selected])
    return {'schema_version': 1, 'authority': 'model_interpretation',
            'items': visible, 'revisions': history,
            'ungrouped_decision_ids': strings(raw.get('ungrouped_decision_ids'))}


BOARD_CONFIG_FIELDS = ("method", "seed", "rounds", "plain_rounds", "warmup_steps", "max_seconds",
                       "max_decision_steps", "runner_round_limit", "dev_count", "test_count", "train_count",
                       "search_count", "validation_count", "model_id", "condition", "candidates_per_round",
                       "minibatch_size", "learning_rate", "global_batch", "max_length")
BOARD_RESOURCE_FIELDS = ("allocated_gpu_count", "model_calls_attempted", "model_call_limit")
BOARD_ID = re.compile(r"[\w.-]{1,200}\Z")


def board_rows(raw, key, limit, text_fields, identity=None):
    value = raw.get(key)
    if not isinstance(value, list):
        return []
    result = []
    for row in value[-limit:]:
        if not isinstance(row, dict):
            continue
        item = reported_text(row, text_fields, 200 if key != "logs" else 1000)
        if identity and not BOARD_ID.fullmatch(item.get(identity, "")):
            continue
        if key == "samples":
            item.update({field: row[field] for field in ("correct", "parsed") if type(row.get(field)) is bool})
            for field in ("predicted", "expected"):
                answer = row.get(field)
                if isinstance(answer, str) and re.fullmatch(r"-?[0-9./]{1,80}", answer):
                    item[field] = answer
        if item:
            result.append(item)
    return result


def board_fields(raw):
    result = {}
    for key, limit, text_fields, identity in (
            ("samples", 500, ("id", "decision_id"), "id"),
            ("artifacts", MAX_ROWS, ("artifact_id", "kind", "decision_id", "model"), "artifact_id"),
            ("logs", 400, ("timestamp", "level", "source", "decision_id", "message"), None)):
        if isinstance(raw.get(key), list):
            result[key] = board_rows(raw, key, limit, text_fields, identity)
    config = raw.get("config")
    if isinstance(config, dict):
        result["config"] = {
            key: value for key, value in pick(config, BOARD_CONFIG_FIELDS).items()
            if type(value) in (int, float) or isinstance(value, str) and BOARD_ID.fullmatch(value)
        }
    resources = raw.get("resources")
    if isinstance(resources, dict):
        result["resources"] = {
            key: value for key, value in pick(resources, BOARD_RESOURCE_FIELDS).items() if type(value) is int and value >= 0
        }
        result["resources"].update(reported_text(resources, ("decision_id",), 200))
        metrics = metrics_only(resources.get("latest_execution_metrics"))
        if metrics:
            result["resources"]["latest_execution_metrics"] = metrics
    return result


def project_snapshot(raw, run_id):
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise SnapshotError(422, "Unsupported monitor schema; expected schema_version=1.")
    if "engine_run_id" in raw and raw.get("run_id") != run_id:
        raise SnapshotError(422, "Run snapshot ID does not match its directory.")
    result = pick(raw, ("objective", "status", "started_at", "updated_at", "engine_run_id"))
    if type(raw.get("simulated")) is bool:
        result["simulated"] = raw["simulated"]
    result.update(board_fields(raw))
    result.update(schema_version=1, run_id=run_id, experiment_name=experiment_name(raw, run_id))
    if type(raw.get("weight_applied")) is bool:
        result["weight_applied"] = raw["weight_applied"]
    if isinstance(raw.get("parent_run_id"), str):
        result["parent_run_id"] = raw["parent_run_id"][:128]
    if isinstance(raw.get("source_run_ids"), list):
        result["source_run_ids"] = [v[:128] for v in raw["source_run_ids"][:100] if isinstance(v, str)]
    for key, projection in (("lifecycle", lifecycle_only), ("history_window", history_window_only)):
        value = projection(raw.get(key))
        if value:
            result[key] = value
    if isinstance(raw.get("evaluations"), list):
        result["evaluations"] = evaluations_only(raw["evaluations"])
    interpreted = attempt_interpretations(raw.get("research_attempts"))
    if interpreted is not None:
        result["research_attempts"] = interpreted
    definitions = metric_definitions_only(raw.get('metric_definitions'))
    if 'metric_definitions' in raw:
        result['metric_definitions'] = definitions
    if 'metric_points' in raw:
        result['metric_points'] = metric_points_only(raw['metric_points'], definitions)
    dashboard = dashboard_only(raw.get('dashboard'))
    if dashboard is not None:
        result['dashboard'] = dashboard
    result["project"] = run_project(raw, run_id)
    result["loops"] = {key: pick(raw.get("loops", {}).get(key), ("status", "summary"))
                       for key in ("decision_making", "exploration")} if isinstance(raw.get("loops"), dict) else {}
    result["models"] = {key: pick(raw.get("models", {}).get(key), ("model_id", "revision"))
                        for key in ("decision_maker", "actor")} if isinstance(raw.get("models"), dict) else {}
    result["workflow"] = pick(raw.get("workflow"), WORKFLOW_FIELDS)
    result["current"] = pick(raw.get("current"), CURRENT_FIELDS)
    result["current"].update(feedback_state_only(raw.get("current")))
    if isinstance(raw.get("current"), dict) and "metrics" in raw["current"]:
        result["current"]["metrics"] = metrics_only(raw["current"]["metrics"])
    for key, fields in (("jobs", JOB_FIELDS),
                        ("candidates", ("candidate_id", "status", "base_revision", "revision", "directive_id")),
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
            if key == "jobs":
                item.update(reported_text(row, ("observed_at", "decision_id")))
            if key == "decisions" and isinstance(row.get("changes"), list):
                # Explicit public action summaries only; never expose arbitrary payloads.
                item["changes"] = []
                for change in row["changes"][:20]:
                    if not isinstance(change, dict) or not isinstance(change.get("field"), str):
                        continue
                    values = {key: change[key][:4000] for key in ("before", "after")
                              if isinstance(change.get(key), str)}
                    if values:
                        item["changes"].append({"field": change["field"][:100], **values})
            if key == "decisions":
                item.update(feedback_state_only(row))
                attempt_id, title = row.get("research_attempt_id"), row.get("research_attempt_title")
                if (isinstance(attempt_id, str) and 0 < len(attempt_id) <= 200 and attempt_id.strip()
                        and isinstance(title, str) and title.strip()):
                    item.update(research_attempt_id=attempt_id, research_attempt_title=title[:500])
                for relation in ("depends_on_decision_ids", "adopted_decision_ids", "source_decision_ids", "adopted_candidate_ids"):
                    if isinstance(row.get(relation), list):
                        item[relation] = [v for v in row[relation][:100]
                                          if isinstance(v, str) and 0 < len(v) <= 200]
            result[key].append(item)
    return result


class RunStore:
    def __init__(self, root):
        # Resolve the explicitly configured root once; every child uses dir_fd +
        # O_NOFOLLOW, so replacing directories/files with links cannot escape it.
        self.root = Path(root).expanduser().resolve()

    def projects(self):
        # list() sorts by authoritative directory ID. First run supplies metadata;
        # the first named run replaces an ID-only name, but never another name.
        projects = {}
        for run in self.list():
            metadata = run["project"]
            project_id = metadata["project_id"]
            if project_id not in projects:
                projects[project_id] = {**metadata, "runs": []}
            project = projects[project_id]
            if project["name"] == project_id and metadata["name"] != project_id:
                project["name"] = metadata["name"]
            project["runs"].append(run)
        return [projects[key] for key in sorted(projects)]

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
                    row = pick(snapshot, ("schema_version", "run_id", "experiment_name", "objective", "status", "started_at", "updated_at", "engine_run_id", "simulated"))
                    row["project"] = snapshot["project"]
                    for field in ("parent_run_id", "source_run_ids"):
                        if field in snapshot:
                            row[field] = snapshot[field]
                    row["counts"] = {key: len(snapshot[key]) for key in ("decisions", "jobs", "candidates", "events")}
                except SnapshotError as error:
                    if error.status == 404:
                        continue  # Ignore unrelated directories and runs without a snapshot.
                    row = {"run_id": name, "experiment_name": name, "status": "unavailable", "error": error.message,
                           "project": run_project({}, name)}
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

    def respond(self, status, body, content_type="application/json; charset=utf-8", *, board=False):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        style = "'self' 'unsafe-inline'" if board else "'self'"
        self.send_header("Content-Security-Policy", f"default-src 'none'; script-src 'self'; style-src {style}; connect-src 'self'; img-src 'self' data:; font-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
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
                return self.respond(200, (STATIC_DIR / filename).read_bytes(), mime, board=path == "/")
            if path == "/api/projects":
                return self.respond(200, {"schema_version": 1, "projects": self.server.store.projects()})
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Run directories containing monitor.json (default: ./runs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
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
