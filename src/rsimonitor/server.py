#!/usr/bin/env python3
"""Dependency-free, read-only RSIComposer monitor. Never serves the runs directory."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[A-Fa-f0-9]{64}\Z")
CONTRACT = json.loads(Path(__file__).with_name('monitor_contract.json').read_text())
MAX_BYTES = CONTRACT['max_bytes']
MAX_VERSION_TEXT = 32_000
MAX_ROWS = CONTRACT['max_rows']
MAX_METRIC_HISTORY = CONTRACT['max_metric_history']
MAX_DASHBOARD_PANELS = CONTRACT['max_dashboard_panels']
MAX_PANEL_METRICS = CONTRACT['max_panel_metrics']
MAX_METRIC_DEFINITIONS = CONTRACT['max_metric_definitions']
MAX_METRIC_POINTS = CONTRACT['max_metric_points']
MAX_EVENT_BYTES = CONTRACT['max_observation_event_bytes']
EVENT_ID = re.compile(r'[A-Za-z0-9_.-]{1,200}\Z')
EVENT_REF = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}\Z')
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app/monitor.js": ("app/monitor.js", "text/javascript; charset=utf-8"),
          "/app/monitor.css": ("app/monitor.css", "text/css; charset=utf-8"),
          "/lucide.min.js": ("lucide.min.js", "text/javascript; charset=utf-8")}
# Retired entry points keep working as links into the single app.
REDIRECTS = {"/legacy": "/", "/board": "/"}
DETAIL_KEYS = {"job_id", "candidate_id", "model_id", "revision", "base_revision",
               "role", "status", "kind", "step", "iteration", "duration_seconds",
               "metric", "value", "baseline", "candidate", "score",
               "decision_id", "call_id", "directive_id", "decision_revision", "actor_revision", "candidate_revision",
               "source", "action", "mode", "reason", "strategy", "round", "rounds", "phase",
               "protocol_ref", "recipe_ref", "method", "review_after_seconds", "next_review_seconds",
               "review_status", "total_steps", "elapsed_seconds", "completed_directives",
               "selection", "applied", "answer_correct", "training_review_count"}
WORKFLOW_FIELDS = ("phase", "round", "rounds", "completed_directives", "elapsed_seconds")
CURRENT_FIELDS = ("directive_id", "mode", "status", "state", "phase", "strategy", "decision_revision", "actor_revision",
                  "job_id", "elapsed_seconds", "review_after_seconds", "next_review_seconds", "review_status",
                  "step", "total_steps", "decision_id", "call_id", "operation", "started_at", "finished_at", "split", "model", "label", "round")
DECISION_FIELDS = ("decision_id", "call_id", "timestamp", "kind", "phase", "source", "round", "directive_id", "mode", "action",
                   "strategy", "reason", "status", "state", "decision_revision", "actor_revision", "job_id", "candidate_id",
                   "protocol_ref", "recipe_ref", "method", "review_after_seconds", "next_review_seconds", "operation", "parent_decision_id",
                   "feedback_summary", "error", "started_at", "finished_at", "step", "total_steps", "split", "model", "label")
ROUND_FIELDS = ("round", "base_revision", "candidate_revision", "selection", "actor_revision", "applied",
                "answer_correct", "training_review_count")
JOB_FIELDS = ("job_id", "kind", "status", "state", "role", "candidate_id", "directive_id", "round", "step", "total_steps", "elapsed_seconds")
PRIVATE_KEY = re.compile(CONTRACT['private_key_pattern'], re.I)


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
        item = reported_text(row, ("evaluation_id", "job_id", "decision_id", "call_id", "candidate_id", "protocol", "role", "evaluation_protocol"))
        item.update(reported_text(row, ("split", "model", "label", "timestamp", "benchmark"), 200))
        for key in ("round", "elapsed_seconds"):
            value = row.get(key)
            if type(value) in (int, float) and scalar(value) is not None and value >= 0:
                item[key] = value
        for key in ("record_sha256", "dataset_sha256", "evaluator_sha256", "protocol_sha256", "submission_sha256"):
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


def probes_only(value):
    if not isinstance(value, list):
        return []
    result = []
    for row in value[:100]:
        if not isinstance(row, dict) or not metric_text(row.get('name'), 100) or not metric_text(row.get('on'), 100):
            continue
        item = {'name': row['name'], 'on': row['on']}
        for key in ('every', 'calls', 'failures'):
            if type(row.get(key)) is int and 0 <= row[key] <= 2**53:
                item[key] = row[key]
        result.append(item)
    return result


def metric_definitions_only(value):
    """Project bounded display metadata; definitions never contain raw data."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, row in list(value.items())[:MAX_METRIC_DEFINITIONS]:
        if (not metric_text(name, 100) or not isinstance(row, dict)
                or not metric_text(row.get('label'))
                or row.get('unit') not in CONTRACT['metric_units']
                or row.get('direction') not in CONTRACT['metric_directions']):
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
    for key in ('timestamp', 'task_id', 'decision_id', 'call_id', 'attempt_id', 'observation_id', 'series'):
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
                or row.get('type') not in CONTRACT['panel_types']
                or row.get('x') not in CONTRACT['panel_x']
                or row.get('group_by') not in CONTRACT['panel_group_by']
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
            ("samples", CONTRACT["max_samples"], ("id", "decision_id", "call_id"), "id"),
            ("artifacts", CONTRACT["max_artifacts"], ("artifact_id", "kind", "decision_id", "call_id", "model"), "artifact_id"),
            ("logs", CONTRACT["max_logs"], ("timestamp", "level", "source", "decision_id", "call_id", "message"), None)):
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
        result["resources"].update(reported_text(resources, ("decision_id", "call_id"), 200))
        metrics = metrics_only(resources.get("latest_execution_metrics"))
        if metrics:
            result["resources"]["latest_execution_metrics"] = metrics
    return result


def version_history_only(raw):
    # IDs are opaque public tokens; contents, arbitrary metadata and paths stay private.
    def ident(value):
        return isinstance(value, str) and re.fullmatch(r'[\w.-]{1,200}', value) is not None

    def refs(values, limit):
        if isinstance(values, list) and len(values) <= limit and all(ident(v) for v in values):
            return list(dict.fromkeys(values))

    versions = {}
    rows = raw.get('versions')
    for row in rows[-MAX_ROWS:] if isinstance(rows, list) else ():
        if not isinstance(row, dict) or not ident(row.get('version_id')):
            continue
        key, label = row['version_id'], row.get('label')
        if not metric_text(label, 100) or '/' in label or '\\' in label:
            continue
        item = {'version_id': key, 'label': label}
        if type(row.get('round')) is int and 0 <= row['round'] <= 2**53:
            item['round'] = row['round']
        for name, limit in (('parent_ids', 8), ('evaluation_ids', MAX_ROWS), ('source_decision_ids', 24)):
            public = refs(row.get(name), limit)
            if public is not None and (name != 'parent_ids' or key not in public):
                item[name] = public
        versions[key] = item
    if not versions:
        return {}
    selections = {}
    rows = raw.get('version_selections')
    for row in rows[-MAX_ROWS:] if isinstance(rows, list) else ():
        if (isinstance(row, dict) and ident(row.get('version_id')) and type(row.get('step')) is int
                and 0 <= row['step'] <= 2**53):
            item = {'version_id': row['version_id'], 'step': row['step']}
            if all(EVENT_ID.fullmatch(row.get(key, '')) if isinstance(row.get(key), str) else False
                   for key in ('event_id', 'scope')) and row.get('action') in ('kept', 'selected', 'applied'):
                item.update(event_id=row['event_id'], scope=row['scope'], action=row['action'])
                identity = (row['scope'], row['event_id'])
            else:
                identity = (row['step'], row['version_id'])
            selections[identity] = item
    return {'versions': list(versions.values()), 'version_selections': list(selections.values())}


def public_event(row, engine_id):
    def ident(value):
        return isinstance(value, str) and EVENT_ID.fullmatch(value) is not None

    if (not isinstance(row, dict) or type(row.get('schema_version')) is not int
            or row['schema_version'] != CONTRACT['observation_event_schema'] or row.get('run_id') != engine_id
            or not ident(engine_id) or type(row.get('sequence')) is not int or not 0 < row['sequence'] <= 2**53
            or not all(ident(row.get(key)) for key in ('event_id', 'scope', 'kind'))
            or row.get('category') not in CONTRACT['observation_categories']
            or row.get('view') not in CONTRACT['observation_views']):
        return None
    item = pick(row, ('sequence', 'event_id', 'scope', 'kind', 'category', 'view', 'timestamp'))
    for key in ('subject', 'call_id'):
        if ident(row.get(key)):
            item[key] = row[key]
    if row.get('content_available') is True:
        item['content_available'] = True
    links = row.get('links')
    if isinstance(links, dict):
        item['links'] = {name: ids for name, ids in list(links.items())[:24] if ident(name)
                         and not PRIVATE_KEY.search(name) and isinstance(ids, list)
                         and len(ids) <= 128 and all(ident(value) for value in ids)}
    fields = row.get('fields')
    if isinstance(fields, dict):
        values = {}
        for name, value in list(fields.items())[:24]:
            if not ident(name) or PRIVATE_KEY.search(name):
                continue
            if (isinstance(value, bool) or isinstance(value, str) and EVENT_REF.fullmatch(value)
                    or type(value) in (int, float) and scalar(value) is not None):
                values[name] = value
            elif isinstance(value, dict):
                values[name] = {key: score for key, score in metrics_only(value).items() if type(score) in (int, float)}
        item['fields'] = values
    return item


def project_snapshot(raw, run_id):
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise SnapshotError(422, "Unsupported monitor schema; expected schema_version=1.")
    if "engine_run_id" in raw and raw.get("run_id") != run_id:
        raise SnapshotError(422, "Run snapshot ID does not match its directory.")
    result = pick(raw, ("objective", "status", "started_at", "updated_at", "engine_run_id"))
    if type(raw.get("simulated")) is bool:
        result["simulated"] = raw["simulated"]
    if metric_text(raw.get("primary_metric"), 100):
        result["primary_metric"] = raw["primary_metric"]
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
    result.update(version_history_only(raw))
    if raw.get('observation_status') == 'degraded':
        result['observation_status'] = 'degraded'
    if type(raw.get('observation_dropped')) is int and 0 < raw['observation_dropped'] <= 2**53:
        result['observation_dropped'] = raw['observation_dropped']
    events = raw.get('observation_events')
    if isinstance(events, list):
        result['observation_events'] = [event for row in events[-CONTRACT['max_observation_events']:]
                                        if (event := public_event(row, raw.get('engine_run_id'))) is not None]
    interpreted = attempt_interpretations(raw.get("research_attempts"))
    if interpreted is not None:
        result["research_attempts"] = interpreted
    if 'probes' in raw:
        result['probes'] = probes_only(raw['probes'])
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
                item.update(reported_text(row, ("observed_at", "decision_id", "call_id")))
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
        self._summaries = {}  # run_id -> ((mtime_ns, size, inode), summary row)

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

    @contextmanager
    def _file(self, run_id, filename, root_fd=None):
        if not RUN_ID.fullmatch(run_id):
            raise SnapshotError(404, "Run not found.")
        owned = root_fd is None
        if owned:
            root_fd = self.open_root()
        directory_fd = file_fd = None
        try:
            directory_fd = os.open(run_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
            file_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise SnapshotError(404, "Monitor snapshot not found.")
            with os.fdopen(file_fd, "rb") as stream:
                file_fd = None
                yield stream, info.st_size
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

    def _read_json(self, run_id, filename, root_fd=None):
        with self._file(run_id, filename, root_fd) as (stream, size):
            if size > MAX_BYTES or len(data := stream.read(MAX_BYTES + 1)) > MAX_BYTES:
                raise SnapshotError(413, "Monitor snapshot exceeds 4 MiB.")
            try:
                return json.loads(data, parse_constant=lambda _: None)
            except (ValueError, UnicodeError, RecursionError):
                raise SnapshotError(503, "Monitor snapshot is being written or is invalid; retry shortly.") from None

    def read(self, run_id, root_fd=None):
        return project_snapshot(self._read_json(run_id, 'monitor.json', root_fd), run_id)

    def events(self, run_id, cursor, category):
        raw = self._read_json(run_id, 'monitor.json')
        engine_id = raw.get('engine_run_id') if isinstance(raw, dict) else None
        if not isinstance(engine_id, str) or not EVENT_ID.fullmatch(engine_id) or raw.get('run_id') != run_id:
            raise SnapshotError(404, 'Event journal not found.')
        events, next_cursor = [], None
        try:
            with self._file(run_id, 'monitor-events.jsonl') as (stream, size):
                if cursor > size:
                    raise SnapshotError(400, 'Invalid event cursor.')
                if cursor:
                    stream.seek(cursor - 1)
                    if stream.read(1) != b'\n':
                        raise SnapshotError(400, 'Invalid event cursor.')
                while stream.tell() - cursor < 4 * MAX_EVENT_BYTES and len(events) < 50:
                    line = stream.readline(min(MAX_EVENT_BYTES + 1, size - stream.tell()))
                    if len(line) > MAX_EVENT_BYTES:
                        raise SnapshotError(422, 'Event journal record exceeds its byte limit.')
                    if not line.endswith(b'\n'):
                        break
                    try:
                        event = public_event(json.loads(line, parse_constant=lambda _: None), engine_id)
                    except (ValueError, RecursionError):
                        event = None
                    if event is not None and (category is None or event['category'] == category):
                        events.append(event)
                    if stream.tell() >= size:
                        break
                else:
                    if stream.tell() < size:
                        next_cursor = stream.tell()
        except SnapshotError as error:
            if error.status != 404:
                raise
        return {'schema_version': CONTRACT['observation_event_schema'], 'events': events, 'next_cursor': next_cursor}

    def version_content(self, run_id, version_id):
        if not EVENT_ID.fullmatch(version_id):
            raise SnapshotError(404, 'Version not found.')
        versions = {row['version_id']: row for row in self.read(run_id).get('versions', ())}
        legacy = None

        def components(key):
            nonlocal legacy
            if not EVENT_ID.fullmatch(key):
                return {}
            try:
                row = self._read_json(run_id, f'monitor-content.{key}.json')
            except SnapshotError as error:
                if error.status != 404:
                    raise
                if key not in versions:
                    return {}
                if legacy is None:
                    try:
                        legacy = self._read_json(run_id, 'monitor-version-content.json')
                    except SnapshotError as legacy_error:
                        if legacy_error.status != 404:
                            raise
                        legacy = {}
                row = legacy.get(key) if isinstance(legacy, dict) else None
            if not isinstance(row, dict) or len(row) > 8:
                return {}
            return {name: text for name, text in row.items()
                    if EVENT_ID.fullmatch(name) and isinstance(text, str)
                    and 0 < len(text) <= MAX_VERSION_TEXT and not re.search(r'[\ud800-\udfff]', text)}

        current = components(version_id)
        if not current:
            raise SnapshotError(404, 'No version content has been published.')
        return {'components': current, 'parents': [
            {'version_id': key, 'components': components(key)} for key in versions.get(version_id, {}).get('parent_ids', ())
        ]}

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
                    stamp = os.stat(f"{name}/monitor.json", dir_fd=root_fd, follow_symlinks=False)
                    identity = (stamp.st_mtime_ns, stamp.st_size, stamp.st_ino)
                except OSError:
                    identity = None
                cached = self._summaries.get(name)
                if identity is not None and cached is not None and cached[0] == identity:
                    rows.append(cached[1])
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
                    identity = None  # Retry unreadable or half-written snapshots on the next poll.
                if identity is not None:
                    self._summaries[name] = (identity, row)
                rows.append(row)
            for name in set(self._summaries) - {row["run_id"] for row in rows}:
                del self._summaries[name]
            return rows
        finally:
            os.close(root_fd)


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runs_dir):
        self.store = RunStore(runs_dir)
        super().__init__(address, MonitorHandler)


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "RSIComposerMonitor/1"

    def log_message(self, format, *args):
        # Do not log request paths, query strings, or monitoring contents.
        pass

    def respond(self, status, body, content_type="application/json; charset=utf-8", *, board=False):
        etag = None
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if status == 200:
                etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
                if self.headers.get("If-None-Match") == etag:
                    status, body = 304, b""
        self.send_response(status)
        if etag is not None:
            self.send_header("ETag", etag)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        style = "'self' 'unsafe-inline'" if board else "'self'"
        self.send_header("Content-Security-Policy", f"default-src 'none'; script-src 'self'; worker-src 'self' blob:; style-src {style}; connect-src 'self'; img-src 'self' data:; font-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
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
            url = urlsplit(self.path)
            path = unquote(url.path, errors="strict")
            if path in REDIRECTS:
                self.send_response(302)
                self.send_header("Location", REDIRECTS[path])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path in STATIC:
                filename, mime = STATIC[path]
                return self.respond(200, (STATIC_DIR / filename).read_bytes(), mime, board=path == "/")
            if path == "/api/schema":
                return self.respond(200, CONTRACT)
            if path == "/api/projects":
                return self.respond(200, {"schema_version": 1, "projects": self.server.store.projects()})
            if path == "/api/runs":
                return self.respond(200, {"schema_version": 1, "runs": self.server.store.list()})
            if path.startswith("/api/runs/"):
                suffix = path[len("/api/runs/"):]
                parts = suffix.split('/')
                if len(parts) == 2 and parts[1] == 'events':
                    query = parse_qs(url.query, strict_parsing=True, max_num_fields=2)
                    if set(query) - {'cursor', 'category'} or any(len(v) != 1 for v in query.values()):
                        raise ValueError('Invalid event query')
                    cursor = query.get('cursor', ['0'])[0]
                    category = query.get('category', [None])[0]
                    if not re.fullmatch(r'[0-9]{1,16}', cursor) or category not in (None, *CONTRACT['observation_categories']):
                        raise ValueError('Invalid event cursor or category')
                    return self.respond(200, self.server.store.events(parts[0], int(cursor), category))
                if len(parts) == 3 and parts[1] == 'versions':
                    if self.server.server_address[0] not in ('127.0.0.1', '::1'):
                        return self.respond(404, {'error': 'Not found.'})
                    return self.respond(200, self.server.store.version_content(parts[0], parts[2]))
                return self.respond(200, self.server.store.read(suffix))
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
    print(f"RSIComposer monitor: http://{args.host}:{server.server_port} (read-only)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
