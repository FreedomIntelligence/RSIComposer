"""Optional native AIDE runtime. No generated program executes in this process.

Resume: after setup and after every Agent.step the worker atomically writes a
snapshot (native Journal serialization, global RNG state, data preview, metric-
augmented task, usage) to resume/snapshot-NNNN.json. Model replies inside the
following step are kept as chained replay records in resume/step-NNNN/. Every
execute event carries the resume point (snapshot, its hash, replies, chain). A
worker started with config['resume'] reloads that snapshot with aideml's own
loads_json, restores the RNG, and replays the recorded replies (and the RNG
state after each) until upstream Agent.step re-emits the identical execute
request; from there it runs live, discarding exchanges beyond the checkpoint.
"""
import json
import os
import random
import sys
import traceback
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from rsicomposer.algorithms.aide.controller import AIDE_REVISION, valid_resume
from rsicomposer.api import (UpstreamWorker, atomic_write, checkout, emit_event, emit_observation,
                             file_hash as digest, observation_id)

SNAPSHOT_FORMAT = 'rsisys-aide-snapshot/v3'


def worker_identity(config):
    """Pin research inputs while allowing the run-owned directory to relocate."""
    return UpstreamWorker.identity(config, protocol='rsisys-aide-worker/v3')


def rng_state():
    version, internal, gauss = random.getstate()
    return [version, list(internal), gauss]


def set_rng_state(state):
    random.setstate((state[0], tuple(state[1]), state[2]))


class ResumeStore:
    """Step-boundary snapshots and per-step reply logs under the run's resume/ directory."""
    def __init__(self, root):
        self.root = Path(root) / 'resume'

    def snapshot_path(self, steps):
        return self.root / f'snapshot-{steps:04d}.json'

    def step_log(self, steps, snapshot_sha, point=None):
        directory = self.root / f'step-{steps:04d}'
        return UpstreamWorker.open_replay(directory, snapshot_sha, point)

    def write(self, steps, payload):
        data = json.dumps({'format': SNAPSHOT_FORMAT, 'revision': AIDE_REVISION, 'steps': steps,
                           **payload}).encode()
        atomic_write(self.snapshot_path(steps), data)
        return sha256(data).hexdigest()

    def read(self, steps, expected):
        path = self.snapshot_path(steps)
        data = path.read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'AIDE snapshot changed after the checkpoint: {path}')
        snapshot = json.loads(data)
        expected_fields = {'format', 'revision', 'steps', 'identity', 'journal', 'rng', 'data_preview', 'task', 'usage'}
        if (set(snapshot) != expected_fields or snapshot.get('format') != SNAPSHOT_FORMAT or snapshot.get('revision') != AIDE_REVISION
                or snapshot.get('steps') != steps):
            raise ValueError(f'Unsupported AIDE snapshot: {path}')
        return snapshot


def prepare_preview(files, preview):
    """Materialize only the public file contents granted by the Lab receipt."""
    if not isinstance(files, dict) or set(files) != {'train.csv', 'validation.csv'}:
        raise ValueError('AIDE preview requires exactly the Lab-approved public files')
    if any(not isinstance(content, str) for content in files.values()):
        raise TypeError('AIDE preview content must be UTF-8 text')
    expected = {name: sha256(content.encode()).hexdigest() for name, content in files.items()}
    if preview.is_symlink():
        raise ValueError('AIDE preview directory may not be a symlink')
    if preview.exists():
        if (any(not path.is_file() or path.is_symlink() for path in preview.iterdir())
                or {path.name: digest(path) for path in preview.iterdir()} != expected):
            raise ValueError('AIDE preview differs from its Lab-provided artifacts')
        return
    preview.mkdir(parents=True)
    for name, content in files.items():
        (preview / name).write_bytes(content.encode())


def normalize_review_metric(result):
    # JSON integers satisfy the number schema, but aideml's reviewer requires float.
    output = result[0]
    if isinstance(output, dict) and type(output.get('metric')) is int:
        return ({**output, 'metric': float(output['metric'])}, *result[1:])
    return result


def run(config, channel=None):
    channel = channel or UpstreamWorker()
    upstream = checkout(config['upstream_root'], AIDE_REVISION)
    sys.path.insert(0, str(upstream))
    import aide.backend as backend
    from aide.agent import Agent, add_task_metric, determine_task_metric
    from aide.interpreter import ExecutionResult
    from aide.journal import Journal
    from aide.utils.config import _load_cfg
    from aide.utils.serialize import dumps_json, loads_json
    from jsonschema import ValidationError

    resume = config.get('resume')
    if resume is not None and not valid_resume(resume):
        raise ValueError('Invalid AIDE resume point')
    # Local chat transport only; native prompts, review schema, parsing, search
    # and metric-based selection are unchanged.
    native_query = backend.provider_to_query_func['openai']
    # Upstream GPT-family models select its Responses client and otherwise
    # ignore OPENAI_BASE_URL. Bind that client to the same host gateway too.
    from aide.backend import backend_openai
    backend_openai.OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', backend_openai.OPENAI_BASE_URL)
    calls = []
    root = Path(config['root'])
    store = ResumeStore(root)
    current = {'log': None}

    def query(**kwargs):
        # Qwen's chat template requires a user turn. Preserve the complete
        # upstream prompt text while representing its system-only query as a
        # user turn; no experiment hints or answer content are injected.
        if kwargs.get('system_message') is not None and kwargs.get('user_message') is None:
            kwargs['user_message'] = kwargs.pop('system_message')
            kwargs['system_message'] = None
        if config.get('disable_thinking', True):
            kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
        kwargs['timeout'] = config.get('timeout', 600)
        trace = {key: value for key, value in kwargs.items() if key != 'func_spec'}
        function = kwargs.get('func_spec')
        if function is not None:
            trace['function'] = {'name': function.name, 'json_schema': function.json_schema}
        log = current['log']
        recorded = log.take('model', trace) if log is not None else None
        if recorded is not None:
            # Same reply and the RNG exactly as it was after the original call
            # (upstream retries may draw backoff jitter from the global RNG).
            result = tuple(recorded['result'])
            set_rng_state(recorded['rng'])
            calls.append({'input_tokens': result[2], 'output_tokens': result[3]})
            return result
        try:
            for attempt in range(3):
                try:
                    result = native_query(**kwargs)
                    if function is not None and function.name == 'submit_review':
                        result = normalize_review_metric(result)
                    break
                except ValidationError as error:
                    if function is None or attempt == 2:
                        raise
                    field = '.'.join(map(str, error.absolute_path)) or 'arguments'
                    key = 'user_message' if kwargs.get('user_message') is not None else 'system_message'
                    kwargs = {**kwargs, key: kwargs[key] + (
                        f'\nThe previous tool reply failed schema validation at {field}. '
                        'Return the same required tool using the declared types. When a field is unknown '
                        'and nullable, use the literal null, not the string "None".')}
        except Exception as exc:
            trace['error'] = type(exc).__name__ + ': ' + str(exc)
            with (root / 'llm-calls.jsonl').open('a') as stream:
                stream.write(json.dumps(trace) + '\n')
            raise
        if log is not None:
            log.record('model', {key: value for key, value in trace.items()},
                       {'result': list(result), 'rng': rng_state()})
        trace.update(output=result[0], elapsed_seconds=result[1], input_tokens=result[2],
                     output_tokens=result[3], provider_info=result[4])
        with (root / 'llm-calls.jsonl').open('a') as stream:
            stream.write(json.dumps(trace) + '\n')
        calls.append({'input_tokens': result[2], 'output_tokens': result[3]})
        (root / 'llm-usage.json').write_text(json.dumps(calls, indent=2))
        return result
    backend.provider_to_query_func['openai'] = query
    cfg = _load_cfg(use_cli_args=False)
    cfg.workspace_dir = str(root / 'preview')
    prepare_preview(config['preview_files'], Path(cfg.workspace_dir) / 'input')
    native = config.get('native_config', {})
    cfg.agent.steps = config.get('iterations', 3)
    cfg.exec.timeout = config.get('execution_timeout', 300)
    cfg.agent.k_fold_validation = native.get('k_fold_validation', 1)
    cfg.agent.data_preview = native.get('data_preview', True)
    cfg.agent.metric_maximize = native.get('metric_maximize', True)
    for key in ('num_drafts', 'debug_prob', 'max_debug_depth'):
        if key in native:
            setattr(cfg.agent.search, key, native[key])
    for stage in (cfg.agent.code, cfg.agent.feedback):
        stage.model = config['model']
        stage.provider = 'openai'
        stage.temp = native.get('temperature', 0.7)
        stage.max_tokens = config.get('max_tokens', 8192)

    def snapshot(agent, journal):
        return store.write(len(journal.nodes), {
            'identity': worker_identity(config), 'journal': dumps_json(journal), 'rng': rng_state(), 'data_preview': agent.data_preview,
            'task': agent.task_desc, 'usage': deepcopy(calls)})

    if resume is None:
        random.seed(config.get('seed', 73))
        task = Path(config['task_file']).read_text()
        metric = determine_task_metric(task, cfg.agent)
        task = add_task_metric(task, metric)
        journal = Journal(metric_maximize=metric.maximize if metric else None)
        agent = Agent(task, cfg, journal)
        snapshot_sha = snapshot(agent, journal)
        current['log'] = store.step_log(0, snapshot_sha)
    else:
        saved = store.read(resume['snapshot'], resume['snapshot_sha256'])
        if saved['identity'] != worker_identity(config):
            raise ValueError('AIDE worker configuration differs from its saved research identity')
        journal = loads_json(saved['journal'], Journal)
        if len(journal.nodes) != resume['snapshot']:
            raise ValueError('AIDE snapshot journal length differs from its step count')
        agent = Agent(saved['task'], cfg, journal)
        agent.data_preview = saved['data_preview']
        set_rng_state(saved['rng'])
        calls[:] = saved['usage']
        snapshot_sha = resume['snapshot_sha256']
        current['log'] = store.step_log(resume['snapshot'], snapshot_sha, resume['replies'])

    def execute(code, reset_session):
        step = len(journal.nodes)
        log = current['log']
        if log.replaying:
            raise RuntimeError('Resumed AIDE step reached execution before replaying all recorded replies')
        candidate = root / f'candidate-{step}.py'
        candidate.write_text(code)
        channel.send({'kind': 'execute', 'candidate': candidate.relative_to(root).as_posix(), 'sha256': digest(candidate), 'step': step,
                      'resume': {'snapshot': step, 'snapshot_sha256': snapshot_sha, 'replies': log.point()}})
        result = channel.receive()
        if result.get('kind') != 'execution_result':
            raise ValueError('Expected AIDE execution callback result')
        return ExecutionResult(**result['execution_result'])
    def observed_node(node):
        return observation_id('aide.node', str(node.id))

    while len(journal.nodes) < cfg.agent.steps:
        previous_nodes = len(journal.nodes)
        agent.step(execute)
        if current['log'].replaying:
            raise RuntimeError('Resumed AIDE step finished with unreplayed recorded replies')
        current['log'].assert_tail_consumed()
        (root / 'journal.json').write_text(dumps_json(journal))
        snapshot_sha = snapshot(agent, journal)
        current['log'] = store.step_log(len(journal.nodes), snapshot_sha)
        for node in journal.nodes[previous_nodes:]:
            ident = observed_node(node)
            emit_observation('aide.node', ident, scope='aide', subject=ident,
                             links={'derived_from': [observed_node(node.parent)] if node.parent else []},
                             fields={'step': node.step, 'stage': node.stage_name,
                                     'score': getattr(node.metric, 'value', None)})
    best = journal.get_best_node()
    initial = next(iter(journal.good_nodes), None)
    evaluations = []
    if best is not None:
        emit_observation('aide.selection', observation_id('aide.selection', str(best.id)), scope='aide',
                         subject=observed_node(best), links={'selected': [observed_node(best)]},
                         fields={'step': best.step, 'score': best.metric.value})
        for label, node in (('initial', initial), ('selected', best)):
            candidate = root / f'candidate-{node.step}.py'
            evaluations.append({'label': label, 'candidate': candidate.relative_to(root).as_posix(), 'sha256': digest(candidate)})
    report = {'nodes': len(journal.nodes), 'good_nodes': len(journal.good_nodes),
              'selected_node': best.id if best else None,
              'selected_step': best.step if best else None,
              'selected_metric': best.metric.value if best else None,
              'stages': [n.stage_name for n in journal.nodes],
              'parents': [n.parent.id if n.parent else None for n in journal.nodes],
              'llm_calls': len(calls)}
    (root / 'native-report.json').write_text(json.dumps(report, indent=2))
    channel.send({'kind': 'finished', 'report': report, 'evaluations': deepcopy(evaluations)})


def main():
    try:
        # Upstream prints diagnostic messages; keep them out of protocol lines.
        with redirect_stdout(sys.stderr):
            run(json.loads(Path(sys.argv[1]).read_text()))
    except BaseException as exc:
        traceback.print_exc()
        emit_event({'kind': 'error', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise


if __name__ == '__main__':
    main()
