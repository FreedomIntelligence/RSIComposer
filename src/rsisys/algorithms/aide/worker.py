"""Optional native AIDE runtime. No generated program executes in this process."""
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import sys
import traceback

from rsisys.algorithms.aide.controller import AIDE_REVISION
from rsisys.controllers.research_bridge import emit_event
from rsisys.labs.harness_runtime import digest
from rsisys.runtime.upstream import checkout


def run(config):
    upstream = checkout(config['upstream_root'], AIDE_REVISION)
    sys.path.insert(0, str(upstream))
    import aide.backend as backend
    from aide.agent import Agent, add_task_metric, determine_task_metric
    from aide.interpreter import ExecutionResult
    from aide.journal import Journal
    from aide.utils.config import _load_cfg
    from aide.utils.serialize import dumps_json

    # Local chat transport only; native prompts, review schema, parsing, search
    # and metric-based selection are unchanged.
    native_query = backend.provider_to_query_func['openai']
    calls = []
    root = Path(config['root'])
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
        try:
            result = native_query(**kwargs)
        except Exception as exc:
            trace['error'] = type(exc).__name__ + ': ' + str(exc)
            with (root / 'llm-calls.jsonl').open('a') as stream:
                stream.write(json.dumps(trace) + '\n')
            raise
        trace.update(output=result[0], elapsed_seconds=result[1], input_tokens=result[2],
                     output_tokens=result[3], provider_info=result[4])
        with (root / 'llm-calls.jsonl').open('a') as stream:
            stream.write(json.dumps(trace) + '\n')
        calls.append({'input_tokens': result[2], 'output_tokens': result[3]})
        (root / 'llm-usage.json').write_text(json.dumps(calls, indent=2))
        return result
    backend.provider_to_query_func['openai'] = query
    random.seed(config.get('seed', 73))
    cfg = _load_cfg(use_cli_args=False)
    cfg.workspace_dir = str(root / 'preview')
    preview = Path(cfg.workspace_dir) / 'input'
    preview.mkdir(parents=True)
    source = Path(config['input_dir'])
    for path in source.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError('AIDE preview inputs must be ordinary public files')
        shutil.copyfile(path, preview / path.name)
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
    task = Path(config['task_file']).read_text()
    metric = determine_task_metric(task, cfg.agent)
    task = add_task_metric(task, metric)
    journal = Journal(metric_maximize=metric.maximize if metric else None)
    agent = Agent(task, cfg, journal)
    def execute(code, reset_session):
        step = len(journal.nodes)
        candidate = root / f'candidate-{step}.py'
        candidate.write_text(code)
        emit_event({'kind': 'execute', 'candidate': str(candidate), 'sha256': digest(candidate), 'step': step})
        line = sys.stdin.readline()
        if not line:
            raise EOFError('Host closed the AIDE callback')
        result = json.loads(line)
        if result.get('kind') != 'execution_result':
            raise ValueError('Expected AIDE execution callback result')
        return ExecutionResult(**result['execution_result'])
    for _ in range(cfg.agent.steps):
        agent.step(execute)
        (root / 'journal.json').write_text(dumps_json(journal))
    best = journal.get_best_node()
    initial = next(iter(journal.good_nodes), None)
    evaluations = []
    if best is not None:
        for label, node in (('initial', initial), ('selected', best)):
            candidate = root / f'candidate-{node.step}.py'
            evaluations.append({'label': label, 'candidate': str(candidate), 'sha256': digest(candidate)})
    report = {'nodes': len(journal.nodes), 'good_nodes': len(journal.good_nodes),
              'selected_node': best.id if best else None,
              'selected_step': best.step if best else None,
              'selected_metric': best.metric.value if best else None,
              'stages': [n.stage_name for n in journal.nodes],
              'parents': [n.parent.id if n.parent else None for n in journal.nodes],
              'llm_calls': len(calls)}
    (root / 'native-report.json').write_text(json.dumps(report, indent=2))
    emit_event({'kind': 'finished', 'report': report, 'evaluations': deepcopy(evaluations)})


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
