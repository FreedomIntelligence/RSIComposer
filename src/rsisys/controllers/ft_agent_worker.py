"""Private FT-Agent worker: native policy pauses at every Lab operation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from .ft_agent_upstream import ChatBackend, Scenario, load_policy
from .research_bridge import emit_event


def run(config):
    root = Path(config['directory'])
    backend = ChatBackend(config['researcher'], root / 'model-calls')
    settings = SimpleNamespace(base_model='target', target_benchmark=config['benchmark'],
        file_path=root / 'assets', force_think_token=False, api_max_workers=1)
    dataset_dir = settings.file_path / 'datasets'
    dataset_dir.mkdir(parents=True, exist_ok=False)
    (dataset_dir / 'dataset_info.json').write_text(json.dumps(config['dataset_info']))

    class Descriptor:
        def describe_dataset_folder(self, path, **kwargs):
            return config['dataset_description']
    api = load_policy(config['upstream_root'], settings=settings, backend=backend,
        parameter_info=config['parameter_info'], dataset_descriptor=Descriptor())
    scen = Scenario(config['scenario'], config['baseline'])
    trace = api.FTTrace(scen)
    generator = api.LLMFinetuneExpGen(scen)
    strategy = api.LLMFinetuneEvolvingStrategy(scen, settings)
    summarizer = api.FTExperiment2Feedback(scen)
    history = []

    def execute(stage, iteration, files):
        emit_event({'kind': 'execute', 'stage': stage, 'iteration': iteration, 'files': files})
        line = sys.stdin.readline()
        if not line:
            raise EOFError('Host closed the FT-Agent feedback channel')
        response = json.loads(line)
        if response.get('kind') != 'feedback' or response.get('stage') != stage or response.get('iteration') != iteration:
            raise ValueError('FT-Agent feedback does not match the pending operation')
        return response['data']

    for iteration in range(1, config['iterations'] + 1):
        exp = generator.gen(trace)
        ws = exp.experiment_workspace
        task = exp.sub_tasks[0]
        query = SimpleNamespace(task_to_former_failed_traces={task.get_task_information(): ([], [])})
        (root / f'hypothesis-{iteration}.json').write_text(json.dumps({
            'hypothesis': str(exp.hypothesis), 'task': task.get_task_information(),
            'parent': exp.local_selection, 'history_length': len(trace.hist)}, indent=2))
        failure = None
        try:
            ws.file_dict.update(strategy.implement_data(task, query, ws))
            prepared = execute('prepare', iteration, ws.file_dict)
            if not prepared.get('ok'):
                raise RuntimeError(prepared.get('error', 'Data preparation failed'))
            ws.file_dict['data_stats.json'] = json.dumps(prepared['data_stats'])
            ws.file_dict.update(strategy.implement_lf_config(task, query, ws))
            trained = execute('train', iteration, ws.file_dict)
            if not trained.get('ok'):
                raise RuntimeError(trained.get('error', 'Training failed'))
            ws.running_info = SimpleNamespace(result=trained['result'], running_time=trained['elapsed_seconds'])
        except Exception as exc:
            # Native error feedback decides what to change in the next proposal.
            failure = exc
        feedback = summarizer.generate_feedback(exp, trace, exception=failure)
        trace.sync_dag_parent_and_hist((exp, feedback), iteration)
        item = {'iteration': iteration, 'hypothesis': str(exp.hypothesis),
            'decision': feedback.decision, 'acceptable': feedback.acceptable,
            'reason': feedback.reason, 'error': str(failure) if failure else None,
            'parent': exp.local_selection, 'files': dict(ws.file_dict)}
        history.append(item)
        (root / f'feedback-{iteration}.json').write_text(json.dumps(item, indent=2))

    selected = trace.get_sota_experiment()
    selected_iteration = trace.exp2idx(selected) + 1 if selected is not None else 0
    emit_event({'kind': 'finished', 'selected_iteration': selected_iteration,
        'iterations': len(history), 'history': history, 'model_calls': backend.calls})


if __name__ == '__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception:
        import traceback
        emit_event({'kind': 'error', 'error': traceback.format_exc()})
        raise
