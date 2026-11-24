"""Private FT-Agent worker: native policy pauses at every Lab operation.

Resume: the native objects (FTTrace with its experiments and feedback, the
proposal generator, coder and summarizer) are exec-compiled upstream classes
without a serialization of their own, so they are rebuilt by re-running this
deterministic loop. Every model reply and every host receipt is a chained
replay record under resume/; each execute event carries the resume point. A
worker started with config['resume'] replays exactly those records, checking
each request is identical, until it re-emits the outstanding execute request;
from there it runs live, discarding exchanges beyond the checkpoint.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from rsicomposer.algorithms.ft_agent.upstream import ChatBackend, Scenario, load_policy
from rsicomposer.api import UpstreamWorker, ReplayMismatchError, emit_event



def replay_log(root, config):
    """Records are chained from the worker configuration, so they belong to this run."""
    base = UpstreamWorker.identity(config, protocol='rsicomposer-ft-worker/v3')
    return UpstreamWorker.open_replay(root / 'resume', base, config.get('resume'))


def run(config, channel=None, pool=None):
    channel = channel or UpstreamWorker()
    root = Path(config['directory'])
    replay = replay_log(root, config)
    backend = ChatBackend(config['researcher'], root / 'model-calls', replay=replay, pool=pool)
    settings = SimpleNamespace(base_model='target', target_benchmark=config['benchmark'],
        file_path=root / 'assets', force_think_token=False, api_max_workers=1)
    dataset_dir = settings.file_path / 'datasets'
    info = json.dumps(config['dataset_info'])
    if dataset_dir.exists():
        # A resumed or restarted worker reuses the run's own assets, unchanged.
        if (dataset_dir / 'dataset_info.json').read_text() != info:
            raise ValueError('FT-Agent dataset assets differ from this run')
    else:
        dataset_dir.mkdir(parents=True)
        (dataset_dir / 'dataset_info.json').write_text(info)

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
        request = {'stage': stage, 'iteration': iteration, 'files': files}
        response = replay.take('feedback', request)
        if response is None:
            channel.send({'kind': 'execute', 'stage': stage, 'iteration': iteration, 'files': files,
                          'resume': replay.point()})
            response = channel.receive()
            if (response.get('kind') != 'feedback' or response.get('stage') != stage
                    or response.get('iteration') != iteration):
                raise ValueError('FT-Agent feedback does not match the pending operation')
            replay.record('feedback', request, response)
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
        except ReplayMismatchError:
            raise
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

    if replay.replaying:
        raise RuntimeError('Resumed FT-Agent finished with unreplayed recorded exchanges')
    replay.assert_tail_consumed()
    selected = trace.get_sota_experiment()
    selected_iteration = trace.exp2idx(selected) + 1 if selected is not None else 0
    channel.send({'kind': 'finished', 'selected_iteration': selected_iteration,
        'iterations': len(history), 'history': history, 'model_calls': backend.calls})


if __name__ == '__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception:
        import traceback
        emit_event({'kind': 'error', 'error': traceback.format_exc()})
        raise
