"""Export a public monitor snapshot; SwanLab stays an optional dependency."""
import argparse
import json
from pathlib import Path

from web.server import project_snapshot


def export(snapshot, *, project, mode='offline', logdir='swanlog', sdk=None):
    if mode not in ('offline', 'online'):
        raise ValueError('mode must be offline or online')
    public = project_snapshot(snapshot, str(snapshot.get('run_id', 'rsisys')))
    if sdk is None:
        import swanlab as sdk
    count = 0
    for job in public['jobs']:
        history = job.get('metric_history', [])
        if not history and job.get('metrics'):
            history = [{'step': job.get('step', 0), 'metrics': job['metrics']}]
        if not history:
            continue
        run = sdk.init(project=project, experiment_name=f"{public['run_id']}-{job.get('job_id', count)}",
                       mode=mode, logdir=str(logdir), config={'run_id': public['run_id'],
                       'job_id': job.get('job_id'), 'kind': job.get('kind')})
        try:
            # Use sample order for SDK monotonicity; preserve optimizer step as a metric.
            for index, point in enumerate(history):
                values = dict(point.get('metrics', {}))
                if point.get('step') is not None:
                    values['optimizer_step'] = point['step']
                if point.get('elapsed_seconds') is not None:
                    values['elapsed_seconds'] = point['elapsed_seconds']
                run.log(values, step=index)
        finally:
            sdk.finish()
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--project', default='RSISys')
    parser.add_argument('--mode', choices=('offline', 'online'), default='offline')
    parser.add_argument('--logdir', type=Path, default=Path('swanlog'))
    args = parser.parse_args()
    print(f"Exported {export(json.loads(args.snapshot.read_text()), project=args.project, mode=args.mode, logdir=args.logdir)} jobs")


if __name__ == '__main__':
    main()
