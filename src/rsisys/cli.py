import argparse
from dataclasses import asdict, replace
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rsisys.application import Application, ApplicationConfig
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot, load_manifest
from rsisys.registry import Registry
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.runtime.storage.files import atomic_write, json_bytes
from rsisys.runtime.orchestration.monitor_snapshot import run_monitored


def main() -> None:
    parser = argparse.ArgumentParser(description='RSISys: RSI controllers, Labs and feedback')
    try:
        installed_version = version('rsisys')
    except PackageNotFoundError:
        installed_version = 'source checkout (not installed)'
    parser.add_argument('--version', action='version', version=f'rsisys {installed_version}')
    commands = parser.add_subparsers(dest='command', required=True)
    catalog = commands.add_parser('catalog', help='List methods and reusable components without running them')
    catalog.add_argument('--plugin', type=Path, action='append', default=[],
                         help='Explicitly trust a registration file or method package directory; repeatable')
    run = commands.add_parser('run', help='Start a new persistent run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--data-dir', type=Path)
    run.add_argument('--experiment-name', help='Name shown in RSIMonitor')
    run.add_argument('--parent-run-id', help='Declare ancestry only; does not restore model or memory state')
    run.add_argument('--literature', type=Path, action='append', default=[])
    run.add_argument('--plugin', type=Path, action='append', default=[],
                     help='Explicitly trust a registration file or method package directory; repeatable')
    run.add_argument('--permissions', type=Path, help='Host ActionPermissions JSON')
    run.add_argument('--durable', action='store_true',
                     help='Checkpoint every step so `rsisys resume --data-dir` can continue this workspace')
    resume = commands.add_parser('resume', help='Continue a durable workspace, or restore a snapshot into a NEW one')
    resume.add_argument('--snapshot', type=Path,
                        help='Restore this quiescent snapshot into --data-dir; omit to continue a durable run')
    resume.add_argument('--data-dir', type=Path, required=True)
    resume.add_argument('--run-id', help='Fork as a new run; parent is taken automatically from the snapshot')
    resume.add_argument('--plugin', type=Path, action='append', default=[],
                        help='Trusted plugin paths for a durable run; defaults to those recorded at launch')
    for command in (run, resume):
        command.add_argument('--probe', type=Path, action='append', default=[],
                             help='Trusted Python observation plugin with register(observations); repeatable')
        command.add_argument('--rounds', type=int)
        command.add_argument('--max-decision-steps', type=int)
    inspect = commands.add_parser('inspect', help='Validate and describe a local snapshot')
    inspect.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'catalog':
            registry = Registry()
            for path in args.plugin:
                registry.load_plugin(path)
            output = registry.describe()
        elif args.command == 'inspect':
            output = asdict(load_manifest(args.snapshot.resolve()))
        else:
            if args.rounds is not None and args.rounds < 1:
                raise ValueError('rounds must be positive')
            if args.max_decision_steps is not None and args.max_decision_steps < 1:
                raise ValueError('max-decision-steps must be positive')
            if args.command == 'run':
                raw = json.loads(args.config.read_text())
                selection = raw.pop('framework', None)
                if selection is None:
                    raise ValueError('run config requires framework; use '
                                     '\"framework\": {\"method\": {\"name\": \"rest_em\", \"config\": {}}} '
                                     'or \"framework\": {\"controller\": {\"name\": \"...\"}, '
                                     '\"labs\": {\"lab\": {\"name\": \"...\"}}}')
                registry = Registry()
                for path in args.plugin:
                    registry.load_plugin(path)
                modules = registry.build(selection)
                config = ApplicationConfig.from_dict(raw, base_dir=args.config.resolve().parent)
                if modules is not None and config.max_decision_steps is None:
                    config = replace(config, max_decision_steps=64)
                if args.data_dir is not None:
                    config = replace(config, data_dir=args.data_dir.resolve())
                if args.parent_run_id is not None:
                    config = replace(config, parent_run_id=args.parent_run_id)
                permissions = (ActionPermissions.from_dict(json.loads(args.permissions.read_text()))
                               if args.permissions is not None else None)
                app = Application(config, modules=modules, action_permissions=permissions, durable=args.durable)
                if args.durable:
                    atomic_write(config.data_dir / 'framework.json', json_bytes(
                        {'framework': selection, 'plugins': [str(path.resolve()) for path in args.plugin]}))
            elif args.snapshot is not None:
                app = restore_snapshot(args.snapshot, args.data_dir, run_id=args.run_id, plugins=args.plugin)
            else:
                if args.run_id is not None:
                    raise ValueError('--run-id forks a snapshot; a durable resume keeps its run identity')
                root = args.data_dir.resolve()
                saved = root / 'framework.json'
                if not saved.exists():
                    raise ValueError(f'{root} was not started with `rsisys run --durable`')
                launch = json.loads(saved.read_text())
                selection = launch['framework']
                registry = Registry()
                for path in args.plugin or [Path(p) for p in launch['plugins']]:
                    registry.load_plugin(path)
                modules = registry.build(selection)
                app = Application.resume(root, modules=modules)
            with app:
                if args.command == 'run':
                    for path in getattr(args, 'literature', ()):
                        app.artifacts.put_file(path, kind='literature')
                result = run_monitored(app, rounds=args.rounds, max_decision_steps=args.max_decision_steps,
                                       framework=selection if args.command == 'run' or args.snapshot is None else None,
                                       experiment_name=getattr(args, 'experiment_name', None), probe_paths=args.probe,
                                       continue_history=args.command == 'resume' and args.snapshot is None)
                snapshot = None
                from rsisys.runtime.state.component_state import snapshot_integration
                snapshot_supported = snapshot_integration(app)
                # Waiting work is explicitly not checkpointable in this first local format.
                if snapshot_supported and result.status != 'waiting' and not app.runner.busy:
                    manifest = LocalSnapshotCapture(app).capture('after-run')
                    snapshot = str(app.config.data_dir / 'snapshots' / manifest.snapshot_id)
                output = {'mode': app.config.resources.get('executor', 'local'), **app.config.source.to_dict(),
                          'simulated': None, 'training_performed': None,
                          'snapshot_supported': snapshot_supported,
                          'snapshot': snapshot, 'experience_records': len(app.store.records()), **asdict(result)}
                output['framework'] = app.modules.configuration()
                report = app.controller_report()
                if report is not None:
                    output['controller_report'] = report
            atomic_write(app.config.data_dir / 'result.json', json_bytes(output))
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (TypeError, ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'rsisys: {exc}\n')
