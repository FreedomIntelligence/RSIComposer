import argparse
from dataclasses import asdict, replace
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rsisys.config import DemoConfig, build_demo
from rsisys.application import Application, ApplicationConfig
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot, load_manifest
from rsisys.registry import Registry
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.runtime.storage.files import atomic_write, json_bytes


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
    demo = commands.add_parser('demo', help='Run the original in-memory mock example')
    demo.add_argument('--config', type=Path, required=True)
    run = commands.add_parser('run', help='Start a new persistent local run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--data-dir', type=Path)
    run.add_argument('--parent-run-id', help='Declare ancestry only; does not restore model or memory state')
    run.add_argument('--literature', type=Path, action='append', default=[])
    run.add_argument('--plugin', type=Path, action='append', default=[],
                     help='Explicitly trust a registration file or method package directory; repeatable')
    run.add_argument('--permissions', type=Path, help='Host ActionPermissions JSON')
    resume = commands.add_parser('resume', help='Restore a quiescent snapshot into a NEW workspace')
    resume.add_argument('--snapshot', type=Path, required=True)
    resume.add_argument('--data-dir', type=Path, required=True)
    resume.add_argument('--run-id', help='Fork as a new run; parent is taken automatically from the snapshot')
    for command in (run, resume):
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
        elif args.command == 'demo':
            config = DemoConfig.load(args.config)
            runner, store = build_demo(config)
            with runner:
                result = runner.run(config.objective, config.rounds)
            output = {'mode': 'demo', 'simulated': True, 'training_performed': False,
                      'snapshot_supported': False, 'experience_records': len(store.records()), **asdict(result)}
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
                if args.plugin and selection is None:
                    raise ValueError('--plugin requires a framework selection in the config')
                registry = Registry()
                for path in args.plugin:
                    registry.load_plugin(path)
                modules = registry.build(selection) if selection is not None else None
                config = ApplicationConfig.from_dict(raw, base_dir=args.config.resolve().parent)
                if modules is not None and config.max_decision_steps is None:
                    config = replace(config, max_decision_steps=64)
                if args.data_dir is not None:
                    config = replace(config, data_dir=args.data_dir.resolve())
                if args.parent_run_id is not None:
                    config = replace(config, parent_run_id=args.parent_run_id)
                permissions = (ActionPermissions.from_dict(json.loads(args.permissions.read_text()))
                               if args.permissions is not None else None)
                app = Application(config, modules=modules, action_permissions=permissions)
            else:
                app = restore_snapshot(args.snapshot, args.data_dir, run_id=args.run_id)
            with app:
                if args.command == 'run':
                    for path in args.literature:
                        app.artifacts.put_file(path, kind='literature')
                result = app.run(rounds=args.rounds, max_decision_steps=args.max_decision_steps)
                snapshot = None
                snapshot_supported = not app.custom_assembly or bool(app.assembly.state_adapters)
                # Waiting work is explicitly not checkpointable in this first local format.
                if snapshot_supported and result.status != 'waiting' and not app.runner.busy:
                    manifest = LocalSnapshotCapture(app).capture('after-run')
                    snapshot = str(app.config.data_dir / 'snapshots' / manifest.snapshot_id)
                output = {'mode': 'local', **app.config.source.to_dict(),
                          'simulated': (None if app.custom_assembly else app.config.execution == 'mock' or any(
                              b.kind == 'mock' for b in app.config.backends.values())),
                          'training_performed': None if app.custom_assembly else False,
                          'snapshot_supported': snapshot_supported,
                          'snapshot': snapshot, 'experience_records': len(app.store.records()), **asdict(result)}
                if app.custom_assembly:
                    output['framework'] = app.modules.configuration()
                report = app.controller_report()
                if report is not None:
                    output['controller_report'] = report
            atomic_write(app.config.data_dir / 'result.json', json_bytes(output))
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'rsisys: {exc}\n')
