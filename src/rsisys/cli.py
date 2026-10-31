import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from rsisys.config import DemoConfig, build_demo
from rsisys.application import Application, ApplicationConfig
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='RSISys: modular loops and local persistence')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Run the original in-memory mock example')
    demo.add_argument('--config', type=Path, required=True)
    run = commands.add_parser('run', help='Start a new persistent local run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--data-dir', type=Path)
    run.add_argument('--literature', type=Path, action='append', default=[])
    resume = commands.add_parser('resume', help='Restore a quiescent snapshot into a NEW workspace')
    resume.add_argument('--snapshot', type=Path, required=True)
    resume.add_argument('--data-dir', type=Path, required=True)
    for command in (run, resume):
        command.add_argument('--rounds', type=int)
        command.add_argument('--max-decision-steps', type=int)
    inspect = commands.add_parser('inspect', help='Validate and describe a local snapshot')
    inspect.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'demo':
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
            if args.command == 'run':
                config = ApplicationConfig.load(args.config)
                if args.data_dir is not None:
                    config = replace(config, data_dir=args.data_dir.resolve())
                app = Application(config)
            else:
                app = restore_snapshot(args.snapshot, args.data_dir)
            with app:
                if args.command == 'run':
                    for path in args.literature:
                        app.artifacts.put_file(path, kind='literature')
                result = app.run(rounds=args.rounds, max_decision_steps=args.max_decision_steps)
                snapshot = None
                # Waiting work is explicitly not checkpointable in this first local format.
                if result.status != 'waiting' and not app.runner.busy:
                    manifest = LocalSnapshotCapture(app).capture('after-run')
                    snapshot = str(app.config.data_dir / 'snapshots' / manifest.snapshot_id)
                output = {'mode': 'local', 'run_id': app.config.run_id,
                          'simulated': app.config.execution == 'mock' or any(
                              b.kind == 'mock' for b in app.config.backends.values()),
                          'training_performed': False, 'snapshot_supported': True,
                          'snapshot': snapshot, 'experience_records': len(app.store.records()), **asdict(result)}
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'rsisys: {exc}\n')
