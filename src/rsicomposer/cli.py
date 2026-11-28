import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import tomllib

from rsicomposer.application import Application, ApplicationConfig
from rsicomposer.snapshot.local import LocalSnapshotCapture, restore_snapshot, load_manifest
from rsicomposer.registry import Registry
from rsicomposer.loops.decision_making.action_space.access import ActionPermissions
from rsicomposer.runtime.storage.files import atomic_write, json_bytes
from rsicomposer.runtime.orchestration.monitor_snapshot import run_monitored


def _read_run_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == '.json':
        config = json.loads(path.read_text(encoding='utf-8'))
    elif suffix == '.toml':
        with path.open('rb') as stream:
            config = tomllib.load(stream)
        try:
            json.dumps(config, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('TOML run config requires JSON-compatible values; '
                             'quote dates and times and use finite numbers') from exc
    else:
        raise ValueError('Run config must have a .toml or .json extension')
    if not isinstance(config, dict):
        raise ValueError('Run config must be an object')
    return config


_DEFAULT_CLI_DECISION_STEPS = 64

def _load_launch(args):
    raw = _read_run_config(args.config)
    selection = raw.pop('framework', None)
    if selection is None:
        raise ValueError('run config requires framework; set framework.method.name '
                         'or configure both framework.controller and framework.labs')
    registry = Registry()
    for path in args.plugin:
        registry.load_plugin(path)
    modules = registry.build(selection)
    config = ApplicationConfig.from_dict(raw, base_dir=args.config.resolve().parent)
    limit_source = 'config' if config.max_decision_steps is not None else 'cli_default'
    if config.max_decision_steps is None:
        config = replace(config, max_decision_steps=_DEFAULT_CLI_DECISION_STEPS)
    if args.data_dir is not None:
        config = replace(config, data_dir=args.data_dir.resolve())
    permissions = (ActionPermissions.from_dict(json.loads(args.permissions.read_text()))
                   if args.permissions is not None else None)
    return config, modules, permissions, limit_source

def _effective_limits(args, config, limit_source):
    rounds = config.rounds if args.rounds is None else args.rounds
    steps = config.max_decision_steps
    if args.max_decision_steps is not None:
        steps, limit_source = args.max_decision_steps, 'flag'
    elif steps is None:
        steps, limit_source = rounds * 8, 'runner_default'
    return {'rounds': rounds, 'max_decision_steps': steps, 'max_decision_steps_source': limit_source}


def _explain_launch(args, config, modules, limit_source):
    # Report only public identity fields: arbitrary plugin configuration can
    # contain secrets and belongs neither in diagnostics nor in terminal logs.
    manifest = modules.method_identity.configuration()['config']
    if 'method' in manifest:
        method = manifest['method']
        framework = {'method': {'name': method['method'], 'version': method['version']}}
    else:
        def named(item):
            return {key: item[key] for key in ('name', 'version')}
        framework = {'controller': named(manifest['controller']),
                     'labs': {alias: named(spec) for alias, spec in manifest['labs'].items()}}
    return {
        'validation': 'configuration_only',
        'framework': framework,
        'plugins': manifest['plugins'],
        'data_dir': str(config.data_dir),
        **_effective_limits(args, config, limit_source),
    }


def _run_output(app, args, limit_source):
    if args.command == 'run' and not args.durable:
        for path in args.literature:
            app.artifacts.put_file(path, kind='literature')
    limits = dict(rounds=args.rounds, max_decision_steps=args.max_decision_steps)
    if args.no_monitor:
        result = app.run(**limits, wait_for_jobs=True)
    else:
        selection = app.modules.registration.framework if not getattr(args, 'snapshot', None) else None
        result = run_monitored(app, **limits,
                               framework=selection,
                               experiment_name=getattr(args, 'experiment_name', None), probe_paths=args.probe,
                               continue_history=args.command == 'resume' and args.snapshot is None)
    output = {'mode': app.config.resources.get('executor', 'local'), **app.config.source.to_dict(),
              'simulated': None, 'training_performed': None,
              'snapshot_supported': None, 'snapshot': None,
              'experience_records': len(app.store.records()), **asdict(result)}
    output['framework'] = app.modules.configuration()
    output['effective_run_limits'] = _effective_limits(args, app.config, limit_source)
    output['postprocessing'] = {'snapshot': {'status': 'pending'}, 'controller_report': {'status': 'pending'}}
    return output


def _add_run_exports(app, output):
    """Attempt both exports; only snapshot errors affect the CLI exit status."""
    phases = output['postprocessing']
    snapshot_error = None
    try:
        from rsicomposer.runtime.state.component_state import snapshot_integration
        output['snapshot_supported'] = snapshot_integration(app)
        if not output['snapshot_supported']:
            phases['snapshot'] = {'status': 'unsupported'}
        # A portable snapshot requires successful quiescence, not just zero Lab tasks.
        elif output['status'] == 'waiting' or app.runner.busy or output['shutdown_complete'] is False:
            phases['snapshot'] = {'status': 'skipped', 'reason': 'not_quiescent'}
        else:
            manifest = LocalSnapshotCapture(app).capture('after-run')
            output['snapshot'] = str(app.config.data_dir / 'snapshots' / manifest.snapshot_id)
            phases['snapshot'] = {'status': 'completed'}
    except Exception as error:
        phases['snapshot'] = {'status': 'failed', 'error_type': type(error).__name__}
        snapshot_error = error

    try:
        report = app.controller_report()
        if report is None:
            phases['controller_report'] = {'status': 'unavailable'}
        else:
            output['controller_report'] = report
            phases['controller_report'] = {'status': 'completed'}
    except Exception as error:
        phases['controller_report'] = {'status': 'failed', 'error_type': type(error).__name__}
        print('rsicomposer: optional Controller report failed; see the run summary', file=sys.stderr)
    return snapshot_error


def _run_and_save(app, args, limit_source):
    path = app.config.data_dir / 'result.json'
    pending = None
    export_error = None
    with ExitStack() as finalization:
        try:
            with app:
                output = _run_output(app, args, limit_source)
                output['cleanup'] = {'status': 'pending'}
                finalization.enter_context(app._hold_workspace_lock())
                atomic_write(path, json_bytes(output))
                pending = output
                try:
                    export_error = _add_run_exports(app, output)
                    atomic_write(path, json_bytes(output))
                except BaseException as error:
                    # Defer export or publication failures until close has its
                    # own outcome. The already published core stays readable.
                    if export_error is not None:
                        error.add_note(f'Snapshot export also failed: {type(export_error).__name__}')
                    export_error = error
        except BaseException as error:
            if export_error is not None:
                error.add_note(f'Run export also failed: {type(export_error).__name__}')
            if pending is not None:
                # Don't replace the original cleanup exception if publication
                # also fails. An intact pending summary is still useful.
                pending['cleanup'] = {'status': 'failed', 'error_type': type(error).__name__}
                try:
                    atomic_write(path, json_bytes(pending))
                except OSError as publication_error:
                    error.add_note(f'Could not update the cleanup status: {publication_error}')
            raise
        output['cleanup'] = {'status': 'completed'}
        atomic_write(path, json_bytes(output))
    if export_error is not None:
        raise export_error
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description='RSIComposer: RSI controllers, Labs and feedback')
    try:
        installed_version = version('rsicomposer')
    except PackageNotFoundError:
        installed_version = 'source checkout (not installed)'
    parser.add_argument('--version', action='version', version=f'rsicomposer {installed_version}')
    commands = parser.add_subparsers(dest='command', required=True)
    catalog = commands.add_parser('catalog', help='List methods and reusable components without running them')
    catalog.add_argument('--plugin', type=Path, action='append', default=[],
                         help='Explicitly trust a registration file or method package directory; repeatable')
    run = commands.add_parser('run', help='Start a new persistent run')
    explain = commands.add_parser('explain', help='Describe effective configuration without starting a run')
    for command in (run, explain):
        command.add_argument('--config', type=Path, required=True, help='Run configuration (.toml or .json)')
        command.add_argument('--data-dir', type=Path)
        command.add_argument('--plugin', type=Path, action='append', default=[],
                             help='Explicitly trust a registration file or method package directory; repeatable')
        command.add_argument('--permissions', type=Path, help='Host ActionPermissions JSON')
    for command in (run, explain):
        command.add_argument('--rounds', type=int)
        command.add_argument('--max-decision-steps', type=int)
    run.add_argument('--experiment-name', help='Name shown in RSIMonitor')
    run.add_argument('--parent-run-id', help='Declare ancestry only; does not restore model or memory state')
    run.add_argument('--literature', type=Path, action='append', default=[])
    run.add_argument('--durable', action='store_true',
                     help='Checkpoint every step so `rsicomposer resume --data-dir` can continue this workspace')
    resume = commands.add_parser('resume', help='Continue a durable workspace, or restore a snapshot into a NEW one')
    resume.add_argument('--snapshot', type=Path,
                        help='Restore this quiescent snapshot into --data-dir; omit to continue a durable run')
    resume.add_argument('--data-dir', type=Path, required=True)
    # Keep parsing the former flag so callers get a specific unsupported error.
    resume.add_argument('--run-id', help=argparse.SUPPRESS)
    resume.add_argument('--plugin', type=Path, action='append', default=[],
                        help='Trusted plugin paths; durable resume defaults to those recorded at launch, snapshot restore requires explicit paths')
    for command in (run, resume):
        command.add_argument('--probe', type=Path, action='append', default=[],
                             help='Trusted Python observation plugin with register(observations); repeatable')
        command.add_argument('--no-monitor', action='store_true', help='Run without the optional Monitor publisher')
    resume.add_argument('--rounds', type=int)
    resume.add_argument('--max-decision-steps', type=int)
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
            limit_source = 'saved_config'
            if args.command in ('run', 'explain'):
                config, modules, permissions, limit_source = _load_launch(args)
                if args.command == 'explain':
                    print(json.dumps(_explain_launch(args, config, modules, limit_source),
                                     ensure_ascii=False, indent=2))
                    return
                if args.parent_run_id is not None:
                    config = replace(config, parent_run_id=args.parent_run_id)
                launch = json_bytes({'plugins': list(modules.plugin_paths)}) if args.durable else None
                app = Application(config, modules=modules, action_permissions=permissions,
                                  durable=args.durable, _cli_launch=launch,
                                  _initial_literature=tuple(args.literature) if args.durable else ())
            elif args.snapshot is not None:
                app = restore_snapshot(args.snapshot, args.data_dir, run_id=args.run_id, plugins=args.plugin)
            else:
                if args.run_id is not None:
                    raise ValueError('Durable resume preserves run identity; --run-id is not supported')
                root = args.data_dir.resolve()
                saved = root / 'framework.json'
                if not saved.exists():
                    raise ValueError(f'No CLI launch metadata in {root}; use the original CLI durable workspace')
                if not (root / 'recovery' / 'checkpoint.json').is_file():
                    raise ValueError(f'Durable startup did not commit an initial checkpoint in {root}; '
                                     'start a new run in a fresh directory')
                launch = json.loads(saved.read_text())
                app = Application.resume(root, plugins=args.plugin or launch['plugins'])
            output = _run_and_save(app, args, limit_source)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (TypeError, ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'rsicomposer: {exc}\n')
