"""Install release wheels offline and run public entrypoints outside the repo."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def check_archive(path, name):
    with ZipFile(path) as archive:
        names = archive.namelist()
        assert any(p.endswith('.dist-info/licenses/LICENSE') for p in names), name
        assert any(p.endswith('.dist-info/licenses/NOTICE') for p in names), name
        assert not any(p.startswith(('tests/', 'experiments/', 'examples/', '.local/')) for p in names)
        if name == 'rsisys':
            assert 'rsisys/labs/SEAL_NOTICE.txt' in names
            assert 'rsisys/controllers/FT_AGENT_NOTICE.txt' in names
            assert 'rsisys/controllers/AIDE_NOTICE.txt' in names
            for module in ('ft_agent', 'ft_agent_worker', 'ft_agent_upstream', 'aide', 'aide_worker', 'research_bridge'):
                assert f'rsisys/controllers/{module}.py' in names
            for module in ('ft_agent', 'ft_train_worker', 'ft_data_sandbox', 'aide_program', 'aide_client'):
                assert f'rsisys/labs/{module}.py' in names
        else:
            for asset in ('index.html', 'app.js', 'style.css', 'i18n.js', 'decisions.js'):
                assert f'rsimonitor/static/{asset}' in names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-wheel', required=True, type=Path)
    parser.add_argument('--monitor-wheel', required=True, type=Path)
    parser.add_argument('--core-sdist', type=Path, help='Also validate extracted source documentation and layout')
    args = parser.parse_args()
    wheels = {'rsisys': args.core_wheel.resolve(strict=True),
              'rsimonitor': args.monitor_wheel.resolve(strict=True)}
    env = {key: value for key, value in os.environ.items()
           if key not in {'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}}
    env.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    with TemporaryDirectory(prefix='rsisys-release-') as temporary:
        root = Path(temporary)
        if args.core_sdist:
            extracted = root / 'source'
            extracted.mkdir()
            with tarfile.open(args.core_sdist.resolve(strict=True)) as archive:
                names = archive.getnames()
                assert not any(any(part in p for part in (
                                   '/.local/', '/docs/review/repros/',
                                   '/docs/review/prompt-snapshots/', '/docs/review/audit-20261104/')) or
                               p.endswith(('docs/development-history.md', '.log', '.safetensors', '.pt'))
                               for p in names)
                archive.extractall(extracted, filter='data')
            source, = extracted.iterdir()
            assert (source / 'src/rsimonitor/server.py').is_file()
            for document in ('docs/methods.md', 'docs/diagrams/framework-loop.svg',
                             'docs/diagrams/build_diagrams.py',
                             'docs/review/weight-reproduction-diagnosis-20261109.json'):
                assert (source / document).is_file(), document
            subprocess.run([sys.executable, 'tools/check_docs.py'], cwd=source,
                           env=env, check=True, timeout=30)
            print('rsisys sdist: source layout and documentation links passed')
        for name, wheel in wheels.items():
            check_archive(wheel, name)
            workspace, venv = root / name, root / f'{name}-venv'
            workspace.mkdir()
            subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True, env=env)
            python, cli = venv / 'bin/python', venv / f'bin/{name}'

            def run(command):
                return subprocess.run(command, check=True, env=env, cwd=workspace,
                                      capture_output=True, text=True, timeout=120).stdout

            run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)])
            run([str(python), '-I', '-c', f'''
import importlib.metadata as m, importlib.util, pathlib, sys
dist = m.distribution({name!r})
assert dist.metadata['License-Expression'] == 'Apache-2.0'
assert not [r for r in dist.requires or () if 'extra ==' not in r], dist.requires
assert not any(' @ ' in r for r in dist.requires or ()), dist.requires
import {name}
assert pathlib.Path({name}.__file__).is_relative_to(sys.prefix)
assert importlib.util.find_spec({'rsimonitor' if name == 'rsisys' else 'rsisys'!r}) is None
'''])
            run([str(cli), '--help'])
            if name == 'rsisys':
                assert run([str(cli), '--version']).startswith('rsisys 0.1.0')
                run([str(python), '-I', '-c', '''
import sys
from rsisys.api import Registry
assert {'rest_em', 'rrsi', 'seal', 'ttrl', 'ft_agent', 'aide'} <= set(Registry().describe()['controllers'])
assert not any(x.split('.')[0] in {'torch', 'transformers', 'peft', 'vllm', 'math_verify', 'aide', 'rdagent', 'yaml', 'jinja2'} for x in sys.modules)
'''])
                shutil.copytree(ROOT / 'examples/registered', workspace / 'registered')
                shutil.copytree(ROOT / 'examples/quickstart', workspace / 'examples/quickstart')
                for config, output, plugin in (
                    ('examples/quickstart/rest_em.json', 'rest-run', None),
                    ('registered/custom-controller.json', 'custom-run', 'registered/plugin.py'),
                ):
                    command = [str(cli), 'run', '--config', config, '--data-dir', output]
                    if plugin:
                        command.extend(['--plugin', plugin])
                    result = json.loads(run(command))
                    assert result['status'] == 'finished', result
                    assert (workspace / output / 'result.json').is_file()
                catalog = json.loads(run([str(cli), 'catalog', '--plugin', 'registered/plugin.py']))
                assert 'target' in catalog['controllers'] and 'counter' in catalog['labs']
            else:
                run([str(python), '-I', '-c', '''
from pathlib import Path
from tempfile import TemporaryDirectory
from rsimonitor import RunStore, MonitorServer
with TemporaryDirectory() as root:
    assert RunStore(root).projects() == []
    server = MonitorServer(('127.0.0.1', 0), Path(root))
    server.server_close()
'''])
            print(f'{name}: isolated install, metadata, resources and public entrypoints passed')


if __name__ == '__main__':
    main()
