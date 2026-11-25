"""Run with a Python environment containing the installed wheel, not an editable checkout."""
import json
import os
from pathlib import Path
import subprocess
import sysconfig
from tempfile import TemporaryDirectory

from rsicomposer import __file__ as installed_package


def main():
    fixture = (Path(installed_package).resolve().parent / 'algorithms' / 'rest_em'
               / 'configs' / 'cpu')
    config = json.loads((fixture / 'rest_em.json').read_text(encoding='utf-8'))
    config['framework']['method']['config']['labs']['math']['manifest'] = str(fixture / 'data' / 'manifest.json')
    cli = Path(sysconfig.get_path('scripts')) / 'rsicomposer'
    # Deliberately do not inherit model keys, PYTHONPATH, or the source working directory.
    env = {'PATH': os.defpath, 'LC_ALL': 'C'}
    with TemporaryDirectory(prefix='rsicomposer-wheel-smoke-') as temp:
        root = Path(temp)
        config['data_dir'] = 'run'
        config_path = root / 'smoke.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')

        def command(*args):
            result = subprocess.run([str(cli), *args], cwd=root, env=env, check=True,
                                    capture_output=True, text=True, timeout=30)
            return json.loads(result.stdout)

        assert 'rest_em' in command('catalog')['methods']
        explanation = command('explain', '--config', str(config_path))
        assert explanation['framework']['method']['name'] == 'rest_em', explanation
        assert not (root / 'run').exists()
        paused = command('run', '--config', str(config_path), '--no-monitor', '--durable',
                         '--max-decision-steps', '4')
        assert paused['status'] == 'step_limit' and paused['cleanup'] == {'status': 'completed'}, paused
        assert paused['effective_run_limits'] == {'rounds': 1, 'max_decision_steps': 4,
                                                   'max_decision_steps_source': 'flag'}, paused
        result = command('resume', '--data-dir', 'run', '--no-monitor')
        assert result['status'] == 'finished' and result['stop_reason'] == 'researcher_finished', result
        assert result['cleanup'] == {'status': 'completed'}, result
        assert result['postprocessing']['snapshot'] == {'status': 'completed'}, result
        assert Path(result['snapshot']).is_dir(), result
        assert result['postprocessing']['controller_report'] == {'status': 'completed'}, result
        assert result['effective_run_limits'] == {'rounds': 1, 'max_decision_steps': 12,
                                                   'max_decision_steps_source': 'saved_config'}, result

        def train_receipt(run):
            receipt, = [event['operation_result'] for event in run['events']
                        if (event.get('operation_result') or {}).get('summary') == 'train completed']
            assert receipt['simulated'] and not receipt['data']['training_performed'], receipt
            return receipt['call_id']

        assert train_receipt(paused) != train_receipt(result), 'the original training operation was replayed'
        assert json.loads((root / 'run' / 'result.json').read_text(encoding='utf-8')) == result
    print('Installed CLI wheel: catalog, explain, and durable ReST^EM run/resume passed')


if __name__ == '__main__':
    main()
