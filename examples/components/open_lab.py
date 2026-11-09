"""Open action outlet: real CPU code, deterministic decision fixture, no training.

Run: PYTHONPATH=src python examples/components/open_lab.py
The fixture exercises wiring, not autonomous model quality or SFT/RL.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

from rsisys.adapters.decision import ContextDecisionPolicy
from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.application import Application, ApplicationConfig
from rsisys.loops.contracts.decision import Decision, OperationCall
from rsisys.research import ResearchAssembly


def make_lab(context):
    return LocalProcessLab(context.sandbox, context.artifacts, timeout_seconds=30)


class FixturePolicy:
    def __init__(self):
        self.stage, self.job = 0, None

    def decide(self, window):
        view = json.loads(window.context)
        if self.stage == 0:
            self.stage = 1
            name, arguments = 'lab_write_file', {'path': 'my_evaluation.py',
                'content': 'import json\nprint(json.dumps({"fixture": True, "value": 2 + 3}))\n'}
        elif self.stage == 1:
            self.stage = 2
            name, arguments = 'lab_start', {'key': 'fixture-code', 'argv': [sys.executable, 'my_evaluation.py']}
        elif self.stage == 2:
            result = view['operation_results'][-1]
            self.job = result['data']['job_id']
            if result['data']['state'] in ('succeeded', 'failed', 'cancelled'):
                self.stage = 3
                name, arguments = 'lab_result', {'job_id': self.job}
            else:
                name, arguments = 'lab_poll', {'job_id': self.job}
        elif self.stage == 3:
            self.stage = 4
            name, arguments = 'lab_export', {'path': 'my_evaluation.py'}
        else:
            return Decision('finish', reason='CPU wiring fixture collected its result and code')
        return Decision('invoke', operation=OperationCall(name, arguments))


def main():
    template = ApplicationConfig.load(Path(__file__).resolve().parents[2] / 'examples/components/configs/local.json')
    with TemporaryDirectory(prefix='rsisys-open-lab-') as directory:
        config = replace(template, data_dir=Path(directory) / 'run', execution='local_process',
            roles={'decision_maker': template.roles['decision_maker']}, objective='Exercise open Lab wiring')
        with Application(config, lab_factory=make_lab, research=ResearchAssembly()) as app:
            fixture = FixturePolicy()
            app.runner.decision_making.harness.policy = ContextDecisionPolicy(fixture)
            until = time.monotonic() + 10
            while time.monotonic() < until:
                step = app.runner.step(config.objective)
                if step.kind == 'finished':
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError('CPU fixture did not finish')
            lab = app.assembly.lab_execution
            result = lab.execute(OperationCall('lab_result', {'job_id': fixture.job}))
            assert json.loads(result.data['stdout']) == {'fixture': True, 'value': 5}
            print(json.dumps({'status': 'fixture_passed', 'shutdown_complete': step.event['run_stop']['shutdown_complete'],
                'native_output': json.loads(result.data['stdout']), 'real_cpu_process': True,
                'real_model_inference': False, 'real_training_performed': False,
                'framework_sft_rl_evaluate_menu': False}, indent=2))


if __name__ == '__main__':
    main()
