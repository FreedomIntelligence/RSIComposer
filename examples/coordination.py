"""Scripted policies exercise real coordination with simulated learning backends."""
from dataclasses import asdict
import json
from pathlib import Path

from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.requests import SFTSpec, RLSpec, EvaluationSpec
from rsisys.loops.decision_making.harness.decision import Decision


class ExampleDecisionPolicy:
    def __init__(self, data):
        self.data = data
        self.stage = 0

    def decide(self, context):
        self.stage += 1
        if self.stage == 1:
            directive = Directive('sft-example', context.objective, 'Produce a candidate', mode='sft',
                                  input_refs=(self.data,), payload=SFTSpec('mock-sft'))
        elif self.stage == 2:
            candidate = context.last_feedback.candidates[0]
            directive = Directive('eval-example', context.objective, 'Inspect the candidate', mode='evaluate',
                                  payload=EvaluationSpec(candidate, 'mock-protocol'))
        elif self.stage == 3:
            # This is a scripted demonstration, not acceptance based on a mock quality score.
            directive = Directive('rl-example', context.objective, 'Exercise two rollout/update iterations',
                                  mode='rl', payload=RLSpec('mock-rl', updates=2))
        else:
            return Decision('finish', reason='Demonstration completed')
        return Decision('explore', directive=directive, simulated=True)


def main():
    config = DemoConfig.load(Path(__file__).resolve().parents[1] / 'configs/demo.json')
    runner, store = build_demo(config, with_learning=True)
    with runner:
        data = runner.decision_making.harness.tools.artifacts.put(b'demonstration data', kind='training_data')
        runner.decision_making.harness.policy = ExampleDecisionPolicy(data)
        events = []
        for _ in range(40):
            step = runner.step('Demonstrate loop coordination')
            events.append({'kind': step.kind, 'directive_id': step.directive_id,
                           'phase': step.event.get('phase')})
            if step.kind == 'finished':
                break
        else:
            raise RuntimeError('Demo did not finish within its tick budget')
        print(json.dumps({'simulated': True, 'training_performed': False, 'quality_evaluated': False,
                          'decision_calls': runner.decision_making.harness.policy.stage,
                          'directives_completed': runner.decision_making.harness.state.completed,
                          'actor_model': asdict(runner.exploration.harness.policy.model),
                          'experience_records': len(store.records()), 'events': events}, indent=2))


if __name__ == '__main__':
    main()
