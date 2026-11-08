"""Independent loop demos. Every learner/evaluator here is simulated."""

from dataclasses import asdict
import json
from pathlib import Path

from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.loops.exploration.harness.evaluation import EvaluationRequest
from rsisys.loops.exploration.harness.rl import RLPlan


def finish(poll, job):
    # Bounded fixture polling only. Production callers choose when to poll.
    for _ in range(8):
        outcome = poll(job)
        if outcome.terminal:
            if outcome.status != JobStatus.SUCCEEDED:
                raise RuntimeError(outcome.error or str(outcome.status))
            return outcome.result
    raise RuntimeError('Mock job did not finish within the fixture budget')


def main():
    config = DemoConfig.load(Path(__file__).resolve().parents[2] / 'configs/demo.json')
    runner, store = build_demo(config, with_learning=True)
    with runner:
        decision, actor = runner.decision_making, runner.exploration
        artifacts = decision.harness.tools.artifacts

        decision.step('Prepare an independent decision-learning example')
        request = DecisionTrainingRequest(
            decision.harness.policy.model,
            tuple(r.record_id for r in store.records() if r.scope == 'decision_maker'),
            'mock-decision-recipe',
        )
        job = decision.harness.submit_learning(request, idempotency_key='demo-decision-learning')
        decision_result = finish(decision.harness.poll_learning, job)
        decision.harness.apply_learning(job)

        data = artifacts.put(b'An explicitly simulated training dataset', kind='training_data')
        job = actor.harness.submit_learning(ActorTrainingRequest(actor.harness.policy.model, (data,), 'mock-sft-recipe'),
                                    idempotency_key='demo-sft')
        sft_result = finish(actor.harness.poll_learning, job)
        actor.harness.apply_learning(job)

        actor.harness.start_rl(RLPlan(Directive('demo-rl', 'Interact with the mock environment', 'echo'),
                             'mock-rl-recipe', updates=2))
        for _ in range(20):
            rl_state = actor.harness.step_rl(action_space=actor.action_space)
            if rl_state.terminal:
                break
        if rl_state.phase != 'completed':
            raise RuntimeError(rl_state.error or f'RL demo ended in {rl_state.phase}')

        eval_job = actor.harness.submit_evaluation(EvaluationRequest(rl_state.results[-1].candidate, 'mock-eval'),
                                           idempotency_key='demo-evaluation')
        evaluation = finish(actor.harness.poll_evaluation, eval_job)
        print(json.dumps({
            'simulated': True,
            'training_performed': False,
            'quality_evaluated': False,
            'decision_learning': asdict(decision_result),
            'actor_sft': asdict(sft_result),
            'actor_rl': {
                'status': rl_state.phase,
                'updates_completed': rl_state.updates_completed,
                'model_revisions': [sft_result.model.revision] + [r.model.revision for r in rl_state.results],
            },
            'evaluation': asdict(evaluation),
            'experience_records': len(store.records()),
            'role_bindings': {'decision_maker': asdict(decision.harness.policy.model), 'actor': asdict(actor.harness.policy.model)},
        }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
