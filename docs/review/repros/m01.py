"""CPU-only M01 audit probes; assertions describe current defects, not desired behavior.

Run from the repository root with PYTHONPATH=src python3.12 docs/review/repros/m01.py.
No model service, training process, or persistent user workspace is used.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from rsisys.application import Application, ApplicationConfig
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.coordination import LoopStepResult, OperationResult
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.runtime.run_policy import TimeRunPolicy
from rsisys.runtime.runner import Runner


class Policy:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return self.decision


class Tools:
    def __init__(self, status):
        self.status = status
        self.calls = 0

    def execute(self, call):
        self.calls += 1
        return OperationResult('executor completed', (), True, status=self.status)


def captured(call):
    try:
        result = call()
        return {'kind': getattr(result, 'kind', type(result).__name__)}
    except Exception as exc:
        return {'error_type': type(exc).__name__, 'error': str(exc)}


def operation_status():
    rows = []
    for status in (ExecutionStatus.COMPLETED, 'completed'):
        tools = Tools(status)
        policy = Policy(Decision('invoke', operation=OperationCall('prepare_data', {'text': 'fixture'})))
        harness = DefaultDecisionHarness(policy, tools, BasicExperienceSystem(InMemoryExperienceStore()))
        space = DemoDecisionActionSpace()
        first = captured(lambda: harness.step('audit', space))
        if isinstance(status, ExecutionStatus):
            assert first['kind'] == 'progressed'
            retry = None
        else:
            retry = captured(lambda: harness.step('audit', space))
            assert first['error_type'] == retry['error_type'] == 'AttributeError'
            assert harness.state.pending_operation_record is not None
            assert tools.calls == policy.calls == 1
        rows.append({'status_type': type(status).__name__, 'first': first, 'retry': retry,
                     'executor_calls': tools.calls, 'policy_calls': policy.calls,
                     'pending_record': harness.state.pending_operation_record is not None})
    return rows


class MinimalHarness:
    """All three required DecisionHarness methods, without optional control hooks."""
    def __init__(self):
        self.calls = 0

    def step(self, objective, action_space):
        self.calls += 1
        return LoopStepResult('progressed')

    def pending(self):
        return None

    def receive_feedback(self, directive, feedback):
        raise AssertionError('No exploration was requested')


class NoExploration:
    def pending(self):
        return None


def optional_control():
    rows = []
    for timed in (False, True):
        harness = MinimalHarness()
        loop = DecisionMakingLoop(harness, DemoDecisionActionSpace())
        runner = Runner(loop, NoExploration(), run_policy=TimeRunPolicy(60) if timed else None)
        first = captured(lambda: runner.step('audit'))
        if timed:
            assert first['error_type'] == 'NotImplementedError'
            assert harness.calls == 0
        else:
            assert first['kind'] == 'progressed' and harness.calls == 1
        rows.append({'timed': timed, 'first': first, 'harness_step_calls': harness.calls})
    return rows


def nonfinite_feedback():
    rows = []
    for metric in (0.5, float('nan'), float('inf')):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(ApplicationConfig.load(Path('configs/local.json')),
                             data_dir=Path(directory), execution='mock')
            with Application(config) as app:
                loop = app.runner.decision_making
                harness = loop.harness
                directive = Directive('audit-feedback', 'audit', 'fixture')
                harness.policy = Policy(Decision('explore', directive=directive))
                dispatched = loop.step('audit')
                assert dispatched.kind == 'dispatch'
                feedback = Feedback(directive.directive_id, ExecutionStatus.COMPLETED,
                                    'custom terminal receipt', True,
                                    quality_metrics={'accuracy': metric})
                first = captured(lambda: loop.receive_feedback(directive, feedback))
                retry = captured(lambda: loop.receive_feedback(directive, feedback))
                finite = metric == 0.5
                assert (harness.pending() is None) == finite
                if finite:
                    assert 'error_type' not in first and 'error_type' not in retry
                else:
                    assert first['error_type'] == retry['error_type'] == 'ValueError'
                    assert harness.state.completed == 0
                rows.append({'metric': 'finite' if finite else ('NaN' if metric != metric else 'Infinity'),
                             'constructor_accepted': True, 'first_delivery': first,
                             'retry_delivery': retry, 'directive_pending': harness.pending() is not None,
                             'completed': harness.state.completed})
    return rows


if __name__ == '__main__':
    print(json.dumps({'operation_status': operation_status(), 'optional_control': optional_control(),
                      'nonfinite_feedback': nonfinite_feedback()}, ensure_ascii=False, indent=2,
                     allow_nan=False))
