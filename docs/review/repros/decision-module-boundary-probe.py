"""Offline replacement-boundary probes; records observations, not live acceptance.

Run from the repository with PYTHONPATH=src. This script deliberately preserves
the current behavior without patching production objects. The original pre-fix
JSON remains historical evidence; new runs write the separate current result.
"""
from dataclasses import asdict
from pathlib import Path
import json

from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import LoopStepResult
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.harness.context import IdentityContextCompactor
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.run_policy import TimeRunPolicy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = DemoConfig.load(ROOT / 'configs/demo.json')
MODEL = ModelRef('offline-fixture', 'meta-agent', 'one')


class ScheduledHarness:
    """Uses all three public methods and validates its directive before return."""
    def __init__(self):
        self.directive = None
        self.done = False

    def step(self, objective, action_space):
        if self.done:
            return LoopStepResult('finished')
        self.directive = Directive('boundary-fixture', objective, 'execute once')
        action_space.validate_directive(self.directive)
        return LoopStepResult('dispatch', directive=self.directive)

    def pending(self):
        return self.directive if not self.done else None

    def receive_feedback(self, directive, feedback):
        if directive != self.directive or feedback.directive_id != directive.directive_id:
            raise ValueError('Unexpected fixture feedback')
        self.done = True


def scheduled_factory(context, exploration):
    return DecisionMakingLoop(ScheduledHarness(), DemoDecisionActionSpace(
        permissions=context.action_permissions))


class PlainPolicy:
    def decide(self, context):
        return Decision('finish')


class FixtureBackend:
    supports_response_schema = False

    def __init__(self, *, invalid_first=False, all_invalid=False):
        self.calls = 0
        self.invalid_first = invalid_first
        self.all_invalid = all_invalid

    def generate(self, request):
        self.calls += 1
        invalid = self.all_invalid or (self.invalid_first and self.calls == 1)
        return GenerationResponse('invalid' if invalid else '{"kind":"finish"}',
            request.model, simulated=True, usage=TokenUsage(11, 3))


class TransparentPolicy:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decide(self, context):
        return self.inner.decide(context)


class RecordingCompactor:
    def __init__(self):
        self.recent_limits = []

    def compact(self, context, *, recent_limit=32):
        self.recent_limits.append(recent_limit)
        return context


def error_value(exc):
    return {'type': type(exc).__name__, 'message': str(exc)}


def main():
    results = {}
    with build_demo(CONFIG, decision_factory=scheduled_factory)[0] as runner:
        result = runner.run('allowed offline fixture', 1)
        results['minimal_harness_positive_control'] = {
            'status': result.status, 'rounds_completed': result.rounds_completed,
            'has_policy': hasattr(runner.decision_making.harness, 'policy'),
            'has_memory': hasattr(runner.decision_making.harness, 'memory')}

    permissions = []
    for mode, policy in (('researcher', None), ('time', TimeRunPolicy(60))):
        with build_demo(CONFIG, decision_factory=scheduled_factory,
                action_permissions=ActionPermissions(directives=()), run_policy=policy)[0] as runner:
            row = {'mode': mode, 'permitted_directives': []}
            try:
                result = runner.run('denied offline fixture', 1, max_decision_steps=4)
                row.update(status=result.status, rounds_completed=result.rounds_completed)
            except Exception as exc:
                row['error'] = error_value(exc)
            permissions.append(row)
    results['directive_permission_modes'] = permissions

    with build_demo(CONFIG)[0] as runner:
        runner.decision_making.harness.policy = PlainPolicy()
        results['minimal_policy_positive_control'] = {
            'step_kind': runner.decision_making.step('plain policy fixture').kind}

    accounting = []
    for all_invalid in (False, True):
        for wrapped in (False, True):
            with build_demo(CONFIG)[0] as runner:
                backend = FixtureBackend(invalid_first=True, all_invalid=all_invalid)
                policy = ModelDecisionPolicy(backend, MODEL, max_validation_attempts=2,
                    context_compactor=IdentityContextCompactor())
                harness = runner.decision_making.harness
                harness.policy = TransparentPolicy(policy) if wrapped else policy
                harness.context_compactor = IdentityContextCompactor()
                row = {'wrapped': wrapped, 'all_invalid': all_invalid}
                try:
                    row['step_kind'] = runner.decision_making.step('usage fixture').kind
                except Exception as exc:
                    row['error'] = error_value(exc)
                row.update(backend_calls=backend.calls,
                    backend_usage=asdict(TokenUsage(backend.calls * 11, backend.calls * 3)),
                    harness_usage=asdict(TokenUsage(harness.state.measured_input_tokens,
                        harness.state.measured_output_tokens)))
                accounting.append(row)
    results['transparent_wrapper_accounting'] = accounting

    compaction = []
    for wrapped in (False, True):
        with build_demo(CONFIG)[0] as runner:
            policy_compactor, harness_compactor = RecordingCompactor(), RecordingCompactor()
            policy = ModelDecisionPolicy(FixtureBackend(), MODEL, context_compactor=policy_compactor)
            harness = runner.decision_making.harness
            harness.policy = TransparentPolicy(policy) if wrapped else policy
            harness.context_compactor = harness_compactor
            harness.memory_limit = 7
            runner.decision_making.step('compaction owner fixture')
            compaction.append({'wrapped': wrapped,
                'policy_compactor_preserved': policy.context_compactor is policy_compactor,
                'policy_compactor_recent_limits': policy_compactor.recent_limits,
                'harness_compactor_recent_limits': harness_compactor.recent_limits})
    results['transparent_wrapper_compaction'] = compaction

    feedback = []
    for explicit_backend in (False, True):
        with build_demo(CONFIG)[0] as runner:
            runner.decision_making.harness.policy = PlainPolicy()
            backend = FixtureBackend()
            row = {'explicit_backend': explicit_backend, 'assembly_requires_model': False}
            try:
                service = assemble_research(runner.decision_making,
                    assembly=ResearchAssembly(context_memory=lambda query: 'current context'),
                    backend=backend if explicit_backend else None, owner=runner.assembly)
                row['assembly_installed'] = True
                row['step_kind'] = runner.decision_making.step('zero inference fixture').kind
                service.observe_output('fixture result', task_id='fixture-task', status='completed')
                row['receipt_captured'] = True
            except Exception as exc:
                row['error'] = error_value(exc)
            row['model_calls'] = backend.calls
            feedback.append(row)
    results['plain_policy_zero_inference_feedback'] = feedback

    output = {'scope': '2026-11-04 current decision harness/policy boundary probes',
        'real_model_inference': False, 'real_training': False,
        'probes': results}
    path = Path(__file__).with_name('decision-module-boundary-probe-current.json')
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(path.read_text())


if __name__ == '__main__':
    main()
