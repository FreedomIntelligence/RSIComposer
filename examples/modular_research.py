"""CPU composition demonstration, not a medical run or model-quality benchmark.

The same host loop composes independently selected data, learning, inference,
RSI scheduling, harness and execution implementations. No LLM, GPU or service.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

from rsisys.assembly import ModuleAssembly, build_default_decision
from rsisys.components import ComponentDefinition, ComponentGraph
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsisys.loops.decision_making.loop import DecisionMakingLoop


class VectorRuntime:
    def __init__(self, precise=False):
        self.precise = precise

    def dot(self, a, b):
        values = [x * y for x, y in zip(a, b, strict=True)]
        return math.fsum(values) if self.precise else sum(values)

    def update(self, weights, gradient, rate):
        return tuple(w - rate * g for w, g in zip(weights, gradient, strict=True))


def gradient_training(data, runtime):
    weights = (0.0, 0.0)
    for _ in range(40):
        errors = [runtime.dot(weights, (x, 1)) - y for x, y in data]
        gradient = (2 * runtime.dot(errors, [x for x, _ in data]) / len(data),
                    2 * sum(errors) / len(data))
        weights = runtime.update(weights, gradient, 0.1)
    return weights


def analytic_training(data, runtime):
    xs, ys = [x for x, _ in data], [y for _, y in data]
    xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
    centered = [x - xmean for x in xs]
    slope = runtime.dot(centered, [y - ymean for y in ys]) / runtime.dot(centered, centered)
    return (slope, ymean - slope * xmean)


def direct_inference(weights, inputs, runtime):
    return [runtime.dot(weights, (x, 1)) for x in inputs]


def rounded_inference(weights, inputs, runtime):
    return [float(round(v)) for v in direct_inference(weights, inputs, runtime)]


@dataclass
class ScheduledResearch:
    attempts: int
    issued: int = 0

    def decide(self, context):
        if self.issued >= self.attempts:
            return Decision('finish', reason='Synthetic schedule finished', simulated=True)
        self.issued += 1
        return Decision('invoke', operation=OperationCall('fit_predict', {}), simulated=True)


class TracingHarness:
    def __init__(self, delegate):
        self.delegate, self.steps = delegate, 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def step(self, objective, space):
        self.steps += 1
        return self.delegate.step(objective, space)


class LinearLab:
    def __init__(self, dependencies):
        self.dependencies, self.results, self.closed = dependencies, [], 0

    def describe(self):
        return (OperationSpec('fit_predict', 'Run the selected synthetic linear pipeline',
            {'type': 'object', 'properties': {}, 'additionalProperties': False}),)

    def execute(self, call):
        if call.name != 'fit_predict' or call.arguments != {}:
            raise ValueError('Expected fit_predict with no arguments')
        d = self.dependencies
        weights = d['training'](d['data'], d['train_engine'])
        predictions = d['inference'](weights, [0.25, 0.75], d['infer_engine'])
        record = {'weights': list(weights), 'predictions': predictions,
                  'scope': 'synthetic composition check; no medical claim'}
        self.results.append(record)
        return OperationResult('Synthetic pipeline executed', (), True, data=record)

    def close(self):
        self.closed += 1


def _callable(value):
    if not callable(value):
        raise TypeError('Selected strategy must be callable')


def _decision_factory(dependencies, traced):
    def factory(context, exploration):
        loop = build_default_decision(context, exploration, policy=dependencies['rsi'])
        return DecisionMakingLoop(TracingHarness(loop.harness), loop.action_space) if traced else loop
    return factory


def catalog():
    """Instance-local host catalog. A JSON file cannot import arbitrary Python."""
    return {
        'linear_data': ComponentDefinition(lambda c, d, p: ((-1, -1), (0, 1), (1, 3)),
            '1', ('linear-data/v1',), stateless=True),
        'python_vectors': ComponentDefinition(lambda c, d, p: VectorRuntime(),
            '1', ('vector-runtime/v1',), stateless=True),
        'precise_vectors': ComponentDefinition(lambda c, d, p: VectorRuntime(precise=True),
            '1', ('vector-runtime/v1',), stateless=True),
        'gradient': ComponentDefinition(lambda c, d, p: gradient_training,
            '1', ('linear-training/v1',), validate=_callable, stateless=True),
        'analytic': ComponentDefinition(lambda c, d, p: analytic_training,
            '1', ('linear-training/v1',), validate=_callable, stateless=True),
        'direct': ComponentDefinition(lambda c, d, p: direct_inference,
            '1', ('linear-inference/v1',), validate=_callable, stateless=True),
        'rounded': ComponentDefinition(lambda c, d, p: rounded_inference,
            '1', ('linear-inference/v1',), validate=_callable, stateless=True),
        'single_pass': ComponentDefinition(lambda c, d, p: ScheduledResearch(1),
            '1', ('rsi-policy/v1',)),
        'repeat': ComponentDefinition(lambda c, d, p: ScheduledResearch(2),
            '1', ('rsi-policy/v1',)),
        'linear_lab': ComponentDefinition(lambda c, d, p: LinearLab(d),
            '1', ('lab-execution/v1',), requires={'data': 'linear-data/v1',
                'training': 'linear-training/v1', 'inference': 'linear-inference/v1',
                'train_engine': 'vector-runtime/v1', 'infer_engine': 'vector-runtime/v1'}, owned=True),
        'default_harness': ComponentDefinition(lambda c, d, p: _decision_factory(d, False),
            '1', ('decision-factory/v1',), requires={'rsi': 'rsi-policy/v1'},
            validate=_callable, stateless=True),
        'tracing_harness': ComponentDefinition(lambda c, d, p: _decision_factory(d, True),
            '1', ('decision-factory/v1',), requires={'rsi': 'rsi-policy/v1'},
            validate=_callable, stateless=True),
    }


def modules(selections):
    return ModuleAssembly.from_components(ComponentGraph(catalog(), selections), decision='harness', lab='lab')


def run(selections):
    config = DemoConfig.load(Path(__file__).resolve().parents[1] / 'configs/demo.json')
    runner, _ = build_demo(config, modules=modules(selections))
    with runner:
        result = runner.run('Check synthetic component composition', 1, max_decision_steps=8)
        return {'status': result.status, 'results': runner.assembly.lab_execution.results,
                'components': runner.assembly.components.configuration()}


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'configs/modular-demo.json'
    print(json.dumps(run(json.loads(path.read_text())), indent=2))
