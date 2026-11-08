"""Select a whole or internally composed RSI method through the same CPU host.

Numerical composition fixture only: no model calls, GPU work or medical claims.
The whole implementation keeps its policy/memory in the meta-agent harness;
the Lab owns its fitting/inference workflow. Internal graph slots are optional.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.components.modular_research import (LinearLab, ScheduledResearch, VectorRuntime,
    analytic_training, direct_inference, modules as composed_modules)
from rsisys.assembly import ModuleAssembly, build_default_decision
from rsisys.config import DemoConfig, build_demo
from rsisys.methods import MethodCatalog, MethodDefinition


ROOT = Path(__file__).resolve().parents[2]


def validate_config(config):
    if set(config) - {'attempts'}:
        raise ValueError('Unknown method parameter')
    if type(config.get('attempts', 1)) is not int or config.get('attempts', 1) not in (1, 2):
        raise ValueError('Synthetic attempts must be 1 or 2')


def whole_method(config):
    def lab(context):
        return LinearLab({'data': ((-1, -1), (0, 1), (1, 3)),
            'training': analytic_training, 'inference': direct_inference,
            'train_engine': VectorRuntime(), 'infer_engine': VectorRuntime()})
    def decision(context, exploration):
        return build_default_decision(context, exploration,
            policy=ScheduledResearch(config.get('attempts', 1)))
    return ModuleAssembly(decision_factory=decision, lab_factory=lab)


def composed_method(config):
    selections = json.loads((ROOT / 'configs/modular-demo.json').read_text())
    selections['training']['component'] = 'analytic'
    selections['rsi']['component'] = 'repeat' if config.get('attempts', 1) == 2 else 'single_pass'
    return composed_modules(selections)


def catalog():
    return MethodCatalog({name: MethodDefinition(factory, '1', validate_config=validate_config)
        for name, factory in (('whole', whole_method), ('composed', composed_method))})


def run(selection):
    modules = catalog().select(selection)
    runner, _ = build_demo(DemoConfig.load(ROOT / 'configs/demo.json'), modules=modules)
    with runner:
        result = runner.run('Check method integration', 1, max_decision_steps=8)
        return {'status': result.status, 'method': runner.assembly.method_identity.configuration(),
                'results': runner.assembly.lab_execution.results}


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'configs/rsi-method-demo.json'
    print(json.dumps(run(json.loads(path.read_text())), indent=2))
