# Compose a system

A run config chooses what to run in its `framework` section. There are two
forms:

```jsonc
// A complete registered method: its Controller and Labs are fixed
"framework": {"method": {"name": "rest_em", "config": {"controller": {…}, "labs": {"math": {…}}}}}

// A registered Controller paired with registered Labs that you choose
"framework": {
  "controller": {"name": "rest_em", "config": {…}},
  "labs": {"math": {"name": "rest_em", "config": {…}}}
}
```

Use the method form to run a built-in algorithm as published. Use the
Controller-and-Labs form to try a Controller on a different Lab, or a Lab with
a different Controller. In the second form, each key under `labs` is an
**alias**. The Controller calls operations as `<alias>.<operation>`, so the
alias must match the name the Controller expects (`math` for the examples
below).

Every example on this page runs on a CPU. Run them from the repository root;
the `data_root` paths are relative to it.

## Pair a registered Controller with a Lab

This config runs the same ReST^EM loop as the [Quickstart](quickstart.md), but
selects the Controller and the Lab separately:

```json
{
  "objective": "Pair the rest_em Controller with the rest_em Lab",
  "data_dir": "runs/compose",
  "framework": {
    "controller": {
      "name": "rest_em",
      "config": {
        "initial_model": "fixture-initial", "train_data": "train.jsonl", "eval_data": "eval.jsonl",
        "iterations": 1, "samples_per_prompt": 4, "max_accepted_per_prompt": 1
      }
    },
    "labs": {
      "math": {
        "name": "rest_em",
        "config": {"engine": "fixture", "data_root": "src/rsisys/algorithms/rest_em/configs/cpu/data"}
      }
    }
  }
}
```

```sh
rsisys run --config compose.json --data-dir runs/compose
```

## Swap the Controller

Replace the Controller and keep the Lab. `single_loop` calls one operation a
fixed number of times; here it only measures the starting model:

```json
"controller": {
  "name": "single_loop",
  "config": {
    "operation": "math.evaluate",
    "arguments": {"round": 0, "model": "fixture-initial", "dataset": "eval.jsonl", "split": "dev"}
  }
}
```

To run on a GPU instead, keep the Controller and change the Lab's `engine` to
`transformers` or `vllm`. The [ReST^EM guide](../src/rsisys/algorithms/rest_em/README.md)
lists the model and data settings.

## Write a Controller and load it as a plugin

A plugin is a Python file or package with a `register(registry)` function. Load
it with `--plugin`; RSISys never imports code named in a config.

Save this as `my_controller/__init__.py`. It runs one self-training round on the
built-in `rest_em` Lab and keeps the better model:

```python
"""One self-training round on the built-in rest_em Lab, keeping the better model."""
from rsisys.api import Decision, ExecutionStatus, OperationCall

STAGES = ('evaluate_base', 'generate', 'filter', 'train', 'evaluate_new')


class OneRoundController:
    def __init__(self, context, config):
        self.base_model = config['model']
        self.train_data, self.eval_data = config['train_data'], config['eval_data']
        self.stage, self.pending, self.outputs = 0, None, {}

    def decide(self, window):
        if self.pending is not None:
            receipt = next(r for r in window.feedbacks if r.call_id == self.pending.call_id)
            if receipt.status != ExecutionStatus.COMPLETED:
                return Decision('finish', reason=f'{STAGES[self.stage]} failed: {receipt.error}')
            self.outputs[STAGES[self.stage]] = receipt.data
            self.stage += 1
        if self.stage == len(STAGES):
            return Decision('finish', reason='Round complete')
        self.pending = OperationCall('math.' + STAGES[self.stage].split('_')[0], self._arguments())
        return Decision.invoke(self.pending)

    def _arguments(self):
        name = STAGES[self.stage]
        if name == 'evaluate_base':
            return {'round': 0, 'model': self.base_model, 'dataset': self.eval_data, 'split': 'dev',
                    'baseline': True}
        if name == 'generate':
            return {'round': 1, 'model': self.base_model, 'dataset': self.train_data,
                    'samples_per_prompt': 4, 'temperature': 0.7, 'condition': 'none'}
        if name == 'filter':
            return {'round': 1, 'samples': self.outputs['generate']['dataset'], 'max_accepted_per_prompt': 1}
        if name == 'train':
            return {'round': 1, 'parent': self.base_model, 'dataset': self.outputs['filter']['dataset']}
        return {'round': 1, 'model': self.outputs['train']['model'], 'dataset': self.eval_data, 'split': 'dev'}

    def report(self):
        scores = {name: self.outputs[name]['quality_metrics']['accuracy']
                  for name in ('evaluate_base', 'evaluate_new') if name in self.outputs}
        keep_new = scores.get('evaluate_new', -1) > scores.get('evaluate_base', -1)
        return {'scores': scores,
                'selected': self.outputs['train']['model'] if keep_new else self.base_model}


def register(registry):
    registry.register_controller('one_round', OneRoundController, version='1')
```

Select it in the run config, with the same `labs` section as above:

```json
"controller": {
  "name": "one_round",
  "config": {"model": "fixture-initial", "train_data": "train.jsonl", "eval_data": "eval.jsonl"}
}
```

```sh
rsisys run --plugin my_controller --config one_round.json --data-dir runs/one-round
```

`controller_report` in `result.json` contains the value that `report()` returns.

Things to know when writing a Controller:

- **Match receipts by `call_id`.** `window.feedbacks` holds the receipts
  delivered at this step.
- **Follow the Lab's rules.** The host checks each call against the Lab's
  schema and profile and returns a failed receipt if a rule is broken. For
  example, the `rest_em` Lab requires a baseline evaluation
  (`"baseline": true`) before it accepts `generate`.
- **Keep `model` free.** If a Policy has a `model` attribute, the host treats
  it as the Policy's own decision model and requires a `ModelRef`.
- **For durable runs**, add `state()` and `restore(state)`; see the
  [method template](../src/rsisys/algorithms/_template/controller.py).

## Package a method

When a Controller and its Labs belong together, register them as one method so
users can select a single name:

```python
from rsisys.api import MethodDefinition

def register(registry):
    registry.register_method('my_method', MethodDefinition.from_factories(
        controller=MyController,
        labs={'math': MyLab},
        version='1',
    ))
```

Both factories receive `(context, config)`. The method config passes
`config['controller']` to the Controller and `config['labs'][<alias>]` to each
Lab. The [method template](../src/rsisys/algorithms/_template/README.md) is a
complete example. It also registers its Controller on its own, so other Labs can
use it.

To write a Lab, start from the [Lab template](../src/rsisys/labs/_template/README.md).

## Add Memory or a custom Harness

Memory and Harness apply to Controllers whose Policy is a model, such as
`ModelDecisionPolicy`. Declare them in a `ControllerDefinition`:

```python
from rsisys.api import ControllerDefinition, HarnessAssembly, ResearchAssembly

def harness(config):
    return HarnessAssembly(ResearchAssembly(context_memory=MyMemory()))

controller = ControllerDefinition(build_policy, harness=harness, endpoint=True)
```

`MyMemory` implements [`ControllerMemory`](../src/rsisys/loops/contracts/feedback_context.py);
`LangMemMemory` is a ready-made option (`.[langmem]`). The
[`meta_agent` registration](../src/rsisys/algorithms/meta_agent/__init__.py)
shows a complete model-driven Controller with Memory.
