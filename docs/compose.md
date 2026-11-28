# Compose a system

For the guarantees and Python interfaces shared by all assembly paths, see
[Framework contracts](framework-contracts.md).

A run config chooses what to run in its `framework` section. There are two
forms:

```jsonc
// A complete registered method: its Controller and Labs are fixed
"framework": {"method": {"name": "rest_em", "config": {"controller": {…}, "labs": {"math": {…}}}}}

// A registered Controller paired with registered Labs that you choose
"framework": {
  "controller": {"name": "single_loop", "config": {…}},
  "labs": {"math": {"name": "template_math", "config": {…}}}
}
```

Use the method form to run a built-in algorithm. Use the Controller-and-Labs
form to try a Controller on a different Lab, or a Lab with a different
Controller. In the second form, each key under `labs` is an **alias**. The
Controller calls operations as `<alias>.<operation>`, so the alias must match
the name the Controller expects (`math` in the examples below).

The examples use the template Controller and Lab, so they need the
`--plugin src/rsicomposer/labs/_template --plugin src/rsicomposer/algorithms/_template`
options and a [Chat Completions endpoint](../src/rsicomposer/labs/_template/README.md#connect-a-gpu-model).
Run them from the repository root; the dataset paths are relative to it.

## Pair a registered Controller with a Lab

[`labs/_template/configs/run.json`](../src/rsicomposer/labs/_template/configs/run.json)
selects the template Controller and the template Lab separately:

```json
"framework": {
  "controller": {"name": "template_prompt_search", "config": {"max_evaluations": 3}},
  "labs": {
    "math": {
      "name": "template_math",
      "config": {
        "datasets": {
          "train": "src/rsicomposer/labs/_template/data/train.jsonl",
          "dev": "src/rsicomposer/labs/_template/data/dev.jsonl"
        }
      }
    }
  }
}
```

```sh
rsicomposer run --plugin src/rsicomposer/labs/_template --plugin src/rsicomposer/algorithms/_template \
  --config src/rsicomposer/labs/_template/configs/run.json --data-dir runs/compose
```

## Swap the Controller

Replace the Controller and keep the Lab. `single_loop` calls one operation a
fixed number of times; here it measures one prompt setting:

```json
"controller": {
  "name": "single_loop",
  "config": {
    "operation": "math.evaluate",
    "arguments": {"split": "dev", "instruction": "Solve the arithmetic problem.", "few_shot": 2}
  }
}
```

At startup, the host checks that the selected Lab alias exposes the operation
configured for `single_loop`. For example, `math.evaluate` requires the `math`
Lab to publish `evaluate`. This check does not validate operation arguments.

## Write a Controller and load it as a plugin

A plugin is a Python file or package with a `register(registry)` function. Load
it with `--plugin` and select its registered name in the framework configuration.
Labs that accept an installed runtime with host access require explicit
`trust = ["host-code"]` in the Lab configuration.

Save this as `few_shot_sweep/__init__.py`. It evaluates one instruction with
several few-shot counts on the template Lab and reports the best count:

```python
"""Try each few-shot count with one instruction and keep the best."""
from rsicomposer.api import Decision, ExecutionStatus, OperationCall


class FewShotSweep:
    def __init__(self, context, config):
        self.instruction = config['instruction']
        self.counts = list(config['few_shot_counts'])
        self.metric = context.lab_catalog.primary_metric('math')
        self.pending, self.scores = None, {}

    def decide(self, window):
        if self.pending is not None:
            receipt = next(r for r in window.feedbacks if r.call_id == self.pending.call_id)
            if receipt.status != ExecutionStatus.COMPLETED:
                return Decision('finish', reason=f'Evaluation failed: {receipt.error}')
            self.scores[self.pending.arguments['few_shot']] = self.metric.value(receipt.data)
            self.pending = None
        remaining = [count for count in self.counts if count not in self.scores]
        if not remaining:
            return Decision('finish', reason='All few-shot counts evaluated')
        self.pending = OperationCall('math.evaluate', {
            'split': 'dev', 'instruction': self.instruction, 'few_shot': remaining[0]})
        return Decision.invoke(self.pending)

    def report(self):
        best = max(self.scores, key=self.scores.get) if self.scores else None
        return {'scores': self.scores, 'best_few_shot': best}


def register(registry):
    registry.register_controller('few_shot_sweep', FewShotSweep, version='1')
```

Select it in the run config, with the same `labs` section as above:

```json
"controller": {
  "name": "few_shot_sweep",
  "config": {"instruction": "Solve the arithmetic problem.", "few_shot_counts": [0, 1, 2]}
}
```

```sh
rsicomposer run --plugin src/rsicomposer/labs/_template --plugin src/rsicomposer/algorithms/_template \
  --plugin few_shot_sweep --config few_shot_sweep.json --data-dir runs/few-shot-sweep
```

`controller_report` in `result.json` contains the value that `report()` returns.

Things to know when writing a Controller:

- **Match receipts by `call_id`.** `window.feedbacks` holds the receipts
  delivered at this step.
- **Declare required operations.** Define `required_operations()` to return a
  tuple of distinct, fully qualified names such as `("math.evaluate",)`. The
  host rejects missing names at startup, before the first decision.
- **Follow the Lab's rules.** The host enforces call permissions and Lab profile
  rules at dispatch. Local operation names cannot contain dots: the host uses
  the dot to separate the Lab alias from the local name. The Lab validates
  domain-specific arguments; an operation
  name passing the startup check does not guarantee valid arguments. Your
  `input_schema` is not enforced for you either: check `properties` and
  `required` at the top of `execute()` and raise `LabToolError` for a
  call the Controller could have got right, so it comes back as feedback rather
  than a failed run.
- **Keep `model` free.** If a Policy has a `model` attribute, the host treats
  it as the Policy's own decision model and requires a `ModelRef`.
- **For durable runs**, add `state()` and `restore(state)`; see the
  [method template](../src/rsicomposer/algorithms/_template/controller.py).
- **Call the `decision_maker` model yourself** when the Controller writes
  candidates, using `context.models` and the `decision_maker` role. A call made
  inside `decide()` is repeated with the same seed if the run stops before the
  decision is checkpointed. To keep sampling while Lab operations run, run the
  calls on your own threads and add `inference_ready()`: the host then calls
  `decide()` as soon as it returns `True`, without waiting for a Lab receipt.
  [FunSearch's Controller](../src/rsicomposer/algorithms/funsearch/controller.py)
  does this with its samplers.

## Package a method

When a Controller and its Labs belong together, register them as one method so
users can select a single name:

```python
from rsicomposer.api import MethodDefinition

def register(registry):
    registry.register_method('my_method', MethodDefinition.from_factories(
        controller=MyController,
        labs={'math': MyLab},
        version='1',
    ))
```

Both factories receive `(context, config)`. The method config passes
`config['controller']` to the Controller and `config['labs'][<alias>]` to each
Lab. The [method template](../src/rsicomposer/algorithms/_template/README.md) is a
complete example. It also registers its Controller on its own, so other Labs can
use it.

To write a Lab, start from the [Lab template](../src/rsicomposer/labs/_template/README.md).

## Add Memory or a custom Harness

A Controller is a Policy plus two optional parts, each with its own factory:

| Part | Factory | What it does |
| --- | --- | --- |
| Harness | `harness(context, config)` | Builds the Policy's `DecisionInput` and supplies local tools |
| Memory | `memory(config)` | Turns recorded evidence into text context |

Omit either one to use the default. Pass them to `ControllerDefinition`:

```python
from rsicomposer.api import ControllerDefinition, MemoryAssembly, ResearchAssembly

def harness(context, config):
    return MyHarness(context, config)

def memory(config):
    return MemoryAssembly(ResearchAssembly(context_memory=MyMemory()))

controller = ControllerDefinition(build_policy, harness=harness, memory=memory)
```

`MyMemory` implements [`ControllerMemory`](../src/rsicomposer/loops/contracts/feedback_context.py);
`LangMemMemory` is a ready-made option (`.[langmem]`). The
[method template](../src/rsicomposer/algorithms/_template/README.md) includes a
working [Harness](../src/rsicomposer/algorithms/_template/harness.py) and
[Memory](../src/rsicomposer/algorithms/_template/memory.py) that both run with it;
select them with `"harness": "scoreboard"` and `"memory": "best_score"` in the
Controller config. The registration code validates those choices and passes only
Policy options to the Policy factory. To contribute Memory tools, return public
`Capability` objects from `capabilities()`; the endpoint exposes them in the
authorized window and binds their handlers without editing the Policy's parser.

Three things to know:

- **A custom Harness replaces the default one**, including its local tools and
  navigation. Memory is separate: the host adds memory text to
  `window.context`, so both work together.
- **Memory needs a `decision_maker` role** in the run config, because the host
  attributes memory observations to it.
- **Mark operations that call a model or change data as `read_only=False`.** The
  generic feedback pipeline does not schedule learning work for read-only receipts.

When using `MetricGoal`, only operations declared as evaluations by the Lab
profile participate. A custom Lab can return the local operation names from
`evaluation_operations()`. A qualifying evaluation receipt includes
`candidate_id`, a versioned `protocol_ref`, exact `input_refs` and the measured
`quality_metrics`. Use a distinct protocol reference per dataset, or provide
`input_refs` in the goal to select exactly which cohort counts. Training scores
and simulated receipts (unless explicitly accepted) cannot satisfy the goal.
