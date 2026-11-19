# GPU method template

This example connects a prompt-search Controller to the reusable
[arithmetic Lab](../../labs/_template/README.md). The Lab calls a GPU-backed
model, scores its answers, and returns development receipts. The Controller
compares prompt and few-shot settings and selects the best measured one. It
changes the prompt configuration; the model weights stay fixed.

| File | What to change |
| --- | --- |
| [`controller.py`](controller.py) | The Controller prototype. Its docstring lists the Policy contract. `decide()` sends `math.evaluate` (the first candidate synchronously, the rest as background jobs), matches each receipt by `call_id`, waits while a job runs, and finishes after the last candidate. `report()` selects from measured candidates. |
| [`__init__.py`](__init__.py) | `register()` binds the Controller and a Lab factory. It resolves the template Lab through the registry so the Lab can also be used independently. Replace the factory if your method needs a different environment. |
| [`configs/run.json`](configs/run.json) | `framework.method` selects the registered method and passes the Controller and Lab options. `backends` and `roles.actor` select the GPU model. |

Start the [GPU endpoint](../../labs/_template/README.md#connect-a-gpu-model). Then copy
this package from the repository root and run it:

```sh
cp -R src/rsisys/algorithms/_template ./my_method
rsisys run --plugin src/rsisys/labs/_template --plugin ./my_method \
  --config ./my_method/configs/run.json --data-dir runs/my-gpu-method
```

Rename `template_prompt_search` in the registration and config when adapting
it. The template also registers that name as a standalone Controller for the
Lab template. Remove the separate registration if you only need a whole method.

## Customize the Controller

The template supplies a Policy and uses the default Harness. Change only the
part your method needs:

| Part | Contract | Where to bind it |
| --- | --- | --- |
| Policy chooses the next action | [`DecisionPolicy.decide`](../../loops/contracts/decision.py) | Pass the Policy factory to `MethodDefinition.from_factories`. |
| Harness prepares inputs and supplies local tools | [`ControllerHarness`](../../loops/contracts/harness.py) | For custom behavior, supply a Controller endpoint with its own Harness. |
| Memory provides context derived from evidence | [`ControllerMemory`](../../loops/contracts/feedback_context.py) | Set `ResearchAssembly.context_memory` in the Controller's `HarnessAssembly`. |

Policy and Lab factories receive `(context, config)`; a custom endpoint
factory also receives the Lab execution port. Use `context` for model and
artifact access. The host keeps the original receipts. For durable runs, implement
`state()` and `restore(state)` on your stateful Policy, Harness, and Labs;
Memory follows its own [recovery hooks](../../loops/contracts/feedback_context.py).
