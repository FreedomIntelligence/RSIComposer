# RSISys method author guide

An RSISys method wires a Controller to one or more named Labs. The Policy decides
what to try; Labs execute operations and report evidence. The host routes calls,
enforces access and budgets, and delivers receipts. The method owns how to choose
candidates, compare results, and stop.

## Start with a runnable method

Use Python 3.12 or 3.13 and `python -m pip install -e .` from the checkout. The
[prompt method template](algorithms/_template/) runs entirely on CPU. Copy that
package to start a method, then replace its literal checker with your own Lab.

## The four ports

Import framework contracts from [`rsisys.api`](api.py). A simple method supplies
a Policy and a Lab; the host builds the decision endpoint. Use a Harness or
Memory only when the method needs them.

| Component | Smallest implementation | Source |
| --- | --- | --- |
| Policy | `decide(DecisionInput) -> Decision`. Use `Decision.invoke(OperationCall(...))` to dispatch and `Decision('finish')` to stop. Match terminal receipts by `call_id`. | [`loops/contracts/decision.py:155`](loops/contracts/decision.py#L155), [`algorithms/_template/controller.py:27`](algorithms/_template/controller.py#L27) |
| Harness | `window(objective, capabilities, feedback) -> DecisionInput`, `observe(event)`, `tools()`, and `execute_tool(call)`. `NullHarness` passes receipts through and declares no tools. | [`loops/contracts/harness.py:11`](loops/contracts/harness.py#L11) |
| Memory | `context(ContextQuery) -> str`; return `''` when there is no context. Add `observe(event)` to accept host evidence or `query(request)` for structured retrieval. | [`loops/contracts/feedback_context.py:64`](loops/contracts/feedback_context.py#L64) |
| Lab | `describe() -> tuple[OperationSpec, ...]` and `execute(OperationCall)`. A Lab receives the local operation name; the Policy addresses it as `alias.operation`. A long operation can return a `JobRef` using the host's explicit job protocol. | [`algorithms/_template/lab.py:26`](algorithms/_template/lab.py#L26), [`loops/contracts/jobs.py`](loops/contracts/jobs.py) |

Use `ControllerDefinition(policy=..., harness=...)` to configure optional research
feedback; `harness(config)` returns a `HarnessAssembly`, which binds a
`ResearchAssembly` to the endpoint. A custom endpoint is declared with
`ControllerDefinition(endpoint=True, policy=factory)`; the factory takes
`(context, lab_execution, config)`. See [`methods.py:76`](methods.py#L76).
Keep model and artifact access on the host capabilities supplied to the factory.

For a reusable Lab, declare a [Lab profile](loops/contracts/profiles.py) and its
instance `MetricSpec`. Put its value in `quality_metrics[metric.name]`. Compare
candidates within the same Lab and evaluation protocol. The host runs sealed
final measurements after selection and keeps them out of Controller evidence.
Use `context.lab_catalog` for operation specifications, aliases and metrics.
An endpoint factory receives the Lab execution port explicitly and must retain
that route. The host checks the underlying Policy's profile requirements in
both assembly styles.

## Assemble, run, and resume

Run this Python example from the checkout. It loads the packaged template's
`register(registry)` function, which supplies its Policy, the `checker` Lab and
their configuration validation. `Registry.build` returns the explicit
`ModuleAssembly`. No API key, model server or GPU is needed.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from rsisys.api import Application, ApplicationConfig, BackendBinding, ModelRef, Registry

registry = Registry()
registry.load_plugin(Path('src/rsisys/algorithms/_template'))
modules = registry.build({'method': {'name': 'custom_prompt', 'config': {
    'controller': {'max_evaluations': 3}, 'labs': {'checker': {}},
}}})

with TemporaryDirectory() as directory:
    config = ApplicationConfig(
        objective='Improve the prompt until the checker is satisfied',
        data_dir=Path(directory) / 'run',
        backends={'mock': BackendBinding('mock', kind='mock')},
        roles={'decision_maker': ModelRef('mock', 'example', '1')},
    )
    with Application(config, modules=modules, durable=True) as app:
        app.run(max_decision_steps=1)
    with Application.resume(config.data_dir, modules=modules) as app:
        app.run()
        print(app.controller_report())
```

For the CLI, use [`algorithms/_template/configs/run.json`](algorithms/_template/configs/run.json)
with `rsisys run --plugin src/rsisys/algorithms/_template --config ... --data-dir ...`.
Every run config supplies `framework.method` or both `framework.controller` and
`framework.labs`. A method package may keep its factories private; the
[`register()` example](algorithms/_template/__init__.py) binds them with
`MethodDefinition.from_factories`.

## Runtime configuration

- **Execution.** `resources.executor` defaults to `local`. For a Lab with GPU
  roles, declare a pool with the available device IDs and bind each role in
  `resources.placement` using `<lab-alias>.<role>`. The
  [ReST^EM GPU preset](algorithms/rest_em/configs/gsm8k-8gpu.json) shows the layout.
  Match the pool inventory to the GPUs visible to the process.
- **Ray.** Install `rsisys[ray]` and set `resources.executor` to `ray` to use a
  Ray cluster. Ray pools declare placement group `bundles` and a placement
  `strategy`. Leases reserve whole bundles; use smaller bundles for independent
  concurrent jobs. GPU sharing is a scheduling quota and does not isolate GPU
  memory. Workers need matching dependencies and shared access to the configured
  paths. RSISys does not distribute environments or start model services. On a
  multi-node cluster, the host uses HTTP for internal calls; keep that traffic
  within a trusted network. Real multi-node GPU placement and recovery have not
  been verified for release.
- **Data splits.** A Lab profile can declare `train`, `train_unlabeled`, `dev`
  and `sealed` roles. Run candidate selection on development evidence. For a
  declared sealed split, the host admits final evaluation after the Controller
  finishes and keeps the results out of Controller and Memory evidence.

## State and recovery

- Implement `state() -> dict` and `restore(state) -> None` on Policy, Harness and
  Labs for durable runs. Return data supported by the framework codec; keep
  machine-local paths, live handles, credentials, and workers out of state.
- **Durable resume** reopens the original workspace. A committed receipt is
  delivered by its original `call_id`; an uncertain effect is blocked unless the
  Lab declares safe retry. Upstream agents must match the committed replay prefix;
  records after the committed checkpoint are discarded and regenerated as needed.
- **Lab concurrency** defaults to one active mutation per stateful Lab. A
  checkpoint taken during that operation sees the state from before it started.
  Set `concurrent_state_safe = True` only when the Lab's state port is thread-safe
  and every value it can capture during overlapping operations is valid after an
  interruption. This includes attempted work only when the Lab explicitly treats
  it as durable, for example a model call charged against a budget.
- **Portable snapshots** use `LocalSnapshotCapture` and `restore_snapshot` into
  a fresh directory. The run must reach a supported boundary with no active work
  to carry across. Registered methods restore through their saved identity; supply
  the original `modules` for a direct Python assembly. Older saved protocols are
  rejected. Open older checkpoints and snapshots with the revision that wrote
  them. Compatible public Monitor records can still be viewed independently.
- **Controller replacement** happens at a checkpoint. Memory inheritance uses
  `seed_version(state)`; it does not move the old version's execution receipts.
  In-flight Lab work keeps its original route. Controller-local tools drain
  before the old version closes.
  A finished host loop only says the run ended. Inspect method reports, Lab
  metrics and artifact references for the experiment outcome.

## Memory ownership

| Owner | Responsibility |
| --- | --- |
| `ControllerMemory` | The only public Memory protocol. Provides context and optionally observes evidence. For recovery, exports/restores owned state or rebuilds from ordered evidence. |
| `HarnessAssembly` | Wires Memory and feedback for each Controller version and captures their state independently of the Harness. |
| Harness and Policy | Harness projects the decision window and owns local tools; Policy owns the strategy and its own algorithm state. |
| Host | Owns canonical evidence, receipts, model access, and accounting. A Memory product is a derived view, not an execution receipt. |
| Provider implementation | Owns notes, indexes or extracted documents. LangMem binds the same instance as `context_memory` and `memory_processor`; explicit restore replaces the branch, and same-workspace recovery preserves verified later commits. |

Install `python -m pip install -e '.[langmem]'` for model-backed extraction with
`LangMemMemory`. To use the optional ReMe provider instead, install its
[upstream requirements](../../requirements/reme.txt) in the appropriate environment.

## Available methods

Run `rsisys catalog` for the live registry and see [`algorithms/`](algorithms/) for
per-method configs. Registration and CPU fixtures establish the runtime contract;
they do not establish paper reproduction or model quality.

- **Prompt and program search:** OPRO, [GEPA](algorithms/gepa/README.md), FunSearch, PromptPG,
  RRSI and Reflexion. `gepa_merge` runs the pinned multi-component optimizer.
- **Weight training:** ReST^EM, STaR, SEAL and TTRL. Real runs need their model,
  data, serving/training dependencies and resource configuration.
- **Upstream agents:** AIDE and FT-Agent require their own prepared checkouts and
  interpreters for native runs.
- **Research scenes:** [Coding Lab](labs/scenes/coding/README.md) provides a reusable
  task environment; [Coding Meta-Agent](algorithms/coding_meta_agent/README.md)
  shows a search method using it. Self-Reflection Lab runs candidate Controllers
  under host budgets and leaves adoption to the acting Controller.

| Verification level | Scope |
| --- | --- |
| Registered | `rsisys catalog` lists 16 method entries, including distinct variants. |
| CPU smoke checked | The ReST^EM Quickstart uses a simulated Lab; the method template uses a deterministic checker. Both exercise execution and reporting without model services. |
| Real environment | Use each native preset as a starting configuration. Model endpoints, pinned model and data inputs, interpreters and GPU placement must be supplied by the operator. The CPU checks do not verify paper scores, GPU recovery or full benchmark runs. |

The [Math Lab](labs/scenes/math/README.md) is reserved for a shared math task
environment. Its verifiers are usable; the `math` Lab itself has no operations.

## Papers and upstream projects

| Area | References |
| --- | --- |
| Weight adaptation | [ReST^EM](https://arxiv.org/abs/2312.06585), [STaR](https://arxiv.org/abs/2203.14465), [SEAL](https://arxiv.org/abs/2506.10943), [TTRL](https://arxiv.org/abs/2504.16084) |
| Search and reflection | [RRSI](https://arxiv.org/abs/2609.24972), [OPRO](https://arxiv.org/abs/2309.03409), [GEPA](https://arxiv.org/abs/2507.19457), [FunSearch](https://github.com/google-deepmind/funsearch), [Reflexion](https://arxiv.org/abs/2303.11366), [PromptPG](https://arxiv.org/abs/2209.14610) |
| Agent integrations | [AIDE](https://github.com/WecoAI/aideml), [RD-Agent / FT-Agent](https://github.com/microsoft/RD-Agent) |
