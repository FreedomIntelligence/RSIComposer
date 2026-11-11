# RSISys

**A runtime for recursive self-improvement research methods.**

[中文](README_zh.md) · [Quickstart](#quickstart) · [Methods](#available-methods) · [Build your method](#register-your-own-method)

Register a complete RSI method, keep its controller and Labs together, and run
it through a shared execution and feedback loop:

**Controller → Lab execution → feedback → Controller**

A method can search over inference harnesses, update model weights with SFT or
RL, or use an LLM meta-agent to choose experiments. The method owns its strategy;
RSISys supplies the execution lifecycle, operation routing and evidence records.

Early research release, **0.1.0**, under [Apache-2.0](LICENSE). The CPU quickstart
requires no GPU, API key or model service. Real-model runs need separately
prepared models, datasets and execution environments.

## Why RSISys

- **Register the complete method.** Keep its controller, Labs, workers and configs
  in one method package, inside or outside the library. Method-private components
  need no separate global registration. Adding a method uses `register(registry)`;
  it does not require adding algorithm branches to the core runner.
- **Keep research decisions in the method.** A controller can be a state machine,
  search policy, LLM meta-agent or hybrid. It determines experiments, candidate
  selection and stopping; the framework does not impose a fixed
  propose–train–evaluate–adopt sequence.
- **Use the same host for different improvement mechanisms.** Existing adapters
  exercise harness search, SFT and RL through the same controller/Lab boundary.
  A Lab exposes domain operations and can call native training or serving stacks.
  Shared harnesses, memory, backends and verifiers are optional building blocks.
- **Inspect what actually happened.** Method identities, operation receipts,
  artifact references and method reports keep execution outcomes traceable.
  `finished` means the host loop ended; real training, validation and gains must
  be established from the method's evidence.

The extension contract is **register a complete method and execute its loop**.
Methods choose their internal structure and dependency injection; arbitrary
components are not required to be interchangeable across every method.

Use RSISys when you want to implement an RSI algorithm with its own experiment
protocol and run it through a common host. Use the native training/serving engine
directly when you only need a standalone training or inference job.

## Quickstart

The core and CPU examples support **Python 3.12 or 3.13** on Linux/macOS.
Use **Python 3.12** for optional upstream integrations unless their dependencies
state otherwise. Install from a source checkout:

```sh
git clone https://github.com/satsuki2486441738/RSISys.git
cd RSISys
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
rsisys catalog
rsisys run --config src/rsisys/algorithms/rest_em/configs/cpu/rest_em.json
```

This executes ReST^EM's schedule against a **simulated arithmetic Lab**, without
LLM training. The command prints a JSON run summary and writes
`runs/rest-em-quickstart/result.json`. Inspect `controller_report` and the Lab
receipts to understand the method outcome; host `status` alone is not a quality
or training result. The core has no mandatory third-party runtime dependencies.
Installation uses a checkout or a locally built wheel, not an assumed PyPI release.

Use a new output directory for each run, for example:

```sh
rsisys run --config src/rsisys/algorithms/rest_em/configs/cpu/rest_em.json \
  --data-dir runs/rest-em-another-run
```

See `rsisys --help` and `rsisys run --help` for CLI options.

## Available methods

The catalog lists **8 complete method registrations**. `ttrl` and `ttrl_verl` are
two execution profiles of TTRL; `meta_agent` is our framework scaffold, not a
paper reproduction. Adaptation scope is part of each method's description.
`reference` and `paper` name adapter profiles, not a claim of full paper reproduction.

| Method | Scope | Configuration |
| --- | --- | --- |
| `rest_em` | Generate, filter, retrain from the initial model; lightweight LoRA adaptation | [GSM8K](src/rsisys/algorithms/rest_em/configs/gsm8k-8gpu.json), [MATH](src/rsisys/algorithms/rest_em/configs/math-8gpu.json) |
| `rrsi` | Fixed-model harness evolution; MATH port with regularization and critic | [Config](src/rsisys/algorithms/rrsi/configs/math.json) |
| `seal` | Self-edits, inner adaptation, selection, outer SFT; reference and paper profiles | [Reference](src/rsisys/algorithms/seal/configs/8gpu.json), [paper](src/rsisys/algorithms/seal/configs/paper.json) |
| `ttrl` | Majority rewards with independent LoRA GRPO | [Config](src/rsisys/algorithms/ttrl/configs/8gpu.json) |
| `ttrl_verl` | Pinned official full-parameter backend with continuous optimizer/reference state | [Config](src/rsisys/algorithms/ttrl/configs/paper.json) |
| `ft_agent` | Partial native RD-Agent policy with local GSM8K/LLaMA-Factory execution; full CoSTEER pipeline excluded | [Config](src/rsisys/algorithms/ft_agent/configs/gsm8k.json) |
| `aide` | Native Agent/search/review/Journal, adapted to fixed-model LLM inference programs | [Config](src/rsisys/algorithms/aide/configs/gsm8k.json) |
| `meta_agent` | Model-driven harness proposals, search feedback and sealed heldout selection; no weight training or implemented self-reflection | [Config](src/rsisys/algorithms/meta_agent/configs/math.json) |

The classic experiment packages (OPRO, GEPA, FunSearch, Reflexion, PromptPG and STaR) are documented in [Classic methods](docs/classic_methods.md).

### Prepare a real-model run

**GPU/model-service configurations are templates**, not prepared environments.
They contain example local paths, served model names and device allocations.
Copy a configuration and edit its JSON; the CLI does not accept arbitrary dotted
configuration overrides.

| Method family | What to prepare |
| --- | --- |
| ReST^EM | Local base weights, train/evaluation data, configured training/inference interpreters and GPU IDs |
| RRSI / meta-agent | Linux with bubblewrap, fixed task-model endpoints, search/heldout data; meta-agent also needs a decision-model endpoint |
| SEAL / TTRL | Local weights, method-specific data and isolated training environment; paper profiles additionally use pinned upstream code |
| FT-Agent / AIDE | Linux with bubblewrap, pinned clean upstream checkouts, model services and the configured worker environments |

For example, prepare the ReST^EM MATH configuration:

```sh
mkdir -p .local/configs
cp src/rsisys/algorithms/rest_em/configs/math-8gpu.json .local/configs/rest-em-math.json
# Edit model/data paths, training/inference interpreters and gpu_ids in this JSON.
rsisys run --config .local/configs/rest-em-math.json --data-dir runs/rest-em-math
```

For that config, edit `framework.method.config.controller.initial_model`,
`train_data` and `eval_data`, plus `framework.method.config.labs.math` options
such as `few_shot_data`, `gpu_ids`, `training_python` and `inference_python`.
Explicit interpreter paths are useful when training and serving use different
environments; otherwise the engines use their configured/default interpreter.
Select `engine` explicitly for real training: the CPU quickstart uses `fixture`.

Path rules matter when copying configurations:

- Top-level `data_dir` is relative to the **JSON file's directory**. CLI
  `--data-dir` is relative to the **current working directory** and overrides it.
- Method-specific paths are resolved by the method, not automatically rebased
  when a config moves. Shipped configs assume the repository working directory;
  use absolute paths for your own models, data, interpreters and upstream code.
  ReST^EM also supports a Lab `data_root` for its input paths.
- Endpoint model names such as `rsisys-fixed` must match your server's served
  model alias. They do not download or identify bundled weights.

Training requires its own PyTorch/Transformers/PEFT environment and vLLM where
configured. Install optional framework dependencies in the environment that
imports them, for example `python -m pip install -e '.[math]'` for Math-Verify.
SEAL/TTRL paper backends use [requirements/seal-ttrl.txt](requirements/seal-ttrl.txt)
in a separate environment; SEAL semantic grading also requires its configured
service. FT-Agent and AIDE use [FT-Agent](requirements/ft_agent.txt) /
[AIDE](requirements/aide.txt) dependencies. These files describe specific worker
stacks; they are not one combined installation for every method.

Data formats belong to each Lab, rather than a single framework-wide schema.
For ReST^EM, see the [synthetic JSONL fixture](src/rsisys/algorithms/rest_em/configs/cpu/data/train.jsonl)
and [Lab implementation](src/rsisys/algorithms/rest_em/lab.py); RRSI/meta-agent
use `id`, `prompt`, `answer` JSONL records with separate search and heldout sets.
Other formats, required upstream revisions and accepted parameters are defined
beside their adapters. Registration does not download data, install GPU stacks
or start external services.

## Register your own method

Start from the runnable [_template](src/rsisys/algorithms/_template/__init__.py):

```text
my_method/
  __init__.py       # register(registry): bind the complete method
  controller.py     # decide what to do from feedback
  lab.py            # execute experiments and return receipts
  configs/run.json  # select the method and provide parameters
```

The package layout is yours; small methods may use one file. A conventional
registration binds private factories accepting `(context, config)`:

```python
from rsisys.api import MethodDefinition
from .controller import MyController
from .lab import MyLab

def register(registry):
    registry.register_method("my_method", MethodDefinition.from_factories(
        controller=MyController,
        labs={"experiment": MyLab},
        version="1",
    ))
```

Select the method as a unit. Controller and Lab parameters remain separate:

```json
{
  "framework": {
    "method": {
      "name": "my_method",
      "config": {
        "controller": {},
        "labs": {"experiment": {}}
      }
    }
  }
}
```

This snippet shows only the method selection; use the template's full run config
for the objective, output path and run budget. Run the bundled template without
GPUs or services:

```sh
rsisys run --plugin src/rsisys/algorithms/_template \
  --config src/rsisys/algorithms/_template/configs/run.json \
  --data-dir runs/custom-prompt
```

It checks literal prompt instructions to demonstrate feedback delivery and
method execution. This is an interface example without LLM inference or training.

To start a method package outside the library:

```sh
cp -R src/rsisys/algorithms/_template ./my_method
# Rename custom_prompt in my_method/__init__.py and my_method/configs/run.json.
rsisys run --plugin ./my_method --config ./my_method/configs/run.json \
  --data-dir runs/my-method
```

The copied template runs with its existing name too; rename it before adding
another copy to the same catalog. `--data-dir` avoids inheriting the template's
repository-relative output path. Keep your implementation in this directory;
its controller, Lab and internal components need no additional global registration.

Alternatively, put it in `src/rsisys/algorithms/my_method/` for built-in discovery.
No core registry or `api.py` edit is needed. `_template` itself is excluded from
automatic discovery. Plugins execute trusted Python code; load packages you trust.

The controller implements `decide(DecisionInput)` and returns `invoke`, `wait`
or `finish`. The Lab implements `describe()` and `execute(call)`, returning
`OperationResult` with status, data and evidence references. Multiple Labs use
`alias.operation` routing. Optional config validators reject parameters before
construction. For custom assembly, implement `RSIMethod.build(config)` with
`MethodDefinition.from_method(...)` or supply a `ModuleAssembly` factory.
See [rsisys.api](src/rsisys/api.py) for the public contracts.

### Components

A method may import shared code directly or resolve registered components in
its factory. Method-private components need no additional registration.

| Role | Built-in IDs | Implementation |
| --- | --- | --- |
| `harness` | `python_step` | Isolated `(harness_path, problem, history)` execution; configurable timeout, requires bubblewrap |
| `backend` | `chat_pool`, `lora` | Chat endpoint pool or prepared-input LoRA engine |
| `memory` | `agent_memory` | Agent-owned notes and research tasks; method binds the harness |
| `verifier` | `strict_numeric`, `math_verify` | Final-answer checking; the latter needs `rsisys[math]` |

Use `registry.resolve_component(kind, name, context, config)`; resolution does
not install dependencies or bind the component to a method automatically.
Custom shared components use `register_component`. DSH can be explicitly bound
with `rsisys.harnesses.decision.dsh.dsh_decision_factory` and the `rsisys[dsh]`
extra. Optional ReMe retrieval uses [requirements/reme.txt](requirements/reme.txt)
and `ResearchAssembly.reme()`; LangMem is a separate optional extra.

Independent controller/Lab registration APIs are available for explicit custom
composition. Default standalone registrations are `single_loop` and
`local_process`; method-private components stay in their method package.
Import concrete method classes from their packages and framework contracts
from `rsisys.api`.

## Workflow and boundaries

![General RSI framework: input, Controller, parallel-capable Labs and feedback form one loop; policies, workflows, harness components and Language Model are method-selected replaceable bindings.](assets/architecture.png)

Controller and Labs are generic roles: each method supplies its decision policy
and experiment workflow. Stacked Labs represent parallel-capable experiments;
methods define asynchronous execution and resource allocation. Dashed slots are
replaceable implementations or components selected by the method. Harnesses,
Tools, Skills, Memory, Context and Language Model are bound as needed; model access includes local or
API-backed access. Methods need not use every depicted component.

| Owner | Responsibility |
| --- | --- |
| Method | Controller strategy, Lab operations, candidate selection, stopping and adoption rules |
| Lab | Domain validation, training/inference execution, metrics and artifacts |
| Host | Registration, run lifecycle, authorization, routing, receipt delivery, resource ownership and explicit recovery integration |

```text
src/rsisys/
  algorithms/       # complete method packages, template and shared implementations
  loops/            # contracts and loop mechanics
  harnesses/        # optional decision/actor execution, memory and feedback services
  labs/             # generic process Lab and reusable scenes
  backends/         # training and serving engines
  runtime/          # host scheduling, execution, evidence and storage
  snapshot/         # capture and explicit recovery integration
  api.py            # public framework imports
  methods.py        # method definitions
  registry.py       # catalogs and trusted plugin loading
```

Controllers and Labs need not use an LLM or a bundled harness. Shared components
are optional; the core owns lifecycle semantics, not the method's science.
A Lab with background resources should implement idempotent `close()` and
`shutdown(mode) -> bool`. Async Labs can expose start/poll/result/cancel; each
method drives its own protocol. Registering a Lab does not poll it automatically.

`runs/` and `runtime-output/` hold local outputs and are ignored by Git. The
`runtime/` and `snapshot/` source packages implement mechanisms, not saved runs.

### Self-reflection: planned extension

[Self-Reflection Lab](src/rsisys/labs/scenes/self_reflection/__init__.py) is an
opt-in architecture scaffold (`0-placeholder`) for improving the meta-agent's
own decision-making from accumulated decision–Lab feedback. Its intended
lifecycle is experience accumulation → candidate construction → long-horizon
validation; the controller would decide whether to adopt a validated version.
Validation can remain pending or inconclusive. The code currently returns
`self_reflection_not_implemented` for valid operations and performs no
experience persistence, training, validation or controller switching.
The runnable `meta_agent` method searches task harnesses; it does not implement
this self-improvement lifecycle. Self-reflection is a future direction and is not implemented in this release.

## Monitor

Optional read-only viewer, installed independently from the core:

```sh
python -m pip install ./src/rsimonitor
rsimonitor --runs-dir runs
```

It binds to loopback by default and has no authentication or multi-tenant isolation.

## Release scope and limits

- This is an early research release; 0.x APIs may change. Pin commits and
  model/data/dependency revisions for experiments.
- Snapshot recovery needs explicit state adapters. Method registration alone
  does not restore arbitrary controller state or external jobs. Native FT-Agent/
  AIDE searches and synchronous training recipes need fresh run directories.
- GPU devices are assigned by the operator. There is no global GPU lease across
  Labs/runs or complete long-job billing and budget handoff.
- Plugins are trusted host Python code. A normal local-process Lab is not an OS
  sandbox; generated untrusted code requires an explicitly isolated executor.
  Custom Labs own external access, charges and resource termination. Credentials
  belong in environment variables; review saved prompts/artifacts before sharing.

## Contributing

```sh
python -m pip install -e '.[dev]'
ruff check --select E9,F63,F7,F82 src
python -m build --outdir dist/core .
python -m build --outdir dist/monitor src/rsimonitor
python -m twine check --strict dist/core/* dist/monitor/*
```

Use [GitHub Issues](https://github.com/satsuki2486441738/RSISys/issues) for bugs,
feature requests and method-integration questions. For a bug, include the commit
or package version, Python/OS, a minimal sanitized config, the failing command
and the error text. Do not attach private data or credentials.

For a new method, keep its implementation and config in one algorithm package,
document adaptation boundaries and upstream attribution, and provide a minimal
command another user can run. PRs should describe behavior changes and relevant
validation. The commands above are available in the public checkout; there is
no public full regression suite. Keep experiment results, process narratives,
downloaded data, weights and raw run logs out of the repository.
Use GitHub private vulnerability reporting when available; otherwise request a
private contact channel without publishing exploit details or secrets.

## License

Copyright 2026 RSISys contributors. Licensed under [Apache-2.0](LICENSE).
Third-party attribution stays beside the relevant adapters; SEAL's adapted code
retains its MIT text in [seal/lab.py](src/rsisys/algorithms/seal/lab.py).
Dependencies, model weights and datasets are obtained under their own terms.
