# Components

This page lists what ships with RSIComposer. `rsisys catalog` prints the current
registry; add `--plugin <path>` to include your own registrations.

A run config selects either one **method**, or one **Controller** together with
named **Labs** (see [Compose a system](compose.md)). **Components** are not
selected in the run config; a method's code resolves them by name.

## Methods

A method is a Controller bound to its own Labs. Select one with
`"framework": {"method": {"name": …}}`. The presets are the JSON files in each
method's `configs/` directory. The **Needs** column lists what the presets
require; check the preset before running it.

| Method | Presets | Needs | Guide |
| --- | --- | --- | --- |
| `rest_em`¹ | `cpu/rest_em`, `gsm8k`, `math`, `gsm8k-8gpu`, `math-8gpu` | `cpu/rest_em`: nothing. Others: CUDA and a local base model; 1 GPU, or 8 GPUs with vLLM | [README](../src/rsisys/algorithms/rest_em/README.md) |
| `star`¹ | `gsm8k` | 4 GPUs, local base model, a math dataset manifest with `train`, `dev` and `test` splits | — |
| `ttrl` | `8gpu` | 8 GPUs, local base model | — |
| `ttrl_verl` | `ttrl/configs/paper` | 8 GPUs, TTRL checkout and separate environment ([`requirements/seal-ttrl.txt`](../requirements/seal-ttrl.txt)) | — |
| `verl_math_grpo` | [TOML example](config.md#online-grpo-with-verl) | Local weights, Math Lab train/dev manifest, GPU and pinned veRL environment | [Configuration](config.md#online-grpo-with-verl) |
| `seal` | `8gpu`, `paper` | 8 GPUs, local base model; `paper` also needs a separate environment ([`requirements/seal-ttrl.txt`](../requirements/seal-ttrl.txt)) | — |
| `opro` | `evolution`, `independent_candidates` | Chat Completions model endpoint, GSM8K files and a math dataset manifest | — |
| `gepa` | `reflective`, `score-only` | Chat Completions model endpoint, GSM8K files and a math dataset manifest | [README](../src/rsisys/algorithms/gepa/README.md) |
| `gepa_merge` | `gepa/configs/merge-screen` | Model endpoint, pinned GEPA checkout in `external/gepa` | [README](../src/rsisys/algorithms/gepa/README.md) |
| `promptpg` | `tabmwp` | Model endpoint, TabMWP files | — |
| `funsearch` | `evolution`, `frozen_parents` | Model endpoint | — |
| `meta_agent` | `math` | Model endpoint and a math dataset manifest with `search` and `heldout` splits | — |
| `rrsi` | `math` | Model endpoint and a math dataset manifest with `search` and `heldout` splits | — |
| `coding_meta_agent` | `coding` | Model endpoint, Docker, Harbor tasks, `.[coding]` | [README](../src/rsisys/algorithms/coding_meta_agent/README.md) |
| `reflexion` | `control`, `reflection` | Model endpoint, HumanEval files | — |
| `aide` | `gsm8k` | AIDE checkout and separate environment ([`requirements/aide.txt`](../requirements/aide.txt)), model endpoint | — |
| `ft_agent` | `gsm8k` | RD-Agent and LLaMA-Factory checkouts, three Python environments ([`requirements/ft_agent.txt`](../requirements/ft_agent.txt)), GPUs, a math dataset manifest with `train`, `dev` and `test` splits | — |

¹ ReST^EM and STaR provide their own Labs for sampling, training, and evaluation.
The [reusable Math Lab](#labs) evaluates a fixed actor and does not train it.

A preset is a starting configuration, not a verified reproduction. Datasets are
not included; prepare the files that each preset names. A dash in the Guide
column means the method has no guide yet; read its `configs/` and
`controller.py`.

Before running a preset for a comparison:

1. Pin the model revision, upstream code commit, dataset splits, scorer, seed,
   and budget in the config.
2. Configure the model endpoints or local model paths. For methods that train
   weights, prepare the training environment and GPU placement.
3. Compare the baseline and the selected candidate with the same protocol. Keep
   the final held-out split out of candidate selection.

Matching a paper's numbers also requires its full protocol and benchmark.

## Controllers

`single_loop` is registered on its own, so a run config can pair it with any
Lab. It calls one Lab operation a fixed number of times and stops on failure;
use it to schedule a pipeline that the Lab implements. Its options are
`operation`, `iterations`, `arguments`, and `stop_on_failure`.

The method template registers one more, `template_prompt_search`, when loaded
as a plugin. The Controllers of the built-in methods are private to their
methods.

## Labs

Select a reusable Lab by its registered name under `framework.labs.<alias>.name`.
The alias is what the Controller uses to call it. Before swapping Labs, check that
the new Lab accepts the operations and arguments the Controller sends and returns
the receipt fields it reads. See [Compose a system](compose.md#pair-a-registered-controller-with-a-lab).

Choose the domain Lab by what the benchmark grades. The same tool, such as a
browser or terminal, can appear in several domains.

| Domain Lab | What the benchmark grades | First benchmark adapters |
| --- | --- | --- |
| [CodingLab](../src/rsisys/labs/coding/README.md) | Executable changes in a code workspace or terminal | SWE-bench Verified and Terminal-Bench 2.1 through Harbor |
| [MathLab](../src/rsisys/labs/math/README.md) | Correctness of mathematical answers | AIME 2024 and 2025 with exact integer verification |
| [SearchLab](../src/rsisys/labs/search/README.md) | Retrieved answers and tables; sources are recorded as evidence | BrowseComp and WideSearch |
| [ComputerUseLab](../src/rsisys/labs/computer_use/README.md) | Whether UI actions produced the required application state | WebArena-Infinity real tasks |
| [ProfessionalWorkLab](../src/rsisys/labs/professional_work/README.md) | Whether a deliverable satisfies the benchmark's rubric | SkillsBench science, office, and finance through Harbor |

Domain Labs expose `inspect` and `evaluate`. Coding, Math, and Professional Work
also expose `read_trace`; Search and Computer Use return trace artifacts.
[`self_reflection`](../src/rsisys/labs/self_reflection/lab.py)
is a framework Lab: it evaluates candidate Controllers in budgeted child runs
using the configured domain Labs.

A method can wrap a shared Lab to add operations specific to its algorithm.
For example, [`coding_meta_agent`](../src/rsisys/algorithms/coding_meta_agent/README.md)
uses Coding Lab for task execution and grading, and adds candidate registration
and selection in its own adapter. A method can also provide its own Lab when it
needs a different operation contract, as ReST^EM does for weight training.

To connect data to a Lab, configure its inputs and declare the split roles.
[`DatasetBinding`](../src/rsisys/loops/contracts/data.py) gives the host a common
way to record the role, format, provenance, and content fingerprint of each
split. The task format stays with the Lab: coding tasks are directories; math
problems are rows. Keep training and development separate from sealed final
evaluation. When a profile requires a primary metric, the Lab declares its
name and scorer. Metric names do not have to match across Labs.

For a new domain, start with one benchmark and its official task loader,
runtime, and grader. Pin the benchmark revision, task splits, grader and
evaluation settings; return the declared metric in `quality_metrics` and
store detailed evidence as artifacts. Verify a complete task with the official
grader before registering the Lab. A second benchmark can then reuse the Lab's
execution path without changing the RSI policy or the framework core.

### Mount benchmarks on a Lab

A benchmark owns its evaluation protocol: its tasks, verifier and budgets. A Lab
mounts benchmarks and does not own, merge or override that protocol, so how a
benchmark is scored never depends on how it is assembled. Each benchmark's protocol
is hashed into a `protocol_id` that appears in receipts and data bindings: the same
benchmark entry has the same `protocol_id` on one Lab that mounts both SWE-bench
Verified and Terminal-Bench 2.1, on a Lab that mounts one, and on two Lab instances
that mount one each. A budget set on the Lab instead of on a benchmark is rejected.

```toml
[labs.coding]
primary_metric = { aggregate = "benchmark-mean" }   # required when several benchmarks are mounted

[labs.coding.benchmarks.swebench-verified]
manifest = "/data/swebench-verified.json"

[labs.coding.benchmarks."terminal-bench-2.1"]
manifest = "/data/terminal-bench-2.1.json"
max_steps = 96
agent_timeout = 3600
```

Results report each benchmark separately. A Lab forms no cross-benchmark total on
its own: with more than one benchmark, `primary_metric` names one benchmark or a
named aggregate, and assembly fails without it. See the
[Coding](../src/rsisys/labs/coding/README.md#connect-a-benchmark) and
[Math](../src/rsisys/labs/math/README.md#benchmarks) Labs. The contract is in
[`loops/contracts/benchmark.py`](../src/rsisys/loops/contracts/benchmark.py).

### RSI policy and domain agent

A research system has two parts that are easy to confuse because both can be "an agent"
with a "harness":

| | Lives in | What it is | Changed by |
| --- | --- | --- | --- |
| **RSI policy** | the Controller (`ControllerDefinition`: a policy, with its harness and memory) | The researcher. It decides what to try next and which domain agent to adopt. | the method's author |
| **domain agent** | the Lab | What does the domain task and is under study: an instruction, a configuration, a program, an agent source tree or a checkpoint. | the RSI policy |

The RSI policy submits and revises domain agents. The Lab pins each one by content hash,
runs it under a fixed protocol and reports the result; it never proposes, edits or selects
a domain agent. The harness of a coding or math domain agent (a `step` program, a Harbor
agent) is part of the domain agent and is not the Controller's `ControllerHarness`.

How a domain agent runs on a benchmark's tasks is the benchmark's **runtime**, recorded in
its `protocol_id` next to the budgets, so a benchmark is run the same way wherever it is
mounted. Everything that decides what a score means (generation parameters, seed policy,
trial counts, timeouts, call budgets) is a setting of the runtime, so changing one changes
`protocol_id`, and the recorded settings are the ones that execute. A runtime declares what a
domain agent may set (any other field is rejected), how
it reaches a model, and a trust level: `data`,
`sandboxed-code` (bwrap or a task container) or `host-code` (imported on a worker host).
`host-code` runs only when the Lab config lists `trust = ["host-code"]`. The contract is
in [`loops/contracts/domain_agent.py`](../src/rsisys/loops/contracts/domain_agent.py);
see the [Coding](../src/rsisys/labs/coding/README.md#plugging-in-a-method-or-coding-agent) and
[Math](../src/rsisys/labs/math/README.md#runtimes) runtimes.

### Connect a question-answer benchmark

A benchmark whose rows hold a question and a numeric or symbolic gold answer
needs no code: bind it to the `math` Lab with a dataset manifest, choose each
split's verifier, and rename row keys with `fields`:

```json
{"schema_version": 1, "splits": {
  "dev":  {"file": "dev.jsonl",  "sha256": "<SHA-256>", "count": 200,
           "verifier": "marked_numeric", "fields": {"prompt": "question"}},
  "test": {"file": "test.jsonl", "sha256": "<SHA-256>", "count": 1319,
           "verifier": "marked_numeric", "fields": {"prompt": "question"}}
}}
```

Any Controller that calls `evaluate(split, instruction)` can then search
instructions on it. See [Connect the datasets](../src/rsisys/labs/math/README.md#connect-the-datasets).
Other answer formats belong in their own Lab, with their own verifiers.

To write a Lab, start with [`template_math`](../src/rsisys/labs/_template/README.md),
a plugin that binds training and development data and runs `evaluate`
synchronously or as a background job.

## Reusable components

A method resolves components with
`registry.resolve_component(kind, name, context, config)`. Supported kinds are
`backend`, `harness`, `memory`, and `scene`.

| Kind | Name | Purpose |
| --- | --- | --- |
| `backend` | `chat_pool` | Pool of Chat Completions clients for batched generation |
| `backend` | `lora` | LoRA training engine |
| `backend` | `trl_sft` | Single-device TRL SFT and PEFT LoRA; currently selectable by ReST^EM |
| `backend` | `verl_sft` | Single-GPU veRL FSDP2 SFT with PEFT LoRA export; selectable by ReST^EM |
| `backend` | `verl_grpo` | Single-node online GRPO with a Lab-supplied reward and a separate rollout engine; used by `verl_math_grpo` |
| `harness` | `python_step` | Runs a domain agent program, `step(problem, history)`, in an isolated Python process. Used by harness-search Labs. This is the domain agent's harness, not the Controller's `ControllerHarness`. |
| `memory` | `agent_memory` | Default Controller Memory: notes written and retrieved by the decision model |

Other implementations are importable from `rsisys.api` without registration:

| Part | Classes |
| --- | --- |
| Policies | `ModelDecisionPolicy` (a model chooses the actions), `dsh_decision_factory` (a Controller built on DeepSeek Harness SDK, `.[dsh]`) |
| Memory | `LangMemMemory` (`.[langmem]`) |
| Model serving | `VLLMBackend`, `SGLangBackend`, `ChatPool` |
| Weight training | `LoRAEngine`, `ParallelLoRAEngine` |

Choose a [model serving binding](config.md#backendsid) in the run config
to send inference requests to a local server or to an endpoint that speaks
OpenAI Chat Completions or Anthropic Messages. For training, the method selects
a registered backend: see [ReST^EM's SFT options](../src/rsisys/algorithms/rest_em/README.md#select-a-trainer)
or [MathLab GRPO](config.md#online-grpo-with-verl), where the training strategy
and in-job rollout engine are configured separately.

## Verifiers

A verifier checks a model answer against the reference answer. Each Lab defines
the verifiers for its own answer format. The math Lab's verifiers
(`strict_numeric`, `marked_numeric`, `competition_integer`, `math_verify`) share
one interface and are selected by name; methods that score math answers in their
own Labs import them from `rsisys.api`. See
[Math Lab: answer verifiers](../src/rsisys/labs/math/README.md#answer-verifiers).
The Coding Lab grades with each benchmark task's own tests.

## Lab profiles

A Lab can declare a profile: a standard set of operations and receipt fields.
A Controller that requires a profile works with any Lab that implements it.

| Profile | For |
| --- | --- |
| `WEIGHT_TRAINING` | Labs that train and evaluate model weights |
| `ROUND_WEIGHT_TRAINING` | Weight training with an explicit round in each call |
| `REFERENCE_WEIGHT_TRAINING` | Round-based weight training that also receives a reference model |
| `ARTIFACT_SEARCH` | Labs that evaluate candidate prompts, harnesses, or other artifacts |
| `PROGRAM_EXECUTION` | Labs that run and score programs |
| `CODING_TASK` | Labs that run coding agents on benchmark tasks |

See [`loops/contracts/profiles.py`](../src/rsisys/loops/contracts/profiles.py)
for the operations each profile requires.
