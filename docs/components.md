# Components

This page lists what ships with RSISys. `rsisys catalog` prints the current
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
| `rest_em` | `cpu/rest_em`, `gsm8k`, `math`, `gsm8k-8gpu`, `math-8gpu` | `cpu/rest_em`: nothing. Others: CUDA and a local base model; 1 GPU, or 8 GPUs with vLLM | [README](../src/rsisys/algorithms/rest_em/README.md) |
| `star` | `gsm8k` | 4 GPUs, local base model, GSM8K files | — |
| `ttrl` | `8gpu` | 8 GPUs, local base model | — |
| `ttrl_verl` | `ttrl/configs/paper` | 8 GPUs, TTRL checkout and separate environment ([`requirements/seal-ttrl.txt`](../requirements/seal-ttrl.txt)) | — |
| `seal` | `8gpu`, `paper` | 8 GPUs, local base model; `paper` also needs a separate environment ([`requirements/seal-ttrl.txt`](../requirements/seal-ttrl.txt)) | — |
| `opro` | `evolution`, `independent_candidates` | Chat Completions model endpoint, GSM8K files | — |
| `gepa` | `reflective`, `score-only` | Chat Completions model endpoint, GSM8K files | [README](../src/rsisys/algorithms/gepa/README.md) |
| `gepa_merge` | `gepa/configs/merge-screen` | Model endpoint, pinned GEPA checkout in `external/gepa` | [README](../src/rsisys/algorithms/gepa/README.md) |
| `promptpg` | `tabmwp` | Model endpoint, TabMWP files | — |
| `funsearch` | `evolution`, `frozen_parents` | Model endpoint | — |
| `meta_agent` | `math` | Model endpoint | — |
| `rrsi` | `math` | Model endpoint and search/held-out files (see preset) | — |
| `coding_meta_agent` | `coding` | Model endpoint, Docker, Harbor tasks, `.[coding]` | [README](../src/rsisys/algorithms/coding_meta_agent/README.md) |
| `reflexion` | `control`, `reflection` | Model endpoint, HumanEval files | — |
| `aide` | `gsm8k` | AIDE checkout and separate environment ([`requirements/aide.txt`](../requirements/aide.txt)), model endpoint | — |
| `ft_agent` | `gsm8k` | RD-Agent and LLaMA-Factory checkouts, three Python environments ([`requirements/ft_agent.txt`](../requirements/ft_agent.txt)), GPUs | — |

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

These Controllers are registered on their own, so a run config can pair them
with any compatible Lab.

| Name | What it does | Config |
| --- | --- | --- |
| `single_loop` | Calls one Lab operation a fixed number of times and stops on failure. Use it to schedule a pipeline that the Lab implements. | `operation`, `iterations`, `arguments`, `stop_on_failure` |
| `rest_em` | Runs ReST^EM rounds: evaluate the baseline, then generate, filter, train, and evaluate. | `initial_model`, `train_data`, `eval_data`, `iterations`, `samples_per_prompt`, `max_accepted_per_prompt` |
| `star` | Runs STaR rounds against the `star` Lab. | `rounds`, `plain_rounds` |

The method template registers one more, `template_prompt_search`, when loaded
as a plugin.

## Labs

| Name | What it runs | Notes |
| --- | --- | --- |
| `rest_em` | Numeric math operations: `generate`, `filter`, `train`, `evaluate`, `select` | `engine`: `fixture` (CPU, simulated), `transformers`, or `vllm` |
| `star` | STaR collection, fitting, and evaluation on GSM8K | GPU |
| `coding` | Evaluates an agent artifact on frozen coding tasks in Harbor containers | [Coding Lab guide](../src/rsisys/labs/scenes/coding/README.md) |
| `local_process` | Runs trusted local programs as jobs | Requires `"execution": "local_process"` in the run config |
| `self_reflection` | Builds candidate Controllers and tests them in nested runs | [Source](../src/rsisys/labs/scenes/self_reflection/lab.py) |
| `math` | Placeholder for a shared math Lab. It publishes no operations yet. | [Status](../src/rsisys/labs/scenes/math/README.md) |

The Lab template registers `template_math`, an arithmetic Lab that scores a
model through a Chat Completions endpoint, when loaded as a plugin.

## Reusable components

A method resolves components with
`registry.resolve_component(kind, name, context, config)`. Supported kinds are
`backend`, `harness`, `memory`, and `scene`.

| Kind | Name | Purpose |
| --- | --- | --- |
| `backend` | `chat_pool` | Pool of Chat Completions clients for batched generation |
| `backend` | `lora` | LoRA training engine |
| `harness` | `python_step` | Runs a candidate agent program, `step(problem, history)`, in an isolated Python process. Used by harness-search Labs. |
| `memory` | `agent_memory` | Default Controller Memory: notes written and retrieved by the decision model |

Other implementations are importable from `rsisys.api` without registration:

| Part | Classes |
| --- | --- |
| Policies | `ModelDecisionPolicy` (a model chooses the actions), `dsh_decision_factory` (a Controller built on DeepSeek Harness SDK, `.[dsh]`) |
| Memory | `LangMemMemory` (`.[langmem]`) |
| Model serving | `VLLMBackend`, `ChatPool` |
| Weight training | `LoRAEngine`, `ParallelLoRAEngine` |

## Verifiers

A verifier checks a model answer against the reference answer. It is part of a
Lab's scoring and lives in the scene whose answer format it handles, so it is
not a registry component. The math scene provides two in
[`labs/scenes/math/verifiers.py`](../src/rsisys/labs/scenes/math/verifiers.py):

| Verifier | Answer format | Example benchmarks |
| --- | --- | --- |
| `StrictNumericVerifier` | A single number | GSM8K |
| `MathVerifyVerifier` | LaTeX or symbolic expressions; needs `.[math]` | MATH, MATH-500 |

A Lab picks one in its own config; for example, the `rest_em` Lab accepts
`"verifier": "numeric"` or `"math_verify"`. A new Lab imports a verifier from
`rsisys.api`, or implements its own scoring for a new answer format.

## Lab profiles

A Lab can declare a profile: a standard set of operations and receipt fields.
A Controller that requires a profile works with any Lab that implements it.

| Profile | For |
| --- | --- |
| `WEIGHT_TRAINING` | Labs that train and evaluate model weights |
| `ARTIFACT_SEARCH` | Labs that evaluate candidate prompts, harnesses, or other artifacts |
| `PROGRAM_EXECUTION` | Labs that run and score programs |
| `CODING_TASK` | Labs that run coding agents on benchmark tasks |

See [`loops/contracts/profiles.py`](../src/rsisys/loops/contracts/profiles.py)
for the operations each profile requires.
