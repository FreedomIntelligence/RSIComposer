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
| `rest_em`¹ | `cpu/rest_em`, `gsm8k`, `math`, `gsm8k-8gpu`, `math-8gpu` | `cpu/rest_em`: nothing. Others: CUDA and a local base model; 1 GPU, or 8 GPUs with vLLM | [README](../src/rsisys/algorithms/rest_em/README.md) |
| `star`¹ | `gsm8k` | 4 GPUs, local base model, GSM8K files | — |
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

| Lab | What it evaluates | How it gets its inputs |
| --- | --- | --- |
| [`coding`](../src/rsisys/labs/scenes/coding/README.md) | Agent artifacts on frozen coding tasks, using Harbor and the benchmark's verifier | A task manifest with named splits and file hashes |
| [`math`](../src/rsisys/labs/scenes/math/README.md) | A fixed actor with a supplied instruction; reports answer accuracy | A manifest of hashed JSONL splits, with an optional 70-question competition preset |
| [`self_reflection`](../src/rsisys/labs/scenes/self_reflection/lab.py) | Candidate Controllers in budgeted child runs. The current Controller decides whether to adopt the candidate. | Scoped parent evidence and the child run's Lab configuration |

Coding and Math are domain Labs. Both expose `inspect`, `evaluate`, and
`read_trace`, but their request schemas and scorers are domain specific.
Self-Reflection is a framework Lab that can be paired with domain Labs. It
does not define its own domain benchmark.

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

A verifier checks a model answer against the reference answer. All verifiers
share one interface, [`AnswerVerifier`](../src/rsisys/labs/verifiers.py), so a
Lab selects one by name in its config instead of importing a class:

```json
"verifier": "math_verify"
"verifier": {"name": "competition_integer", "maximum": 999}
```

| Name | Class | Answer format | Example benchmarks |
| --- | --- | --- | --- |
| `strict_numeric` | `StrictNumericVerifier` | A final `####` number or numeric box | GSM8K |
| `marked_numeric` | `MarkedNumericVerifier` | The last marked number (`####`, box, or "the answer is") | GSM8K (GEPA, STaR) |
| `competition_integer` | `CompetitionIntegerVerifier` | A terminal boxed integer up to `maximum` (999 or 99999) | AIME, AIMO3 Reference |
| `math_verify` | `MathVerifyVerifier` | LaTeX or symbolic expressions; needs `.[math]` | MATH, MATH-500 |

The math scene (per split, in its manifest), ReST^EM, FT-Agent and the harness-search Labs
accept a `verifier` option. A Lab calls `parse(text)` and
`is_correct(completion, answer)`, or `check(completion, answer)` when it also
records whether the answer parsed. `answer_format` is the instruction text that
asks a model for the format the verifier reads.

For a new answer format, subclass `AnswerVerifier` from `rsisys.api`, set
`name` (a versioned identity that becomes the primary metric's scorer), and
implement `parse`. Override `parse_reference` when gold labels use a different
format from completions, and set `isolated = True` if grading must run on a
process main thread. Pass the instance to a Lab factory that accepts
`verifier=`, as the ReST^EM, TTRL and harness Labs do.

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
