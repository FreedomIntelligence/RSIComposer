# Math task environment

`MathLab` is a reusable Lab registered as `math`. It loads frozen math
problems, calls the target model bound to the `actor` role, verifies answers,
and records scores, token usage, and evidence. The method supplies the solver
instruction. Its Controller and Harness own proposal generation, search,
training, selection, and stopping. A method can compose this Lab with its own
training Lab, as described in the [Lab template](../_template/README.md).
The division of responsibilities follows the [Coding Lab](../coding/README.md).

## Runtime requirements

Use Python 3.12 or later and a configured actor backend. For a deployed
OpenAI-compatible actor, set `json_mode: false` so the model can return LaTeX
text. Configure the model, revision, server context window, timeout, and GPU
allocation at deployment. The Lab requests up to 32,768 output tokens per
question by default, using one generation and temperature zero. The server
context window must accommodate the entire input plus the output budget.
Setting the Lab budget does not change the deployed server's limits.

The competition sets and integer-labeled development data use only Python's
standard library. Install `pip install -e '.[math]'` to use the existing
Math-Verify scorer on symbolic answers. The Lab gives symbolic grading a
separate, time-bounded Python process when the Lab runs on a host thread.

## Connect the datasets

Configure the Lab with a dataset manifest and explicit roles. The example below
shows the entry for the `framework.labs.math` alias of a run config.

```json
{
  "name": "math",
  "config": {
    "manifest": "/data/experiments/math-datasets.json",
    "split_roles": {"development": "dev", "official": "sealed"},
    "max_output_tokens": 32768
  }
}
```

`grading_workers` (default 1) grades with verifiers that must run on a main
thread, such as `math_verify`, in that many worker processes.

Use a separate JSON object for the dataset manifest.

```json
{
  "schema_version": 1,
  "splits": {
    "development": {
      "file": "development.jsonl",
      "sha256": "<development.jsonl SHA-256>",
      "count": 128
    },
    "official": {
      "benchmarks_manifest": "/data/datasets/math-benchmarks/v1/manifest.json",
      "sha256": "<benchmark manifest SHA-256>"
    }
  }
}
```

Use an absolute path for the dataset manifest, or make it relative to the
working directory. Entries inside the dataset manifest resolve relative to that
manifest. Hash the exact file bytes, for example with
`shasum -a 256 /path/to/file`. The Lab checks the hashes and counts at
startup. It rejects repeated IDs or normalized question text within or across
the bound splits and checks the data, model and scoring bindings during resume.
The Lab declares each split as a host dataset binding. The host retains a
fingerprint for the sealed split without exposing its row IDs or source.

A split can bind a JSONL file using `file`, `sha256`, and `count`, whether its
role is `dev` or `sealed`. Each row has `id`, `prompt`, `answer`, and optionally
`benchmark` (defaults to `math`), which groups per-benchmark metrics. If your
files use other keys, rename them with `fields`, for example
`"fields": {"prompt": "question"}` for GSM8K. Each split selects a
[verifier](#answer-verifiers); the default accepts nonnegative integer labels
through 99,999. For symbolic labels use `"math_verify"` and supply bare LaTeX
answers or boxed references. The training
and rollout data stay in the method's own Lab or preparation pipeline. Keep that
data, the development inputs, and the sealed assessment disjoint.

A run can omit the sealed split for development-only work. To run the official
competition benchmarks, use `benchmarks_manifest` on the sealed split. The bundle
can hold any of the frozen benchmarks below, each checked for its count and hash.
Set `"benchmarks": ["aime2024", "aime2025"]` to mount just the two AIME years.
Omitting `benchmarks` selects all three and requires a complete bundle. Use the
JSONL form to bind a different sealed benchmark.

| Benchmark | Count | Scoring |
| --- | ---: | --- |
| `aime2024` (AIME 2024 I and II) | 30 | Terminal boxed integer from 0 to 999 |
| `aime2025` (AIME 2025 I and II) | 30 | Terminal boxed integer from 0 to 999 |
| `aimo3_reference` (AIMO3 Reference) | 10 | Terminal boxed integer from 0 to 99,999 |

Use the prepared benchmark manifest and the JSONL files for the selected benchmarks.

## Benchmarks

A source in the manifest mounts one benchmark: a named problem set together with
its evaluation protocol, the answer verifier and the generation budget
(`max_output_tokens`, default 32,768, for Labs whose actor answers each problem).
Name it with `"benchmark": "<name>"` (default `math`). The protocol is hashed into
a `protocol_id` that every receipt reports, so a benchmark is scored the same way
however it is mounted: alone on one Lab, together with others, or on several Lab
instances. The Lab sets no protocol of its own, and a split may list several sources:

```json
"development": [
  {"file": "a.jsonl", "sha256": "...", "count": 50, "benchmark": "a", "verifier": "marked_numeric"},
  {"file": "b.jsonl", "sha256": "...", "count": 30, "benchmark": "b", "verifier": "math_verify",
   "max_output_tokens": 8192}]
```

Metrics are reported per benchmark (`<name>_accuracy`, `<name>_correct`,
`<name>_count`). `accuracy`, the primary metric, is the one benchmark's accuracy when
an evaluation covers one benchmark. When it covers several, the Lab config must
declare what `accuracy` means with `primary_metric`: `{"benchmark": "<name>"}` or
`{"aggregate": "pooled-accuracy"}` (correct answers over all problems) or
`{"aggregate": "benchmark-mean"}` (the unweighted mean of the benchmarks'
accuracies). A Lab never forms a cross-benchmark total on its own, and a benchmark
cannot be mounted twice with different protocols.

## Runtimes

A runtime is how a domain agent answers a batch of problems. The domain agent is what
the Lab measures; the RSI policy in the Controller submits and revises it.

**Everything that decides what a score means is a field of the runtime**: temperature,
token limits, seed policy, trial counts, timeouts and call budgets. The runtime's
`settings` is exactly what is recorded in each benchmark's protocol, and `run()` reads the
same object, so the protocol that is recorded and the one that executes cannot differ.
Changing any of those settings changes `protocol_id`; resources that do not change a score
(`workers`, `sandbox_workers`, `grading_workers`) do not. The domain agent's content hash
is on each receipt. The Lab picks its runtime; a domain agent cannot, and a runtime rejects
a domain agent that sets fields it does not accept.

| Runtime | Domain agent | Settings recorded in the protocol | Trust | Used by |
| --- | --- | --- | --- | --- |
| `instruction` | An instruction; one actor call per problem | `temperature`, `max_output_tokens` | `data` | `MathLab` |
| `prompt-template` | An instruction rendered into a method's own pinned prompt, answered by the actor under a call budget | `template`, `system`, `temperature`, `top_p`, `seed`, `max_output_tokens`, `max_model_calls` | `data` | GEPA, GEPA-Merge, OPRO |
| `step-program` | A `step(problem, history)` program run in a bwrap sandbox, with the Lab calling a fixed model pool for it | `max_calls`, `seed`, `temperature`, `max_tokens`, `timeout`, `trials`, `sandbox_timeout` | `sandboxed-code` | the RRSI and Meta-Agent harness Lab |
| `checkpoint` | A checkpoint reference; an engine the Lab owns generates | `engine`, `temperature`, `max_new_tokens` | `data` | ReST^EM, STaR, FT-Agent |

`MathLab.solve()` runs the Lab's runtime (`make_runtime()`) through `run_runtime()`, which
checks the domain agent first and hands the runtime a `Run`: this call's own request, trial
count and lent resources. Nothing about one evaluation is stored on the Lab, so two
evaluations in flight cannot see each other. A Lab that needs a different runtime overrides
`make_runtime()` and calls `run_runtime()` from `solve()`; it never runs one runtime while
recording another.

GEPA, GEPA-Merge and OPRO keep their prompts identical to the pinned upstream by supplying
`render()` to `PromptSearchLab`, which builds a `prompt-template` runtime. The template name
is recorded (OPRO's includes `instruction_pos`) and the rendering function is not, so a change
to the upstream prompt format must change the template name.

## Answer verifiers

A verifier checks a model answer against the reference answer. The verifiers
in [`verifiers.py`](verifiers.py) share one interface, `AnswerVerifier`, so a
split selects one by name instead of importing a class:

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

Methods that score math answers in their own Labs (ReST^EM, STaR, TTRL,
FT-Agent, AIDE, GEPA and the harness-search Labs) import these verifiers from
`rsisys.api`; ReST^EM, FT-Agent and the harness Labs accept the same `verifier`
option. A Lab calls `parse(text)` and
`is_correct(completion, answer)`, or `check(completion, answer)` when it also
records whether the answer parsed. `answer_format` is the instruction text that
asks a model for the format the verifier reads.

For a new answer format, subclass `AnswerVerifier` from `rsisys.api`, set
`name` (a versioned identity that becomes the primary metric's scorer), and
implement `parse`. Override `parse_reference` when gold labels use a different
format from completions, and set `isolated = True` if grading must run on a
process main thread. Pass the instance to a Lab factory that accepts
`verifier=`, as the ReST^EM, TTRL and harness Labs do.

## Build a method Lab on MathLab

A method that works on math tasks subclasses `MathLab` instead of writing its
own data loading, verification and traces. The base class keeps the dataset
manifest, split roles, verifiers, metrics and `read_trace`; the subclass changes
only what its candidate is and how it answers:

| Override | Default | Change it to |
| --- | --- | --- |
| `solve(problems, arguments)` | Ask the actor each problem after `arguments['instruction']` (`ask_actor`) | Run a harness program, generate with a trained checkpoint, or render the method's own prompt template |
| `problems(arguments)` | The split, or the listed development `task_ids` | Select rows another way, such as integer offsets or one row per trial |
| `verify(problems, attempts)` | Each answered attempt's verifier, with isolated verifiers in `grading_workers` processes | Grade another way |
| `bind_actor(context, config)` | The `actor` role | Another model reference, such as a served model pool |
| `evaluate_schema()` | `split`, `instruction`, optional `task_ids` | The method's candidate arguments, such as `candidate_id` or `model` |
| `result_data(records, ref, arguments)` | No extra fields | Add the per-example fields the method's Controller reads |
| `verifiers` | The four shared verifiers | Add the method's own answer format, for example an upstream evaluator |
| `describe()` / `execute()` | `inspect`, `evaluate`, `read_trace` | Add operations such as `register`, `select`, `generate` or `train` |

`solve` returns one `Attempt` per problem (completion, status, usage, and an
`extra` dict of method fields that the trace record keeps);
`attempt(request, generate)` makes one from a model call, so a subclass can pass
its own budgeted `generate`. `score()` verifies the attempts and returns the
receipt. Traces record `binding()`, the data, actor and verifier identity; a
subclass that keeps more state extends `state()` and checks `binding()` when it
restores. Dataset manifests select a verifier by name from the subclass's
catalog, so a method can keep its upstream scoring as the default and still let
a run choose another verifier.

Prompt-optimization methods use `PromptSearchLab`
([`algorithms/shared/math_search.py`](../../algorithms/shared/math_search.py)),
a `MathSearchLab` (MathLab speaking the artifact-search protocol) whose candidates are actor prompts: `register`,
`evaluate` by integer offsets into `search`, `validation` and the sealed `test`
split, and `select`. GEPA and OPRO subclass it and only render their prompts;
OPRO adds its author grader to the catalog as `opro_official`. The harness Lab
used by RRSI and Meta-Agent ([`harness_lab.py`](../../algorithms/shared/harness_lab.py))
is another `MathSearchLab`: its candidate is a program, `solve()` runs each
problem for several trials against a fixed model pool, and the manifest has
`search` and sealed `heldout` splits.

Weight-training methods split a manifest into `train` and `dev` splits
(`split_roles` accepts `train`, `dev` and `sealed`). ReST^EM's Lab
([`algorithms/rest_em/lab.py`](../../algorithms/rest_em/lab.py)) samples from the
`train` split with its engine, filters with `verify()`, trains, and measures
checkpoints on `dev` with MathLab's `evaluate()`: `bind_actor()` records the
engine as the measured model, and `solve()` asks the engine instead of the actor.
STaR's Lab ([`algorithms/star/lab.py`](../../algorithms/star/lab.py)) does the
same with GPU workers and adds a sealed `test` split that the host measures after
the Controller finishes. `split_named(ref, role)` resolves a Controller's
`dataset` argument, a split name or the split file's path, to a manifest split.
FT-Agent's Lab ([`algorithms/ft_agent/lab.py`](../../algorithms/ft_agent/lab.py))
is the same shape with LLaMA-Factory training; it adds `math_verify_gsm8k` to its
verifier catalog, which grades GSM8K `#### n` references with Math-Verify.
TTRL keeps its own Lab: its training prompts are unlabeled and gold answers must
never reach rollouts or rewards, which a manifest that requires an answer per row
cannot express. It still uses the shared verifiers.

## Public operations

| Operation | Arguments and result |
| --- | --- |
| `inspect` | Empty object. Returns development IDs, benchmark names, model binding and generation budget. |
| `evaluate` | `split`, `instruction`, optional development-only `task_ids`. Returns overall accuracy, per-benchmark correct counts and accuracy, inference duration, format errors, truncations and token usage when the provider reports it. |
| `read_trace` | Development `split`, the `evaluation` reference from `data.evidence_refs[0]`, zero-based sample `index`, and an optional character `offset`. Returns a bounded completion page with `next_offset` for the next page. |

The primary metric is correct answers divided by the number of questions
evaluated. A wrong answer, an unparseable answer, or a provider-reported
truncation counts as incorrect. A provider transport or model identity failure
fails the operation. Token totals are included when all questions report that
token field. The receipt also records how many questions reported each field.

During development a Controller can issue the namespaced operation, assuming
the Lab alias is `math`.

```python
from rsisys.api import OperationCall

call = OperationCall('math.evaluate', {
    'split': 'development',
    'instruction': 'Solve the problem and check the result.',
})
```

For formal evaluation, the method's Lab adapter exposes
`final_evaluations()` after saving its selected instruction, following
[Coding Meta-Agent's adapter](../../algorithms/coding_meta_agent/lab.py).
It returns one local `OperationCall('evaluate', {'split': 'official',
'instruction': selected_instruction})`. The RSISys host executes the full
sealed split after the Controller finishes and withholds per-question data
from the Controller. Public Monitor shows the overall and per-benchmark scores. Aggregate
timings and token counts are in the host receipt. Authorized host artifact
access is required for the detailed official evidence. A direct Python call
bypasses orchestration, so use the host route for formal comparisons.

## Check the Lab registration

To confirm that the `math` Lab appears in the catalog, run this from the repository root:

```sh
PYTHONPATH=src python -B -m rsisys catalog
```
