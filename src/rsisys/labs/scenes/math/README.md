# Math task environment

`MathLab` is a reusable scene registered as `math`. It is the
[`evaluation` Lab](../../../../docs/components.md#connect-a-question-answer-benchmark)
with math defaults: rows group by `benchmark`, the default verifier is
`competition_integer`, the output budget is 32,768 tokens, and a sealed split
can use the official competition bundle. It loads frozen math
problems, calls the target model bound to the `actor` role, verifies answers,
and records scores, token usage, and evidence. The method supplies the solver
instruction. Its Controller and Harness own proposal generation, search,
training, selection, and stopping. A method can compose this scene with its own
training Lab, as described in the [Lab template](../../_template/README.md).
The division of responsibilities follows the [Coding Lab](../coding/README.md).

## Runtime requirements

Use Python 3.12 or later and a configured actor backend. For a deployed
OpenAI-compatible actor, set `json_mode: false` so the model can return LaTeX
text. Configure the model, revision, server context window, timeout, and GPU
allocation at deployment. The scene requests up to 32,768 output tokens per
question by default, using one generation and temperature zero. The server
context window must accommodate the entire input plus the output budget.
Setting the Lab budget does not change the deployed server's limits.

The competition sets and integer-labeled development data use only Python's
standard library. Install `pip install -e '.[math]'` to use the existing
Math-Verify scorer on symbolic answers. The scene gives symbolic grading a
separate, time-bounded Python process when the Lab runs on a host thread.

## Connect the datasets

Configure the Lab with a scene manifest and explicit roles. The example below
shows the entry for the `framework.labs.math` alias of a run config.

```json
{
  "name": "math",
  "config": {
    "manifest": "/data/experiments/math-scene.json",
    "split_roles": {"development": "dev", "official": "sealed"},
    "max_output_tokens": 32768
  }
}
```

Use a separate JSON object for the scene manifest.

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

Use an absolute path for the scene manifest, or make it relative to the
working directory. Entries inside the scene manifest resolve relative to that
manifest. Hash the exact file bytes, for example with
`shasum -a 256 /path/to/file`. The scene checks the hashes and counts at
startup. It rejects repeated IDs or normalized question text within or across
the bound splits and checks the data, model and scoring bindings during resume.
The scene declares each split as a host dataset binding. The host retains a
fingerprint for the sealed split without exposing its row IDs or source.

A split can bind a JSONL file using `file`, `sha256`, and `count`, whether its
role is `dev` or `sealed`. Each row has `id`, `prompt`, `answer`, and optionally
`benchmark` (defaults to `math`); `fields` renames them as in the `evaluation`
Lab. The default verifier accepts nonnegative integer labels through 99,999.
Set `"verifier"` in the split entry to any [shared verifier](../../../../docs/components.md#verifiers);
for symbolic labels use `"math_verify"` and supply bare LaTeX answers or boxed
references. The training
and rollout data stay in the method's own Lab or preparation pipeline. Keep that
data, the development inputs, and the sealed assessment disjoint.

A run can omit the sealed split for development-only work. To run the official
70-question preset, use `benchmarks_manifest` on the sealed split. This form
requires all three frozen datasets and checks their individual counts and
hashes. Use the JSONL form to bind a different sealed benchmark.

| Benchmark | Count | Scoring |
| --- | ---: | --- |
| AIME 2024 I and II | 30 | Terminal boxed integer from 0 to 999 |
| AIME 2025 I and II | 30 | Terminal boxed integer from 0 to 999 |
| AIMO3 Reference | 10 | Terminal boxed integer from 0 to 99,999 |

Use the prepared benchmark manifest and the three JSONL files from the deployment.

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
[Coding Meta-Agent's adapter](../../../algorithms/coding_meta_agent/lab.py).
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
