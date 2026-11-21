# GEPA

The `gepa_merge` method runs the pinned upstream GEPA optimizer with two GSM8K
prompt components, `strategy` and `verification`. Upstream controls parent and
component selection, reflection, merge scheduling, acceptance, and evaluation
budget. The Controller answers reflection prompts with the `decision_maker`
model ([`proposer.py`](proposer.py)); the Lab scores candidates with the `actor`
and records receipts; the host evaluates the seed and selected candidate on the
sealed split after the controller finishes. The `gepa` method serves the paired
`reflective` / `score-only` experiment: its Controller mutates the parent from
the minibatch trace the Lab stored, and its fixed spending protocol is unchanged.

## Run a small cohort

From the repository root, place a clean checkout of `gepa-ai/gepa` at
`external/gepa`, using commit `99c3b0ac3b4750cde6a2a88547ed4450930d681b`.
Set the endpoint and model IDs in [`merge-screen.json`](configs/merge-screen.json)
for your serving process. Supply `data/gepa-merge-screen/{search,validation,test}.jsonl`
with 24, 12, and 32 rows respectively. Each row has string fields `id`, `question`,
and `answer` (a numeric value or a GSM8K answer ending in `#### <number>`).
Use disjoint questions and IDs across all splits. Sample search and validation
from the training set and reserve the test set for final measurement. Keep the
same splits and seed when comparing `use_merge: true` with `use_merge: false`.

The Lab is a [MathLab](../../labs/math/README.md), so it reads the splits from a
dataset manifest, `data/gepa-merge-screen/manifest.json`, that binds each file's
hash and row count and selects its answer verifier:

```json
{"schema_version": 1, "splits": {
  "search":     {"file": "search.jsonl", "sha256": "<sha256sum of the file>", "count": 24,
                 "verifier": "marked_numeric", "fields": {"prompt": "question"}},
  "validation": {"file": "validation.jsonl", "sha256": "...", "count": 12,
                 "verifier": "marked_numeric", "fields": {"prompt": "question"}},
  "test":       {"file": "test.jsonl", "sha256": "...", "count": 32,
                 "verifier": "marked_numeric", "fields": {"prompt": "question"}}}}
```

`marked_numeric` is GEPA's GSM8K grader; any verifier in the MathLab catalog can
replace it. The `reflective` and `score-only` presets read `data/gepa/manifest.json`
in the same form. The OPRO presets read `data/opro/manifest.json` and select
`opro_official`, the pinned author grader.

```sh
PYTHONPATH=src python -m rsicomposer run --durable \
  --config src/rsicomposer/algorithms/gepa/configs/merge-screen.json \
  --data-dir runs/gepa-merge-screen
```

`max_metric_calls` is GEPA's stop threshold, checked between iterations: the
last iteration and initial validation may take the count above it. A reflective
iteration can use up to two minibatches plus full validation; a merge iteration
can use five validation examples plus full validation. The Lab's
`max_model_calls` is a hard limit on actor calls; include two sealed evaluations
when sizing it. The Controller's `max_reflection_calls` limits reflections
(at most one per round). Report actual spends for
each run, since merge and no-merge runs may stop at different counts.

`rsicomposer resume --data-dir runs/gepa-merge-screen` reconstructs the upstream
optimizer from checkpointed exchanges without repeating evaluations with
committed receipts or reflections already in the Controller's journal. Replay
runs CPU work from the beginning at each exchange; measure its cost before
increasing the round limit substantially. A reflection interrupted before its
decision is checkpointed is asked again with the same seed. Because a Lab call
contacts the model service, an evaluation interrupted with no committed receipt
requires reconciliation by the host before continuing.

## Compare runs

Run the same preset twice in separate workspaces with `use_merge` enabled and
then disabled. Check the baseline and selected candidate on the same sealed
cohort, and record the actual metric and model calls for each run. A small
cohort checks the adapter under a controlled budget; it does not establish the
paper's benchmark scores.
