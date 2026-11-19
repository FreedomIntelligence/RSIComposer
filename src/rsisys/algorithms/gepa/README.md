# GEPA

The `gepa_merge` method runs the pinned upstream GEPA optimizer with two GSM8K
prompt components, `strategy` and `verification`. Upstream controls parent and
component selection, reflection, merge scheduling, acceptance, and evaluation
budget. RSISys executes model calls in the Lab, records receipts, and evaluates
the seed and selected candidate on the sealed split after the controller finishes.
The existing `gepa` method serves the paired `reflective` / `score-only`
experiment. Its candidate shape and fixed spending protocol are unchanged.

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

```sh
PYTHONPATH=src python -m rsisys run --durable \
  --config src/rsisys/algorithms/gepa/configs/merge-screen.json \
  --data-dir runs/gepa-merge-screen
```

`max_metric_calls` is GEPA's stop threshold, checked between iterations: the
last iteration and initial validation may take the count above it. A reflective
iteration can use up to two minibatches plus full validation; a merge iteration
can use five validation examples plus full validation. `max_model_calls` is a
separate hard Lab limit, covering every model call. Include up to one reflection
per round and two sealed evaluations when sizing it. Report actual spends for
each run, since merge and no-merge runs may stop at different counts.

`rsisys resume --data-dir runs/gepa-merge-screen` reconstructs the upstream
optimizer from checkpointed exchanges without repeating model calls with
committed receipts. Replay runs CPU work from the beginning at each exchange;
measure its cost before increasing the round limit substantially. Because a Lab
call contacts the model service, an interruption with no committed receipt
requires reconciliation by the host before continuing.

## Verify behavior

```sh
PYTHONPATH=src RSISYS_GEPA_SOURCE=external/gepa \
  python -m unittest tests.test_gepa_merge
```

The deterministic fixture compares RSISys to direct upstream execution,
including accepted and rejected merges, parent lineage, evaluation batches,
model call counts, final selection, and resume after the merge minibatch. This
validates algorithm behavior for the GSM8K adapter. Paper benchmark scores and
real model performance need separate runs.
