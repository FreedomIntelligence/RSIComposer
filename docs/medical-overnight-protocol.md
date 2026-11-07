# Medical Open Lab deployment protocol

The host owns capabilities, device boundaries, lifetime, accounting, evaluation
data and scoring. The meta-agent writes training and inference programs and
candidate bundles inside the Lab. A successful program is evidence of execution;
training improvements require optimizer/loss, changed weights and matching
host-verified evaluation receipts.

## Fixed official medical evaluation

Use `prepare_official_benchmarks` with pinned source bytes. The closed test
sets contain all **2450 MedXpertQA Text** and **1273 MedQA USMLE English**
questions in the official release's original order. The host does not silently
drop duplicates or upstream metadata inconsistencies from the test denominator.
Public development uses MedQA's official dev split with four exact test overlaps
removed (1268 rows). MedXpertQA has no separate official development split in
this setup. Training uses a deterministic 1024-row MedQA training selection
after exact question overlap exclusion; this does not establish semantic or
pretraining contamination freedom.

`--evaluation-score-protocol medxpertqa-official-zero-shot-ao-v1` selects the
pinned MedXpertQA evaluation repository's common MedQA/MedXpertQA zero-shot
answer-only prompts and answer extraction. Full upstream questions, including
their original Answer Choices suffix, are supplied once. Host accuracy includes
every sample; invalid answers count as incorrect. The original
`exact-option-letter-v1` remains available under its distinct identity. Scores
across these protocols or different datasets must not be treated as comparable.
Published model reports can use different inference settings; reproducing these
rules alone does not reproduce every published score.

The model submits `submission.json`, an inference entrypoint and optional
checkpoint/adapter. The host freezes and hashes this bundle and its registered
base. Inference strategy remains candidate-owned. Use the same inference bundle
when isolating a training effect; otherwise the score measures the whole system.

`closed_gateway` runs in the operator filesystem. Its private config, source
cache, labels, raw predictions and diagnostics must not be mounted into research
or inference workers. The candidate receives gold-free input over stdin inside
the host's filesystem, PID, network and exact device boundary. Only aggregate
accuracy, sample count and status return to the meta-agent. Independent durable
host and gateway ledgers permit one closed submission per 1800 seconds across a
campaign, including failed accepted submissions. Unknown accepted effects are
reconciled, never automatically replayed.

## Reflection without meta-agent training

The built-in optimizer accepts:

```json
{"method":"open_lab","config":{"reflect":true,"reflect_interval_seconds":1800,"meta_model_updates":false}}
```

At the first decision boundary after a review becomes due, the same meta-agent
calls `reflect_begin`. This journalled operation fixes the evidence snapshot and
a wall deadline: at most one hour, or the absolute host deadline if earlier.
`reflect_max_seconds` may configure 1..3600 seconds. Context projection does not
start a timer, and retry/resume cannot renew it.

The agent keeps authorized archive, code, execution, polling, Memory and system
activation capabilities throughout the review. There is no review-operation
count cap. It may interleave reading with building and testing a `meta.prompts`
or `meta.harness` candidate. Completing `reflect_report` requires complete raw
operation records for all failures and the latest ordinary operation in the
snapshot, plus valid distinct citations and the actual activated system version.
An evidence-based `no_change` is valid. A timeout is incomplete, retains backlog
and waits until the next original cadence before an explicit continuation.

Pending producer receipts also require explicit Memory consideration: write or
update source-cited notes, or record a reasoned `memory_review` disposition.
No independent model or automatic note author is added. Retrieval, writing and
actual downstream use are different facts. Research tasks and accounting persist.
`meta_model_updates=false` still installs no meta-agent model binding port;
Lab agent training remains available.

The wall deadline gates decision/report admission and rejects late decisions;
it does not add service-side cancellation of an in-flight model HTTP request.
Running Lab jobs retain their own timeouts and the absolute host lifetime.
Host supervisors still enforce process cleanup. Review text, source coverage,
accepted reports and activated configurations are not evidence of improvement.

An explicit code-changing repair attempt may supply the original
`reflect_started_at` epoch to retain campaign cadence. Missed periods never
create a burst of catch-up model calls or extend the experiment deadline.
Reflection state/config identity changed: do not resume an old frozen run with
this source. Use an explicit continuation with retained evidence and accounts.
See [the component design](review/research-reflection-memory-20261107.md) for
Memory handling and the separate, future parallel-worker boundary.

## Deployment and recovery

Verify the actual target model revision, all weight shard digests, device UUIDs,
isolated GPU execution, structured decisions and native memory tool calls before
research admission. A 4+4 allocation uses separate meta inference and Lab device
capabilities. The Lab cannot acquire meta devices by changing CUDA environment
variables. Installed Python, CUDA SDK and compiler helper directories must all
be present on the launched process PATH.

An absolute UTC deadline applies to every attempt and is not extended by
initialization, repairs or resume. Preserve failed attempts, the campaign
evaluation ledger and cumulative model calls. Same-directory recovery requires
the same source identity; code-changing repairs need explicit migration or a
new attempt with retained evidence and the original remaining campaign budget.
Never clear a journal to hide unknown external effects. At the deadline, stop
only PID/UID/start-time-matched owned processes and verify device release before
marking cleanup complete. Dashboard processes are observational and do not
authorize restarting research.

Sources: TsinghuaC3I/MedXpertQA commit
`5f1db162d0d2b03d62da2d4fe086076a36105e18` (`eval/main.py`,
`eval/config/prompt_templates.py`, `eval/utils.py`, prepared test files), and
the pinned MedQA source identities in `prepare_official_benchmarks.SOURCES`.
