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

The same meta-agent sees a scheduled review in its live runtime context, reads
its existing decision graph and raw archive, and may build and activate a
`meta.prompts` or `meta.harness` candidate through versioned host ports. It then
calls `reflect_report` with decision references, analysis and either `changed`
or `no_change`. No separate summarizer/model is created. Explicit agent_memory,
research tasks and accounting persist. No meta-agent model binding port is
installed with `meta_model_updates=false`; Lab agent training remains available.

Reflection receipts bind cited decision-record digests and the actual component
version. They are self-reports, not proof that the cited evidence was read or
that research performance improved. The monitor must check archive reads and
subsequent behavior separately. Review hints refresh at decision boundaries;
they do not preempt an in-flight model call. Missed periods do not cause a burst
of catch-up requests. Durable state detects an interrupted receipt write rather
than resetting the schedule.

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
