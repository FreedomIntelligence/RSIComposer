# Open Lab data paths and bounded decision format recovery

Run `medical-lion-night-05-20261104T161442Z` ended after 1219.863 seconds,
282 model calls and seven compactions. Its last response used a smart quote in
the JSON key delimiter and an unterminated argument string. The serving adapter
correctly rejected it as `invalid_json`; Open Lab had omitted the earlier medical
launcher's bounded format recovery, so the first such rejection ended research.
No action from the rejected response executed. The run failed before the operator
could send a stop signal. Original requests, responses and generated programs
remain remote; cleanup had no errors, no GPU compute processes remained, and the
owned Pod reached `Failed`.

The deployment also advertised `available_data.path=train.medqa.jsonl` while
seeding the file as `data/train.medqa.jsonl` in the persistent Lab workspace.
Real generated programs failed on that advertised path and on invented absolute
paths. The reference evaluator already uses the correct `data/` prefix. This is
an environment-description defect, separate from errors in model-written code.

The launcher now publishes workspace-relative `available_data` paths with their
existing hashes and row counts, declares `path_base=lab_workspace` and
`data_root=data`, and explains that the unchanged task's dataset paths are
relative to `data_root`. It changes no dataset, task, evaluation or acceptance
criterion.

`RecordedBackend` permits at most two additional generations for an explicitly
retryable rejected decision JSON response. Each generation re-enters the same
call and deadline admission and records the request, raw wire and structured
error with its response attempt. A static system correction preserves the
original prompt, model, schema, tools and output limit; rejected text is never
promoted into instructions. Native feedback tools retain their existing
recovery lifecycle. Identity mismatches, transport errors, refusals and other
non-retryable errors remain fatal. This never replays an executed Lab action or
restarts a run after ordinary recovery exhaustion.

CPU regressions first reproduced the missing advertised file and immediate
format-rejection failure against the old implementation. The path test builds
the real launcher's seeded Lab and checks the first decision prompt's paths and
hashes, including a nested input path. Local HTTP fixtures cover the observed
smart-quote error form, successful correction, three-call exhaustion, shared
call admission, deadline admission, and no retry for identity/transport/native
feedback failures. Fixture recovery is distinct from live service recovery.
Detailed test counts and frozen revisions are recorded in campaign state.

All 282 run5 raw-wire prompt counts, including rejected responses, match local
counts. Eleven memory jobs completed, 54 failed, five were cancelled and ten
encountered a full queue. These bounded feedback failures are retained. Accepted
memory falsely calls `/home/jovyan/train.medqa.jsonl` correct; valid references do
not verify that claim. Summaries also assert prompt score improvements without
a verified candidate/protocol lineage. This repair does not claim to solve
those semantic failures. No optimizer steps, saved weights, real RL or adoption
were verified. Development observations do not establish medical generalization.
