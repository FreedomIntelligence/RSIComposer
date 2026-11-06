# Candidate inference evaluation protocol

The host owns datasets, scoring, resource/time budgets and submission admission.
The meta-agent owns its checkpoint/adapter and inference implementation, including
loading, tokenizer, prompts, thinking/decoding and batching. Evaluation is a
protocol boundary rather than a prescribed model implementation.

## Submission

`medical_evaluate(key, candidate_id, split, submission_path)` accepts a directory
inside the Lab workspace. `split` is `public` or `closed`. A minimal bundle is:

```text
candidate/
  submission.json
  infer.py
  weights/...
```

```json
{
  "schema": "candidate-inference-bundle-v1",
  "entrypoint": "infer.py",
  "base_model_path": "/operator/registered/base",
  "checkpoint_path": "weights"
}
```

Use `adapter_path` instead of `checkpoint_path` for an adapter. Omit both for a
baseline using the registered base. Candidate paths are bundle-relative; symlinks
and escaping paths are rejected. Helper files and inference configuration can
live in the same directory. The host copies the bundle and verifies every file's
SHA before admission. Its identity binds code, weights/configuration and the
registered base's weight/tokenizer manifests. Later workspace edits cannot modify
an accepted candidate. Mutation of the frozen copy invalidates the score.

The installed Python executes the entrypoint with the frozen directory as its
working directory. The script reads one JSON object from stdin:

```json
{"protocol":"exact-option-letter-v1","items":[{"benchmark_id":"example","id":"1","messages":[{"role":"system","content":"Choose the best answer. Return exactly one option letter, with no explanation."},{"role":"user","content":"Full question\nA. first option\nB. second option"}]}]}
```

The script can transform prompts for its model and must write one JSON object:

```json
{"predictions":[{"benchmark_id":"example","id":"1","text":"B"}]}
```

Return one prediction per supplied identity. All provided option letters are
valid choices, including A–J where present. The host accepts a whole final letter
with optional whitespace/period, and never scans explanation text for letters.
Missing/duplicate/unknown predictions fail evaluation. A malformed answer counts
as incorrect. Scores and adoption cannot be supplied by candidate code.

The protocol/source/budget hash and dataset SHA identify a comparison. Inference
code belongs to candidate identity, so different inference methods may compete
under one host protocol. This measures the whole candidate pipeline; claims about
training alone require a comparison with the same inference implementation.

## Public and closed data

Public development questions/answers are available to the agent for debugging.
Public evaluation may return detailed evidence and does not consume closed quota.
Training uses only the designated training partition.

Closed questions and answers remain at the operator's isolated evaluation service.
The research meta-agent has no dataset file, individual answers or per-item logs.
The submitted script necessarily sees the current inference input inside that
service, without gold. Its writable files, logs and network cannot carry data
back to the research environment. Only aggregate scores and fixed status codes
cross back through the gateway; private diagnostics stay at the service.

The existing local Lab is trusted execution under one UID. It cannot enforce
this boundary against research programs with the same filesystem/process access.
Therefore closed evaluation is disabled unless an independent gateway is supplied.
No same-UID local fallback exists. This repository defines the client/receipt
contract and admission ledger; deploying and verifying the isolated service is a
separate operator step. It has not been provisioned by changing this code.

Existing visible development data is **not** silently relabelled as closed data.
No private dataset is discovered or read by the Open Lab launcher. A combined task
manifest is reduced to public `train` and `development` fields before seeding.

## Campaign quota and recovery

Closed submissions are accepted at least 1800 seconds apart, shared by all
candidates and runs of the operator-assigned campaign. The first can be accepted
immediately. The host clock and persistent locked ledger are authoritative for
local admission. Clock rollback cannot grant an extra slot.

Acceptance is recorded before execution. Candidate failure, timeout, cancellation
and an infrastructure failure after acceptance all retain the reservation. The
same key returns the original job; changed arguments under that key are rejected.
A crash with an uncertain effect requires host reconciliation of that submission,
never automatic resubmission or clearing the journal. The gateway independently
enforces the same campaign quota and idempotency, including any transport retry.
Its private admission record protects the quota even if a research-side process
can modify an ordinary local file.

For a shared closed campaign, provide `--evaluation-campaign-id`, a persistent
`--evaluation-ledger` outside the individual run directory, and
`--closed-evaluator-config`. These are host settings, not model tool arguments.
The gateway config contains no closed dataset paths:

```json
{
  "argv": ["/operator/bin/isolated-medical-evaluation-client"],
  "service_id": "operator-assigned-service",
  "dataset_sha256": {"closed-benchmark": "<64 hexadecimal characters>"}
}
```

The command is an operator-owned transport to the isolated service, not the
submitted inference entrypoint. It receives JSON on stdin containing campaign,
submission and candidate IDs, protocol hash, frozen bundle path/manifest/hash,
and the execution budget. It imports the bundle into immutable service storage,
validates hashes and registered bases, enforces quota, executes the inference
worker under the isolation described above, and writes one receipt to stdout.
Transport/import details depend on the deployment. It must never execute submitted
code with access to service gold files, private logs or research network/storage.

A success receipt echoes `protocol_sha256`, `campaign_id`, `submission_id`,
`candidate_id`, `submission_sha256`, the configured `service_id` and
`dataset_sha256`; it adds `status: "succeeded"` and
`metrics: {"closed-benchmark": {"accuracy": 0.5, "n": 2}}`.
A failed candidate returns the same identities and `status: "failed"`.
The client reconstructs an allowlisted record, drops all other fields and
suppresses transport stdout/stderr and exception text from Lab logs. Gateway
identity, dataset identity and metric types/ranges are checked before publication.

## Validation scope

CPU fixtures exercise real submitted Python execution, hashing, public host
rescoring, ledger concurrency/restart, result recovery, and closed-receipt/log
redaction. Synthetic gateway fixtures prove the adapter contract, not OS isolation,
real model inference, medical accuracy or training improvement. GPU experiments
and closed-service deployment require separate execution authorization.
