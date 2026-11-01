# Independent medical assembly

This example composes `DefaultDecisionHarness`, `DefaultExplorationHarness`, and
`Runner`. Its normal policy is `ModelDecisionPolicy` with three bounded validation
attempts (generation/parsing only, before effects); there is no medical
controller or scripted production research plan. The tests use scripted HTTP
responses and training receipts **only for integration verification**. Synthetic
symbol questions are not medical quality evidence. No source datasets are shipped.

Use Python 3.12+ from the RSISys directory with `PYTHONPATH=src:.`.

## Local format and read-only preparation

The example owns this UTF-8 JSONL schema, with exactly these fields per line:

```json
{"id":"unique-source-id","benchmark_id":"medqa","question":"<user-provided question>","options":{"A":"<option>","B":"<option>","C":"<option>","D":"<option>"},"gold":"A"}
```

IDs must be unique within a file and across development files. MedXpertQA uses
`medxpertqa` and A–J; MedQA `medqa` and MedMCQA `medmcqa` use A–D. PubMedQA
`pubmedqa` uses `yes/no/maybe` with `options: {}`. Its question contains the
question and permitted evidence context, without conclusion, long answer or gold
fields. Users must select legitimate labeled source splits, preserve source
versions/licenses, and convert their own native sources. In particular, do not
use MedMCQA hidden-test placeholder answers or confuse zero-based HF labels with
one-based source labels. The example does not infer those mappings.

A local task JSON references train and development files relative to itself:

```json
{
  "train": "/absolute/user/train.jsonl",
  "development": [
    {"benchmark_id":"medxpertqa","role":"primary","path":"/absolute/user/primary.jsonl"},
    {"benchmark_id":"medqa","role":"guardrail","path":"/absolute/user/guardrail.jsonl"}
  ],
  "guardrails": true,
  "max_guardrail_drop": 0.02,
  "max_output_tokens": 32
}
```

```bash
PYTHONPATH=src:. python3.12 -m examples.medical.run prepare --config /absolute/task.json
PYTHONPATH=src:. python3.12 -m examples.medical.run check --config /absolute/task.json
```

Both commands read, validate and print the SHA256-bound frozen task; neither
writes files, downloads data, trains, calls models, nor starts a service. Exactly
one development benchmark is primary. There is no final/holdout configuration in
this researcher-facing task. Unknown fields are rejected.

`MedicalTask` and nested reference records are frozen. Each new measurement and
training preparation checks original bytes against their frozen digests. The
protocol identity covers the data refs, benchmark membership, output limit and
acceptance settings. Exact grading trims outer whitespace only: explanations,
prefixes, lowercase multiple-choice labels and altered PubMedQA capitalization
score wrong. `score` is primary accuracy; per-benchmark accuracy, macro F1 over
all legal labels and invalid-format counts remain separate. No aggregate accuracy
mixes benchmarks.

## Explicit run

For a prompt-only run against already configured Chat Completions servers, supply
a backend JSON with `decision_binding`, `decision_model`, `actor_binding`, and
`initial_model`. Each binding follows `BackendBinding` (backend_id, kind, endpoint,
optional credential_env); model refs contain backend_id, model_id and revision.
Use `json_mode: false` on the actor binding for exact label answers. Credentials
are environment variable names, never secret values in files. Endpoint roots
include `/v1` when the server uses that prefix. No endpoint or model is guessed.

A settings JSON explicitly provides `objective`, `max_runner_steps`, and optionally
`poll_interval_seconds` (default 1). Then:

```bash
PYTHONPATH=src:. python3.12 -m examples.medical.run run \
  --config /absolute/task.json --backends /absolute/backends.json \
  --settings /absolute/settings.json --output-directory /absolute/new-run-artifacts
```

This starts baseline development inference, then lets the model choose tools and
directives. It reports `done` when the researcher/run policy finishes. Step-budget
exhaustion or baseline inference failure reports `needs_input` (CLI exit 2).
Pending jobs are not automatically cancelled. Polling sleeps while waiting.
A fresh CLI invocation creates a new in-memory assembly; it is **not** a resume
command. Use a new output directory, and reconcile pending work before another run.

Python is the native training/deployment injection entry:

```python
from examples.medical.run import build, RunSettings
from examples.medical.evaluation import MedicalEvaluator
from examples.medical.native import MedicalTrainingBackend, MedicalServing
from rsisys.adapters.serving import VLLMBackend, VLLMDeploymentBackend
from rsisys.runtime.artifacts import FileArtifactStore

artifacts = FileArtifactStore(output_directory)
candidate_backend = VLLMBackend(actor_binding)
deployments = VLLMDeploymentBackend(
    candidate_backend, deployment_directory,
    adapter_paths=operator_adapter_paths,
    allow_runtime_lora=runtime_lora_opt_in,
)
# For a new trained candidate, resolve the published native_model_manifest
# to an immutable path visible to the serving host. Return None for the baseline.
serving = MedicalServing(deployments, adapter_path=resolve_candidate_server_path)
native_learner = MedicalTrainingBackend(
    native_job_directory, artifacts,
    tokenize=tokenize_samples,
    conversion_id=tokenizer_revision_and_template_identity,
    recipe_template=native_recipe_template,
)
assembly = build(
    task, artifacts=artifacts,
    decision_backend=researcher_backend, decision_model=researcher_model,
    initial_model=expert_model,
    inference=serving,
    system_activator=serving,
    learner=native_learner,
    bind_training_inputs=native_learner.bind_inputs,
    training_initializer=expert_model,
    run_policy=explicit_run_policy,
    action_permissions=permissions,
    prepare_inference=serving.prepare_inference,
    evaluator_factory=MedicalEvaluator,  # Optional; this is the default.
    trusted_exclusions=operator_only_exclusion_rows,
)
result = assembly.run(RunSettings(objective="<explicit research objective>",
                                  max_runner_steps=100, poll_interval_seconds=1))
```

`build` writes immutable task/input artifacts but makes no model calls and starts
no jobs. The optional `evaluator_factory(task, artifacts, systems, candidates,
inference, *, prepare_inference=None)` receives the same explicit dependencies.
Its public contract consists of `task`, an `evidence` dictionary mapping visible
artifact IDs to managed refs, `establish_anchor()`, `measurement(candidate_id)`
(returning a copied EvaluationResult or None), `interpret(candidate_id)`,
`completed_evaluation(candidate_id)`, and the evaluator submit/poll/cancel/result
job methods. Application reads only these public evaluator interfaces.

`prepare_inference(system)` is an optional trusted serving-readiness hook, called
before baseline and each candidate's first evaluated sample. It must confirm the
actual requested model route, or raise `ModelActivationPending` while deployment
is pending/uncertain. The same frozen system is retried on later polls. A hook
that merely records a model name without checking the live route is insufficient.
The default evaluator scores at most **one sample per poll**. Pending preparation
returns RUNNING without querying a sample; completed rows remain accumulated and
are not repeated. Cancellation can take effect between requests and preserves
partial raw predictions. A single synchronous provider request is still bounded
by that provider's timeout, not interruptible mid-request.

`establish_anchor()` is intentionally an explicit synchronous baseline call. It
walks the baseline samples before autonomous decisions begin; it is not called by
prepare/check/build. If baseline serving is pending, `run` returns `needs_input`;
retain the same assembly and resume after readiness, without repeating already
scored baseline rows. The caller owns adapters, their device allocations and cleanup. Inference
is an explicit callable `GenerationRequest -> GenerationResponse`, or a
`ModelBackend` adapted with `backend_inference`. It must serve the **requested
candidate model**, including before application. Its sampling/server configuration
must be fixed by the operator; the generic GenerationRequest carries the frozen
output limit, not a portable provider sampling configuration. Candidate UTF-8
prompt instructions are prepended to the fixed legal-label instruction. Gold is
never in the inference request. Transport identity checks do not attest weight
contents unless the serving provider independently verifies them.

`system_activator.activate(SystemVersion, idempotency_key=...)` confirms the same
manifest when ready; use `ModelActivationPending` until readiness. Reference-only
activation supports prompt changes with an unchanged model. `SystemRegistry`
merges component candidates, enforces their explicit parent and applies via CAS.
A prompt proposal creates a candidate without mutating the incumbent. Evaluation
uses `EvaluationSpec`, so the default harness retains candidate evaluation context.
Application rechecks strict primary improvement against the current incumbent.
Guardrails, enabled by default, compare against the **initial fixed anchor**, never
a gradually weaker incumbent. `guardrails: false` is a choice made before freezing
the task. No candidate is automatically selected on a tie. A known candidate that
fails acceptance, has no successful evaluation, or has failed interpretation
receives FAILED application feedback plus a rejection artifact. The incumbent stays
unchanged and the researcher chooses its next action; rejection does not exit the
runner.

The configure_prompt, evaluate and apply_candidate directive specifications declare
`accepts_input_refs: false`. Omitted `input_ids` means empty for those modes;
explicit nonempty input IDs are a validation error before effects. Their backends
also reject unexpected input refs. Training alone consumes explicit data refs.

Activation pending remains a nonterminal `applying` receipt, and subsequent runner
ticks reconcile the same candidate and idempotency key. Cancellation before an
activation intent starts prevents activation. Once activation has been issued,
cancellation cannot discard an uncertain remote effect: the same receipt stays
pending until readiness is confirmed. If application is confirmed, feedback says
it completed despite the late cancellation, with `cancel_requested` and
`activation_confirmed` evidence. No rollback is claimed. `SystemActivationRejected` is converted to FAILED
application feedback only when the activator confirms that application did not
occur (or was rolled back). Unknown external errors still propagate with the
application intent/receipt retained.

Training requests always carry `TrainingOrigin`. Default restart initialization is
`training_initializer` (default the explicit baseline model), independent of the
current experimental parent. Explicit origins must name the current parent and an
allowed baseline/current initializer; restart still requires the fixed initializer.
Checkpoint refs must be accessible through the trusted artifact store. Native
capabilities reject unsupported resume modes. The adapter must return the same
parent in its candidate; the example never adds a parent after training.

`bind_native_inputs(request, artifacts) -> ActorTrainingRequest` is **required when
a learner is supplied**. This trusted callback validates/materializes the actual
native recipe and training inputs against `request.data_refs`. It may replace
`recipe_ref` with the bound native config identity; data refs, base, method and
origin must remain identical. The native command must read that bound config,
consume precisely those validated source bytes (or their explicitly documented
conversion), and retain source digests, conversion/tokenizer identity and actual
consumption evidence. Do not validate one file while a preconfigured trainer reads
another. For `ProcessTrainingBackend`, inject a native `command(request, job_dir)`
that consumes this binding; its default tokenized-JSONL worker cannot consume this
example's question/options JSONL unchanged. No universal converter, dotted dynamic
imports or hidden source-data fallback is installed.

The bundled `MedicalTrainingBackend` keeps caller source refs/origin intact,
materializes its configured native recipe and tokenized input once per submission,
and records source digests/IDs, conversion identity, native recipe/data hashes and
native training evidence. `tokenize_samples(tuple[Sample, ...])` must return native
feature dictionaries with `input_ids` and `labels`; it is an explicitly supplied
operator conversion, not an automatic benchmark importer. Its fixed `recipe_ref`
is a template-content hash. `build` advertises that ref as the training schema's
single allowed recipe; the backend independently rejects other refs. Learners
without an explicit string `recipe_ref` retain the general recipe schema.

`MedicalServing` is both the inference callable and SystemActivator. Pass its
**`prepare_inference`** method as the evaluator hook. `VLLMDeploymentBackend`
checks the serving alias and probes it before candidate evaluation; adapter paths
must be explicitly mapped to the requested model. For dynamic candidates, the
optional `adapter_path(system)` callback may read the managed `model` manifest
(`native_ref` identifies the native output) and map it to a serving-host path.
This callback owns cross-host accessibility; the framework does not copy weights
or guess mount mappings. It cannot change an existing model-to-path binding.
Runtime LoRA mutation requires
explicit opt-in; static/shared routes are not restarted or replaced. The hook and
inference must use the same serving object/backend. Wrapper-level durable training
receipts do not make the example's in-memory loop state durable.

The trusted overlap validator excludes normalized, casefolded, whitespace-collapsed
question matches and IDs against all development samples and programmatically
supplied private exclusions, before training submission. Private exclusion rows do
not enter the task, model prompts, tools or experience. This is exact separation,
not semantic deduplication, pretraining contamination detection or medical fact
validation. The caller must keep private files/artifacts out of any researcher
workspace; this assembly installs no arbitrary filesystem tool.

## Evidence and lifecycle limits

Raw artifacts contain each actual development output, legal-format flag, grade,
model/prompt manifest and usage. Acceptance interpretations are separate artifacts
with raw source refs. `read_artifact` accesses only published train/prompt/development
evidence. External component references (including `file:///` weight directories)
return reference metadata and an explicit external-resolver requirement; the read
tool does not pass them to the artifact store or traverse directories.
ResearchTools shares the decision harness's memory and agenda; archived
observations and researcher notes retain their distinct roles. The installed action
catalog exposes only real data storage, visible artifact reads and research tools.

Malformed model/data results and transport failures become terminal evaluation
failures, with completed predictions retained as partial raw evidence.
Interpretation errors also terminate the evaluation as failed and preserve links to
its completed raw measurements. A failed interpretation cannot authorize application. No hidden
network retries or repeated queries occur when that failed job is polled again.
These partial observations are not a complete benchmark score. A new evaluation
implementation/assembly is required for an explicit retry after reconciling failure.

With `FileArtifactStore`, task manifests, predictions and interpretation artifacts
persist. Loop state, experience, evaluator receipts and SystemRegistry are currently
in memory. Retain the same assembly to resume a bounded Python run; process-crash
recovery requires caller-owned coordinated persistence. This example does not claim
durable evaluation or medical efficacy. Native backend/deployment proof is separate
from the explicitly labeled fixture verification:

```bash
PYTHONPATH=src:. python3.12 -m unittest tests.test_medical_example tests.test_framework_adapters -v
```

The operator reported a real vLLM attempt5 completing in 10.68 seconds:
prompt proposal → development evaluation → rejected application → researcher
finish, with the baseline unchanged. This is a live lifecycle check, not evidence
of medical improvement or a fresh holdout result. The fixture tests independently
verify the same rejection behavior, native source/conversion provenance and
candidate-serving routing without claiming model training quality.
