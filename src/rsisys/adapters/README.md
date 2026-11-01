# Framework boundaries

These optional adapters run trusted native jobs and call an existing vLLM API.
Importing them needs only the standard library. Install neither frameworks nor
services through the adapter. Harnesses continue to own policy and role binding.

## Training

```python
from rsisys.adapters.training import ProcessTrainingBackend

learner = ProcessTrainingBackend('/absolute/durable/jobs')
job = learner.submit(request, idempotency_key='experiment-001')
outcome = learner.poll(job)  # explicit observation; never submits another job
# A new ProcessTrainingBackend with the same root can reattach(job).
# After a successful poll: learner.result(job). To stop: learner.cancel(job).
```

The default executes `python -m rsisys.adapters.training transformers-peft RECIPE`
with the controller's interpreter; that environment must already contain torch,
Transformers and PEFT. `request.recipe_ref` is an absolute native JSON file:

Optional native recipe `num_processes` (default `1`) selects single-node torchrun
DDP. For example, `8` requires eight scheduler-allocated visible CUDA GPUs and
loads one full model replica per GPU. This is not tensor parallelism or FSDP.
The worker sets each local rank's device before loading, synchronizes checkpoint
saves, and publishes progress/results from rank zero only. Training evidence
includes actual per-rank and summed sample/token exposure. The caller owns GPU
allocation; the adapter never requests resources or shares an active inference
engine's GPUs.

```json
{
  "initial_model": {"backend_id":"vllm","model_id":"tiny-gpt2","revision":"random-smoke"},
  "model_config": {"model_type":"gpt2","vocab_size":32,"n_positions":32,"n_embd":16,"n_layer":1,"n_head":2},
  "lora_config": {"r":2,"lora_alpha":4,"target_modules":["c_attn"],"task_type":"CAUSAL_LM"},
  "train_dataset": "tokenized.jsonl",
  "training_args": {"use_cpu":true,"max_steps":3,"per_device_train_batch_size":1,"save_strategy":"no","logging_steps":1,"report_to":"none"},
  "seed": 0
}
```

Each JSONL row is a native tokenized feature dictionary, for example
`{"input_ids":[1,2,3,4],"labels":[1,2,3,4],"attention_mask":[1,1,1,1]}`.
The native default collator requires compatible row shapes. No source-data
conversion or tokenization is inferred. Medical/domain code must supply its own
explicit converter and native references. If `request.data_refs` is nonempty,
every URI must identify this local tokenized file and every supplied SHA256 digest
must match. A remote, raw, or different source reference is rejected.

`initial_model` must equal `request.base`; this is an explicit recipe identity
binding, not verification of base weights. The random `model_config` example is
only a functional training smoke. For pretrained weights use
`model_name_or_path` and optional native `model_kwargs` (including `revision`).
`lora_config` is passed to PEFT `LoraConfig`, and `training_args` to Transformers
`TrainingArguments`. Optional `result_model` specifies the output ModelRef.
Default output is the unique job's `adapter/`; any existing output directory is
rejected, even if empty. The worker never overwrites another adapter directory.

The worker writes plain `dataclasses.asdict(TrainingResult)` JSON to
`RSISYS_RESULT_FILE`. Its candidate includes the adapter and `training_evidence`
artifacts. Evidence records recipe/data SHA256, actual dataset rows, observed
training microbatch rows and input-token exposures, optimizer steps, initialization
kind and rank scope. It does not assert a weight digest or medical improvement.
The parent system comes from `request.origin.parent_system` independently of the
base model. Restart and PEFT weights-only initialization are supported; the latter
loads an explicit local PEFT checkpoint and starts a fresh Trainer. Full-state
resume, running hyperparameter updates, and checkpoint export are unsupported.

For another native worker supply
`ProcessTrainingBackend(root, command(request, job_dir), *, capabilities=..., result_reader=...)`.
The callable returns `TrainingCommand(argv, cwd, env={}, timeout_seconds=3600)`.
`transformers_peft_command(config_ref, *, cwd, python=..., num_processes=1)` and
`llamafactory_command(config_ref, *, cwd, executable='llamafactory-cli')` build
explicit argv; no shell expansion occurs. LLaMA-Factory retains its native YAML
and dataset registry. Its unmodified CLI does not emit our result format: provide
`result_reader(request_namespace, job_dir) -> TrainingResult` that reads actual
native outputs after exit zero. Declare only semantics the configured command
implements. Custom commands default to SFT/restart capabilities, not resume.

Receipts, exact argv/request, PID/start identity, logs, cancellation and exit state
live under the durable root. Detached supervisors and filesystem locks prevent
relaunching the native command on duplicate submit, including concurrent clients.
A controller may exit and another instance reattach. Lost supervisor identity is
`UNKNOWN`; a prepared receipt without observed startup remains queued and an
explicit repeated submit can attach a supervisor. No poll/result call restarts
training. `result()` reads only the cached validated successful result, so poll
first. Process identity uses Linux boot ID/start ticks, or POSIX `ps` start time
(second precision) on macOS. This is a single-host POSIX bridge, not a scheduler,
security sandbox, host migration mechanism or power-loss training resume.
A lost supervisor can leave a native child requiring operator cleanup; cancellation
is a durable request to the supervisor, not a claim of cancellation after its loss.

The native child receives a small environment whitelist including accelerator
visibility, distributed rank/master variables, numerical thread limits and cache
locations. Explicit `TrainingCommand.env` adds overrides. It is recorded in the
receipt: do not put secrets there. Logs persist on disk without automatic rotation.

## Serving

```python
from rsisys.adapters.serving import VLLMBackend, VLLMDeploymentBackend
from rsisys.loops.contracts.model_service import BackendBinding

backend = VLLMBackend(
    BackendBinding('vllm', 'http', 'http://127.0.0.1:8001/v1', json_mode=False),
    generation_options={'chat_template_kwargs': {'enable_thinking': False},
                        'temperature': 0, 'max_tokens': 16})
# backend.generate(GenerationRequest(...)); backend.list_models()
deployments = VLLMDeploymentBackend(backend, '/absolute/durable/deployments')
# deployments.prepare(model, idempotency_key=...) probes an existing model.
```

`VLLMBackend(binding, models=None, *, generation_options=None)` accepts optional
`ModelRef -> server alias` bindings. Native generation options cannot override
model/messages/stream. `max_tokens` must be a positive integer, is capped by
`request.max_output_tokens`, and readiness always uses one token.
`max_completion_tokens` is rejected to avoid conflicting budgets. HTTP responses
must identify the requested alias. Ordinary vLLM responses leave
`revision_verified=False`; a model name or deployment label is not a checkpoint
attestation. A `length` finish is returned as provider text, so policy callers must
still validate their required response format.

Deployment `prepare/query/reattach/release` implement the optional core contracts.
Default preparation is read-only: list the model and confirm an actual generation
request. Static/shared deployment release is unsupported. Runtime loading requires
`allow_runtime_lora=True` **and** `adapter_paths={ModelRef: server_native_path}`.
The operator must already have an appropriately configured LoRA server. Each root
has a persisted random namespace; each deployment has an isolated name. The adapter
never replaces existing names, uses in-place reload, restarts an engine or unloads
base/shared models. LoRA readiness requires matching model-list root/path and a
successful generation probe. Uncertain load responses are reconciled by query,
without blindly repeating the load. Paths must be immutable for a model revision;
path metadata is not a digest of remote weights.

`VLLMLoRAActivator(deployments, adapter_path=None)` implements
`activate(result, *, idempotency_key) -> ModelRef`; an optional callback maps an
actual result to a server-native adapter path. It raises `ModelActivationPending`
until readiness and never changes role policies. Query/reattach do not load or
unload. Release affects only the adapter recorded under this root and checks its
path before unloading. No method mutates shared engine configuration.

## Verification and primary references

`tests/test_framework_adapters.py` uses real local subprocesses and a real local
HTTP server. Its worker/CLI/vLLM payloads are **fixtures**, not framework training,
LLaMA-Factory installation, GPU execution or real vLLM deployment verification.
Remote framework proof and medical evaluation belong to their explicit runs.

Boundaries checked against primary docs:
[Trainer train/resume](https://huggingface.co/docs/transformers/main/en/main_classes/trainer),
[PEFT checkpoint format](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint),
[LLaMA-Factory native CLI](https://github.com/hiyouga/LLaMA-Factory#quickstart),
[vLLM LoRA load/unload and lineage](https://docs.vllm.ai/en/latest/features/lora/).

Native Transformers/PEFT recipes may explicitly set `device_type: "npu"`
(default: `"cuda"`). Select allocated Ascend devices with
`ASCEND_RT_VISIBLE_DEVICES` before launch and use a Python with matching
`torch_npu`/CANN. The worker validates visible device count, maps each local rank
to one logical NPU, and uses HCCL for distributed jobs. Single-NPU execution
needs no distributed launcher. This does not install dependencies or allocate
cluster resources. CANN and device visibility are preserved by the process
adapter; optional frameworks remain imported only inside the worker.

`VLLMBackend.count_tokens(request)` 调用服务根路径 `/tokenize`，使用相同 model alias、
messages 和 chat_template_kwargs，计数包含 generation prompt。原生 tokenizer 不可用或返回无效
count 时明确报错；实现其他服务的 adapter 时，可提供同名计数方法。计数调用不产生模型决策，
但消耗正常运行时间。模型生成、摘要调用仍由原有预算管理。

`VLLMBackend.supports_response_schema` advertises native structured-output
transport. `GenerationRequest.response_schema` becomes `response_format` with
`type=json_schema`, `strict=true` and that exact schema. Schema property order
is preserved in the HTTP body, native generation options and medical request
logs: constrained generation grammars may enforce that order. Canonical sorting
for training/artifact hashes must not be reused for inference serialization.
Successful schema compilation does not prove value fidelity. The tested
modelmate service truncated references under an unanchored `\S` and altered
legal newlines, quotes and backslashes under native string patterns/length bounds.
`VLLMBackend(..., native_schema_mode="portable")` is an explicit provider profile:
it removes `pattern`, `minLength`, `maxLength` and `uniqueItems` from schema
positions on the wire. On objects with their own properties it also omits
`allOf`/`anyOf`/`oneOf` conditional refinements: the tested compiler otherwise
closes partial branches and loses sibling fields. Base types, enum/const, numeric
bounds, array size, base required fields, field order, closed envelopes and the
outer discriminated decision union are preserved. Canonical schemas are unchanged.
The default `"exact"` profile transmits them unchanged. Neither profile retries a
failed request with weaker constraints or silently corrects returned parameters.

Portable grammar is a decoding aid, not semantic validation. Callers must validate
against the canonical schema and their Action Space before effects; the bundled
parsers retain nonblank, NUL, single-line, duplicate-ID, reference, length and cross-field checks.
The medical request log records both canonical and wire schemas and the profile.
Native object properties are emitted in schema order; policy instructions and
canary target presentation follow that order without changing target values.

For open dictionaries, `additionalProperties: true` is explicit: the tested strict
compiler otherwise closes an undeclared map and rejects `minProperties: 1`.
The opt-in live suite checks exact generated values and production parsers,
including a complete medical train envelope prepared without submitting a job.
Set `RSISYS_LIVE_SCHEMA_MODE=portable` for modelmate. It can first compile with the
installed provider's xgrammar via `RSISYS_LIVE_NATIVE_PREFLIGHT=1`; this protects
the endpoint from compiler errors but does not replace generation/value checks.
Default tests do not contact any endpoint. `RSISYS_LIVE_EVIDENCE` keeps requests,
wire schemas, raw responses and the prepared-but-unsubmitted training request.
This is distinct from
`json_object`, which only requests JSON. A server rejection is surfaced without
retrying unconstrained. Conflicting native format/tool options, truncated replies
and duplicate JSON keys fail before execution. Server support for the particular
schema is still required; domain validation and execution permissions remain in
force. See [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).
