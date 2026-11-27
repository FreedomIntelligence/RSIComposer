# RSIComposer framework contracts

This is the reference for extending the Python library. The [composition guide](compose.md)
shows how to choose components; this page specifies what the Host guarantees and what
each component must implement. Import public types from `rsicomposer.api`.

| Boundary | Public interfaces | Who owns it? |
| --- | --- | --- |
| Assembly | `Application`, `ApplicationConfig`, `Registry`, `MethodDefinition`, `ModuleAssembly`, `ControllerPort`, `ComponentDefinition`, `ComponentGraph` | The caller selects implementations; the Host validates bindings and owns acquired resources. |
| Decision and execution | `ControllerDefinition`, `DecisionPolicy`, `ControllerHarness`, `LabExecution`, `OperationSpec`, `OperationCall`, `OperationResult` | The Controller chooses operations; the Host authorizes and dispatches them; the Lab validates domain inputs and executes. |
| Authorization | `ActionPermissions`, `AccessRequest`, `SensitivePolicy` | The Host policy applies in addition to endpoint-local rules. |
| Run control and background work | `RunPolicy`, `RunContext`, `RunDecision`, `RunResult`, `JobRef`, `JobOutcome`, `JobStatus`, `ExecutionLifecycle` | The Host controls admission and waits for owners to confirm settlement. Owners must expose and stop their work. |
| State and evidence | `RecoveryContract`, `SnapshotContract`, `RecoveryRequired`, `LocalSnapshotCapture`, `restore_snapshot`, `MetricGoal` | The Host owns run identity, recorded effects, and goal admission. Components own explicit state and Lab evidence. |

## Assemble components without starting work

- Select one complete method or a Controller and a set of named Labs through
  `Registry.build(selection)`. For Python composition, pass an explicit `ModuleAssembly`
  or use `ModuleAssembly.from_components(graph, decision=..., lab=...)`. A graph
  decision slot supplies `decision-factory/v1`; a graph Lab slot supplies
  `lab-execution/v1`. These IDs identify compatibility, not authority.
- For a registered Controller, `ControllerDefinition(policy, harness=..., memory=...)`
  takes a Policy factory `(context, config)`, an optional Harness factory with that
  signature, and an optional Memory factory `(config)`. A Policy implements
  `decide(window) -> Decision` (or is a callable with that signature). A custom
  Harness implements `window`, `observe`, `tools`, and `execute_tool`.
- For direct composition, the decision factory takes `(context, lab_execution)`.
  Use `build_default_decision(..., policy=...)` or, with a custom Harness,
  `build_harness_decision(..., policy=..., harness=...)`. For a registered
  `ControllerDefinition(endpoint=True)` the factory takes
  `(context, lab_execution, config)` and supplies its own Harness. Keep the
  supplied Host and Lab routing; do not borrow a live endpoint from another run.
- A run has a replaceable Controller only when its assembly supplies
  `controller_factory`, returning a `ControllerPort` that names the initial
  instance and its version. That port is what makes the Controller Host exist,
  and the Host is what serves `host.adopt_controller` and rebuilds a catalog
  entry, so `controller_catalog` is rejected without it: a catalog no port can
  build from would list Controllers nothing could ever activate. Without the
  port the decision factory's loop is the Controller for the whole run, and
  adoption is unavailable by construction rather than failing at call time. The
  port's instance must be the loop the decision factory returns, because that
  instance is what the Host activates.
- A Lab implements `describe() -> tuple[OperationSpec, ...]` and
  `execute(call) -> OperationResult | JobRef`. A registered Lab uses local names
  without dots; the Host publishes them as `<alias>.<operation>`. Direct
  `LabExecution` supplies its own installed operation names. A Host-wired
  Controller can declare `required_operations()`; the check is against installed
  operations, not permission to call them. Nonempty `required_profiles()` also need
  a profile-aware Lab provider, such as the Registry's grouped Labs. Bare
  `LabExecution` providers cannot advertise sealed splits; use the grouped Lab
  route for Host-enforced split roles.
- `OperationSpec.input_schema` describes the arguments to the Controller; it is
  not enforced at dispatch. The Host checks only the transport envelope — an
  object with string keys — and forwards the call. The Lab enforces the schema
  inside `execute()`, immediately before effects, and rejects a hopeless call
  with `OperationArgumentError` so the Host can record it as correctable
  feedback instead of a failure. An operation whose schema nothing enforces is
  the Lab's own gap, so a Lab that accepts structured arguments should check
  `properties` and `required` against them at entry.
- Keep selection factories and config validators free of resource acquisition.
  Construct resources at launch, but do not start research or re-submit effects
  from a factory. Make owned `close()` methods retryable. The registered path owns
  the Policy it constructs. For direct composition, register cleanup for a
  resource-owning Policy with the assembly context, unless the graph owns it.
  A trusted Python
  registration module is executable code: `catalog` and `explain` can still import
  and run its registration code. `explain` is not a live Lab/provider check.

## Keep authorization at the Host boundary

Pass `ActionPermissions` to `Application(..., action_permissions=...)`, or use the
CLI `run --permissions` input. `operations=None` leaves the operation-name filter
open; an empty collection permits none. The Host filters discovery and validates
normalized calls, intersecting its policy with endpoint restrictions. Schemas,
component contracts, model proposals and run-control hints do not grant access.
An `AccessRequest` has no action during discovery, so an argument-dependent rule
can reject a concrete call even if its operation is visible.

This protects the Controller action route provided by the framework; it is **not**
a process sandbox for host-supplied Python. The host-only sealed evaluation phase
is separate: a grouped Lab must supply both the sealed split and a
`final_evaluations()` plan. The built-in `ResearcherRunPolicy` runs that plan on
normal final completion. A resumable research round or a hard stop does not
start it. Host reports can retain sealed aggregates; Controller feedback,
Memory, archives, and learning references do not include them.

## Own background work through settlement

`JobRef(job_id, backend)` identifies unresolved work. For `JobOutcome`, only
`SUCCEEDED`, `FAILED`, and `CANCELLED` are terminal. A request to cancel, a running
observation, and a terminal receipt are different facts. For a Controller's
explicit wait, the Host uses the cancellation hook to send the request once.
If it receives a valid nonterminal acknowledgement, it uses `poll_job(ref)` on subsequent ticks until the job is
terminal or cleanup times out. After the acknowledgement has been checkpointed,
durable resume polls the same job instead of submitting the request again.

| Work owner | Extension points | Settlement rule |
| --- | --- | --- |
| Lab operation | Return a terminal `OperationResult`, or return a `JobRef` with a Lab `poll_job(job)` that reports `JobOutcome`/progress through the registered Lab adapter. Implement `cancel_job(job)` or checkpoint control when supported, and `reattach_job(job)` for jobs a durable workspace can reconnect to. Direct executors implement the call-aware Host-facing ports instead. | Preserve call and job identity; do not return a terminal outcome before work exits. |
| Controller Policy | Return `Decision("wait", job=ref)` with `can_poll_job(ref)` and `poll_job(ref)`, plus `cancel_job(ref)` for cancellation. Background work that can coexist with other decisions also needs `pending_research_job()` and appropriate `drain_research()`/`cancel_research()` hooks. | A ready Controller wait lets the Policy continue; it is not a Lab receipt. Keep unresolved work visible until it settles and stop admitting new work when stopping. |
| Lab-owned resources | Optional `ExecutionLifecycle.shutdown(mode) -> bool` and `close()`. | Shutdown closes admission and returns `True` only after owned work exits. Keep dependencies usable if cleanup must be retried. |

A `RunPolicy.evaluate(context) -> RunDecision` may continue, restart, or stop the
run. A hard stop prevents **new** dispatch, an uncommitted Controller adoption, and
new sealed evaluation. It cannot preempt a synchronous callback already entered.
`Application.close()` can outlast the Runner’s cleanup window. Check
`RunResult.shutdown_complete` for the latter; CLI `cleanup.status` reports the
separate close outcome.

`max_decision_steps` caps new decisions, not the number of Lab receipts. The
standard Host can settle already-admitted, independently pollable Lab work with
`wait_for_jobs=True` without sampling a new decision or adopting a Controller.
Opaque or checkpoint-held work can still return `waiting`. Requesting a timeout
or reaching a cap does not authorize retrying dispatch with an unknown outcome.

## Restore state without widening permissions or repeating effects

- `Application.resume(data_dir, ...)` continues the original durable workspace.
  It requires matching framework source, Python major/minor, and run/component
  identity. A portable snapshot instead restores a supported, quiescent bundle
  into a new directory with `restore_snapshot(...)`. It preserves the run ID, does
  not fork, and needs explicit plugin paths or the original Python assembly for
  custom components. Do not treat a copied live workspace as a portable snapshot.
  For a new CLI durable run, the first checkpoint commits startup after the
  launch metadata, initial literature, and recovery journal have been written.
  An interrupted launch without that checkpoint is not resumable.
- Supply explicit state adapters or `RecoveryContract` (`contract_id`, `capture`,
  `restore`); the portable contract additionally implements `validate_snapshot`.
  The convenience `restore_snapshot` API uses the bundled assembly recovery
  contract; implementing a custom capture contract does not add a custom restore
  entrypoint. Capture detached data; restore identities without re-submitting work.
  A registered Lab's optional `validate_snapshot()` can veto portable capture or
  restore; it does not veto writing durable checkpoints. For explicit Python
  factories the supplied assembly and its adapters must validate external
  component identity; the framework source hash alone cannot do so.
  Saved Host permissions are restored before component construction. Supported
  older state without saved permissions denies Controller operations; arbitrary
  Python permission callbacks are not durable or portable.
- A stable `call_id` ties together dispatch and receipt. Treat it as an opaque
  identifier, not a filename: stored receipt names are derived from a digest,
  and the receipt itself keeps the original ID. Mark an operation
  `idempotent=True` only if repeating the same call ID is safe. Even a matching
  terminal receipt does not prove the enclosing interrupted step committed.
  `RecoveryRequired` means stop and use verified reconciliation, not automatic
  replay or journal deletion. Local record deduplication is not an exactly-once
  guarantee for external observers.

## Declare evaluation evidence explicitly

Use `GoalRunPolicy(MetricGoal(...))` for a threshold on measurements. The Host
checks the delivered operation against the installed Lab declaration, then checks
its candidate, `protocol_ref` and exact `input_refs` against the original request
where present. Built-in profiles declare their evaluation operations; a custom
Lab can expose `evaluation_operations()` with catalog names (local for registered
Labs, full names for direct execution). A receipt needs those identity fields and
finite numeric `quality_metrics`. An evaluation declaration or a built-in profile
alone is not sufficient: a Lab that emits benchmark metadata but not the
canonical candidate, protocol and input references (for example, the reusable
Math Lab) does not drive a `MetricGoal` without an explicit evidence integration.
A simulated receipt is excluded by default; an
explicitly invalid or unscored receipt is not a qualifying measurement. Use a
distinct protocol reference for each dataset or pin `MetricGoal.input_refs`.
A threshold does not by itself apply the candidate or replace a sealed evaluation.

## Check a new extension

- Assemble the same minimal Policy/Lab pair through each supported entrypoint.
- Verify the same allow/deny rules before and after supported recovery.
- Exercise normal completion, cancellation acknowledgement, late settlement,
  and timeout with a blocked fake provider. No keys or accelerators are needed.
- Check both a valid evaluation and a receipt that has metric fields but comes
  from a non-evaluation operation.

The repository’s [library contracts](../verification/test_library_contracts.py),
[resource boundaries](../verification/test_resource_boundaries.py),
[run control](../verification/test_run_control.py), and
[durable recovery](../verification/test_durable_recovery.py) tests are executable
examples. Run them from the repository root:

```sh
PYTHONPATH=src python -m unittest discover -s verification -v
```
