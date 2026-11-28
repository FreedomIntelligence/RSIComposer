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
- A registered assembly retains one rebuildable selection: the root method or
  pair, whether built-ins were enabled, plugin names and digests, and effective
  Lab bindings and Controller slots. Launch, nested workers and automatic restore
  use the same Registry resolver. `modules.registration` is a detached view for
  inspection. Method identity records what ran; it is not a recipe for importing
  code. Replacing a factory in Python makes the result an explicit assembly.
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
- Choose exactly one Controller source. A decision factory (or graph decision
  slot) builds a fixed loop. For a hosted Controller, supply `controller_factory`
  **instead of** a decision factory: it returns a `ControllerPort(instance, version)`.
  The Host takes ownership immediately and activates `instance.endpoint` after
  the same routing, compatibility and permission checks as any Host-wired loop.
  `controller_catalog` requires this port. A registered Controller can run without
  offering `host.adopt_controller`: adoption requires Policy and Harness state
  ports, and any installed Memory must support export/restore or rebuilding from
  the Host's evidence log.
- A Lab implements `describe() -> tuple[OperationSpec, ...]` and
  `execute(call) -> OperationResult | JobRef`. A registered Lab uses local names
  without dots; the Host publishes them as `<alias>.<operation>`. Direct
  `LabExecution` supplies its own installed operation names. A Host-wired
  Controller can declare `required_operations()`; the check is against installed
  operations, not permission to call them. Nonempty `required_profiles()` also need
  a profile-aware Lab provider, such as the Registry's grouped Labs. Bare
  `LabExecution` providers cannot advertise sealed splits; use the grouped Lab
  route for Host-enforced split roles.
- For a constraint that depends on *both* components' settings, a Policy can
  return `{alias: json_object}` from `required_lab_bindings()`. At startup and on
  replacement the Host sends each requirement to the mounted Lab's
  `validate_binding(requirement)`; it must return `None` or raise. The same
  checks apply to a seeded or recovered Controller's restored state and to the
  fully bound replacement before it becomes active. When recovering an assembly,
  the Host checks the resulting pair after all state adapters have run, including
  when a custom adapter restores the Lab last. A direct
  provider implements `validate_bindings(requirements)` to route these checks.
  The Policy receives neither a live Lab nor its private data. MetaRSI uses this
  port for matched training data, independent development cohorts and release thresholds.
- The Controller's read-only `lab_catalog.describe()` uses installed names.
  Bare Labs need no group attributes. A grouped provider can expose
  `catalog_metrics()` mapping aliases to primary metrics (or `None`); requesting
  a metric that was not declared fails at lookup.
- `Application` supplies the parent configuration for nested runs. When calling
  `modules.build(context)` directly with a Lab that needs `nested_run()`, supply
  `data_dir` and `AssemblyContext.host_config` with `application` (the parent's
  `ApplicationConfig.to_dict()`) and `registry_source` (the trusted assembly's
  `registry_source` for registered children). In a nested experiment, `labs=None`
  inherits the immediate parent's effective Lab bindings, including explicit
  overrides. A child cannot silently widen its parent's Controller catalog or
  action permissions. Custom permission callbacks cannot cross the worker boundary.
  Starting a child starts fresh unless the parent explicitly supplies a Controller
  version and approves its inherited evidence. Labs that do not start nested runs
  need no parent configuration.
- The run owns Lab execution, the model service, executor, budget, evidence and
  artifact storage. Each hosted Controller version owns its Policy, Harness, Memory,
  workspace and cleanup callbacks. Retiring a version closes its parts in dependency
  order; a failed owned close is retried without closing the run's services.
- A Memory can provide only `context(query)`. To add tools, implement
  `capabilities()` returning `Capability` objects (specification, validator and
  handler together). The endpoint installs them under the same action permissions
  as other tools. A custom action space needs `with_definitions(definitions)` only
  to accept extra tools. Policies read the authorized `window.capabilities`; neither
  a private parser nor a particular Harness implementation is required.
- `OperationSpec.input_schema` describes the arguments to the Controller; it is
  not enforced at dispatch. The Host checks only the transport envelope — an
  object with string keys — and forwards the call. The Lab validates its domain
  arguments before effects in `execute()`. Return a failed `OperationResult` or
  raise `LabToolError` for a correctable call; the registered Lab path journals
  the failed receipt under the original `call_id`. `OperationArgumentError` is
  a signal for pre-dispatch validators, not for `Lab.execute()`. Ordinary Lab
  exceptions remain errors and do not authorize replay.
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

For a trained model, the Host stages the Controller change. A training result must
name only `decision_maker` and match the version's model, whether it is registered
by the learner or supplied directly. Its artifact list is provenance, not a patch
for arbitrary components. The model service checks
whether its backend can address the exact `ModelRef` before deployment and verifies
the route again after the activator returns that ref. A backend with special routing
rules implements `validate_activation_route(base, model, *, prepared)`; return `None`
or raise without deploying or granting access. A plain model-ID-only endpoint cannot
serve two revisions under the same request name. Parent-delegated pins remain binding.

This protects the Controller action route provided by the framework; it is **not**
a process sandbox for host-supplied Python. The host-only sealed evaluation phase
is separate: a grouped Lab must supply both the sealed split and a
`final_evaluations()` plan. Every planned operation must be an evaluation
declared by the Lab or its profile; its spelling alone is not a declaration.
The built-in `ResearcherRunPolicy` runs that plan on
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
  Assembly marker schema 5 records the registry selection and its resolved public
  Lab bindings. Restore compares JSON configuration by type as well as value,
  including `false` versus `0`. It rejects a missing or older marker;
  use the original framework revision for an older run.
- With no `modules=`, `Application.resume` and `restore_snapshot` rebuild the
  registered selection. Supply trusted plugin paths in the original order via
  `plugins=`; a relocated plugin may retain its name and content. A portable
  snapshot never loads a plugin path or imports Python factory references supplied
  by the bundle. Direct Python compositions need their original `modules=`.
  Run state separately records an explicit initial Controller version and approved
  evidence so restore can recreate the right workspace before applying checkpoint
  state. A worker bound to a parent's budget or delegated model grant cannot be
  restored independently by turning it into a portable snapshot.
- Before publishing a candidate in a durable run, the Host verifies its bound
  Controller state can be captured and encoded. A rejected state does not replace
  or close the active Controller. A manual port may archive any state its own
  restore hook can read and the codec can encode, not just a dictionary. A
  Controller version ID cannot be reassigned, including the initial ID.
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
- Portable files are owned by the component state adapter. If state alone is not
  portable, implement both `export_snapshot_files(directory, state)` and
  `import_snapshot_files(directory, state)` (return `None`). The directory is
  private to that adapter. Import runs after identity validation and before
  state restore; neither hook should start work or copy a live workspace.
- A stable `call_id` ties together dispatch and receipt. Treat it as an opaque
  identifier, not a filename: stored receipt names are derived from a digest,
  and the receipt itself keeps the original ID. Mark an operation
  `idempotent=True` only if repeating the same call ID is safe. Even a matching
  terminal receipt does not prove the enclosing interrupted step committed.
  `RecoveryRequired` means stop and use verified reconciliation, not automatic
  replay or journal deletion. Local record deduplication is not an exactly-once
  guarantee for external observers.
- Registered Labs include applied receipts in portable snapshots, so moving to
  the new workspace neither grants nor revokes an explicit idempotent retry.
  A recorded async terminal result and its final progress are replayed without
  polling the backend again. A history-dependent profile rejection also retains
  its `call_id`; use a new ID for a new attempt after meeting the prerequisite.
  `once_per` reserves the key while a job is outstanding. Host final evaluation
  plans require globally distinct call IDs across Labs before the first effect.

## Declare evaluation evidence explicitly

Use `GoalRunPolicy(MetricGoal(...))` for a threshold on measurements. The Host
checks the delivered operation against the installed Lab declaration. A custom Lab
can expose `evaluation_operations()` and `candidate_identity_fields()` using local
operation names (full names for direct execution). For example, a Lab that varies
both encoder and index can return `{'score': ('encoder', 'index.sha256'),
'select': ('chosen_encoder', 'chosen_index')}`. Field positions must describe the
same target across operations. Dot-separated paths address object keys, not code.
Single nonempty string IDs pass through; tuples, other JSON values and strings in
the reserved `candidate-json-sha256:` namespace are content-addressed. Never use
a mutable reference as an identity. Installed profiles can supply these fields;
the actual Lab may explicitly specialize them for its experiment.

`min_candidates` and `measured_selection` require a grouped Lab (or the Registry)
and bindings for all evaluations and the measured selection operation. A missing
part is rejected before dispatch. Direct execution supports stateless scoring but
cannot promise Host-maintained candidate history.
The same identity binds successful measurement, distinct counts, selection,
sealed preflight and RunPolicy. A conflicting receipt cannot credit the request.
A Lab without these invariants may omit candidate fields, but a receipt from that
operation will not satisfy `MetricGoal`, even if it supplies `candidate_id`.

A qualifying receipt needs a bound candidate, `protocol_ref`, exact `input_refs`,
and finite numeric `quality_metrics`. The Host can fill missing facts on delivery
without rewriting the Lab journal. It never takes a scorer from the Controller's
request, or an identity from an undeclared argument like `model`. Benchmark metadata on an
undeclared operation (for example, the reusable Math Lab on its own) is not goal
evidence. Simulated receipts are excluded by default. When supplied, `valid`
and `scored` must be booleans; an invalid or unscored receipt cannot qualify.
Use a distinct protocol reference for each dataset or pin `MetricGoal.input_refs`.
Request `input_refs` constrain the expected measurement; they are not proof of
what ran. An opaque cohort ID must come from the Lab's terminal receipt. If the
Lab returns only a score, the Host can derive exact refs only from its known
selectors (`split`, `task_ids`, `indices`, `dataset`): `split=dev, task_ids=[q1]`
is not the entire dev split. Conflicting or incomplete identities cannot satisfy
the goal. This applies equally to direct Labs, grouped Labs, jobs and replay; the
underlying Lab journal is not relabeled. A threshold does not by itself apply the
candidate or replace a sealed evaluation.

Sealed final reports and their journal entries expose only finite numeric metric
maps (`quality_metrics`, `training_metrics`, `execution_metrics`, `metrics`),
boolean `valid`/`scored` flags and a nonnegative integer `count`. A malformed
aggregate cannot yield a successful sealed measurement. Only correctly typed
values are published, even on failure; native data, error text and per-item
results remain in the restricted `sealed-evaluation` artifact.

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
