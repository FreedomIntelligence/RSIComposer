# M02 / M13 implementation audit — 2026-11-04

Scope: current working tree of RSISys, including the existing feedback pipeline,
LangMem, M03 prompt and run-control changes. This audit owns runtime runner/run
policy/recovery, snapshot, experience, and artifact/filesystem persistence only.
It does not establish model quality, clinical semantics, real training, remote
recovery, or GPU/NPU behavior. All verification used CPU fixtures, temporary local
workspaces and, in existing tests, loopback HTTP fixtures/local child processes.
No packages were installed, no experiment/training or external model call was
launched, and no commit, push, service or automation change was made.

Four confirmed bug groups were reproduced and fixed:

| ID | Reproduction and impact | Fix | Dedicated regression evidence |
| --- | --- | --- | --- |
| RR-01 | An explicit recovery contract starts with valid state, then exports `float('nan')`. `mutate`'s `encode` preflight accepts it; the callback executes, then JSON checkpoint serialization fails. A known invalid state therefore crosses the external-effect boundary. | Serialize the encoded preflight through `json_bytes` before writing intent or invoking the callback. The rejection leaves the prior journal and intervention unchanged. | `test_recovery_rejects_nonfinite_contract_before_effect_or_intent` asserts no callback, no intent change, no intervention. |
| RR-02 | Register a context processor on the default harness, capture a snapshot and restore it. The original capture succeeds, but the restored processor list is empty. Feedback listeners, receipt listeners, cleanup callbacks and proposal resolver have the same unrecorded configuration boundary. | Built-in snapshot/recovery rejects these hooks unless the existing explicit decision recovery extension owns their reconstruction. No callable serialization or architecture substitution was added. | `test_snapshot_and_builtin_recovery_reject_uncontracted_decision_hooks` exercises all five hook locations and verifies rejection before snapshot creation. |
| RR-03 | A snapshot with a consistent file digest but invalid compactor settings publishes its target before validation fails. Failure while attaching restored state also leaves an unreachable live Application/lock. Corrupt SQLite initialization can strand a target too. | Validate agenda/compactor before publication; close an initialized Application on attachment failure; remove this operation's failed target only after nonblocking acquisition of its retained staging `.lock` inode and matching directory/lock identities. Preserve the target if a competing owner holds it or either identity changed; retain the original exception with cleanup notes. | Invalid-compactor, corrupt-database and postconstruction failure tests; real `flock` contention fixture using separate file descriptions; replaced directory and replaced lock inode fixtures. All paths operate only in TemporaryDirectory. Successful restore also retains its live Application lock. |
| RR-04 | In-memory Experience finds `Straße` using `STRASSE` and Greek uppercase using lowercase. Reopened SQLite misses those records because SQLite `lower()` folds only ASCII. With `limit=1`, it can return the wrong newest matching record. | Register a deterministic connection-local Python `casefold` SQL function; filter with it before order/limit. No schema migration or package is required. | `test_experience_unicode_casefold_matches_reopened_sqlite_and_recent_limit` compares both stores across four queries and two limits after reopen. |

Before fixes, the initial 20-test dedicated suite failed in 14 assertions/subtests
covering those bug groups, including leaked-lock ResourceWarnings. The final
dedicated file contains 27 test methods, including additional negative boundary
and publication/locking cases.

Actual source review and evidence against the module checklist:

| Checklist / reviewed source | Boundary or negative scenario reviewed | Evidence / disposition |
| --- | --- | --- |
| M02.1 `runtime/runner.py` | Receive before acknowledge; preserve exact outbox across either response loss; mismatched directive receipts; endpoint discovery and ownership; parallel slot retirement after acknowledged delivery/view failure. | New receive/ack and wrong-owner tests; `test_coordination`, `test_operation_retry`, `test_parallel_experiments`, `test_durable_recovery`. No additional Runner bug confirmed. |
| M02.2 `runner.py`, `run_policy.py` | Default researcher finish, host timed restart/cooldown, legacy per-call round/decision limits, optional run-control capability checks, pending external research jobs. | `test_run_policy`, `test_research_cycles`, `test_qwen_run_policy`. Runner never sleeps; the caller drives later ticks. Pending research jobs require external handoff/cancellation support; no automatic replay/polling was introduced. |
| M02.3 `run_policy.py`; read-only inspection of medical `DecisionBackend`, `ExperimentRunPolicy`, `RecipeLearner` | Deadline precedence, invalid numeric budgets, stale/simulated metric goals, cumulative model-call and training reservations, format-recovery attempts charged through host admission. | `test_run_policy`, `test_durable_recovery`, `test_medical_response_recovery`, `test_medical_experiment`. Call/output/training admission remains with the host adapters; the generic Runner does not invent a second accounting system. |
| M02.4 `runner.py`; read-only lifecycle/process call sites | UNKNOWN is pending, cancel request is not confirmation, late success retains actual status, cleanup deadline reports unresolved work, KeyboardInterrupt retains ownership and original error. | Five new lifecycle tests plus existing cancel/review/parallel/recovery cases. Cleanup callback behavior in engine and native process termination belongs to other audit owners. |
| M13.1 `experience/records.py`, `provenance.py`, `storage.py`, `retrieval.py`, `__init__.py`; read-only MemoryBuffer/ResearchAgenda and feedback ownership call sites | Experience vs loop-local history/memory vs model notes vs artifacts; source scope is not access control; source references do not resolve themselves; run identity is enforced at BasicExperienceSystem. | New conflict/detachment/scope/run-id test; `test_experience_boundary`, `test_memory`, `test_research_agenda`, `test_run_provenance`. LineageLink is only a data relation; automatic provenance-graph validation remains unimplemented. |
| M13.2 `experience/sqlite.py`, storage/retrieval; `runtime/artifacts.py`, `files.py` | Identical-ID no-op vs conflicting content, detached nested data, latest filtered order, Unicode, unsupported SQLite schema, foreign/forged artifact refs, corrupt bytes, failed metadata publication/rename, temporary cleanup, symlink refusal. | New persistence cases plus `test_foundation`, `test_external_artifacts`, `test_operation_retry`, `test_experience_boundary`. RR-04 fixed; artifact staging and same-directory atomic writes passed targeted fault fixtures. Limits bound record count, not bytes or retrieval cost. |
| M13.3 `runtime/recovery.py` | Pre-effect serializability/intent, checkpoint as commit, journal generation mismatch, uncertain delivery and submission, interruption, evidence-backed failed/successful reconcile, downtime and first-step budget start, custom contract identities, mock job reattachment without submit replay. | RR-01/RR-02; new intent-write, journal-corruption and reconciliation tests; `test_durable_recovery`. Recovery intentionally blocks after uncertain effects. It neither rolls back independently persisted evidence nor automatically retries an interrupted external operation. |
| M13.4 `snapshot/local.py`, `manifest.py`, `codec.py` | Inventory/digest/source/Python checks; safe codec allowlist; preserved adaptive compactor targets/increments/summary/fingerprints, context limits and model bindings; new-target-only restore; invalid configuration and publication/lock races. | RR-02/RR-03; new snapshot/codec/fault tests; `test_foundation`, `test_harness_composition`, `test_system_directives`, `test_durable_recovery`, `test_run_provenance`. Pre-existing M03 prompt rejection and feedback proposal codec compatibility edits were preserved. |
| M13.4 `snapshot/capture.py`, `restore.py`, `storage.py`, `__init__.py` | Abstract protocol vs local implementation; placeholders must fail explicitly; invalid snapshot ID traversal. | New placeholder/store-boundary test. Generic capture/restore adapters are unimplemented features, not silent-success bugs. |

Portable snapshots still explicitly reject a live feedback service, custom
uncontracted hooks/components, concurrent runner, nondefault run policy/budget,
pending delivery/jobs/session and unsupported policy configuration. New tests
verify feedback-service rejection and preservation of the current full adaptive
compactor state; this audit does not claim feedback/LangMem recovery support was
implemented. An explicit extension/contract remains responsible for custom state.
External model references are not saved weights; random state is not captured.
Source revision and Python compatibility checks remain strict.

The restore locking fix protects cooperating Application owners. It does not
claim a transaction across hostile filesystem replacement, an OS crash between
rename and Application construction, or multiple independently writing stores.
On contention or replaced identity, the failed target is deliberately retained
for its owner/operator instead of deleted. SQLite/artifact/history persistence
remains separate from the recovery checkpoint; interrupted effects require
evidence-backed reconciliation. These are explicit design boundaries.

Files changed by this audit:

- `src/rsisys/runtime/recovery.py`
- `src/rsisys/snapshot/local.py`
- `src/rsisys/experience/sqlite.py`
- `tests/regressions/test_audit_runtime_recovery.py`
- `docs/review/audit-20261104/runtime-recovery.md`

`runtime/runner.py` and `snapshot/codec.py` already had uncommitted edits at the
start; this audit did not modify them. Parent-owned engine/state/context and
`tests/decision/test_m03_decision_policy.py` were not edited.

Validation was performed with `.local/langmem-validation/bin/python` and
`PYTHONPATH=src` (or `src:tests` for existing tests' sibling-fixture imports).
The initial targeted baseline passed 155 tests; an intermediate wider run passed
267 tests. An expanded 323-test run then produced one failure and eight errors
through `local_process._watch/_kill` (`os.killpg` PermissionError and missing
watcher outcome), including subprocess-based snapshot/CLI cases. That module was
being audited concurrently and was not changed here. This failed run is retained
as negative evidence; earlier passing runs do not establish native process
cleanup correctness. Final run results are recorded below.

Final validation after the lock/inode protection was complete:

- Dedicated audit file: **27 tests passed** (0.339 s).
- Final focused integration run: **283 tests passed** (33.456 s), including
  recovery, snapshot boundaries, lifecycle/parallel coordination, Experience,
  medical host admission and response-recovery fixtures.
- `git diff --check` on the changed files: passed.
- This is a focused audit result, not a claim that the entire repository suite
  or native process cleanup has passed.

Reproduction commands (from the RSISys directory):

```sh
PYTHONPATH=src .local/langmem-validation/bin/python -m unittest tests.regressions.test_audit_runtime_recovery

PYTHONPATH=src:tests .local/langmem-validation/bin/python -m unittest \
  tests.regressions.test_audit_runtime_recovery tests.runtime.test_run_policy tests.decision.test_research_cycles \
  tests.runtime.test_coordination tests.runtime.test_parallel_experiments tests.runtime.test_operation_retry \
  tests.runtime.test_durable_recovery tests.runtime.test_exploration_recovery tests.memory.test_experience_boundary \
  tests.runtime.test_external_artifacts tests.memory.test_memory tests.runtime.test_run_provenance \
  tests.decision.test_action_permissions tests.runtime.test_system_directives tests.decision.test_research_agenda \
  tests.experiments.medical.test_medical_response_recovery tests.experiments.medical.test_medical_experiment
```
