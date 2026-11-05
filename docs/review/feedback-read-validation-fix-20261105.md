# Feedback read argument rejection

The lion overnight run `medical-lion-night-04-20261104T161442Z` stopped after
1,886.749 seconds and 419 accounted model calls. Its last response selected
`feedback_learning_read` with `view=jobs` and `target=context_memory`. `target`
filters proposals, so that combination is invalid. The validator raised an
ordinary `ValueError`, which was recorded and then escalated out of decision.
The controller and owned services exited; supervisor cleanup reported no errors.

`OperationArgumentError` is an optional validator signal for arguments the model
can correct. The default shared executor converts this explicit rejection into
a failed operation result before invoking any handler. The original call and
error remain in history and reach the next decision. There is no automatic
retry or replay. An ordinary validation exception retains its host semantics.
The same exception raised after validation, inside a tool, also remains a host
fault. Permission, identity, accounting and infrastructure errors must not be
classified as argument errors.

The built-in feedback read validator uses this signal for its static argument
checks. Its tool adapter also returns a failed receipt for direct invalid reads
without touching storage. The tool schema now explains that `target` requires
`view=proposals` and must be omitted for job reads.

The new application regression reproduced the original fatal exception before
the fix. After the fix it checks that the invalid read returns failed feedback,
the next model request includes the error, a corrected read succeeds, and no
Lab process is submitted. Boundary tests retain host errors for permissions,
identity/storage failures and exceptions from an executor. Local targeted
validation: 82 tests passed, without model inference. The deployment control
directory records the separate full lion regression and frozen source manifest.

This repair does not validate model-written summary prose, make malformed
LangMem patches succeed, or establish training/evaluation/adoption correctness.
The prior run's semantic audit and bounded feedback failures remain evidence.

## Archive cursor rejection (2026-11-06)

The single-meta six-hour run stopped after 552 calls when `research_archive`
combined a cursor from the full history with a different, nonexistent record ID.
The stale cursor raised `ValueError` inside the handler and escaped the decision
loop, despite being a correctable read request with no external effect. All
owned processes exited and both cleanup receipts confirmed teardown.

Archive cursor syntax, selection mismatch and changed-record checks now return
a native failed receipt with an explicit `read_path` whose cursor is null. They
never silently restart or mix pages from different selections. Original records
stay unchanged. The next decision can inspect the error and choose a new read.
Provider/storage exceptions still propagate as host failures. Regressions cover
the original fatal path through the decision harness, malformed/stale cursors,
exact pagination after rejection, unchanged originals and real provider errors.
