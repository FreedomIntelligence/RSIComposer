# Coding Meta-Agent method

This method optimizes the instruction of a fixed coding agent. Its controller
chooses proposals and final selection. Its private `CodingSearchLab` owns the
candidate registry, proposal budget and sealed heldout policy, delegating task
execution to the reusable [CodingLab environment](../../labs/coding/README.md).
It is an adaptation of Meta-Agent harness search, not a weight-training algorithm
or a reproduction of RRSI. Other RSI methods can use the environment independently.

## Installation

Use Linux with working Docker/Compose, enough disk space for benchmark images,
and OpenAI-compatible local model endpoints supporting `response_format=json_object`.
Recent benchmark Dockerfiles also require working BuildKit/Buildx. Install the optional dependencies:

```bash
python3.12 -m venv .venv-coding
.venv-coding/bin/pip install -e '.[coding]'
docker run --rm hello-world
```

The core package, public API and catalog still import without Harbor or OpenAI.
Only the trial worker imports those optional dependencies. Models run outside
containers; the agent can execute commands only in the benchmark environment.
Benchmark graders and reference solutions are never sent to the coding model.

## Experiment inputs

Set `framework.method.name` to `coding_meta_agent`. Put Controller settings in
`framework.method.config.controller` and select benchmarks in
`framework.method.config.labs.coding.benchmarks`. The Lab combines the selected
manifests into disjoint `search` and `heldout` splits. Each benchmark carries its own
budgets (`max_steps`, `max_tokens`, `total_tokens`, `agent_timeout`, `verifier_timeout`,
`thinking`) next to its `manifest`; the Lab sets none, so a benchmark is measured under the
same protocol however it is mounted. Mounting several requires `primary_metric`:
`{ benchmark = "<name>" }` or `{ aggregate = "benchmark-mean" }`. Each manifest row contains:

```json
{
  "id": "task-name",
  "benchmark": "terminal-bench-2.1",
  "path": "/absolute/path/to/task",
  "files": {
    "task.toml": "sha256-of-the-file",
    "instruction.md": "sha256-of-the-file"
  }
}
```

Include hashes for **all** task files when preparing real runs, pin the upstream
revision, and record container image digests. Select tasks before inspecting model
scores. The same `(benchmark, id)` cannot occur twice or in both splits. A subset
is a development experiment; its score is not a full benchmark leaderboard score.
Terminal-Bench 2.1 has shorter tasks suited to rapid iterations; Terminal-Bench
4.1 uses a separate verifier and much longer task budgets. Keep their results
separate and document any shorter host budgets.

Before scoring models, run official reference-solution calibration in an isolated
host-only job with the intended images, network, certificates and verifier budget.
A zero reward can reflect expired CA roots, unavailable external sites or rate
limits as well as model errors. Retain the original logs and mark unresolved
environment limitations. If an environment repair is needed, version and hash it
as a new task variant before freezing the experiment; do not change tests, bypass
TLS verification, remove failures after seeing scores, or send reference solutions
to either model. Keep calibration rewards separate from model measurements.

The coding Lab config requires `experiment_dir`, `benchmarks`, `python` (the absolute
virtualenv interpreter path), `endpoints` and explicit `model`. Optional Lab settings are
`workers`, `seed` (73), `primary_metric`, `min_candidates` (default 3) and `max_candidates` (6).
Each benchmark entry takes `max_steps` (48), `max_tokens` (4096 per call), `total_tokens`
(98304 output tokens per task), `agent_timeout` (1800 seconds), `verifier_timeout`
(1800-second cap) and `thinking` (false). Worker budgets cannot be changed by a candidate. The worker uses constrained JSON
responses. Each command runs under a container-side 120-second `timeout`, with
a five-second kill grace period. A command timeout returns exit code 124 as
feedback so the agent can revise its approach; it does not abort the entire trial.
The host execution timeout includes a separate cleanup margin.
Worker history keeps the original system/task prefix exactly once and bounds the
complete retained content to 120000 characters and 48000 UTF-8 bytes, including
the prefix and omission marker. The byte limit conservatively bounds the deployed
byte-level tokenizer within its 64K context while reserving output space; it also
applies to binary-looking output and Unicode, even after an oversized early response.

Use the real virtualenv interpreter path; resolving its symlink with
`Path.resolve()` can accidentally select the system interpreter.

## Running the method

Start the configured controller model service and worker model services first. Use the
[TOML configuration template](configs/coding.toml). Keep only the benchmarks you want,
replace the absolute paths and model identifiers, then run:

```bash
PYTHONPATH=src .venv-coding/bin/python -m rsisys run \
  --config /absolute/path/to/coding.toml \
  --data-dir /absolute/path/to/new-controller-attempt
```

The controller owns the research sequence:

1. Inspect persisted Lab state and evaluate the baseline.
2. Read search trajectories, propose a general instruction and hypothesis, and
   evaluate that candidate. Repeat for at least the configured number of distinct
   candidates. The model chooses the experiments and final candidate.
3. Call `select(candidate_id)` for a measured candidate, then finish.
4. The host seals search before measuring baseline and selected on heldout with
   the same budgets. These receipts never go back to the Controller or Memory.
   Improvement is not guaranteed.

Candidates cannot contain benchmark task identifiers. The baseline cannot count
as a proposed round. A command's zero exit code or an agent's claim of completion
is not a solved task: only the official verifier supplies the score.

## Results and recovery

The experiment directory contains immutable `candidates/*.json`, `binding.json`,
method summaries under `evaluations/`, shared per-task receipts and original Harbor
trial artifacts under `environment/measurements/`,
`selection.json`, and host-owned sealed receipts in the controller run. Full agent traces retain model
responses, command observations and token counts. Controller feedback contains
aggregate scores and compact per-task search status. The selection receipt confirms only the choice. Host sealed receipts expose
aggregate metrics; full per-task records stay in sealed artifacts.
Only search traces are exposed
to the meta-agent. A trace is paginated; an excerpt is not the complete record.

The binding includes the manifest, endpoint/budget settings and executable worker
source hashes (including history assembly). Changing them requires a new experiment directory. Completed task
receipts are reused when a fresh controller attaches the same experiment. Missing
or infrastructure-failed grades are retried twice and remain incomplete if both
attempts fail; they are not silently dropped from the score denominator. A total
agent time-budget exhaustion with an official valid reward is a measured result,
including a genuine zero; it is not retried as infrastructure. The
controller attempt uses a **new** RSISys data directory. This reattachment is
explicit Lab recovery, not arbitrary controller or framework snapshot recovery.
A filesystem lock prevents concurrent controllers mutating one experiment.

Run with `--durable` and use `rsisys resume --data-dir <controller-run>` after
an interruption. The host restores its selected candidate, sealed plan, stable
call IDs and cursor, and replays completed receipts before continuing missing
heldout work. Search and selection changes stay closed after sealing. Preserve unfinished trial directories when investigating
external side effects; do not replay a live task merely because its receipt has
not appeared yet. Each submission has an `inflight` record written before spawn.
Reattachment matches the unique trial name and receipt path against the live
Linux process command line. A leftover container without a live worker is marked
uncertain and requires host reconciliation before another submission.

## Eight-GPU deployment

A practical allocation for the requested models is one GPU for a Qwen3.8-27B
Meta Agent and seven independent one-GPU Qwen3.5-9B services for Lab agents.
Fourteen concurrent tasks distribute two tasks per worker service by task index.
Model weights stay fixed across baseline and candidates. Endpoint identity,
GPU mapping, model revisions, inference settings, source/config hashes, task
manifest, image digests, watchdog events and exact resume commands belong in the
run evidence, not in the reusable library defaults.

## Environment separation and migration

`register`, `select`, the candidate budget and the equal-family search objective
belong to this method. Its `artifact-search/v1` declaration is the environment's primary
metric: `<benchmark>/task_reward` for one benchmark, or `benchmark_reward` with scorer
`harbor-official-reward/benchmark-mean/v1` when the config declares that aggregate. Baseline,
measured selection, minimum candidates and split access are host invariants. The environment
declares `coding-task/v1` with per-benchmark task-mean rewards; it measures immutable artifacts,
named task subsets and repeat trials. It does not choose candidates.

The modular implementation uses a new policy/environment binding. Start a new
experiment directory; archived runs and their executable snapshots remain intact.
The optional host-only post-selection entry point moved to:

```bash
python -m rsisys.algorithms.coding_meta_agent.frozen --help
```

Pass `--run-dir <completed-durable-controller-run>` to the post-selection runner.
It requires successful host heldout receipts for the baseline and selected
candidate; an ordinary selection file is insufficient. The former scene-level
`frozen` entry point reports this migration explicitly.
GPU counts, worker counts and model choices in the deployment example above are
examples, not requirements of the environment or the method.
