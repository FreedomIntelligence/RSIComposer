# AIDE: native research-agent search over LLM inference programs

This recipe registers the **actual upstream AIDE Agent**, pinned to
`ad73d227af481bea3de06f884f41b2993dd6453a`, as controller `aide` and its
execution environment as Lab `aide_program`. The task is a **GSM8K LLM
inference-program adaptation**, not a reproduction of AIDE's original tabular
benchmarks or their reported gains. Model weights stay fixed.

## What runs through the framework

1. Native `Agent.step` selects a parent with its original `search_policy`, then
   calls its original draft, improve or debug generation method.
2. At the native execution callback, the worker yields the immutable candidate
   reference to the registered RSISys controller.
3. The controller invokes the registered Lab. The Lab executes the candidate,
   scores its predictions and returns an `ExecutionResult` in a typed receipt.
4. The controller delivers that result to native `parse_exec_result`; upstream
   review, Journal updates and the next tree decision continue.
5. After the configured search budget, native `Journal.get_best_node()` selects
   the final program. The Lab evaluates the first valid and selected programs
   on a separate heldout split. These scores do not influence selection.

The integration does not run an entire external CLI as one opaque Lab operation.
Neither the host nor the maintenance agent edits proposed strategies, selects a
winner, supplies pseudo-labels or changes the metric based on observed results.
The upstream reviewer still interprets the terminal output; its reported metric
and the Lab's measured metric are separate evidence, not guaranteed identical.

## Dependencies and setup

Linux, Python 3.12+, working bubblewrap/user namespaces, a local fixed-model
OpenAI-compatible service, and a separately installed AIDE runtime are required.
The core package has no additional mandatory dependencies.

```sh
git clone https://github.com/WecoAI/aideml /absolute/path/to/aideml
git -C /absolute/path/to/aideml checkout ad73d227af481bea3de06f884f41b2993dd6453a
python3.12 -m venv /absolute/path/to/aide-env
/absolute/path/to/aide-env/bin/python -m pip install -r experiments/aide/requirements.txt
```

The checkout must have the exact revision and no tracked edits. We import the
native Agent, backend, review schema, Journal and selection implementation from
that checkout. See [upstream attribution](UPSTREAM_NOTICE.txt).
The optional runtime requirements are the tested Agent import subset; the broad
upstream scientific benchmark environment is not needed by this standard-library
inference task. These pins are independent of the vLLM serving environment.

Prepare existing disjoint GSM8K JSONL files, each row containing `id`, `prompt`
and `answer` (the standard `####` final numeric answer). The preparation script
freezes the first 8 public examples, 32 validation questions and 64 heldout
questions before any new evaluation. It rejects overlapping IDs/text and refuses
to overwrite a data directory.

```sh
python experiments/aide/prepare.py \
  --train /absolute/path/to/few-shot.jsonl \
  --validation /absolute/path/to/dev.jsonl \
  --heldout /absolute/path/to/test.jsonl \
  --output runs/aide-data
```

`public/` contains only labeled training examples and unlabeled validation
questions. Keep the two private scoring JSONLs outside the public input directory.
No model weights or dataset records are distributed with this recipe.

For the tested four-replica Qwen deployment, use a separate vLLM environment:

```sh
python experiments/aide/serve.py \
  --python /absolute/path/to/vllm-env/bin/python \
  --model /absolute/path/to/local-qwen-model \
  --gpus 4,5,6,7 --log-dir runs/aide-serving
```

The service enables tool choice with `qwen3_xml` parsing because native AIDE submits
structured reviews with forced tool calls. It uses `qwen3` reasoning parsing, 24,576 context
tokens and four independent replicas. Adjust model-specific parsers for other
model families. Local-provider compatibility converts native system-only prompts
to a user turn with exactly the same text and disables Qwen thinking by default;
both differences are explicit, rather than silently claiming identical API
behavior to the upstream paper's models.

Fill the checkout, worker Python, model endpoint and data paths in
[gsm8k.json](gsm8k.json), then launch from the repository root:

```sh
rsisys run --config experiments/aide/gsm8k.json
```

Stop the owned serving launcher after the run to release its replicas. AIDE's
planning/review calls are sequential; the Lab can parallelize candidate inference
over all replicas. High peak memory allocation does not mean the GPUs are busy
during every stage. The four-card split supports concurrent independent methods;
we do not add redundant inference merely to inflate utilization.

## Execution and scoring boundary

Generated programs execute in bubblewrap with a private process/network
namespace, installed Python runtime mounts, their own writable workspace and one
explicit Unix broker socket. They receive no direct GPU devices, host TCP/network,
API credentials, evaluator code, private labels or other run state. The broker
only permits fixed-model chat requests with bounded output tokens and call count.
The default limits are four calls per question and 1,536 output tokens per call.
The supplied candidate client is [aide_client.py](../../src/rsisys/labs/aide_client.py).

Candidates write `working/submission.csv` (`id,answer`). The Lab checks complete,
unique IDs and independently scores strict numeric final answers. Each candidate
gets a fresh workspace. No candidate receives heldout questions during search;
final programs receive unlabeled heldout questions only after selection. The
host appends its measured validation accuracy to native terminal feedback.
This is still development-set search and needs final heldout reporting.

There is no training in this recipe. Snapshot recovery of the running native
process is not supported; restarting requires a fresh run directory. GPU services
are owned by their serving launcher, not automatically leased by registration.

## Verification

The standard offline tests cover exact receipt matching, candidate identity,
private scoring, input immutability, inference budgets and worker cleanup:

```sh
python -m unittest tests.registration.test_aide -v
```

Run the native protocol test on Linux with the optional environment:

```sh
AIDE_UPSTREAM_ROOT=/absolute/path/to/aideml \
AIDE_WORKER_PYTHON=/absolute/path/to/aide-env/bin/python \
  python -m unittest tests.registration.test_aide -v
```

That additional test executes the real upstream Agent against a scripted HTTP
fixture and real bubblewrap, verifies draft → improve → debug, final native
selection and that gold-label files are absent. It measures interface behavior,
not LLM research ability. Actual GPU evidence and its limits are recorded in
[RESULTS.md](RESULTS.md).
