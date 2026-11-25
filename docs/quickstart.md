# Quickstart

## Install

RSIComposer requires Python 3.12 or 3.13 on Linux or macOS. The core package has no
third-party dependencies.

```sh
git clone https://github.com/FreedomIntelligence/RSIComposer.git
cd RSIComposer
python -m pip install -e .
rsicomposer catalog        # lists the registered methods, Controllers, Labs, and components
```

Most methods need more than the core. Install the
[optional extras](../pyproject.toml) that the method uses, such as
`pip install -e '.[math]'` or `pip install -e '.[coding]'`. Methods that run an
upstream project in a separate environment list its packages in
[`requirements/`](../requirements/).

If you are upgrading from an earlier checkout, install this repository in a fresh
environment and use `rsicomposer` for commands and `rsicomposer.api` for
plugin imports. Keep the original checkout and environment for resuming existing
checkpoints or restoring snapshots made before the package migration.

## Verify a checkout

Run the core integration and fault-injection checks without model credentials, GPUs, or optional extras:

```sh
PYTHONPATH=src python -m unittest discover -s verification -v
```

CI also installs the wheel into a fresh environment and runs the ReST^EM CPU fixture through
the installed console script from a temporary working directory. It pauses after four decisions
and resumes the durable workspace, checking packaged resources and real CLI behavior rather than
only imports from the source tree.

## Run without a GPU

This run executes the ReST^EM loop on a few fixture problems: evaluate, then
generate, filter, train, and evaluate for two rounds. The fixture engine
simulates the model, so no GPU or model download is needed, and no real
training happens. Use it to check the installation and to see what a run
produces.

```sh
rsicomposer run --config src/rsicomposer/algorithms/rest_em/configs/cpu/rest_em.json \
  --data-dir runs/quickstart
```

The command prints `result.json` when it finishes. Use a new `--data-dir` for
each run.

## Read the results

`result.json` stores every receipt in `events`. This prints each operation and
its metrics:

```sh
python - <<'EOF'
import json
result = json.load(open('runs/quickstart/result.json'))
print(result['status'], result['stop_reason'])
for event in result['events']:
    receipt = event.get('operation_result') or {}
    print(receipt.get('summary'), receipt.get('data', {}).get('quality_metrics', ''))
EOF
```

```text
finished researcher_finished
evaluate completed {'accuracy': 1.0, 'eval_samples': 1}
generate completed
filter completed
train completed
evaluate completed {'accuracy': 1.0, 'eval_samples': 1}
…
```

The [Concepts](concepts.md#what-a-run-writes) page describes the other files in
the run directory.

## Run with a GPU model

The method template searches prompt and few-shot settings for a model behind an
OpenAI-compatible Chat Completions endpoint, such as vLLM.

1. [Start or configure the endpoint](../src/rsicomposer/labs/_template/README.md#connect-a-gpu-model).
2. Run the template from the repository root:

   ```sh
   rsicomposer run --plugin src/rsicomposer/labs/_template --plugin src/rsicomposer/algorithms/_template \
     --config src/rsicomposer/algorithms/_template/configs/run.json \
     --data-dir runs/gpu-quickstart
   ```

The two `--plugin` options load the template Lab and the template method, which
are not built in. `controller_report` in `result.json` shows each candidate's
score and the selected setting.

## Watch runs in RSIMonitor

RSIMonitor is an optional read-only dashboard for decisions, metrics, samples,
and logs.

```sh
python -m pip install ./src/rsimonitor
rsimonitor --runs-dir runs    # then open http://127.0.0.1:8765/
```

## Continue an interrupted run

Start the run with `--durable` to save a checkpoint after every step, then
continue it in the same directory:

```sh
rsicomposer run --durable --config path/to/run.json --data-dir runs/my-run
rsicomposer resume --data-dir runs/my-run
```

Durable runs require the selected components to support checkpoint recovery.
The CLI rejects `--durable` at startup if the assembly does not supply that
recovery contract. `snapshot_supported` in `result.json` is a separate check:
it reports whether the current components support portable snapshots. If a
snapshot was saved, `snapshot` gives its path. Restore it into a new directory
with `rsicomposer resume --snapshot <snapshot-dir> --data-dir <new-run-dir>`.

## Next steps

- [Concepts](concepts.md): the terms used in the rest of the docs.
- [Components](components.md): what ships with RSIComposer.
- [Compose a system](compose.md): combine existing parts or load your own.
