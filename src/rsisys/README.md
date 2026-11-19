# Using and extending RSISys

A method binds a Controller to one or more named Labs. The Controller decides
what to try; the Labs execute operations and report results. Install RSISys via
the [Quickstart](../../README.md#getting-started) and run commands from the
repository root.

| I want to… | Start here |
| --- | --- |
| Run an existing method with the paper's setup | [Reproduce an existing method](#reproduce-an-existing-method) |
| Write my own RSI algorithm or customize a Controller | [Define a method](#define-a-method) |
| Reuse a Lab, attach my data, or write a new Lab | [Define a Lab or connect data](#define-a-lab-or-connect-data) |

## Reproduce an existing method

1. Run `rsisys catalog` to find its registered name. Copy a preset from the
   method's directory under [`algorithms/`](algorithms/).
2. Pin the model revision, upstream code, dataset splits, scorer, seed, and
   budget. Configure inference endpoints or local model paths, and supply the
   training environment and placement if the method updates weights.
3. Run the experiment, inspect `result.json`, and compare the baseline and
   selected candidate with the same protocol. Keep a final held-out split
   separate from candidate selection.

```sh
rsisys run --config path/to/experiment.json --data-dir runs/my-experiment
```

For methods that support recovery, start the run with `--durable` and continue
it with `rsisys resume --data-dir runs/my-experiment`.

| Example | GPU workload |
| --- | --- |
| [GEPA-Merge](algorithms/gepa/README.md) | Inference on a small GSM8K cohort with the pinned upstream optimizer. The recipe specifies data format, model setup, and paired merge/no-merge runs. |
| [ReST^EM](algorithms/rest_em/README.md) | GSM8K sampling, evaluation, and fine-tuning. Choose the single-GPU or eight-GPU preset. |
| [Coding Meta-Agent](algorithms/coding_meta_agent/README.md) | Inference for instruction search against the Coding Lab. The benchmark tasks execute in Harbor containers. |

Running a preset is a starting point. Claiming paper reproduction requires
matching the paper's protocol and measuring the corresponding benchmark.

## Define a method

Copy the [method template](algorithms/_template/README.md). It compares three
prompt and few-shot settings on the same model and development benchmark, then
selects the best measured candidate. [Connect a GPU model](labs/_template/README.md#connect-a-gpu-model)
and run:

```sh
rsisys run --plugin src/rsisys/labs/_template --plugin src/rsisys/algorithms/_template \
  --config src/rsisys/algorithms/_template/configs/run.json \
  --data-dir runs/gpu-method-template
```

Change the Policy in [`controller.py`](algorithms/_template/controller.py),
bind the Labs in [`register()`](algorithms/_template/__init__.py), and select the
method in `framework.method`. The [method template](algorithms/_template/README.md#customize-the-controller)
shows how Policy, Harness, and Memory fit together. Import the framework
interfaces from [`rsisys.api`](api.py).

## Define a Lab or connect data

To reuse a Lab, keep its operation contract and configure its benchmark,
datasets, and scorer. The Lab owns data loading, execution, and scoring:

| Lab | How to connect its inputs |
| --- | --- |
| [Coding Lab](labs/scenes/coding/README.md#connect-a-benchmark) | Bind a frozen coding task manifest and configure its splits. |
| [ReST^EM's math Lab](algorithms/rest_em/README.md#connect-training-and-evaluation-data) | Bind separate training and development JSONL files and select the verifier. |
| [Math Lab](labs/scenes/math/README.md) | Shared scene placeholder with verifier helpers. It does not load data or run a benchmark yet. |
| [Self-Reflection Lab](labs/scenes/self_reflection/lab.py) | Runs candidate Controllers against child Labs, which supply their own benchmarks and data. |

To write a Lab, copy the [Lab template](labs/_template/README.md). It uses
`train.jsonl` for examples and `dev.jsonl` for evaluation. This run selects the
Controller and Lab separately via `framework.controller` and `framework.labs`:

```sh
rsisys run --plugin src/rsisys/labs/_template --plugin src/rsisys/algorithms/_template \
  --config src/rsisys/labs/_template/configs/run.json \
  --data-dir runs/gpu-lab-template
```

Implement `describe()` and `execute()` for the operations your Controller
calls. Declare split roles and a primary `MetricSpec`; publish its value in
`OperationResult.data['quality_metrics']`. Each Lab defines its own data format
and scoring logic.

A Lab can declare fixed training or evaluation inputs with
`data_bindings() -> tuple[DatasetBinding, ...]`. Import
[`DatasetBinding`](loops/contracts/data.py) from `rsisys.api` and use the same
role as the declared split. The host records the bindings in the Lab identity
for recovery. The Lab still loads and validates its own data. Record data
created during a run in artifacts and receipts.
