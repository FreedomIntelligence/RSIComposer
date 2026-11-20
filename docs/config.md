# Run configuration

`rsisys run --config run.toml` and `rsisys run --config run.json` accept the same
fields. See the [Coding Meta-Agent TOML preset](../src/rsisys/algorithms/coding_meta_agent/configs/coding.toml)
or the JSON example below. In TOML, quote dates used as revision strings and
keys that contain a dot (for example, `"terminal-bench-2.1"`). Benchmark
manifests and saved run state still use JSON.

```json
{
  "objective": "Compare prompts on a small arithmetic dataset",
  "data_dir": "runs/my-run",
  "max_decision_steps": 8,
  "backends": {
    "gpu": {"kind": "chat_completions", "endpoint": "http://127.0.0.1:18098/v1"}
  },
  "roles": {
    "actor": {"backend_id": "gpu", "model_id": "my-model", "revision": "2026-01-01"}
  },
  "budget": {"seconds": 3600},
  "framework": {
    "method": {
      "name": "template_prompt_search",
      "config": {
        "controller": {"max_evaluations": 3},
        "labs": {"math": {"datasets": {"train": "data/train.jsonl", "dev": "data/dev.jsonl"}}}
      }
    }
  }
}
```

This example runs the method template, so it needs
`--plugin src/rsisys/labs/_template --plugin src/rsisys/algorithms/_template`.

## Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `objective` | yes | The run's goal in plain text. Model-driven Controllers receive it in their input. |
| `data_dir` | yes | Output directory. A relative path is resolved from the config file's directory. `--data-dir` overrides it and is resolved from the current directory. |
| `framework` | yes | What to run: a method, or a Controller with Labs. See [Compose a system](compose.md). |
| `max_decision_steps` | no | Maximum number of Controller decisions. Default: 64. |
| `backends` | no | Model endpoints, keyed by an ID that `roles` refers to. |
| `roles` | no | Which model fills each role: `actor` (the model a Lab uses) or `decision_maker` (the model a model-driven Controller uses). |
| `budget` | no | Run-wide limits: `seconds`, `model_calls`, `gpu_seconds`. The host stops the run when one is reached. |
| `resources` | no | GPU and CPU pools and their placement by role, for local or Ray execution. See `rest_em/configs/gsm8k-8gpu.json` for an example. |
| `execution` | no | Process sandbox for Controller tools: `mock` (default) or `local_process` (trusted local processes, not OS isolation). |
| `rounds` | no | Host rounds to run. Default: 1. Same as `--rounds`. |
| `run_id`, `parent_run_id` | no | Run identity. Generated if omitted. |

### `backends.<id>`

| Field | Default | Meaning |
| --- | --- | --- |
| `kind` | — | `chat_completions` (OpenAI-compatible), `http`, or `mock` |
| `endpoint` | — | Base URL; for `chat_completions` it ends in `/v1` |
| `credential_env` | — | Name of the environment variable that holds the API key. Never put the key itself in the config. |
| `timeout_seconds` | 60 | Request timeout |
| `json_mode` | true | Request JSON responses |
| `max_response_bytes` | 8 MiB | Response size limit |

### `roles.<role>`

| Field | Meaning |
| --- | --- |
| `backend_id` | A key in `backends` |
| `model_id` | The model name the endpoint serves |
| `revision` | The exact model revision, recorded with the results |

### `framework`

| Form | Shape |
| --- | --- |
| Method | `{"method": {"name": …, "config": {"controller": {…}, "labs": {<alias>: {…}}}}}` |
| Controller and Labs | `{"controller": {"name": …, "config": {…}}, "labs": {<alias>: {"name": …, "config": {…}}}}` |

The options inside each `config` belong to the selected Controller or Lab.
Each one validates its options at startup and reports unknown keys. Relative
data paths in a Lab's options are usually resolved from the current directory
or from the Lab's `data_root`; see the Lab's guide.

## Command-line options

| Command | Option | Meaning |
| --- | --- | --- |
| `rsisys run` | `--config` | TOML or JSON run config file |
| | `--data-dir` | Output directory; overrides `data_dir` |
| | `--plugin` | Load a trusted registration file or package; repeatable |
| | `--durable` | Save a checkpoint after every step so `rsisys resume` can continue the run |
| | `--experiment-name` | Name shown in RSIMonitor |
| | `--max-decision-steps`, `--rounds` | Override the config values |
| | `--probe` | Load an observation plugin for RSIMonitor; repeatable |
| `rsisys resume` | `--data-dir` | Continue a durable run in this directory |
| | `--snapshot` | Restore a snapshot into a new `--data-dir` instead |
| `rsisys catalog` | `--plugin` | Include plugin registrations in the listing |
| `rsisys inspect` | `--snapshot` | Validate and describe a snapshot |
