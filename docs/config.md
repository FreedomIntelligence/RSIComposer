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
| `kind` | — | `openai_chat_completions`, `anthropic_messages`, `chat_completions` (generic compatible servers), `vllm`, `sglang`, `http`, or `mock` |
| `endpoint` | — | API root (normally ending in `/v1`); leave off `/chat/completions` or `/messages` |
| `engine_version` | — | Required for `sglang`. The adapter checks the exact version reported by the server before serving requests. |
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

For a separately deployed SGLang service, bind the role without naming the
engine in a Lab or Controller:

```toml
[backends.local_actor]
kind = "sglang"
endpoint = "http://127.0.0.1:30000/v1"
engine_version = "0.5.23"
json_mode = false

[roles.actor]
backend_id = "local_actor"
model_id = "your-served-model-name"
revision = "your-model-revision"
```

The native adapter supports structured generation and provider token counts;
tool calling also needs the appropriate tool parser on the server. For Meta-Agent
harness research, bind `roles.decision_maker` the same way and set the
Controller's `backend = "native"` for provider token budgeting and compaction.
Keep each server in its own pinned environment. The server version check does
not attest to the loaded weights: deploy an immutable model revision and
record its tokenizer and chat template with the experiment. Training adapters
are selected by the method that owns the training operation; see the
[ReST^EM trainer example](../src/rsisys/algorithms/rest_em/README.md#select-a-trainer).

To call a hosted provider or a gateway that implements its protocol, select the
wire format in `backends`. Bind either backend to `actor` or `decision_maker` as
required by the chosen method:

```toml
[backends.openai]
kind = "openai_chat_completions"
endpoint = "https://api.openai.com/v1"
credential_env = "OPENAI_API_KEY"
json_mode = false

[backends.anthropic]
kind = "anthropic_messages"
endpoint = "https://api.anthropic.com/v1"
credential_env = "ANTHROPIC_API_KEY"
json_mode = false

[roles.decision_maker]
backend_id = "anthropic"
model_id = "your-exact-model-id"
revision = "your-model-release"
```

Both adapters support non-streaming text, function calls and JSON Schema when
the selected model supports them. Set `json_mode = false` for Anthropic and
let methods request an explicit schema when they need structured output.
Anthropic supports input token counting before generation. OpenAI Chat
Completions reports usage after generation; it does not provide the input
token counter required by methods that use a provider tokenizer for hard
limits. Use `chat_completions` for compatible endpoints that expect the older
`max_tokens` option. Credentials are read at runtime, and nested runs use
temporary host gateway tokens for delegated calls.

### `framework`

| Form | Shape |
| --- | --- |
| Method | `{"method": {"name": …, "config": {"controller": {…}, "labs": {<alias>: {…}}}}}` |
| Controller and Labs | `{"controller": {"name": …, "config": {…}}, "labs": {<alias>: {"name": …, "config": {…}}}}` |

The options inside each `config` belong to the selected Controller or Lab.
Each one validates its options at startup and reports unknown keys. Relative
data paths in a Lab's options are usually resolved from the current directory
or from the Lab's `data_root`; see the Lab's guide.

### Online GRPO with veRL

`verl_math_grpo` scores a local baseline on the development split, trains on
the training split with the Math Lab's verifier as the reward, exports a full
checkpoint, and evaluates it on the same development split. Keep a separate
test split for reporting final results.

```toml
objective = "Compare baseline and GRPO checkpoints on the Math Lab dev split"
data_dir = "runs/math-grpo"

[framework.method]
name = "verl_math_grpo"

[framework.method.config.labs.math]
manifest = "/path/to/math/manifest.json"
initial_model = "/path/to/local/model"
inference_python = "/path/to/evaluation-env/bin/python"

[framework.method.config.labs.math.backend]
name = "verl_grpo"

[framework.method.config.labs.math.backend.config]
training_python = "/path/to/verl-env/bin/python"
training_strategy = "fsdp2"
verl_version = "0.9.1"
rollout_engine = "vllm"
rollout_version = "0.24.0"
gpus = 1
samples_per_prompt = 4
steps = 2

[resources.pools.gpu]
nodes = 1
gpus = [0]
gpus_per_node = 1

[resources.placement."math.rollout"]
pool = "gpu"
gpus = 1

[resources.placement."math.trainer"]
pool = "gpu"
gpus = 1
```

veRL owns the rollout engine and weight synchronization inside the training
job. Keep its Python environment separate from serving and evaluation.
`rollout_engine` accepts `vllm` or `sglang`; `rollout_version` must match the
package in `training_python`. The example's FSDP2 and vLLM stack was exercised
on one B200. The veRL 0.9.1 SGLang extra and its pinned SGLang 0.5.12 declare
conflicting Transformers versions, so that combination has not been verified.
This adapter currently uses the local RSISys executor and a single GPU node.

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
