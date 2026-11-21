# SearchLab

Mount BrowseComp for answer accuracy and WideSearch for table accuracy and
row/item precision, recall, and F1. The Lab saves retrieved sources alongside
the response as evidence. The official graders score answers and tables.

Install `rsisys[search]` in the evaluator environment. Pin the upstream
`openai/simple-evals` and `ByteDance-Seed/WideSearch` checkouts, and prepare a
JSON manifest for each dataset. The manifest has `schema_version: 1`,
`benchmark`, and `splits`; each split lists `{id, input, reference}` records.
`input` contains the public question (and optionally language or answer format).
BrowseComp's `reference` contains the decoded `answer`. WideSearch's contains
`language`, `evaluation`, `answer_file` (a relative gold CSV path), and its
`sha256`. Keep these manifests and gold tables on the evaluator host.

```toml
[framework.labs.search]
name = "search"

[framework.labs.search.config]
trust = ["host-code"]
runner = "my_search_runner:solve"
runner_revision = "v1"
split_roles = { sealed = "sealed" }
primary_metric = { benchmark = "widesearch" }

[framework.labs.search.config.benchmarks.browsecomp]
manifest = "/data/search/browsecomp.json"
sha256 = "<manifest SHA-256>"
checkout = "/opt/simple-evals"
revision = "<upstream commit>"
python = "/opt/evaluator/bin/python"
judge_endpoint = "http://127.0.0.1:8000/v1"
judge_model = "<judge model revision or pinned deployment name>"
judge_key_env = "JUDGE_API_KEY"

[framework.labs.search.config.benchmarks.widesearch]
manifest = "/data/search/widesearch.json"
sha256 = "<manifest SHA-256>"
checkout = "/opt/WideSearch"
revision = "<upstream commit>"
python = "/opt/evaluator/bin/python"
judge_endpoint = "http://127.0.0.1:8000/v1"
judge_model = "<judge model revision or pinned deployment name>"
judge_key_env = "JUDGE_API_KEY"
```

The runner is trusted deployment code. Its `solve(*, instruction, task, context)`
function uses your actor and retrieval service, then returns `{"response":
text, "sources": [...]}`. The Lab passes only public task inputs to the runner.
For BrowseComp, return explanation, exact answer, and confidence. For WideSearch,
follow the requested Markdown table format. The separate grader worker connects
to the configured Chat Completions endpoint. Use `judge_key_env = "-"` for a
local endpoint that needs no key.

`evaluate` takes `split`, `instruction`, and optionally benchmark-qualified
`task_ids` on a development split. Mounting both benchmarks requires choosing
the primary metric. Each receipt reports the available benchmark metrics
separately, including `browsecomp/accuracy` and `widesearch/f1_by_item`.
