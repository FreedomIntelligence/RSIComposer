# RSISys

[中文](README_zh.md)

A reusable RSI framework with registered implementations of existing algorithms
and ready-to-configure LLM benchmark recipes: ReST^EM, RRSI, SEAL, TTRL and AIDE.
FT-Agent is available as an explicitly partial adaptation.

**RSI controller → Lab execution → feedback → RSI controller**.
The library owns method implementations and shared execution; `recipes/` provides
method-specific data preparation, dependencies, configurations and measured scope.
Harnesses, memory, verifiers and training engines remain replaceable components.

Early research release. CPU examples need no GPU, API key or model service.
Real-model recipes require their own models, data and optional environments.
A completed loop does not establish improvement or reproduce a paper's gains.

## Quickstart

Python **3.12 or 3.13**, Linux or macOS. From a checkout:

```sh
git clone https://github.com/satsuki2486441738/RSISys.git
cd RSISys
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
rsisys catalog
rsisys run --config recipes/rest_em/cpu/rest_em.json
```

This runs ReST^EM against a **simulated arithmetic Lab**, with no LLM training.
Inspect `runs/rest-em-quickstart/result.json` and its operation receipts.
Use a fresh directory for another run: `--data-dir runs/another-run`.
Installation uses a checkout or locally built wheel; no PyPI release is assumed.

## Architecture

```mermaid
flowchart LR
    C["RSI controller"] -->|Decision| L["Lab"]
    L -->|Feedback| C
```

The controller owns strategy, candidate selection and stopping decisions. Labs
own domain execution and return structured metrics, artifacts and log references.
The host provides registration, configuration, operation authorization, routing,
receipts, evidence references, lifecycle management and owned resource cleanup.
It does not choose an algorithm's winner from scores.

New algorithms normally need only external registration. Changes to shared
scheduling, delivery, cancellation or recovery semantics belong in the core.
`recipes/<method>/` contains maintained algorithm launch recipes, not generic
application demos. `src/rsisys/controllers/` and `src/rsisys/labs/` contain their
registered implementations. Supported tasks currently cover GSM8K, MATH and
SEAL's SQuAD knowledge adaptation. Medical and toy-model experiments are outside
this release. Formal protocols live in
[source](src/rsisys/loops/contracts/rsi.py); public imports use
[rsisys.api](src/rsisys/api.py).

## Register your own method

Existing methods run without custom plugins. To develop a new method, save the
following public-interface implementation as `plugin.py`:

```python
from rsisys.api import Decision, LabRequest, LabFeedback, OperationSpec

class Controller:
    def __init__(self, context, config):
        pass

    def decide(self, window):
        if window.feedback is not None:
            return Decision("finish", reason="Received Lab result")
        return Decision("invoke", operation=LabRequest("demo.step", {}))

class Lab:
    def __init__(self, context, config):
        pass

    def describe(self):
        return (OperationSpec("step", "Run one experiment", {
            "type": "object", "properties": {}, "additionalProperties": False
        }),)

    def execute(self, call):
        return LabFeedback("Example completed", (), False, data={"score": 1})

def register(registry):
    registry.register_controller("custom", Controller, version="1")
    registry.register_lab("custom", Lab, version="1")
```

Save as `plugin.py`, and select it with `experiment.json`:

```json
{
  "objective": "Run my RSI method",
  "data_dir": "runs/custom",
  "max_decision_steps": 8,
  "framework": {
    "controller": {"name": "custom", "config": {}},
    "labs": {"demo": {"name": "custom", "config": {}}}
  }
}
```

Run `rsisys run --config experiment.json --plugin plugin.py`.
`config` is passed to each factory alongside its assembly context. Optional
`validate_config` callbacks reject bad parameters before construction.
Multiple Labs use `alias.operation` names; each Lab receives its native name.
`DecisionInput` includes `objective`, latest `feedback`, `last_operation`,
`capabilities`, `context` and `references`. Decisions are `invoke`, `wait` or
`finish`. Domain arguments must be validated before side effects.

A Lab with background resources should implement idempotent `close()` and
`shutdown(mode) -> bool`, returning true only after its own work has stopped.
For async jobs, expose start/poll/result/cancel operations and deliver the final
receipt; registration does not automatically poll arbitrary external jobs.
Optional `required_operations()` checks operation availability, and `report()`
adds JSON method state to `result.json.controller_report`.

## Components

- `single_loop` provides fixed iteration; `meta_agent` uses the configured model
  with `harness=default|dsh`; set actual backend, role and model identities
  in your own JSON configuration. DSH also needs its native home/workspace/session
  configuration and `python -m pip install -e '.[dsh]'`.
- Memory, engines and scorers are injected inside module factories.
  Default memory is explicit/literal. Optional ReMe BM25 retrieval uses
  `python -m pip install -r requirements/reme.txt` and
  `ResearchAssembly.reme()` from `rsisys.research`; it adds no embedding service
  or autonomous memory agent. LangMem is a separate optional extra.
- Whole-assembly integrations can register a `MethodDefinition` through
  `registry.register_method`; select `framework.method` instead of
  `framework.controller/labs` when custom state adapters are needed.

## Methods and evidence

| Controller / Lab | Scope | Configuration / evidence |
| --- | --- | --- |
| `rest_em` / `math` | Generate, filter, retrain from the initial model; lightweight LoRA adaptation | [GSM8K](recipes/rest_em/configs/gsm8k-8gpu.json), [MATH](recipes/rest_em/configs/math-8gpu.json) |
| `rrsi` / `harness_math` | Fixed-model harness evolution; MATH port with regularization and critic | [Config](recipes/rrsi/math.json), [results](recipes/rrsi/results-20261108.json) |
| `seal` / `seal` | Self-edits, inner adaptation, selection, outer SFT; reference and paper profiles | [Reference](recipes/seal/8gpu.json), [paper](recipes/seal/paper.json), [results](recipes/seal/results-20261109.json) |
| `ttrl` / `ttrl` | Majority rewards with independent LoRA GRPO | [Config](recipes/ttrl/8gpu.json), [results](recipes/ttrl/results-20261109.json) |
| `ttrl_verl` / `ttrl_verl` | Pinned official full-parameter backend with continuous optimizer/reference state | [Config](recipes/ttrl/paper.json) |
| `ft_agent` / `llm_finetune` | Partial native RD-Agent policy with local GSM8K/LLaMA-Factory execution | [Config](recipes/ft_agent/gsm8k.json), [validation](recipes/ft_agent/validation-20261109.json) |
| `aide` / `aide_program` | Native Agent/search/review/Journal, adapted to fixed-model LLM inference programs | [Config](recipes/aide/gsm8k.json), [validation](recipes/aide/validation-20261109.json) |

SEAL and reference TTRL completed two-round, eight-GPU Qwen2.5-7B Base runs;
**neither reproduced paper gains**. SEAL's paper profile has small native-training
validation using a lexical smoke scorer, not an accepted GPT-4.1/full-paper run.
Native TTRL has a 32-target-question, one-epoch GPU check, not the full 500-question,
10-epoch experiment; its target metric is transductive.

FT-Agent completed three native SFT experiments; heldout accuracy fell from
48/64 to 45/64. It omits the full CoSTEER debugging/knowledge pipeline.
AIDE completed three search nodes; heldout accuracy rose from 28/64 to 34/64
with more calls (29 → 67) and tokens. This is not an equal-compute comparison or
AIDE's original benchmark. Neither adapter trains meta-agent weights.
These pilots validate integration; they do not justify blaming the papers for
unreproduced gains. JSON records retain run scope and failure evidence.

## Real-model setup

Run commands from the repository root. Use fresh data/run directories and edit
model paths, interpreters, endpoints and disjoint GPU allocations in configs.
The core installs no ML stack. Keep optional GPU environments separate.

For Qwen2.5-7B Base ReST^EM, install PyTorch, Transformers, PEFT, pyarrow and
huggingface-hub in a compatible CUDA environment; add vLLM for the eight-GPU
inference recipes and the math extra for MATH scoring:

```sh
python -m pip install -e '.[math]'
python recipes/rest_em/prepare_gsm8k.py \
  --model-dir models/Qwen2.5-7B-d1497293 --data-dir runs/gsm8k-data
python recipes/rest_em/prepare_math.py --data-dir runs/math-data
rsisys run --config recipes/rest_em/configs/math-8gpu.json
```

Preparation downloads pinned model/data revisions and records hashes. Configs
need to match those output paths. Eight-GPU recipes parallelize inference over
replicas and train with DDP; smaller configs are also under `configs/`.

SEAL/TTRL paper profiles use [seal-ttrl.txt](requirements/seal-ttrl.txt)
in a separate Python 3.12 environment. Tested stack: torch 2.7.0+cu128,
Transformers 4.52.4, TRL 0.18.1, PEFT 0.15.2, vLLM 0.9.1; native TTRL also
needs compatible flash-attn. Do not mix this with the reference training stack.
Pinned upstream revisions are in configs and the corresponding source adapters.

```sh
python recipes/seal/prepare.py --paper --output runs/seal-paper-data
python recipes/ttrl/prepare.py --paper --output runs/ttrl-paper-data
# Set paths and install each backend's dependencies before launching:
rsisys run --config recipes/seal/paper.json
rsisys run --config recipes/ttrl/paper.json
```

Native TTRL additionally requires a clean checkout of
`PRIME-RL/TTRL@5806e119789132fcb43552beaf2532d7d0447213`.
SEAL semantic scoring needs `OPENAI_API_KEY` and a compatible GPT-4.1 service.

RRSI, FT-Agent and AIDE need Linux bubblewrap/user namespaces plus an
OpenAI-compatible model service. Their `prepare.py` and `serve.py` entrypoints
provide `--help`. FT-Agent uses separately installed clean checkouts of
`microsoft/RD-Agent@484776c211e4fbbeef03e0ec00d6bbee7362a4f4` and
LLaMA-Factory v0.9.3 (`ca75f1edf3cb50343ed1c98605141c3e22075b5f`), plus
[worker requirements](recipes/ft_agent/requirements.txt), PyYAML and Math-Verify.
LLaMA-Factory's training environment is separate from SEAL's TRL environment.
AIDE uses `WecoAI/aideml@ad73d227af481bea3de06f884f41b2993dd6453a` and its
[worker requirements](recipes/aide/requirements.txt). Search/review remains
native; the host handles execution and feedback. Prepare disjoint public inputs
and private validation/heldout labels before running either method.

## Monitor

Optional, independent of the core package and GPU stack:

```sh
python -m pip install ./src/rsimonitor
rsimonitor --runs-dir runs
```

The viewer is read-only and binds to loopback by default. It has no authentication
or multi-tenant isolation; external exposure needs access control.

## Limits and security

0.x APIs may change; pin commits, model/data revisions and dependencies for long
experiments. Snapshots require explicit state/recovery adapters. Ordinary
registration does not restore arbitrary controller memory or external jobs;
FT-Agent/AIDE native searches and synchronous training recipes need fresh runs.
There is no global GPU lease across Labs/runs or complete long-job cost/budget
integration. Assign non-overlapping devices yourself. Snapshot metadata does not
freeze the installed dependency environment.

Plugins execute trusted host Python. Ordinary local-process Labs are not OS
sandboxes; use an explicit isolated executor for untrusted generated programs.
Custom Labs own network access, external billing and resource termination.
Supply credentials through environment variables. Saved prompts, logs and
artifacts can contain private data; review them before sharing.
`finished` describes host lifecycle, not training success or quality improvement.
Check method reports, receipts, optimizer/weight evidence and fixed heldout metrics.

For vulnerabilities, use GitHub's private reporting facility when available;
otherwise request a private channel without posting exploit details or secrets.

## Development

```sh
python -m pip install -e '.[dev]'
ruff check --select E9,F63,F7,F82 src recipes
python -m build --outdir dist/core .
python -m build --outdir dist/monitor src/rsimonitor
python -m twine check --strict dist/core/* dist/monitor/*
```

Regression suites and validation scripts are maintained privately;
this repository has no GitHub Actions workflows.
PRs should describe behavior, relevant validation and method deviations; use
synthetic fixtures, preserve attribution and keep weights, datasets and logs out
of Git. Contributions use Apache-2.0; keep discussion respectful and specific.

## License

Copyright 2026 RSISys contributors. Licensed under [Apache-2.0](LICENSE).
Third-party origins are linked in the
corresponding adapters; adapted SEAL code retains its MIT text in
[seal.py](src/rsisys/labs/seal.py). External dependencies, weights and datasets
are obtained separately under their own terms.
