# RSISys

A reusable research framework for recursive self-improvement (RSI):
**controller → Lab execution → feedback → controller**.

Implement an algorithm as a controller, expose its experiments through one or
more Labs, and register both in your own Python file. Select them with an
independent JSON config. Harnesses, memory, verifiers and training engines are
replaceable components inside these two boundaries.

**Early research release.** CPU quickstarts run without a GPU, API key, model
service or deployment. Real-model recipes have separate requirements and
explicit limits. A completed loop does not establish improvement or reproduce
a paper's results. [中文使用指南](docs/usage.md) · [Documentation](docs/README.md)

## Quickstart

Python **3.12 or 3.13**, Linux or macOS. From a checkout:

```sh
git clone https://github.com/satsuki2486441738/RSISys.git
cd RSISys
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
rsisys catalog
rsisys run --config examples/quickstart/rest_em.json
```

This runs the registered ReST^EM controller with a **simulated arithmetic Lab**,
not a trained language model. Inspect `runs/rest-em-quickstart/result.json` and
the operation receipts. For another run, pass a fresh `--data-dir`.
Installation currently uses a checkout or locally built wheel; this guide does
not assume a published PyPI package.

To develop your own method, copy `examples/registered` into your project:

```sh
rsisys run --config examples/registered/custom-controller.json \
  --plugin examples/registered/plugin.py
rsisys catalog --plugin examples/registered/plugin.py
```

That example runs on CPU through the same host loop. See the
[registration guide](docs/registration.md) for complete interfaces, validation,
multiple Labs, component injection and advanced runtime integration.

## What the framework provides

- Versioned registration, configuration, operation discovery and namespaced Lab
  routing; public types are exported by `rsisys.api`.
- Host operation authorization, structured receipts, evidence references,
  lifecycle management and owned resource cleanup.
- Reusable controllers and Lab components; optional default/DSH harnesses and
  explicit memory backends for model-driven research.
- Snapshot and recovery facilities for explicitly adapted implementations.
  Arbitrary plugin state and external jobs are not automatically recoverable.
- A separate, read-only monitor with no core or GPU dependency:

```sh
python -m pip install ./src/rsimonitor
rsimonitor --runs-dir runs
```

The monitor binds to loopback by default. Plugins and the local process Lab run
trusted host code; see [security boundaries](SECURITY.md).

## Registered methods and recipes

See [method support and evidence](docs/methods.md) for tested configurations,
upstream fidelity and known deviations. These entries include both reference
implementations and native adapters; no full-scale paper result is claimed.

| Controller | Purpose | Implementation scope | Entry |
| --- | --- | --- | --- |
| `single_loop` | Fixed iterations | Template | [CPU example](examples/registered/README.md) |
| `meta_agent` | Model-driven decisions | Default / DSH harness template | [Registration](docs/registration.md) |
| `rest_em` | Generate, filter, SFT, evaluate | Lightweight LoRA implementation | [CPU quickstart](examples/quickstart/README.md), [GSM8K/MATH](experiments/rest_em/README.md) |
| `rrsi` | Fixed-model harness evolution | Algorithm port to MATH | [Recipe and evidence](experiments/rrsi/README.md) |
| `seal` | Self-edits, inner adaptation, outer SFT | Reference and paper-profile training adapters | [Recipe](experiments/seal/README.md), [results](experiments/seal/RESULTS.md) |
| `ttrl` | Majority rewards and GRPO | Independent LoRA reference backend | [Recipe](experiments/ttrl/README.md), [results](experiments/ttrl/RESULTS.md) |
| `ttrl_verl` | Continuous full-parameter adaptation | Pinned official training backend | [Configuration](experiments/ttrl/paper.json), [environment](experiments/paper-environment.md) |
| `ft_agent` | Fine-tuning research | Partial native policy / local Lab adaptation | [Scope](experiments/ft_agent/README.md), [results](experiments/ft_agent/RESULTS.md) |
| `aide` | Inference-program research | Native Agent / adapted LLM task | [Scope](experiments/aide/README.md), [results](experiments/aide/RESULTS.md) |

SEAL and TTRL completed two-round, eight-GPU runs using Qwen2.5-7B Base.
**Neither small run reproduced the paper's performance gains.** Their data,
scoring and backend differences are documented. Those historical runs used
independent reference trainers. The SEAL paper profile and registered `ttrl_verl`
backend provide native training paths; their integration validation is separate
from reproducing paper-level gains.

## Extend and contribute

Start with [architecture](docs/architecture.md), [public contracts](docs/api-contracts.md)
and [registration](docs/registration.md). Shared scheduling, delivery, recovery
and resource semantics belong in the core; algorithm strategy and domain
execution belong in controllers and Labs. See [compatibility and support](docs/compatibility.md)
for the current boundary.

`examples/` contains tutorials, `experiments/` contains model/data recipes,
and `tests/` contains maintained regression coverage. Development and test
commands are in [CONTRIBUTING.md](CONTRIBUTING.md). Release checks and remaining
roadmap items are in [the release guide](docs/releasing.md).

## License

Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights and external datasets
retain their own terms and are not included in the release.
