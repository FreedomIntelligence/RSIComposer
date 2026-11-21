<h1 align="center">RSIComposer</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-009688.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/FreedomIntelligence/RSIComposer/stargazers"><img src="https://img.shields.io/github/stars/FreedomIntelligence/RSIComposer?logo=github" alt="GitHub stars"></a>
  <img src="https://img.shields.io/badge/arXiv-Coming%20soon-B31B1B?logo=arxiv" alt="arXiv paper: Coming soon">
</p>

<p align="center"><strong>A general framework for recursive self-improvement: choose what to improve and build your own RSI system from interchangeable components</strong></p>

<p align="center">English · <a href="README_zh.md">简体中文</a></p>

<p align="center"><a href="#framework-overview">Overview</a> · <a href="#news">News</a> · <a href="#highlights">Highlights</a> · <a href="#getting-started">Getting started</a> · <a href="#domain-labs-and-benchmarks">Labs</a> · <a href="docs/README.md">Docs</a> · <a href="#adapted-methods">Methods</a> · <a href="#citation">Citation</a></p>

RSIComposer runs recursive self-improvement (RSI) experiments as a loop. A Controller decides what to try next, Labs run the experiments and return receipts, and the Controller reads the evidence to choose its next step or the candidate to adopt. Prompt search, harness search, supervised fine-tuning and reinforcement learning share this loop, so you can swap a Controller, reuse a Lab, or add a method without changing the core.

## Framework overview

![RSIComposer architecture, in two panels. (a) The Controller requests operations from Domain Labs, which return trajectories, metrics, and artifacts as evidence. (b) The Self-Reflection Lab constructs a Controller update and can validate it in a nested trial. The current Controller decides whether to adopt the update.](assets/rsisys-overview.png)

A **Controller** decides what to try and what to adopt. Its Policy makes decisions, its Harness prepares inputs and tools, and optional Memory provides context. A **Lab** executes operations such as evaluation, sampling, or training against its configured inputs. The host routes calls, enforces budgets and split rules, and preserves receipts for recovery.

## News

- **[2026-11]** RSIComposer is open-sourced, with 16 built-in method registrations and support for durable runs.

## Highlights

**A general RSI framework built around the research loop.** RSIComposer automates the human–AI research loop: the Controller chooses what to try, Labs run the experiments, and accumulated evidence guides what to try next or what to adopt. [Different RSI methods](#adapted-methods) use this structure with their own optimization algorithms.

**Choose what to improve.** A method can change an agent's prompts, programs, harness, or model weights. It can also change the RSI policy that decides how to improve. The Self-Reflection Lab can test a candidate Controller in a nested run, and the current Controller decides whether to adopt it based on the evidence.

**Swap and reuse components, including training and inference.** Controllers, Labs, policies, harnesses, and memory compose through public interfaces. A Lab can reuse its benchmarks with different Controllers, while compatible methods can select training backends such as TRL or veRL and bind inference to vLLM, SGLang, or an OpenAI Chat Completions or Anthropic Messages endpoint. Online RL configures its rollout engine separately from the trainer. [See the available components and backends](docs/components.md#reusable-components).

## Getting started

Install with Python 3.12 or 3.13 on Linux or macOS. The core has no third-party dependencies; methods list their [optional extras](docs/components.md#methods).

```sh
git clone https://github.com/FreedomIntelligence/RSIComposer.git
cd RSIComposer
python -m pip install -e .
```

**Run a first experiment without a GPU.** This runs two ReST^EM rounds on fixture data with a simulated model. It checks the installation and shows what a run produces.

```sh
rsicomposer run --config src/rsicomposer/algorithms/rest_em/configs/cpu/rest_em.json --data-dir runs/quickstart
```

**Run with a GPU model.** [Start a Chat Completions endpoint](src/rsicomposer/labs/_template/README.md#connect-a-gpu-model), then run the method template, which searches prompt and few-shot settings:

```sh
rsicomposer run --plugin src/rsicomposer/labs/_template --plugin src/rsicomposer/algorithms/_template \
  --config src/rsicomposer/algorithms/_template/configs/run.json --data-dir runs/gpu-quickstart
```

The [Quickstart](docs/quickstart.md) explains how to read the results, watch runs in RSIMonitor, and resume interrupted runs.

## Domain Labs and benchmarks

Choose a Lab and its benchmarks in your run configuration. Each Lab guide lists the required task assets, runner, and scoring setup.

| Lab | Benchmarks | Validation so far |
| --- | --- | --- |
| [CodingLab](src/rsicomposer/labs/coding/README.md) | SWE-bench Verified, Terminal-Bench 2.1 | One official reference solution scored in Harbor per benchmark |
| [MathLab](src/rsicomposer/labs/math/README.md) | AIME 2024, AIME 2025; also supports AIMO3 Reference | Correct and incorrect answers checked across both AIME sets |
| [SearchLab](src/rsicomposer/labs/search/README.md) | BrowseComp, WideSearch | Scoring checked on benchmark tasks with a test judge; live search agent run pending |
| [ComputerUseLab](src/rsicomposer/labs/computer_use/README.md) | WebArena-Infinity (real tasks) | Chromium and state verification checked on a sample task; model-driven browser agent run pending |
| [ProfessionalWorkLab](src/rsicomposer/labs/professional_work/README.md) | SkillsBench v1.0 (science, office, finance) | Official verifier passed on one finance task; full Harbor trial pending |

These are integration checks. Full benchmark runs with models and RSI methods are still pending for this set of adapters.

## Documentation

| I want to… | Read |
| --- | --- |
| Learn the terms: Controller, Lab, operation, receipt, split | [Concepts](docs/concepts.md) |
| See the built-in methods, Controllers, Labs, and components | [Components](docs/components.md) |
| Run a Controller on a different Lab, or load my own Controller | [Compose a system](docs/compose.md) |
| Write a new method | [Method template](src/rsicomposer/algorithms/_template/README.md) |
| Write a Lab or connect my own benchmark | [Lab template](src/rsicomposer/labs/_template/README.md) |
| Look up a config field or CLI option | [Run configuration](docs/config.md) |
| Inspect and compare runs | [RSIMonitor](src/rsimonitor/README.md) |

## Adapted methods

| Family | Methods | Controller (research) | Lab (exploration) | What improves |
| :-- | :-- | :-- | :-- | :-- |
| Self-training | [ReST^EM](https://arxiv.org/abs/2312.06585) · [STaR](https://arxiv.org/abs/2203.14465) · [TTRL](https://arxiv.org/abs/2504.16084) · TTRL-verl | Schedules generate, filter and train rounds | The model samples answers, is scored and fine-tuned | Model weights |
| Self-editing | [SEAL](https://arxiv.org/abs/2506.10943) | Schedules self-edit rounds | The model writes its own training data and is updated | Model weights (unified) |
| Prompt search | [OPRO](https://arxiv.org/abs/2309.03409) · [GEPA](https://arxiv.org/abs/2507.19457) · GEPA-Merge · [PromptPG](https://arxiv.org/abs/2209.14610) | Searches over candidate prompts | A task model is evaluated with each prompt | Prompts |
| Program search | [FunSearch](https://github.com/google-deepmind/funsearch) | Evolves a population of programs | Programs are executed and scored | Programs |
| Harness search | [RRSI](https://arxiv.org/abs/2609.24972) · Meta-Agent · [Coding Meta-Agent](src/rsicomposer/algorithms/coding_meta_agent/README.md) | A model proposes harness changes | An agent solves tasks with the candidate harness | Agent harness |
| Reflection | [Reflexion](https://arxiv.org/abs/2303.11366) | Decides when to retry | The same agent writes code, runs tests and reflects on failures | Its own reflections (unified) |
| Research agents | [AIDE](https://github.com/WecoAI/aideml) · [FT-Agent](https://github.com/microsoft/RD-Agent) | The upstream agent drafts, debugs and selects solutions | Candidate programs and fine-tuning jobs run | Solutions and fine-tuned models |
| Self-reflection | [Self-Reflection Lab](src/rsicomposer/labs/self_reflection/lab.py) (a Lab, paired with a Controller) | The current Controller | Candidate Controllers run in nested experiments | The Controller itself |

Run `rsicomposer catalog` to list the registered methods. [Components](docs/components.md#methods) lists each method's presets, hardware, and dependencies. Guides exist for ReST^EM, GEPA, and Coding Meta-Agent; the other methods have presets but no guide yet. A preset is a starting point, not a verified reproduction of the paper.

## Contributing

Start with the [method template](src/rsicomposer/algorithms/_template/) or the [Lab template](src/rsicomposer/labs/_template/), and see [Compose a system](docs/compose.md) for plugins. For a new method or Lab, include a runnable configuration, state which results it has been checked against, and cite the source of any upstream adaptation. Import framework interfaces from `rsicomposer.api`.

## Citation

If you find RSIComposer useful, please cite:

```bibtex
@misc{rsicomposer2026,
  title        = {RSIComposer: A Modular Runtime for Recursive Self-Improvement Research},
  author       = {{RSIComposer contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/FreedomIntelligence/RSIComposer}}
}
```

## License

RSIComposer is released under the [Apache 2.0 License](LICENSE). Adapted upstream code keeps its attribution next to the implementation.
