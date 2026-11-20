<h1 align="center">RSISys</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-009688.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/satsuki2486441738/RSISys/stargazers"><img src="https://img.shields.io/github/stars/satsuki2486441738/RSISys?logo=github" alt="GitHub stars"></a>
  <img src="https://img.shields.io/badge/arXiv-Coming%20soon-B31B1B?logo=arxiv" alt="arXiv paper: Coming soon">
</p>

<p align="center"><strong>A general framework for recursive self-improvement: research, explore, and improve the whole system</strong></p>

<p align="center">English · <a href="README_zh.md">简体中文</a></p>

<p align="center"><a href="#framework-overview">Overview</a> · <a href="#news">News</a> · <a href="#highlights">Highlights</a> · <a href="#getting-started">Getting started</a> · <a href="docs/README.md">Docs</a> · <a href="#adapted-methods">Methods</a> · <a href="#citation">Citation</a></p>

RSISys runs recursive self-improvement (RSI) experiments as a loop. A Controller decides what to try next, Labs run the experiments and return receipts, and the Controller reads the evidence to choose its next step or the candidate to adopt. Prompt search, harness search, supervised fine-tuning and reinforcement learning share this loop, so you can swap a Controller, reuse a Lab, or add a method without changing the core.

## Framework overview

![RSISys architecture, in two panels. (a) The Controller requests operations from Domain Labs, which return trajectories, metrics, and artifacts as evidence. (b) The Self-Reflection Lab constructs a Controller update and can validate it in a nested trial. The current Controller decides whether to adopt the update.](assets/rsisys-overview.png)

A **Controller** decides what to try and what to adopt. Its Policy makes decisions, its Harness prepares inputs and tools, and optional Memory provides context. A **Lab** executes operations such as evaluation, sampling, or training against its configured inputs. The host routes calls, enforces budgets and split rules, and preserves receipts for recovery.

## News

- **[2026-11]** RSISys is open-sourced, with 16 built-in method registrations and support for durable runs.

## Highlights

**One loop for every RSI method.** RSISys takes a reinforcement learning view: a Controller chooses an action, a Lab runs it, and the evidence guides the next decision. Single-agent RSI is a special case: the same agent decides what to try and performs the task in the Lab. The same loop also lets a separate research agent direct experiments for a task agent. [Different methods](#adapted-methods) can use this loop with their own optimizers.

**Choose what to improve.** A method can change an agent's prompts, programs, harness, or model weights. It can also change the RSI policy that decides how to improve. The Self-Reflection Lab can test a candidate Controller in a nested run, and the current Controller decides whether to adopt it based on the evidence.

**Swap and reuse components.** Controllers and Labs compose through public interfaces, and their policies, harnesses, memory, and backends can be replaced independently. Reuse the same Coding Lab with different compatible Controllers, or connect a new dataset or benchmark through a Lab, without changing the core. [See how to compose a system](docs/compose.md).

## Getting started

Install with Python 3.12 or 3.13 on Linux or macOS. The core has no third-party dependencies; methods list their [optional extras](docs/components.md#methods).

```sh
git clone https://github.com/satsuki2486441738/RSISys.git
cd RSISys
python -m pip install -e .
```

**Run a first experiment without a GPU.** This runs two ReST^EM rounds on fixture data with a simulated model. It checks the installation and shows what a run produces.

```sh
rsisys run --config src/rsisys/algorithms/rest_em/configs/cpu/rest_em.json --data-dir runs/quickstart
```

**Run with a GPU model.** [Start a Chat Completions endpoint](src/rsisys/labs/_template/README.md#connect-a-gpu-model), then run the method template, which searches prompt and few-shot settings:

```sh
rsisys run --plugin src/rsisys/labs/_template --plugin src/rsisys/algorithms/_template \
  --config src/rsisys/algorithms/_template/configs/run.json --data-dir runs/gpu-quickstart
```

The [Quickstart](docs/quickstart.md) explains how to read the results, watch runs in RSIMonitor, and resume interrupted runs.

## Documentation

| I want to… | Read |
| --- | --- |
| Learn the terms: Controller, Lab, operation, receipt, split | [Concepts](docs/concepts.md) |
| See the built-in methods, Controllers, Labs, and components | [Components](docs/components.md) |
| Run a Controller on a different Lab, or load my own Controller | [Compose a system](docs/compose.md) |
| Write a new method | [Method template](src/rsisys/algorithms/_template/README.md) |
| Write a Lab or connect my own benchmark | [Lab template](src/rsisys/labs/_template/README.md) |
| Look up a config field or CLI option | [Run configuration](docs/config.md) |
| Inspect and compare runs | [RSIMonitor](src/rsimonitor/README.md) |

## Adapted methods

| Family | Methods | Controller (research) | Lab (exploration) | What improves |
| :-- | :-- | :-- | :-- | :-- |
| Self-training | [ReST^EM](https://arxiv.org/abs/2312.06585) · [STaR](https://arxiv.org/abs/2203.14465) · [TTRL](https://arxiv.org/abs/2504.16084) · TTRL-verl | Schedules generate, filter and train rounds | The model samples answers, is scored and fine-tuned | Model weights |
| Self-editing | [SEAL](https://arxiv.org/abs/2506.10943) | Schedules self-edit rounds | The model writes its own training data and is updated | Model weights (unified) |
| Prompt search | [OPRO](https://arxiv.org/abs/2309.03409) · [GEPA](https://arxiv.org/abs/2507.19457) · GEPA-Merge · [PromptPG](https://arxiv.org/abs/2209.14610) | Searches over candidate prompts | A task model is evaluated with each prompt | Prompts |
| Program search | [FunSearch](https://github.com/google-deepmind/funsearch) | Evolves a population of programs | Programs are executed and scored | Programs |
| Harness search | [RRSI](https://arxiv.org/abs/2609.24972) · Meta-Agent · [Coding Meta-Agent](src/rsisys/algorithms/coding_meta_agent/README.md) | A model proposes harness changes | An agent solves tasks with the candidate harness | Agent harness |
| Reflection | [Reflexion](https://arxiv.org/abs/2303.11366) | Decides when to retry | The same agent writes code, runs tests and reflects on failures | Its own reflections (unified) |
| Research agents | [AIDE](https://github.com/WecoAI/aideml) · [FT-Agent](https://github.com/microsoft/RD-Agent) | The upstream agent drafts, debugs and selects solutions | Candidate programs and fine-tuning jobs run | Solutions and fine-tuned models |
| Self-reflection | [Self-Reflection Lab](src/rsisys/labs/scenes/self_reflection/lab.py) (a Lab, paired with a Controller) | The current Controller | Candidate Controllers run in nested experiments | The Controller itself |

Run `rsisys catalog` to list the registered methods. [Components](docs/components.md#methods) lists each method's presets, hardware, and dependencies. Guides exist for ReST^EM, GEPA, and Coding Meta-Agent; the other methods have presets but no guide yet. A preset is a starting point, not a verified reproduction of the paper.

## Contributing

Start with the [method template](src/rsisys/algorithms/_template/) or the [Lab template](src/rsisys/labs/_template/), and see [Compose a system](docs/compose.md) for plugins. For a new method or Lab, include a runnable configuration, state which results it has been checked against, and cite the source of any upstream adaptation. Import framework interfaces from `rsisys.api`.

## Citation

If you find RSISys useful, please cite:

```bibtex
@misc{rsisys2026,
  title        = {RSISys: A Modular Runtime for Recursive Self-Improvement Research},
  author       = {{RSISys contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/satsuki2486441738/RSISys}}
}
```

## License

RSISys is released under the [Apache 2.0 License](LICENSE). Adapted upstream code keeps its attribution next to the implementation.
