<h1 align="center">RSISys</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-009688.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/satsuki2486441738/RSISys/stargazers"><img src="https://img.shields.io/github/stars/satsuki2486441738/RSISys?logo=github" alt="GitHub stars"></a>
</p>

<p align="center"><strong>A general framework for recursive self-improvement: research, explore, and improve the whole system</strong></p>

<p align="center">English · <a href="README_zh.md">简体中文</a></p>

<p align="center"><a href="#framework-overview">Overview</a> · <a href="#news">News</a> · <a href="#highlights">Highlights</a> · <a href="#getting-started">Getting started</a> · <a href="#adapted-methods">Methods</a> · <a href="#citation">Citation</a></p>

RSISys runs recursive self-improvement (RSI) experiments as a loop. A Controller decides what to try next, Labs run the experiments and return receipts, and the Controller reads the evidence to choose its next step or the candidate to adopt. Prompt search, harness search, supervised fine-tuning and reinforcement learning share this loop, so you can swap a Controller, reuse a Lab, or add a method without changing the core.

## Framework overview

![RSISys architecture, in two panels. (a) The Controller requests operations from Domain Labs, which return trajectories, metrics, and artifacts as evidence. (b) The Self-Reflection Lab constructs a Controller update and can validate it in a nested trial. The current Controller decides whether to adopt the update.](assets/rsisys-overview.png)

A **Controller** decides what to try and what to adopt. Its Policy makes decisions, its Harness prepares inputs and tools, and optional Memory provides context. A **Lab** executes operations such as evaluation, sampling, or training against its configured inputs. The host routes calls, enforces budgets and split rules, and preserves receipts for recovery.

## News

- **[2026-11]** RSISys is open-sourced, with 16 built-in method registrations and support for durable runs.

## Highlights

**One loop for every RSI method.** RSISys describes recursive self-improvement as a single loop: the Controller decides what to try, Labs run experiments in the target environment, and the evidence they return drives the next decision. What improves is not fixed. It can be a model's weights, a prompt, a program, an agent harness, or the Controller itself. Self-training, prompt and program search, harness evolution and autonomous research agents all fit this loop without a second structure; the [method table](#adapted-methods) shows how each one maps.

**Research and exploration are roles, not extra components.** The Controller plays the research role; each Lab plays the exploration role. Assign the roles to different agents, for example a strong model deciding how to train a small one, and the system is decoupled. Assign both to the same agent and you get the usual single-agent self-improvement loop. RSISys expresses both through the same interfaces, so you can move between them without rewriting the method.

**The Controller can improve itself.** In most systems the self-improvement algorithm is fixed. Here the Controller is a replaceable component: it can trial new versions of itself in nested experiments and switch to one at a checkpoint, so the research loop is itself open to improvement.

## Getting started

Clone the repository and install it with Python 3.12 or 3.13 on Linux or macOS. The core has no third-party dependencies; install [optional extras](pyproject.toml) for the components your method uses.

```sh
git clone https://github.com/satsuki2486441738/RSISys.git
cd RSISys
python -m pip install -e .
```

**Run a first GPU experiment.** [Start or configure a GPU model endpoint](src/rsisys/labs/_template/README.md#connect-a-gpu-model), then run prompt search against the arithmetic Lab. Results are written to `runs/gpu-quickstart/result.json`; use a new `--data-dir` for each run.

```sh
rsisys run --plugin src/rsisys/labs/_template --plugin src/rsisys/algorithms/_template \
  --config src/rsisys/algorithms/_template/configs/run.json \
  --data-dir runs/gpu-quickstart
```

**Watch the run.** RSIMonitor is an optional read-only dashboard. It shows the decision graph, metrics, samples and logs, and compares experiments.

```sh
python -m pip install ./src/rsimonitor
rsimonitor --runs-dir runs    # then open http://127.0.0.1:8765/
```

## Guides

- [Reproduce an existing method](src/rsisys/README.md#reproduce-an-existing-method)
- [Write or customize a method](src/rsisys/README.md#define-a-method)
- [Define a Lab or connect a benchmark or dataset](src/rsisys/README.md#define-a-lab-or-connect-data)
- [Inspect and compare runs with RSIMonitor](src/rsimonitor/README.md)

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
| Self-reflection | Self-Reflection Lab | The current Controller | Candidate Controllers run in nested experiments | The Controller itself |

Run `rsisys catalog` to list the registered methods. For each method’s benchmark, model and training setup, see the [reproduction guide](src/rsisys/README.md#reproduce-an-existing-method).

## Contributing

Start with the [GPU method template](src/rsisys/algorithms/_template/) or the [Lab template](src/rsisys/labs/_template/). For a new method or Lab, include a runnable configuration and the source of any upstream adaptation. Import framework interfaces from `rsisys.api`.

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
