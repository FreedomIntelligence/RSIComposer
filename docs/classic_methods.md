# Classic RSI methods

These packages register with the normal RSISys catalog. Run commands from the repository root. Copy the selected JSON to a local path before changing data paths, deployments or budgets. Use --data-dir to give each run a new output directory.

| Method | Evolving object | Profiles |
| --- | --- | --- |
| OPRO | Solver instruction | [Score history](../src/rsisys/algorithms/opro/configs/evolution.json), [independent candidates](../src/rsisys/algorithms/opro/configs/independent_candidates.json) |
| GEPA | Solver instruction | [Reflective feedback](../src/rsisys/algorithms/gepa/configs/reflective.json), [score-only feedback](../src/rsisys/algorithms/gepa/configs/score-only.json) |
| FunSearch | Bin-packing priority expression | [Evolving population](../src/rsisys/algorithms/funsearch/configs/evolution.json), [frozen parents](../src/rsisys/algorithms/funsearch/configs/frozen_parents.json) |
| Reflexion | Per-task reflection and Python solution | [With reflection](../src/rsisys/algorithms/reflexion/configs/reflection.json), [without reflection](../src/rsisys/algorithms/reflexion/configs/control.json) |
| PromptPG | Example selector weights | [TabMWP](../src/rsisys/algorithms/promptpg/configs/tabmwp.json) |
| STaR | Task-model text weights | [Four-round GSM8K profile](../src/rsisys/algorithms/star/configs/gsm8k.json) |

Example after configuring the model endpoint and data:

    rsisys catalog
    rsisys run --config src/rsisys/algorithms/gepa/configs/reflective.json \
      --data-dir runs/gepa-reflective-17

The controllers schedule operations and select candidates or checkpoints. The Labs execute one operation per request and return feedback. HTTP configurations bind an OpenAI-compatible model endpoint through RSISys. Configure credentials, when needed, through a backend's credential_env field. The included endpoints and model names are examples. The endpoint must return the configured model identity and support the requested sampling parameters.

## Sources and data

- **OPRO** uses the [author's prompt construction and GSM8K evaluator](https://github.com/google-deepmind/opro/tree/a76bdce2cbf6d4a0d1e570a6fcfe17be9c2abdd7). Set source_dir to a clean checkout of that revision. The included local profile uses a separate validation split for selection. It uses the author helpers and the local Controller search loop.
- **GEPA** uses the [author's candidate selection, minibatch sampler and instruction signature](https://github.com/gepa-ai/gepa/tree/99c3b0ac3b4750cde6a2a88547ed4450930d681b). Set source_dir to a clean checkout of that revision and install its Python dependencies. The local profile uses one GSM8K instruction and matched scoring and mutation budgets. The reflective arm receives task output and feedback, while the score-only arm receives correctness bits.
- **OPRO and GEPA** take disjoint search, validation and test JSONL files with id, question and answer string fields. Use numeric answer strings and set the declared counts to the file lengths. To reproduce a historical run, use its frozen cohorts. OPRO needs NumPy, pandas and the author evaluator dependencies. Responses rejected as truncated by the host text transport count as missing answers or proposals.
- **FunSearch** uses the [author's population database](https://github.com/google-deepmind/funsearch/tree/cc53f274237d7ab05c19df939edbc1f9616a7c19) and requires NumPy. Set source_dir to the author checkout. The three used source files are verified against their hashes. Its bin-packing instances are generated deterministically, and candidates run through the method's restricted numeric expression evaluator.
- **Reflexion** uses the [pinned author generator and Python executor](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0). Set source_dir to the clean checkout and install the author generator dependencies. Both profiles use the same first 64 official HumanEval tasks, with six internal tests at most and two attempts. Generated programs require a working RSISys OS isolation driver. Linux requires bubblewrap and working namespaces.
- **PromptPG** uses [PromptPG's TabMWP data and prompt/scoring helpers](https://github.com/lupantech/PromptPG) from the 5a1a522 experiment checkout. The four used source/data files are hash-checked. Configure source_dir and encoder_path for the local frozen BERT encoder. Install PyTorch and Transformers. The Lab updates the selector for one batch at a time, while the actor remains frozen. The profile compares the selected selector, random examples and nearest neighbors on the same held-out set.
- **STaR** is an adaptation of [the author's algorithm](https://github.com/ezelikman/STaR) to local Qwen3.5-9B. Configure base_model and disjoint train, dev and test JSONL files with id, question and answer fields. The profile retains four rounds with rationalization in the last two, global batch 8, and one-node, four-rank FSDP2 full-text training. It requires PyTorch with FSDP2, Transformers with Qwen3.5 support, Accelerate, safetensors, and four allocated GPU UUIDs in NVIDIA_VISIBLE_DEVICES. This profile uses no packing and a ten-step warmup. The controller chooses the trained checkpoint by development accuracy.

The CLI prints a controller report. Detailed evidence belongs in the run directory. Store external checkouts, model weights, datasets and run outputs outside the files staged for contribution.
