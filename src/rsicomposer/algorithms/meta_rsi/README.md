# MetaRSI-derived workflow

This method implements the [MetaRSI operator/scheduler paradigm](https://arxiv.org/html/2609.06396v2#S4) on RSIComposer. The original paper's weights, training setup, and benchmark gains are **not** reproduced by this project. The default CPU preset is a mechanism test; the optional local Transformers path runs supervised LoRA, not the paper's full training recipe.

RSIComposer discovers this built-in algorithm through the package’s `register(registry)` entry; no `--plugin` flag is needed. Use `framework.method.name = "meta_rsi_model"` for model-authored scheduling, or `"meta_rsi"` for the rule-based reference. The model-driven method uses the stock Controller Harness for native cross-term controller adoption.

For the prompt-only, multi-term AIME experiment, see the [AIME protocol and run instructions](<AIME.md>).

## Run the CPU fixture

Run from the RSIComposer checkout root with Python 3.12+:

```bash
PYTHONPATH=src python -m rsicomposer run \
  --config src/rsicomposer/algorithms/meta_rsi/configs/cpu/meta_rsi.json \
  --data-dir runs/meta-rsi-fixture --no-monitor
```

Use a fresh run directory. No API key, weights, or training packages are needed. The arithmetic engine deliberately has staged capabilities: accuracy goes from 1/3 to 2/3 after the selected Domain Harness change, then to 1 after a **simulated** Model operator. The scheduler comparison starts both arms from the same snapshot: the incumbent Data-first arm gains 0, the Harness-first challenger gains 1/3 in the same one-proposal horizon. The Host adopts the challenger. The vertical proposal-profile and Controller Harness trials retain the incumbent when there is no extra gain or no change in trial actions. An isolated ablation (set `meta_targets` to `["controller_harness"]`) holds policy settings fixed and lets the evidence projection alone change the next action from Data to Harness. The bounded-ledger variant gains 1/3 over the recent-observation variant's 0 and is adopted in that ablation. These synthetic scores are not research results.

## What can actually change

| Surface | Mechanism | What a run can establish |
|---|---|---|
| RSI Policy | Scheduler order (horizontal), and per-operator proposal profile (vertical) | A config diff, two matched receipts, and a Host adoption only after positive excess next-round gain |
| Controller Harness | Projection of development evidence into the policy context: recent observation vs bounded ledger | A separately tested projection; no changes to memory backend, skills, tools, or permissions |
| Domain Harness | Bounded prompt template and optional use of verified training exemplars in that prompt | A different immutable candidate and a measured selection (or rejection); no tool, skill, MCP, or memory-mechanism change |
| Domain Agent | An adapter trained from verified trajectories on a fixed local base model | A positive optimizer-step receipt plus adapter digests in real mode; selection is separate, so training does not imply adoption |

The Data operator generates trajectories on the train cohort, verifies them with the fixed MathLab verifier, and deduplicates them. Neither evaluation labels nor Controller history are training data. A non-regressing Data change with more verified trajectories may be retained even when it has no immediate accuracy gain; this is an enabling state change, not an accuracy gain. Operators mutate only one of data, domain harness, or model. Search scores are diagnostic; development validation determines selection. A tie never adopts a Controller revision.

The parent captures an adapter-free `S1` after warmup. Each meta pair is executed sequentially through SelfReflection in isolated child Applications from the same candidate ID, with equal proposal horizon and allocated budget. The train cohort is the same (portable verified trajectories); child search/validation and parent search/validation/test are disjoint. The parent reads nested **Lab** receipts, checks protocol identity, normal finish, baseline equality, and work completion, and compares excess improvement. Its report separates child budget/observed wall time from task-operator measurements. The native `host.adopt_controller` operation installs a new Controller version with resumable parent state; a successful child job does not adopt by itself. Child applications have no sealed split and cannot recurse into SelfReflection. On normal finish, the Host alone scores baseline and selected candidates on the sealed test, after all decisions are closed.

This is a bounded implementation: the meta comparison excludes Model-RSI and adapter transfer, reuses a small development cohort for meta comparisons, and compares equal requested horizons, not identical compute per operator. Measured wall times include process overhead. The local LoRA engine does not use the provider's model-call/token metering; caps on rows, generated tokens, optimizer steps, and elapsed budgets should be set for the intended machine. A single tiny evaluation set cannot establish statistical significance.

## Run with real weights

1. Prepare the adjunct environment (e.g. an isolated Python 3.13 venv) with `torch==2.8.0`, `transformers==4.56.2`, and `peft==0.17.1` for the small CPU example. These are not required dependencies of RSIComposer.
2. Opt in to data preparation. This downloads and verifies the upstream sources from [the pinned OpenAI GSM8K commit](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c), then writes both manifests and provenance. Add `--source-dir` to use a pinned local upstream checkout instead. The destination must not exist.

   ```bash
   PYTHONPATH=src python -m rsicomposer.algorithms.meta_rsi.prepare_gsm8k \
     --output-dir data/meta-rsi-gsm8k
   ```

3. Place verified local Transformers files at `models/smollm2-135m-instruct-12fd25f/` (or override both Lab paths). The example uses [SmolLM2-135M-Instruct at `12fd25f77366fa6b3b4b768ec3050bf629380bac`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac). Copy regular files for config, tokenizer, and `model.safetensors` (SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`), not symlinks to a cache. The Lab fingerprints the resulting local model tree and does not fetch weights.
4. Run the [real GSM8K preset](<configs/gsm8k.json>) from the checkout root. Use absolute paths for the two manifests and model directories if launching elsewhere.

   ```bash
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m rsicomposer run \
     --config src/rsicomposer/algorithms/meta_rsi/configs/gsm8k.json \
     --data-dir runs/meta-rsi-gsm8k --no-monitor
   ```

### Checked small GSM8K pilot

A real CPU pilot used 16 shared training questions and three examples per development/test cohort (`--train-count 16 --dev-count 3 --sealed-count 3`). Data-RSI yielded no verifier-passing trajectories, so the Model operator was not eligible in that run. The three matched meta comparisons retained the incumbent. The sealed baseline and selected accuracies were both 0/3. This is a reproducibility finding for this small model and setup, not evidence of gains or evidence against the original method.

### Verify just the real adapter path

The [real LoRA smoke preset](<configs/cpu/real_lora_smoke.json>) uses the same pinned base but does **not** require GSM8K preparation. It seeds three labeled trajectories from the arithmetic **train** cohort, which the Lab re-verifies, and proposes one real LoRA update. It does not test self-generated data collection or run meta trials.

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m rsicomposer run \
  --config src/rsicomposer/algorithms/meta_rsi/configs/cpu/real_lora_smoke.json \
  --data-dir runs/meta-rsi-real-lora --no-monitor
```

On the checked CPU run, the worker completed one optimizer step (one microstep, seven completion tokens used, finite loss 2.72), wrote both hashed adapter files, and evaluated the adapter candidate on search and validation. Validation remained 0/3, so the incumbent was correctly kept. These are mechanism checks, not efficacy results.

If no generated trajectories pass verification, Model-RSI has no input and must not fabricate a training receipt. Check `controller_report.attempts`, nested `evidence_ref` entries, Host final receipts, and the checkpoint's `training.json` / `checkpoint.json` under the run's `meta-rsi/models/` directory. Insufficient evidence retains the incumbent. More samples, seeds, or a larger base model are necessary for a claim about generalization.
