# MetaRSI on AIME

This is a public reconstruction of the AIME part of [MetaRSI v2](https://arxiv.org/html/2609.06396v2#S5.SS1), not the authors' exact program or a result reproduction. The paper reports **55.0 → 68.3%** in one term on AIME and a **+3.3 percentage point** second-term meta advantage from the same released starting state. The five-term curve is a four-benchmark average, not an AIME curve. See [Table 3](https://arxiv.org/html/2609.06396v2#S5.T3) and [§5.4](https://arxiv.org/html/2609.06396v2#S5.SS4).

## Protocol and changes

| Component | Public reconstruction |
|---|---|
| Dataset | Training: bounded, verifier-compatible MATH counting/probability train questions. Four disjoint dev roles: 2022–24 AIME. Sealed: all **60** questions from 2025/26 I and II, grouped by exam; matched answer keys and pinned source hashes, with no contest questions vendored. One upstream variant ([2026 II #10](https://live.poshenloh.com/past-contests/aime/2026II/problem/10)) asks a different quantity; the preparer hash-checks the source and restores the wording and answer used by the published exam/key before verifying all four key digests. A separate pilot option explicitly marks the reduced test cohort. |
| Scorer | One greedy decode for each question, terminal boxed 0–999 integer, pooled accuracy. AIME diagram markup/ASCII is preserved verbatim; no rendering or image model is used. This is a fixed text reconstruction, not a claim of parity with the paper's renderer/decoder. |
| Data-RSI | Produce candidate trajectories on the frozen train cohort, verify answers, deduplicate, log identity and freshness. Dev and sealed answers never enter training. Opt-in affine synthesis constructs a new train-derived integer goal, tests if the target already knows it, and then requires two independent blind target-model answers checked against a derived numeric label. The author is a bounded transform, not the paper's open-ended model author. It does not check derivations, run semantic near-deduplication, or train on a consolidation lane. |
| Harness-RSI | Change system prompt and, when proposed, cite immutable verified train exemplars in that prompt. Tools, Skills, MCP, verifier, and memory are fixed. Dataset changes alone do not add dynamic exemplars to the current prompt. |
| Model-RSI | Require a fresh verified dataset for the current Prompt/adapter basis. Fit a fresh LoRA from the fixed base with assistant/completion-only loss, hash and evaluate the candidate, retain it only on improvement. The public backend's bounded recipe/profile choices are not the unknown paper recipes. |
| Five handoffs | D→H uses cited verified experience; D→M materializes cumulative train records after a freshness check; H→D and M→D require recompilation against the changed state; M→H can propose removing a pinned exemplar and runs a frozen development replay before offering the patch. Each handoff can decline when its preconditions fail. |
| Horizontal / vertical | Horizontal chooses among current admissible operations with the selected routing rule and budget. Vertical proposals change only a single operator's declared proposal profile (collection breadth, prompt/exemplars, or training steps), and are accepted only after a matched development probe beats the old profile. This is a bounded declarative rewrite, not an LLM freely rewriting each operator's instructions. |
| Cross-term meta | Finish and record a release on development data, then try the revised scheduler in a distinct dev-only child starting from that **same released state** with equal effective budget and proposal horizon. Only improved proposals get native Host adoption; the final Host-only sealed scores cannot train the next policy. Optional `--controller-harness` separately tests `recent` → `ledger` evidence projection (context assembly, not tools, long-term memory or the Domain prompt); leave it off for a scheduler-only contrast. |

A trained checkpoint eligible for transfer is matched to its content/recipe hashes and the same local base. No child receives a sealed split. A selected candidate must have a valid development measurement; score ties keep existing Prompt/Model/Controller choices. A non-regressing data-only change can still be retained when verified coverage grows. Every adopted target and observed gain should be read from the receipts, not inferred from the presence of a config or operator.

## Run

From the RSIComposer checkout root (Python 3.12+), the arithmetic fixture exercises the AIME campaign protocol but is **not** an AIME evaluation:

```bash
PYTHONPATH=src python -m rsicomposer run \
  --config src/rsicomposer/algorithms/meta_rsi/configs/cpu/aime_campaign.json \
  --data-dir runs/aime-campaign-fixture --no-monitor
```

To prepare the 60 official questions and disjoint dev cohorts, install PyArrow in a separate conversion environment or use `uv run --with pyarrow==23.0.1`. Omit `--pilot-sealed-count` for the 60-item benchmark. `--list-sources` shows the immutable source names and sha256s, and `--source-dir` accepts exactly matching offline files instead of `--fetch`. The manifest and provenance bound your local copy; upstream access does not grant permission to redistribute third-party problems or proofs.

```bash
PYTHONPATH=src uv run --with pyarrow==23.0.1 python \
  -m rsicomposer.algorithms.meta_rsi.prepare_aime \
  --fetch --output-dir data/meta-rsi-aime
```

Stage [Qwen2.5-Math-1.5B-Instruct at aafeb0fc](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct/tree/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35) as regular local files. Verify `model.safetensors` SHA-256 `08faf1e950173bb44c11d26df28c55da7237dc55c77dbc1cbae4a5d7584540d9`. Use a separate runtime with PyTorch, Transformers and PEFT (e.g. 2.8.0, 4.56.2, 0.17.1); the run does not download weights. This model predates the sealed exams' release and is materially smaller than the paper's **Qwen3.5-35B-A3B**. Dates and prompt deduplication do not rule out all contamination. The latter MoE requires an architecture-compatible accelerator backend and does not fit this CPU configuration.

```bash
PYTHONPATH=src python -m rsicomposer.algorithms.meta_rsi.aime_run_config \
  --bundle data/meta-rsi-aime --model models/qwen2.5-math-1.5b-aafeb0f \
  --run runs/aime-adaptive-23 --arm adaptive --terms 2 --seed 23 > aime-adaptive-23.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD/src" python -m rsicomposer run \
  --config aime-adaptive-23.json --durable --no-monitor
```

The CUDA backend can also load the paper's model family as a pinned local [Qwen3.5-35B-A3B checkpoint](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/2508b2437e382c605eb0fc52b83f5545960f6b4a). That public revision is a reconstruction choice, not an author-confirmed pin. On a Blackwell GPU with enough memory for the ~72 GB BF16 weights and runtime allocations, add `--device cuda --model-kind qwen3_5_moe --thinking --max-length 65536 --generation-tokens 4096 --generation-batch-size 2` when generating the config. Verify native sm_100 CUDA support and check inference, a finite adapter update/reload and truncation on **development only** before freezing the cap; 65,536 is the context window, not an instruction to generate that many tokens per item. The runner uses the MoE attention projections for assistant-only LoRA; no quantization or cache approximation is needed for the BF16 run. Continue with `--durable` and run a single `--seed 23` arm for initial validation.

On the same CUDA MoE checkpoint, `--rollout-backend vllm_graph --generation-batch-size 16` opts into single-GPU graph batching for inference only. The application interpreter must have vLLM v0.32, HF Transformers and PEFT; the Host still supervises each worker and trains the adapter with the same Transformers/PEFT code. Graph rollout uses stable per-request seeds and preserves the distinction between EOS and length or prompt-budget truncation. It is a separate decoding protocol, so preflight it on train/dev only, measure cold worker start and actual long decode, and freeze it for both sealed evaluations. Do not mix its results with a Transformers baseline or switch an existing run in place.

The default verified-trajectory storage limit is 4,096 characters, independently of the model's generation limit. For long reasoning runs, set `--max-completion-chars` explicitly (at most 131,072). The Lab stores the full verified completion, refuses truncated or wrong answers, and records how many completed answers exceeded the storage limit. Long rows remain available for SFT but are not inserted as prompt exemplars; the 4,096-character per-exemplar limit is unchanged. A dev-only `--arm model-after-data --terms 1 --steps 2` preflight can verify the D→M path without opening sealed scores. The LoRA trainer refuses sequences that would truncate the assistant answer.

For an interrupted durable run, use `python -m rsicomposer resume --data-dir runs/aime-adaptive-23 --no-monitor` with the same offline environment and local model. Do not change decode or split settings in place.

Run **adaptive** and **frozen** with the same model, data, seeds, device, budgets, and decoding. For a scheduler-only comparison pass `--scheduler-only` on the adaptive track: both tracks then use the same operator policy in term 1; verify that `terms[0].released_id` matches before comparing term 2. The internal matched forks also export their starting candidate ID and allocated budget; a fork with clipped/different allocation does not justify adoption. Do not select a seed, profile or arm using sealed scores. Each outer seed is a fresh run; changing a seed is not redecoding the same selected model.

Pass `--data-policy affine` on all arms in a synthesis comparison. The default `rollout` mode is useful to first verify LoRA and scheduling without requiring the smaller target to solve a newly transformed problem twice.

Additional controls: `reference` (no changes), `data`, `prompt`, `model-after-data`, `fixed-order`, `uniform-admissible`, and `diagnostic-router`. A literal empty-data “Model-only” arm has no dataset; `model-after-data` explicitly enables the D→M prerequisite. The paper does not disclose its Model-only bootstrap; the corresponding results are not interchangeable. Similarly this Data-only arm does not directly change inference. None of the controls makes a promise of identical realized tokens or verifier load.

## Read a run

Summarize aggregate-only results with:

```bash
PYTHONPATH=src python -m rsicomposer.algorithms.meta_rsi.aime_results \
  runs/aime-adaptive-23/result.json --frozen runs/aime-frozen-23/result.json
```

Without `--frozen`, this prints each campaign, per-exam counts, the baseline-to-selected sealed difference, actual task proposal changes, and each meta comparison's development advantage (null for unmatched or incomplete trials). With it, the tool rejects mismatched outer seeds or measured S1, and returns Term 2 **development** and final **sealed** advantages separately. Also verify the full input configs and dev protocol before treating those tracks as a compute-matched experiment.

For the full pooled experiment, add `--require-full-aime` to verify both real Host receipts contain all four 2025/2026 I/II exam cohorts with 15 scored attempts per exam. A pilot-sized or simulated result cannot satisfy this check.

For a budget-limited or interrupted result that did not seal successfully, pass `--dev-only` instead. It reports recorded development evidence without any held-out scores; the normal reporter still requires both complete Host evaluations. When reporting from the original run directory, add `--optimizer-evidence` to validate the trained parent candidates' adapter files and emit numeric loss, optimizer steps and supervised token counts. It does not expose the training rows. A checkpoint's `samples` counts eligible input rows, not necessarily the number consumed under a fixed optimizer-step budget.

- `controller_report.terms`: starting/released system, development scores, operator calls, elapsed wall time and measured task-operation/vertical-probe costs.
- `controller_report.attempts`: each actual proposal, transition, diagnostic basis, whether it trained and whether it became the incumbent. Absence of accepted trajectories means **no training**, not an optimizer failure or a zero-gain adapter.
- `controller_report.meta`: stage and term, specific config diff, both candidate receipts, allocated budget/realized usage and adoption reason. Matched development gains are not held-out accuracy.
- Host `host_finish` receipts: aggregate baseline/final results on sealed data. They are not exposed to the controller. Report per-exam counts, pass@1 and paired gains; use multiple outer seeds before generalizing. A four-item CPU pilot is only an integration check.
- Budget caps and actual metering are distinct. Local generation does not increment provider model-call counters; nullable token counters are **unmeasured**, not zero. A complete paper compute match would require the unreleased numerical ledgers, exact initial/updated policies, training/dedup recipes, and inference artefacts. Do not treat the published +13.3 or +3.3 points as targets guaranteed by this preset.

## Development-only training pool size probe

This probe is deliberately separate from the campaign's 24-row Experience limit and cannot load a sealed role. It tests one seed and one topic rather than changing topic coverage or enabling new-question synthesis at the same time. Prepare a private bundle with the pinned MATH counting/probability and historical AIME sources only:

```bash
PYTHONPATH=src uv run --with pyarrow==23.0.1 -m rsicomposer.algorithms.meta_rsi.prepare_aime_scale \
  --fetch --train-count 256 --output-dir data/aime-pool-scale-private
PYTHONPATH=src python -m rsicomposer.algorithms.meta_rsi.aime_scale_probe \
  --manifest data/aime-pool-scale-private/manifest.json --check-only
```

For the single-arm experiment, add `--wide-only` and a fresh output directory. It generates one answer per large-pool question and trains once on every distinct, correctly verified complete rollout that fits the storage and model-context limits, using all causal assistant labels. It measures only the trained adapter on historical development questions; no baseline or small-pool arm is launched.

For a four-GPU retry of a stopped, unscored single-arm run whose rollout stages have all completed, use a fresh output directory and add `--gpus 0 1 2 3 --rollouts-from <owned-run-directory> --global-batch-size 16`. Completed rollout content is revalidated and its hashes bound into the new plan; no scores, adapters, or optimizer states are imported. Each rank uses one full example per microstep; four DDP waves form a 16-example batch with loss normalized by the batch's total supervised tokens. Only the order within each batch is changed to put similarly long examples in the same wave. The final partial batch is retained when divisible by four ranks. The resulting adapter is scored on four disjoint 15-item historical dev shards through separate Host leases.

For an optional matched diagnostic, on B200 append the pinned local base and an output directory with `--model`, `--output` and the leased physical GPU index with `--gpu`; omit `--check-only`. Both arms get 256 sampled original-question rollouts: 32 per question on the same original eight training questions versus one per question on 256 distinct questions. Both require 48 distinct verified trajectories within the same label-length window. The large arm must actually train on 48 distinct questions; the small arm must actually cover all eight. Each arm starts fresh from the same base and performs exactly 48 optimizer updates and 49,152 causal assistant-label tokens, with deterministic within-completion label masking and unchanged prompt/response text. Input sequence length, sampled output tokens and actual GPU time need not match. The fixed scorer reports greedy numeric pass@1 and paired gains/losses on the same 60 historical 2022–2024 development questions, not on the 2025–2026 final test. Only the aggregate report is shareable; the stage directory contains private training material. An inadequate rollout yield or corrupt receipt fails closed rather than silently reducing the budget.

**Completed large-pool run (B200 physical GPUs 0–3, seed 23).** Of 256 distinct MATH counting/probability questions, 228 yielded complete verified trajectories; 27 hit the decode limit and one failed answer formatting. DDP trained once on each of the 228 full trajectories, consuming 2,760,753 causal assistant labels over 57 global microstep waves and 15 optimizer updates at global batch size 16 (final batch 4). The optimizer loop took 24.6 minutes. The four ranks ended with identical trainable weights, and 4 × 15 vLLM adapter evaluations on the historical development cohort scored **49/60 (81.67% pass@1)**, with 10 length-limited outputs. This is a single-arm historical-development result, not a gain estimate or a second evaluation of the 2025–2026 test. No baseline or small-pool adapter was scored.

## Wide-pool closed loop (development only)

The wide loop starts from the verified batch-16 adapter above, not a newly scored bare base. Both parent and nested policy trials share the same 256 train originals and the same 228 verified starting trajectories. The 60 already-scored historical problems are divided into disjoint domain search/validation (22/23) and nested meta search/validation (7/8); the latter are independent of the domain splits, but neither is an unseen final test. The two-term adaptive configuration has no sealed or standalone control arm.

The Data operator rechecks previous train rollout non-admissions on the current candidate. Fresh failures nominate bounded affine transformation tasks with reference answers computed from the original numeric verifier. An item already solved greedily is skipped; otherwise two separate same-model samples must both pass the numeric verifier before admission. Recovered original trajectories are counted separately from admitted new items. This is not the paper’s open-ended, model-authored capability taxonomy and synthesis. The Prompt operator proposes a system instruction and eligible short verified examples. The Model operator fits fresh attention LoRA on cumulative verified rows at global batch 16; the nested model policy comparison changes the learning rate, not the batch or supervision. Policy and schedule challengers are adopted by the Host only when the matched nested trials show a validation improvement.

Prepare the dedicated bundles with `python -m rsicomposer.algorithms.meta_rsi.prepare_aime_wide_loop`, emit a run config with `python -m rsicomposer.algorithms.meta_rsi.aime_wide_loop`, and invoke the ordinary `rsicomposer run --durable --no-monitor` with the pinned local model and physical GPUs 0–3. The optional `rsicomposer.algorithms.meta_rsi.aime_wide_loop_audit` module exports aggregate-only evidence from the result, the Host’s selection/activation journals and optimizer checkpoints; `--live` reads durable progress without resuming the run or launching workers.

The score gate is intentionally strict: a pure Data proposal can survive a validation tie if it appends verified rows, but not a measured regression. A fresh Model fit is eligible only from a current candidate carrying a bound data-refresh proof. A nested policy comparison is discarded if the arms measure different starting scores even for the same candidate and cohort. Record such mismatches separately from effects of a prompt or adapter change; fixed greedy decoding alone does not assert bitwise reproducibility.

An optional `--model-stage-from` continuation is only for a completed, separately preserved campaign that produced no Model fit and retained its exact starting system. It stages the earliest integrity-checked verified Data candidate from that run without claiming it was selected, trains one alternate LoRA from the same base with optimizer seed 23 at batch 16, and measures it on the same historical development roles. The continuation does not promote its measured challenger or open any sealed role; it is not a replacement for the original campaign or a standalone baseline.

## Model-driven MetaRSI (development only)

Add `--model-driven` to the wide-loop configuration command to select the registered `meta_rsi_model` method instead of the rule reconstruction. It uses the same pinned target (and the current selected adapter when present) for structured decisions, bounded non-thinking proposal generation, solver rollouts and blind Anchor. The original rule entry point and the isolated model stage remain separate.

- The controller sees bounded, development-only signal and declared capabilities. It chooses a typed horizontal action/program, a vertical operator rewrite, or a stop. Each planned step is independently rechecked against current artifacts and remaining Host budget. A patch specifies its exact before/after diff and cannot edit protected evaluation, release, resource or artifact settings.
- The domain agents re-run observed train gaps and can author new tasks in a versioned, independently solvable counting family. A private reference solver constructs the checkable answer; the current solver and both independent blind draws see only the problem. Prompt proposals can revise short solving guidance and cite vetted train exemplars. Model proposals choose a bounded LoRA recipe. These typed AIME proposals do not purport to recreate the undisclosed open-ended author/diagnosis stack.
- A verified Data-only proposal stays staged without a standalone score or release. Its precise state hash and fresh dataset proof are passed to Model. Only the trained system is compared with the measured incumbent on fixed development. A later Prompt or Model change invalidates stale training data. Full-row coverage, optimizer seed 23 and global batch 16 remain Host constraints in the wide preset.
- After each term is released, the outer agent reads the complete term summary and can propose a bounded scheduler patch. Activation goes through the native Host Controller version API. A contract-valid activation is not evidence of a measured score gain; report actual quality of released systems separately. This method does not mount any sealed evaluation, does not read earlier sealed data, and does not launch a standalone baseline.

## Original MetaRSI framework (arXiv v2 public specification)

Use `--paper-kernel` in place of `--model-driven` when rendering a new wide-loop configuration. This is an opt-in protocol, not a migration or a resume of an existing run. The arithmetic adapter mounts development only and publishes no standalone Data score. The general kernel is callable independently of AIME; a domain supplies signal compilation, verification, execution and selection. It does not impose thresholds, retries or routing preferences.

| Mechanism | Executable contract |
| --- | --- |
| Common kernel | The executable domain kernel runs Observe → Diagnose → Propose → Validate → Execute → Select → Export with one proposal/adjudication boundary. Operator cards declare one writable surface. Selected changes recompile the signal, and audit records bind actual compiled evidence. Data Export means verified input, not a capability score or deployment. |
| Typed composition | Runtime handoffs bind Experience→HarnessPatch, Dataset→masked shards and recipe, Genome→Signal, Checkpoint→Signal and learning signatures, and TrainingReport→HarnessPatch. The D→D and H→H self-loops use the learning signal directly; H→M and M→M are invalid. Freshness is checked against the live capability binding, not a hash from a former model. |
| Two axes | A horizontal block declares purpose, expected evidence, per-step allocation and stop intent. A vertical directive scopes a single operator policy and binds its requested surfaces, contract ID and exact diff. The kernel does not force Data→Model or auto-resubmit a rejected model reply. |
| Diagnose and Data | The same target model ranks typed hypotheses over aggregate observed failures; the system does not infer the cause of an incorrect answer from a numeric score. A proposed subset-residue task is canonicalized and solved by a private reference enumerator. In this mode Data revisits all still-unverified original train rows from the frozen source pool, not a fixed eight-item operator sample. Original recoveries, frontier admissions and independently already-held consolidation rows are separate. Two independent blind samples must pass the numeric and executable histogram checks for frontier credit. The checker does not validate arbitrary freeform reasoning. |
| Harness | The portable genome has prompt, tools, skills, MCP and policies (including memory). The AIME adapter enables only prompt changes and replays its candidates on a common failure shard before independent validation. A typed M→H reconciliation may keep a pure removal that holds validation accuracy. |
| Model | A bounded set of up to two fresh LoRA recipes starts from the same fixed base and exactly the verified staged data. Two recipes compete on frozen development search; only the winner sees validation. The wide worker preserves curriculum membership per batch of 16, the uneven tail, the optimizer seed and full row coverage. Independently already-held rows have separate retention observations; these are training-set monitoring, not generalization evidence. |
| Release and Meta | Independent inner selection is separate from the end-of-term Host deployment verdict. Meta sees the completed term, retained or rejected attempts, and resource use; it can change only scheduler policy (diagnosis, routing, budget, stopping, axis and guidance for fixed adapter endpoints). The outer role itself is fixed. The AIME runner ships with no additional regression cap or forced routing. |

The framework is domain-independent; AIME is the development adapter and a non-AIME in-memory fixture verifies portability. The paper's arXiv v2 calls the method MetaRSI-v1. The released description does not specify the exact task generator, diagnosis taxonomy, budgets, recipe portfolio, deduplication implementation, term prompts or full benchmark runner. AIME task families, exact training recipes and experimental metric weights are domain or reproduction choices, not the authors' unpublished internals. This mode does not implement an executable benchmark agent for Terminal-Bench/SWE-bench or claim any paper score, unpublished internal equivalence, or a run with 512 questions. No earlier sealed cohort is read or used for novelty checking.

## Checked mechanism fixtures

- The bundled arithmetic fixture selects verified train rows, a system-prompt patch and a **simulated** adapter in Term 1; development accuracy progresses from 1/3 to 3/3. Term 2 starts from the exact release. It exercises M→H replay and rejects the changed prompt on a tie. Its matched vertical and cross-term trials retain the incumbent when neither arm earns extra development gain.
- A separate one-proposal-per-term positive control creates a strict matched gain by trying the prompt first after Term 1 collects data. Host adopts the alternative policy and Term 2 actually executes Harness-RSI instead of Data-RSI. This demonstrates policy change, not improvement on AIME data.
- An earlier real local LoRA smoke completed one optimizer step on verified arithmetic completions and then rejected the adapter on a validation tie. It is evidence that the backend trains, not evidence of a released AIME adapter.

## Checked B200 development qualification (seed 23)

The vLLM graph protocol used the pinned 35B BF16 model, 65,536-token context, 32,768 new tokens and batch 16. Its eight-item parent validation returned **5/8**, with one length truncation. With the legacy storage bound, 23/24 training generations passed answer verification but none fit in 4,096 characters. With the qualified long-storage option, all 23 verified completions entered Experience; one adapter finished one real optimizer update using 9,742 assistant-label tokens and reloaded for validation. Its validation score tied the parent, so it was not selected. The two-term scheduler qualification compared a Data-first and a System-Prompt-first action, each allocated 2,700 seconds starting from the same candidate. Both arms scored **7/8 → 7/8** on the separate meta-validation cohort; the scheduler was not changed. These are development-only results, not sealed AIME efficacy.

## Completed B200 sealed AIME (seed 23)

The first frozen Transformers attempt exhausted its 12-hour budget before opening the held-out split. After two nonsealed qualifications, the same seed and pinned BF16 Qwen3.5-35B-A3B revision ran the frozen vLLM graph protocol (65,536 context, 32,768 output cap, batch 16), with two terms of five proposals. Host successfully recorded both 60-item greedy numeric pass@1 evaluations; the campaign took 7.30 wall hours (6.97 GPU lease hours).

| Sealed exam | Baseline | Selected |
| --- | ---: | ---: |
| 2025 AIME I | 10/15 | 11/15 |
| 2025 AIME II | 13/15 | 11/15 |
| 2026 AIME I | 14/15 | 11/15 |
| 2026 AIME II | 12/15 | 11/15 |
| **Pooled** | **49/60 (81.67%)** | **44/60 (73.33%)** |

The selected release lost **8.33 percentage points**. The baseline had 9 truncated completions and one invalid-format attempt; the selected release had 13 truncated completions and no invalid-format attempts. Truncated attempts count as incorrect; the higher truncation count alone does not establish the cause of the regression.

- **RSI Policy:** Host installed the Model proposal profile `compact → detailed` (1→2 optimizer steps, +12.5 points against its matched independent development arm) and the schedule `Data/Harness/Model → Model/Data/Harness` (+12.5 points against its matched cross-term arm). Term 2 actually started with Model-RSI. Its later Model-profile comparison was excluded: the two evaluations of the same starting candidate returned 8/8 and 7/8.
- **Controller Harness:** `recent → ledger` was not installed. Matched arms both scored 7/8 with identical actions.
- **Domain Harness:** Experience grew from 0 to 24 verified completions; System Prompt remained at the seed instruction with no exemplars. No tool, skill, MCP or memory-slot changes were made.
- **Domain Agent:** Host released the Term 1 LoRA after 2 real optimizer steps and 44,360 supervised assistant tokens (token-weighted loss 0.2538). The parent development score improved 5/8→6/8. Term 2 trained three more adapters but none replaced that release; development stayed 6/8.

This run demonstrates actual adaptation and selection, but not improved held-out accuracy or reproduction of the paper's five-seed gain. No held-out score was used to select or modify the protocol.

## Checked real CPU diagnostic

Qwen2.5-Math-1.5B-Instruct at the pinned revision, seed 23, two terms, one task proposal per term and one per matched scheduler arm, 24 generation tokens; two training questions, one per development role, and four sealed AIME questions (one per exam). Every problem was scored with one fixed greedy decode. The two term releases kept the initial candidate at 0/1 development accuracy. The matched trials tested different first proposal actions with the same released candidate and allocation, both gained zero, and the revised schedule was not adopted. Meta trials took 86.2 seconds combined as observed by the parent. No new trajectory passed the verifier, so no AIME LoRA training ran. Host scored baseline and selected at **0/4 → 0/4**; all four outputs hit the generation limit in each assessment. This checks real weights, orchestration, and sealing, not AIME solving quality.

A separate 160-token, three-proposal-per-term run completed its first term and first nested arm, but did not complete the matched pair or reach sealing. The parent hit a local worker-completion publication race; worker terminal JSON is now published atomically and late publication around the PID probe is covered by regression tests. An unfinished arm contributes no gain. The full 60-item/five-seed efficacy matrix has **not** been run.


