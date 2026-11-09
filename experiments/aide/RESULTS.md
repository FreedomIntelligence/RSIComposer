# AIDE registered integration: 2026-11-09

The native AIDE Agent completed a real, fixed-model GSM8K program-search loop
through the registered `aide` controller and `aide_program` Lab. This validates
the integration and this task adaptation. It does **not** reproduce AIDE's
original tabular benchmarks or establish its paper's reported gains.

## Protocol and model

- Upstream `WecoAI/aideml` revision:
  `ad73d227af481bea3de06f884f41b2993dd6453a`.
- Built-in registration, no external plugin. Five real Lab receipts completed:
  three validation executions and two final heldout evaluations.
- Same local checkpoint `Qwen3.8-27B-8c0b3d30` for research/review and solving;
  its config declares `Qwen3_5ForConditionalGeneration`. No weight training.
- Frozen disjoint prefixes: 8 public labeled examples, 32 validation questions
  and 64 heldout questions. Private labels remain outside candidate execution.
- Seed 73; three search steps, one initial draft, one fold. Upstream defaults
  are a larger 20-step search, five initial drafts and five folds. This reduced
  recipe is an integration run, not an equivalent benchmark configuration.
- Four calls per question maximum; 1,536 output tokens per call. Research calls
  allow 8,192 output tokens. Generated code can also answer questions directly.

Native search produced **draft → improve → improve**. Both improvements chose
the first draft as their parent. Upstream Journal selection chose node
`7e92adc2e84d45a29ba3d7c0b474401d` (step 2). The maintenance agent did not edit
candidate strategies, choose the winner or use heldout scores to guide search.
All three native reviewer metrics matched independent host validation scoring.

| Evaluation | Correct | Accuracy | Model calls | Total tokens | Execution seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Draft validation | 14/32 | 43.75% | 15 | 10,793 | 9.03 |
| First improvement validation | 13/32 | 40.625% | 56 | 39,817 | 47.12 |
| Selected improvement validation | 19/32 | 59.375% | 42 | 33,828 | 22.19 |
| Initial program heldout | 28/64 | 43.75% | 29 | 19,552 | 5.32 |
| Selected program heldout | 34/64 | 53.125% | 67 | 48,762 | 14.63 |

The heldout difference is **+9.375 percentage points** on one seed and 64
questions. The selected program also spent more inference: 67 versus 29 calls
and 48,762 versus 19,552 tokens. This is not an equal-compute comparison or a
statistically established improvement. Execution times cover each candidate's
Lab execution, not the complete research-agent latency. Token counts are the
serving API's reported total tokens.

## GPU use and service compatibility

The run used GPU indices 4–7 on the user's eight-card B200 machine; the device
query reported NVIDIA B20Z. Four independent vLLM 0.32.0 replicas used 24,576
context tokens and a 0.70 memory-utilization limit. Across 77 samples, roughly
two seconds apart over 162.65 seconds, utilization averaged 74.0%, 21.83%,
34.61% and 35.73%; each GPU peaked at 100%. Planning/review is sequential on
the first endpoint; candidate calls round-robin across all four replicas.
The short sequential search did not sustain full four-card utilization.

Initial failures are retained alongside the successful evidence:

1. The first launcher omitted the serving environment's bin directory from
   PATH, making `ninja` unavailable. The launcher now preserves it.
2. `run-v1`: Qwen rejected native system-only prompts. The local transport moves
   the exact prompt text to a user turn and explicitly disables thinking.
3. `run-v2`: native forced `submit_review` required a tool parser.
4. `run-v4`: in vLLM 0.32.0, the parser also required
   `--enable-auto-tool-choice`. An independent forced-tool probe passed before
   `run-v5`, using `qwen3_xml` tools and `qwen3` reasoning parsing.

These are explicit interface adaptations. Upstream search, review schema,
parent choice, Journal and final selection remain native.

## Evidence and checks

[validation-20261109.json](validation-20261109.json) records the metrics,
candidate/source hashes, upstream revision, serving configuration and evidence
digests. Raw artifacts remain ignored at:

```text
b200:/home/snchen/xinyuan/RSISys/.local/meta-agents-20261109/aide/
```

The real run is `run-v5`, run ID `c6be4bfb3b3741e1b13929bbf19cca52`.
Its immutable `source-v5.tar.gz` SHA256 is
`468b866a1553ce2ca40a26c7e81f36ba87d136f27ceec4339691c32477c4809f`.
Evidence includes the host result, native Journal, complete LLM request/response
log, generated source, Lab receipts/submissions and GPU telemetry.

The original run snapshot passed 11 native adapter tests without skips. Final
hardening passed **16 Linux native adapter tests, zero skips**, from immutable
`source-final6-aide.tar.gz` (SHA256
`87c92fa9b4811ff7506b404cc026f1b7d77144e207b5cd9c6dcfa7b66f1abde1`).
The local suite passed 15 tests with the Linux native test skipped; fatal Ruff
checks also passed. Final regressions cover failure reporting, simulated-receipt
rejection, mutated-candidate metrics and process-group cleanup. The suite runs the real upstream Agent with a
scripted endpoint and real Linux bubblewrap, covering draft → improve → debug,
native selection, separate final scoring, inaccessible private labels and host
TCP ports, immutable identity, budgets and worker cleanup. These scripted tests
validate interfaces, independently of the real GPU run's model behavior.

The owned four-replica serving launcher was stopped after collection. GPUs
4–7 were verified at 0 MiB and 0% utilization; no owned serving/worker process
remained. The other method's GPUs 0–3 were left untouched.
