# FT-Agent registered reference adaptation: 2026-11-09

The registered `ft_agent` controller and `llm_finetune` Lab completed three
native LLaMA-Factory SFT experiments, native feedback/selection and independent
final inference on the selected merged checkpoint. This validates the local
reference adaptation, **not** the full FT-Dojo implementation or paper gains.

## Fixed protocol

- RD-Agent: `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`.
- LLaMA-Factory v0.9.3: `ca75f1edf3cb50343ed1c98605141c3e22075b5f`.
- Target: local Qwen2.5-7B Base, `Qwen2ForCausalLM`.
- Researcher: local `Qwen3.8-27B-8c0b3d30` directory; its config declares
  `Qwen3_5ForConditionalGeneration`. This directory name is not an independently
  verified public model identifier. The same researcher served AIDE.
- Frozen GSM8K prefixes: 64 training, 32 development and 64 heldout questions,
  selected in existing order before evaluation; split hashes are in the
  [machine-readable evidence](validation-20261109.json).
- Every experiment trains from the configured base model. Native hypothesis,
  data/config generation, acceptance and FTTrace parent/SOTA selection run in
  the optional worker. The host does not modify generated strategies or choose
  checkpoints using heldout scores.
- Identical Qwen user/assistant evaluation template for base and trained models;
  greedy generation, 512-token cap, fixed EOS stops and Math-Verify 0.9.0.
  GSM8K's final `####` numeric gold is converted to equivalent boxed notation
  only for scoring; original training reasoning is preserved.

## Observed results

| Model | Development | Native acceptance | Parent experiment |
| --- | --- | --- | --- |
| Base | 24/32 = 75.00% | Baseline | — |
| Experiment 1 | 27/32 = 84.375% | Accepted | Base |
| Experiment 2 | 29/32 = 90.625% | Accepted | Experiment 1 |
| Experiment 3 | 22/32 = 68.75% | Rejected as SOTA | Experiment 2 |

Native FTTrace selected experiment 2. Only after selection, the Lab evaluated
the base and selected model on the frozen heldout split:

| Heldout evaluation | Correct | Accuracy |
| --- | --- | --- |
| Base | 48/64 | 75.00% |
| Selected experiment 2 | 45/64 | 70.3125% |

Heldout accuracy **decreased by 4.6875 percentage points**. This single-seed,
small-split run does not establish a general algorithmic effect. Development
improvement did not translate into heldout improvement; small development-set
selection and the adapted training/task budget are plausible explanations,
not isolated causal findings. It is insufficient to attribute a failure to
reproduce paper gains to the original paper.

All eight real Lab receipts completed successfully: baseline, three preparation
and training pairs, and final evaluation. Each experiment completed a separate
one-step native execution check and 32 full-training optimizer steps on three
GPUs. Merged bfloat16 weights changed numerically in 1,691,633,994;
2,484,656,284; and 3,077,340,443 elements, respectively. Final inference loaded
the selected merged checkpoint. LoRA ranks 64/64/128 and alpha 32/64/128 were
proposed by the native research policy. Total run time was about 547 seconds.

GPU0 served research; GPUs1–3 performed native DDP training and independent
inference. Continuous GPU telemetry was not captured for this final run because
the monitoring launcher produced invalid Python. Native world-size and worker
artifacts verify device participation; sustained utilization is not established.
All owned services were released afterward, with all eight devices at 0 MiB.

## Earlier failures and fixes

The ignored evidence directory preserves unsuccessful attempts as well:

- `run-v2`: incomplete dataset descriptions caused native scripts to filter all
  rows. The adapter now supplies actual schema/examples and processing logs.
- `run-v3`: the named-definition loader omitted the `yaml` dependency; fixed
  and covered by native configuration/feedback tests.
- `run-v4`: scientific-notation YAML strings were rejected as nonnumeric; they
  now normalize to the identical numeric value before validation. One candidate
  also generated an empty dataset. Its last native training process stalled
  after best-checkpoint loading and was stopped; the root cause was not isolated.
  The final run's three native best-checkpoint paths completed normally.
- `run-v5`: native SFT and numerical merge succeeded, but the adapter compared
  boxed model answers with unconverted GSM8K gold, invalidating search metrics.
  This run was stopped and explicitly marked invalid. Real-verifier positive and
  negative regression checks now precede the corrected `run-v6`.
- Earlier raw-prompt evaluations differ from the final fixed Qwen chat protocol
  and are not pooled with final results.

These were local implementation or execution issues. No generated hypothesis,
data script, training schedule or acceptance decision was manually repaired.

## Reproducibility and limits

Raw evidence is ignored under
`b200:/home/snchen/xinyuan/RSISys/.local/meta-agents-20261109/ft-agent/`.
The valid run is `run-v6`, ID `8b33633fc1a644c785f708f4b21892df`.
Its immutable source archive SHA256 is
`184f9cc8dd856de89d98e6623b17ca081d135d81108bbf330351e3044d292132`.
The final source adds report fields and shared worker-cleanup hardening after
that snapshot; those changes do not change the recorded policy or evaluation.

Training used Python 3.12.3, Torch 2.7.0+cu128, Transformers 4.52.4,
PEFT 0.15.2, TRL 0.9.6, Datasets 3.6.0 and Accelerate 1.7.0 in an isolated
training environment. Native policy and sandbox tests passed **9/9** on Linux,
including the actual optional Math-Verify adapter and pinned upstream methods.
The complete FT-Dojo provisioning, CoSTEER knowledge/debugging pipeline,
OpenCompass benchmark and meta-agent weight training are outside this recipe;
see the [adaptation boundaries](README.md).
