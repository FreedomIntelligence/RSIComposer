# ReST^EM

This adapter runs rounds of sampling, filtering, and training on model-generated
solutions. Start with the single-GPU GSM8K preset, or use the eight-GPU preset
when inference and training need separate placement.

## Connect training and evaluation data

The Lab is a [MathLab](../../labs/math/README.md), so its data is a math dataset
manifest with a `train` split (problems to sample from) and a `dev` split
(measured every round). Start from [`configs/gsm8k.json`](configs/gsm8k.json) for
a native training pilot: set `framework.method.config.controller.initial_model`
to the local base model revision and `framework.method.config.labs.math.manifest`
to your manifest. The Controller's `train_data` and `eval_data` name the splits
(`train` and `dev`).

```json
{"schema_version": 1, "splits": {
  "train": {"file": "train.jsonl", "sha256": "<sha256sum of the file>", "count": 7473,
            "verifier": "marked_numeric"},
  "dev":   {"file": "dev.jsonl", "sha256": "...", "count": 500, "verifier": "marked_numeric"}}}
```

Each row has a unique string `id`, a `prompt` and an `answer` the split's verifier
accepts; the [fixture manifest](configs/cpu/data/manifest.json) shows the numeric
answer format. The verifier is chosen per split: `strict_numeric` (the CPU
fixture), `marked_numeric` (GSM8K) or `math_verify` (the MATH presets). MathLab
rejects a problem that appears in both splits.

`few_shot_data`, for completion prompts, is a separate example file in the same
row format; relative paths start at the working directory. The GSM8K preset uses
CUDA and its Transformers engine; prepare the corresponding model and training
environment before running. The [eight-GPU preset](configs/gsm8k-8gpu.json) shows
process and device placement.

```sh
rsicomposer run --config path/to/rest-em-gsm8k.json --data-dir runs/rest-em-gsm8k
```

Use the configured dev split to compare the baseline and trained candidates.
This preset does not schedule a sealed test evaluation. To report a final
benchmark score, measure the selected model on a separately prepared test set
with the intended evaluation protocol.

## Select a trainer

The Lab's `engine` selects its rollout path and the default trainer. Keep the
same rollout path and override only training by selecting `trl_sft`:

```toml
[framework.method.config.labs.math.trainer]
name = "trl_sft"

[framework.method.config.labs.math.trainer.config]
trl_version = "1.17.1"
```

Use this with `engine = "transformers"` or `engine = "vllm"` and set the Lab's
`training_python` to a separate environment with the pinned TRL release, PyTorch,
Transformers, Datasets, Accelerate, and PEFT. Allocate one GPU to `math.trainer`.
The trainer applies loss only to accepted completion tokens and saves a PEFT
adapter; the next evaluation loads that adapter over the original model. The
candidate records the actual training package versions. This adapter does not
support distributed SFT or optimizer-state resume.

To use veRL's FSDP2 SFT trainer instead, set `training_python` under
`framework.method.config.labs.math` to the interpreter in a separate veRL
environment. Replace the trainer selection with:

```toml
[framework.method.config.labs.math.trainer]
name = "verl_sft"

[framework.method.config.labs.math.trainer.config]
verl_version = "0.9.1"
```

Allocate one CUDA GPU to `math.trainer`. Start with `train_batch_size = 1`
and `gradient_accumulation_steps = 1`: their product must divide the number
of accepted examples so the trainer cannot silently drop examples. The veRL
worker verifies the installed version, trains only on completion tokens, and
exports a PEFT adapter that the existing rollout engine can evaluate. The
receipt records the installed versions, training settings, and adapter file
digests. This path does not use veRL's RL trainer or resume optimizer state.
