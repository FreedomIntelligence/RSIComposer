# ReST^EM

This adapter runs rounds of sampling, filtering, and training on model-generated
solutions. Start with the single-GPU GSM8K preset, or use the eight-GPU preset
when inference and training need separate placement.

## Connect training and evaluation data

Start from [`configs/gsm8k.json`](configs/gsm8k.json) for a native training pilot.
Set `framework.method.config.controller.initial_model` to the local base model
revision. Set `train_data` and `eval_data` in the same section to separate JSONL
files. Each row has a unique string `id`, a `prompt`, and a string `answer`
parseable by the Lab's verifier. The [train](configs/cpu/data/train.jsonl) and
[dev](configs/cpu/data/eval.jsonl) fixtures show the numeric answer format.

The `math` Lab resolves relative input paths under `framework.method.config.labs.math.data_root`
(default: the working directory). If you use `few_shot_data` with completion
prompts, point it at a separate example file using the same row format. The
GSM8K preset uses CUDA and its Transformers engine; prepare the corresponding
model and training environment before running. The
[eight-GPU preset](configs/gsm8k-8gpu.json) shows process and device placement.

```sh
rsisys run --config path/to/rest-em-gsm8k.json --data-dir runs/rest-em-gsm8k
```

Use the configured dev split to compare the baseline and trained candidates.
This preset does not schedule a sealed test evaluation. To report a final
benchmark score, measure the selected model on a separately prepared test set
with the intended evaluation protocol.
