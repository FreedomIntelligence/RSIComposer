# GPU Lab template: connect a dataset and a benchmark

The [arithmetic Lab](lab.py) implements `evaluate` using GPU inference.
`few_shot` controls how many rows from `train` appear in the prompt. The Lab
scores the model's answers on `dev`. The template keeps model weights fixed.

## Connect a GPU model

The templates use a GPU-backed Chat Completions endpoint. The Lab sends a
[`GenerationRequest`](../../loops/contracts/model_service.py) through the host
model service; the backend and model are selected in the run config.

If you already have an endpoint, set `backends.gpu.endpoint` (base URL ending in
`/v1`), `roles.actor.model_id`, and `roles.actor.revision` in the run config.
If the server requires a key, set `backends.gpu.credential_env` to the name of
its environment variable. The GPU can be on a separate serving machine.

Example on one GPU with vLLM and a local Qwen3.5-9B snapshot:

```sh
MODEL_DIR=/path/to/Qwen3.5-9B-snapshot
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name template-model --host 127.0.0.1 --port 18098 \
  --language-model-only --enforce-eager --max-model-len 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.30 --default-chat-template-kwargs '{"enable_thinking":false}'
```

Leave the server running in a separate terminal. The template configs already
point to this endpoint and model name. Replace `local-snapshot` in the config
with the actual model revision. Adjust the memory limit for your GPU and model.

## Copy and configure the Lab

1. From the repository root, copy the Lab with
   `cp -R src/rsisys/labs/_template ./my_lab`.
2. Register a Lab name in [`__init__.py`](__init__.py) and select it in
   [`configs/run.json`](configs/run.json) under `framework.labs.math.name`.
   `math` is the run-local alias used by the template Controller. Give the
   matching scene registration the same name if a method factory will resolve it.
3. Set `framework.labs.math.config.datasets.train` and `.dev` to your input
   files. The template resolves relative paths from the current directory.
   To use the copied fixtures, set the paths to `my_lab/data/train.jsonl`
   and `my_lab/data/dev.jsonl`. See the [training](data/train.jsonl) and
   [development](data/dev.jsonl) inputs for the JSONL format:

   ```json
   {"id": "dev-1", "question": "What is 34 plus 18?", "answer": 52}
   ```

4. Run the copied Lab with the template Controller:

   ```sh
   rsisys run --plugin ./my_lab --plugin src/rsisys/algorithms/_template \
     --config ./my_lab/configs/run.json --data-dir runs/my-gpu-lab
   ```

Replace the loader, model call, and scorer as needed for your domain. Keep the
operation names, arguments, and receipt fields if reusing an existing Controller.
The Controller calls `math.evaluate`; the Lab receives `evaluate` and returns
its declared primary metric (`accuracy` in this example).

Keep train and dev disjoint. For blind final evaluation, declare a `SEALED`
split and implement `final_evaluations()` to give the host a final-evaluation
plan; see [Coding Meta-Agent’s Lab](../../algorithms/coding_meta_agent/lab.py).
The template declares its inputs in `data_bindings()`: one
[`DatasetBinding`](../../loops/contracts/data.py) per split, with the split role,
source path, format version, content hash, and row IDs. The host records the
bindings with the Lab's identity and rejects a resume with changed inputs. For a
`sealed` split it stores only a fingerprint. Keep the binding's `role` equal to
the role that `split_roles()` declares for that split.
For larger benchmarks, store detailed records as artifacts instead of returning
all rows in the receipt. Record the benchmark revision, scorer, and settings
alongside model results.
