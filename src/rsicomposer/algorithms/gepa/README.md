# GEPA

`gepa` runs the pinned optimizer from the authors' paper artifact. It selects
parents, builds reflection prompts, proposes instructions, and decides which
candidates advance. The Controller connects its model calls and checkpoints to
RSIComposer. The MathLab runs and scores the authors' DSPy CoT predictor and
returns evidence. The host evaluates the initial and selected instruction on
sealed AIME 2025 after search finishes.

The [AIME preset](configs/aime.toml) uses the authors' 45 search and 45 validation
problems and the 1,839 metric-call stop threshold. It uses a minibatch of 16
instead of the paper's 3. Final
evaluation samples each of the 30 AIME 2025 problems five times per instruction.
`use_merge` is a setting on the same optimizer; the preset uses `false`.

From the repository root, use Python 3.12 and place these upstream checkouts at
the paths configured in the preset:

| Checkout | Commit |
| --- | --- |
| `gepa-ai/gepa-artifact` | `cbefbc1aa0f43dd39874ec4bf42211365dbda42e` |
| `gepa-ai/dspy` (`gepa_study`) | `62dc3b634d7dc0c4889abcf905bb4c391ea6b396` |

Install the pinned DSPy checkout in the experiment environment. Configure a
Chat Completions endpoint to serve the model named in the preset, with
`top_k=20` on the server and room for a 16,384-token completion plus its prompt.
Reflection packs the batch's traces into one prompt; a 32k context can be too
short for this preset.

```sh
python -m pip install .local/upstream/dspy-gepa-study 'datasets==3.6.0' 'litellm==1.74.9' 'wandb==0.19.10'
PYTHONPATH=src python -m rsicomposer.algorithms.gepa.prepare_aime \
  .local/gepa-aime-paper/data --artifact-dir .local/upstream/gepa-artifact \
  --dspy-dir .local/upstream/dspy-gepa-study
PYTHONPATH=src python -m rsicomposer run --durable \
  --config src/rsicomposer/algorithms/gepa/configs/aime.toml --data-dir runs/gepa-aime
```

This run uses fresh Qwen3.5-9B samples from vLLM instead of the paper's Qwen3-8B
and Arbor/MIPRO cache. The paper checks its metric threshold between iterations
and excludes skipped reflection batches from that counter. RSIComposer records
all Lab evaluations separately and bounds them with `max_lab_calls`.

Resume with `rsicomposer resume --data-dir runs/gepa-aime`. The Controller
replays committed evaluation receipts and reflections to reconstruct optimizer
state. Checkpoints from earlier GEPA method versions require a new run. The
current benchmark binding includes `max_model_calls`, since the budget can
affect whether every example is evaluated.
