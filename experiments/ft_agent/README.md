# FT-Agent: registered LLM fine-tuning research

This is a **local reference adaptation** of Microsoft RD-Agent's FT-Agent,
not a reproduction of the FT-Dojo paper's performance or complete infrastructure.
The controller runs the pinned upstream hypothesis generator, native code-generation
methods, feedback acceptance and FTTrace traversal. Each data-preparation or
training request pauses at the framework's controller → Lab → feedback boundary.
The host does not choose a hypothesis, repair generated strategy, accept a
candidate or select a checkpoint using test labels.

## Components and scope

- Registered controller: `ft_agent`; registered Lab: `llm_finetune`.
- RD-Agent pin: `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`.
- Native training: LLaMA-Factory v0.9.3,
  `ca75f1edf3cb50343ed1c98605141c3e22075b5f`.
- Task: fixed GSM8K train/dev/test partitions. The research model sees development
  feedback; final test inference runs only after the native acceptance decisions.
- Data transformations execute in Linux bubblewrap with no network, host home,
  credentials, GPU, verifier or development/test data. The installed capability
  is Python standard library processing and LoRA SFT. Unsupported proposals fail
  and are returned to the method as feedback; they are never silently clamped.
- A one-step native training check precedes full native training. Publication
  requires optimizer steps and a numerical change at the merged model dtype.

The adapter compiles only named definitions from the clean pinned checkout,
retaining upstream policy code and adjacent prompt templates without importing
the scenario initializer that downloads every registered FT-Dojo dataset.
Scenario/workspace/API transports are explicit local shims. Native generation
and acceptance still run in a separate optional-dependency worker.

Differences from the full upstream method: local GSM8K scenario; Qwen research
model instead of a hosted frontier model; bounded LoRA-only capability and
stdlib data processing; framework Lab validation instead of the complete CoSTEER
knowledge/debugging pipeline; exact mathematical scoring instead of OpenCompass;
one-step execution check instead of upstream test-parameter interpretation;
no meta-agent weight training, autonomous deployment or resumable search state.
`test_params.yaml` is retained as policy evidence; the Lab's fixed one-step check
does not consume it as an executable configuration.

## Setup and run

Use Linux with bubblewrap/user namespaces, Python 3.12 and CUDA training/inference
environments. Clone the two external repositories and check out the exact pins
above. Tracked upstream modifications are rejected. Install controller-worker
dependencies with `pip install -r experiments/ft_agent/requirements.txt`.
The framework interpreter also needs PyYAML and the default Math-Verify scorer:
from this repository, run `python -m pip install -e '.[math]' PyYAML==6.0.3`.
If the controller uses a separate interpreter, install its requirements there too.
Install native LLaMA-Factory and its compatible training dependencies in a
separate interpreter; v0.9.3 uses Transformers <=4.52.4, PEFT <=0.15.2 and
TRL <=0.9.6. Keep this separate from SEAL's TRL 0.18 environment.

Prepare full GSM8K data using the [ReST^EM recipe](../rest_em/README.md), then freeze
the pilot before model evaluation:

```sh
python experiments/ft_agent/prepare.py --source-dir runs/gsm8k-data \
  --output-dir runs/ft-agent-data
```

Serve an instruction-following research model at an OpenAI-compatible endpoint.
Edit `gsm8k.json` to set actual model paths, endpoint/model identity, interpreters,
data and device allocation. Example four-card allocation: GPU0 serves research,
GPUs1–3 run native DDP training and data-parallel target inference. Do not allocate
those cards to another job. Then launch through the normal registry:

```sh
rsisys catalog
rsisys run --config experiments/ft_agent/gsm8k.json
```

Artifacts include each model exchange, native hypothesis/parent, generated files,
raw Lab receipts, effective training parameters, native logs, numerical publication
checks and final independent test metrics. A completed research budget can include
failed experiments and no accepted improvement. Examine `trained_experiments`
and individual receipts before claiming SFT completed.

The [validated run and failures](RESULTS.md) include three real native SFT
experiments, final heldout degradation, version pins and artifact hashes.
