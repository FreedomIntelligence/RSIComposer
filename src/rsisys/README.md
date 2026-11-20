# Source layout

This page maps the `rsisys` package. For how to use RSISys, start with the
[documentation](../../docs/README.md).

Import framework interfaces from [`rsisys.api`](api.py). Other modules are
internal and may change between releases.

| Path | Contents | Read when |
| --- | --- | --- |
| [`api.py`](api.py) | The public interface | Writing a Controller, Lab, or plugin |
| [`algorithms/`](algorithms/) | One package per built-in method, each with `configs/` presets. [`_template/`](algorithms/_template/README.md) is the method template; `shared/` holds code reused by several methods. | Running or adapting a method |
| [`labs/`](labs/) | [Reusable Lab catalog](../../docs/components.md#labs) and [Lab template](labs/_template/README.md). Method-owned Lab adapters live under `algorithms/`. | Choosing a Lab, binding data, or writing a Lab |
| [`loops/contracts/`](loops/contracts/) | The contracts: decisions and operations, receipts, splits and metrics, Lab profiles, Harness, Memory, data bindings, model service | Checking exactly what an interface requires |
| [`harnesses/`](harnesses/) | Model-driven decision policies, Memory implementations, and the `python_step` program harness | Building a model-driven Controller |
| [`backends/`](backends/) | Model serving and weight-training engines (vLLM, Chat Completions pools, LoRA) | Methods that serve or train models |
| [`host/`](host/), [`runtime/`](runtime/) | The host: call routing, budgets, receipts, resources and Ray execution, recovery, storage | Debugging the framework itself |
| [`snapshot/`](snapshot/) | Portable run snapshots for `rsisys resume --snapshot` | |
| [`adapters/`](adapters/) | Conversions to other libraries' model interfaces | |
| [`registry.py`](registry.py), [`methods.py`](methods.py) | The registry and method definitions behind `rsisys catalog` and `--plugin` | Registering methods or components |
| [`application.py`](application.py), [`cli.py`](cli.py) | Run configuration and the `rsisys` command | |
