# Math Lab

`math` is a reserved Lab name in the registry. The placeholder in
[`lab.py`](lab.py) can be assembled with an empty config, but publishes no
operations and produces no measurements. Existing methods continue to use
their own Labs.

Implement the shared math task environment in `MathLab`. Choose the operation
contract and primary metric when wiring the first consumer, and bump the
registration version when the implementation is ready. The existing
[`verifiers.py`](verifiers.py) provides strict numeric and symbolic answer
verification. Candidate generation, search, and selection belong to the method.

Define benchmark and training inputs when wiring the first consumer,
and keep their splits explicit. Keep the math problem format and scoring
logic in the scene. See the [Lab template](../../_template/README.md) for the data binding and
operation contracts. The [ReST^EM adapter](../../../algorithms/rest_em/README.md#connect-training-and-evaluation-data)
shows a separate math Lab that already accepts training and evaluation JSONL.
