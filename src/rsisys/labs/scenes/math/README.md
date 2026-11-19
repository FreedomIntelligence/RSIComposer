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
logic in the scene. See the [method author guide](../../../README.md) for the
Lab contract.
