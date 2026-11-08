# Contributing

RSISys is an early research framework. Small, independently reviewable changes
are welcome. Describe the problem, resulting behavior and validation in a pull
request. For large changes to scheduling, recovery or public contracts, open
an issue explaining the intended behavior first.

## Development

Use Python 3.12 or 3.13 on Linux or macOS. From a checkout:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m unittest discover -s tests -t . -q
node --test tests/monitor/frontend/*.test.js
ruff check --select E9,F63,F7,F82 src tests examples experiments tools
```

The default tests require no GPU, model endpoint, API key or downloaded weights.
Optional integrations are explicitly skipped when their dependencies or opt-in
flags are missing; report skips with results. Node 22 is used in CI for the
monitor tests. See [tests](tests/README.md) for integration commands.

## Where changes belong

- New algorithms implement a controller and a Lab through `rsisys.api`. Prefer
  a library-external registration file for your own method. Reusable built-ins
  belong in `src/rsisys/controllers` and `src/rsisys/labs`.
- Harness, memory, scorer and training engine choices live inside those modules.
  The controller owns algorithm decisions; the host must not select candidates
  from scores on the algorithm's behalf.
- Shared scheduling, delivery, persistence, cancellation and resource ownership
  belong in the core. Registration does not implement new runtime semantics.
- `examples` contains quickstarts and interface tutorials; `experiments` contains
  model/data recipes and honest result reports; `tests` contains maintained
  regression checks. Formal protocols belong under `src`, never in test fixtures.

Add focused regression coverage for changed behavior, especially failures,
resource cleanup and state transitions. Use synthetic fixtures. Keep model
weights, raw datasets, credentials, temporary probes and run outputs out of Git.
Do not rewrite unrelated code to satisfy a style preference.

## API and experiment changes

Document public behavior changes in [CHANGELOG.md](CHANGELOG.md). During 0.x,
pin a commit or release for long experiments; see [compatibility](docs/compatibility.md).
Do not claim automatic recovery without a state adapter and reconciliation test.
Separate simulated execution, actual optimizer updates and measured improvement.
Record algorithm deviations, model/data revisions, splits, seeds and hardware.

Contributions are accepted under the project's Apache-2.0 license. Preserve
upstream notices and describe the source of adapted material. Never include
third-party code, data or weights unless their redistribution terms permit it.
Be respectful and specific in reviews; report security issues through
[SECURITY.md](SECURITY.md), without posting credentials or private logs publicly.
Community expectations are described in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
