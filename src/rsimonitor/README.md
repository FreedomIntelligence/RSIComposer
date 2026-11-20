# RSIMonitor

RSIMonitor is the optional read-only viewer for RSISys experiments. It is an
independent Python distribution with a bundled Vue dashboard and no mandatory
third-party Python runtime dependencies. A run is produced by RSISys; the viewer
reads its public `monitor.json` and serves a bounded projection to the browser.

See the [RSISys package guide](../rsisys/README.md) for method execution and state
recovery, or the [project overview](../../README.md) ([中文](../../README_zh.md))
for the architecture diagram and recent updates. The framework guide includes
the method catalog and adaptation scope.

## Quickstart

Use Python 3.12/3.13 on Linux/macOS. All repository paths and development commands
below assume the repository root as the working directory. The
[GPU example](../../README.md#getting-started) writes a run that the dashboard can open.

```sh
python -m pip install ./src/rsimonitor
rsimonitor --runs-dir runs
```

Open `http://127.0.0.1:8765/` and select a run. The dashboard shows the decision
graph, metrics, samples, logs, and experiment comparisons in English and Chinese.
Give each experiment a distinct output directory, for example
`rsisys run --config config.json --data-dir runs/experiment-002 --experiment-name 'Experiment 002'`.
Point `--runs-dir` at the common parent of these directories. `rsisys run` publishes `monitor.json`;
the dashboard follows newly started experiments and displays the observations reported for each run.

It binds to loopback by default and has no authentication or multi-tenant isolation.


The Monitor exposes its versioned scalar wire limits and enums at `/api/schema`.
Evaluation records retain reported split, protocol identity and validated protocol/submission digests.
For application-level integrations, `app.runner.add_observer(observer)` accepts an object with
`on_step(result)` and returns an idempotent detach callback. Each observer receives a detached
step after the execution transaction commits; observation failures warn without replaying work.
The bundled decision endpoint also provides `observed_decisions()` as a detached ledger view.

## Metrics and probes

The metrics workspace ports native chart components from
[SwanLab-Dashboard](https://github.com/SwanHubX/SwanLab-Dashboard). Sources live in
`src/rsimonitor/frontend/swanboard/`, with revision and license attribution in code.
Search, namespace groups, pin/hide, smoothing, linked legends, zoom and image export
run directly inside RSIMonitor. Data subscriptions use RSISys observations. No
SwanLab account, SDK, separate dashboard service or iframe is required. The data
adapter currently supports scalar metrics; media observations are not wired yet.
Layouts are browser-local and do not modify experiment artifacts.

Use **Add chart** to choose one or more Y metrics and an X axis (observation order,
reported step, elapsed/record time, or another metric). Each card's **Settings**
button opens a per-chart settings drawer for data, axes and smoothing: title, group,
axis labels, linear/log scales, optional bounds, Time Weighted EMA, Running Average
or Gaussian Smoothing. Smoothing parameters and the raw-curve overlay are saved
independently for each chart and also apply to zoom/export. The workspace toolbar
manages chart creation/removal; axis and smoothing controls live inside each chart.
Custom X metrics first match values from the same observation. For separate log
calls, record X before Y; alignment uses either the same step or the most recent X,
within the same run and operation. Unmatched points are omitted. Log
axes omit nonpositive points. **Manage charts** removes panels or restores removed
automatic panels; these changes only affect the current browser's layout.

```sh
rsisys run --plugin src/rsisys/labs/_template --plugin src/rsisys/algorithms/_template \
  --config src/rsisys/algorithms/_template/configs/run.json \
  --probe src/rsisys/algorithms/_template/probes.py --data-dir runs/custom-observed
rsimonitor --runs-dir runs
```

Keep probes with their method and load with `--probe path/to/probes.py`:

```python
def register(observations):
    def sample(event):
        receipt = event.get("operation_result") or event.get("feedback") or {}
        return receipt.get("data", {}).get("quality_metrics") or None
    observations.register("quality", sample, on="feedback", every=1)
```

Any code in a run, including training or inference worker processes, can log like SwanLab:
`import rsisys; rsisys.log({"train/loss": loss, "note": "text"}, step=step)`. Numbers are charted,
text appears in the run log, and worker-process records reach the board within about a second.
Outside a monitored run it is a no-op returning `False`.

Controller/Lab code can also call `observe({"train/loss": measured_loss}, step=optimizer_step)`
or register a probe with `register_probe(name, callback, on="checkpoint", every=2)`
and invoke `trigger_probe("checkpoint", payload)`. Import these from `rsisys.api`.
Register each name once per run; remove with `current_observations().remove(name)`.
Host events are `start`, `step`, `feedback` and `finish`. Sampling starts with the
first event, then every N events. Callbacks receive an event copy and return finite
numeric metrics or `None`; keys are prefixed with the probe name.

Callbacks execute synchronously at event boundaries. Keep them short; schedule
expensive evaluation as an explicit Lab operation. Browser refreshes never trigger
probes or adopt candidates. Host-launched workers can use `rsisys.log()` with the
inherited observation environment; other subprocesses return measurements through
Lab receipts or progress. Observations append to the run's `observations.jsonl`; snapshots retain
the latest 10,000 points. The default axis is observation order; actual reported
steps and elapsed time are also selectable. Probe failures remain visible in
status/logs while the method continues running.

Built static assets ship with the repository. After editing the chart frontend,
run `npm ci && npm run build` in `src/rsimonitor/frontend/`. Node is a development
dependency only.

## Run discovery and HTTP API

`--runs-dir` points at the parent of run directories, each containing a public
snapshot. Directory names identify runs in the viewer:

```text
runs/
  experiment-001/monitor.json
  experiment-002/monitor.json
```

The viewer reads immediate children; point it at the common parent rather than
at an individual run. Unrelated directories without a snapshot are skipped.
Reported project metadata groups runs into projects. A run with an unsupported
or invalid snapshot appears unavailable rather than being interpreted as a
successful experiment.

Compatible `monitor.json` files from different framework revisions can share
this directory. New views populate only when a run reports the associated data.
When present, `call_id` identifies one operation and joins its stage, evaluation,
samples and metrics; `decision_id` links it to the issuing decision. A decision
can issue more than one operation. Host-only final evaluations publish aggregate
quality metrics without per-item samples or restricted artifact references.
Programmatic consumers should use the `split` reported on each evaluation rather
than treating the original closed list of evaluation split names as an enum.
Viewing a run does not exercise the framework's checkpoint or restore protocol.
Continuing a durable run in place preserves the history in its existing
`monitor.json`. A portable execution snapshot does not bundle that public file.

| GET/HEAD endpoint | Response |
| --- | --- |
| `/` | Dashboard entry page |
| `/api/schema` | Public wire contract, limits and enums |
| `/api/projects` | Project groups and run summaries |
| `/api/runs` | Run summaries |
| `/api/runs/<directory-name>` | Projected public run snapshot |

JSON responses support ETags. Write methods are rejected. The run directory is
never exposed as a general file server, so artifact files and internal journals
cannot be fetched by guessing their paths. The producer and viewer keep matching
copies of `monitor_contract.json`; protocol changes must update both.

Override the local port or use an absolute runs path with:

```sh
rsimonitor --runs-dir /absolute/path/to/runs --host 127.0.0.1 --port 8766
```

For a viewer on a remote machine, forward its loopback port from your local machine:

```sh
ssh -N -L 8765:127.0.0.1:8765 user@remote-host
```

Then open the local dashboard at `http://127.0.0.1:8765/`. The server has no
built-in authentication or multi-tenant isolation. Browser requests are checked
for local/same-origin access; any remote access needs an appropriate deployment
boundary. Refreshing the viewer never starts training, resumes a run or adopts a
candidate. Displayed metrics and evaluation outcomes come from reported evidence.

## Source layout

| Path | Responsibility |
| --- | --- |
| `server.py` | Run discovery, public projection, read-only HTTP API and static delivery |
| `monitor_contract.json` | Versioned wire limits shared with the producer |
| `frontend/` | Vue application, chart components, translations and build tooling |
| `frontend/swanboard/` | Adapted upstream chart sources with attribution |
| `static/` | Committed browser assets served by the Python package |
| `pyproject.toml` | Independent distribution and `rsimonitor` console entry point |

## Develop and package

Build the standalone wheel from the repository root:

```sh
python -m pip wheel --no-deps --wheel-dir dist/monitor ./src/rsimonitor
```

To try the viewer locally, open a run from the [GPU Quickstart](../../README.md#getting-started).

After a frontend edit, rebuild and include the generated `static/app` assets in
the same change. Use Node 24 and run `npm ci` followed by `npm run build` in
`src/rsimonitor/frontend/`. Installing or serving the Python wheel needs no
Node toolchain and does not rebuild the frontend.

The viewer's package metadata uses this README. Installing `rsimonitor` does not
install `rsisys`; run production and run inspection can use separate environments.

## Troubleshooting

- **No runs listed:** check that `--runs-dir` is the parent directory and that
  child runs contain `monitor.json`. A fresh viewer can start before runs exist.
- **Run unavailable:** check snapshot validity, supported schema and public size
  limits from `/api/schema`. A partially written snapshot is retried on later reads.
- **No curves or samples:** inspect what the run actually reported. Scalar metrics
  are supported; media observations are not wired into the chart adapter.
- **Frontend changes are missing:** rebuild `frontend/` and reinstall the viewer
  if it is running from a wheel rather than the checkout.

## License

Apache-2.0; see [LICENSE](LICENSE). Attribution for adapted chart sources stays
with those files and in the bundled frontend. Dependency terms remain separate.
