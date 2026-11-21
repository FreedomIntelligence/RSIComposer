# ComputerUseLab

Run WebArena-Infinity real tasks with a fresh local app server and Chromium
context per trial. The actor interacts with the app, and the Lab runs the
upstream state verifier after the actor finishes. Each receipt includes success
rate and an action trace artifact.

Install `rsisys[computer-use]`, run `python -m playwright install chromium`, and
check out the chosen WebArena-Infinity revision. Build a JSON manifest with
`schema_version: 1`, `benchmark: "webarena-infinity"`, and `splits`. Each split
lists `{id, input, reference}` records. Use `<app>/<task id>` as the ID,
`{app, instruction}` as the input, and `{app, verify}` as the reference, taking
the fields from the app's `real-tasks.json`. Keep the checkout and manifests on
the host; the test app serves only public UI assets.

```toml
[framework.labs.computer_use]
name = "computer_use"

[framework.labs.computer_use.config]
trust = ["host-code"]
runner = "my_browser_runner:solve"
runner_revision = "v1"
split_roles = { sealed = "sealed" }

[framework.labs.computer_use.config.benchmarks.webarena-infinity]
manifest = "/data/browser/webarena-infinity.json"
sha256 = "<manifest SHA-256>"
checkout = "/opt/webarena-infinity"
revision = "<upstream commit>"
```

The trusted runner implements `solve(*, instruction, task, context, page)` and
returns a JSON action trace. `page` is a synchronous Playwright page. Expose UI
observations and actions to your actor, and wait for the app's UI changes to
complete before returning. The Lab keeps the verifier separate from the runner.
Call `evaluate` with `split` and `instruction`. Optional `task_ids` are available
for development splits; the primary metric is `webarena-infinity/success_rate`.
