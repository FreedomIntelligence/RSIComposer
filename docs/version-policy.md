# Canonical RSISys version

Each working environment has one canonical checkout on `main`:

| Environment | Checkout |
| --- | --- |
| Local | `design/RSISys` |
| b200 | `/home/snchen/xinyuan/RSISys` |
| lion | `/home/jovyan/projects/RSISys` |
| modelmate | `/home/work/user-job-dir/app/slai/user-xiexinyuan/xinyuan/projects/RSISys` |
| GitHub | `satsuki2486441738/RSISys`, branch `main` |

Synchronize the same commit and verify a clean tracked tree before claiming
alignment. A model name, directory suffix or deployment label is not source
identity. Push without rewriting remote history. Preserve and reconcile local
edits before fast-forwarding; never reset a dirty checkout to make it match.

The November 6, 2026 consolidation preserves old source archives, dirty patches,
Git bundles and per-file SHA manifests under
`.local/version-unification-20261106T051207Z/` on the corresponding hosts. Its
final receipt records exact source commits and cleanup results. Historical
experiment records still refer to their original source paths; resolve retired
paths through these archives rather than substituting newer code into old records.

Old deployment and integration worktrees are retired after source comparison,
archive verification and live-process checks. Retain run data, model weights,
environments and evidence separately from the working source. Legacy monitor
releases are runtime artifacts under ignored `monitor-ui/`, not another development
checkout. Do not infer that retained runtime artifacts are the current framework.

Future experiment provenance can use the canonical commit plus a verified source
archive/manifest under `.local/`. Do not create sibling `RSISys-*-fixN` working
copies or new deployment branches without a reason agreed with the operator.
Updating source does not authorize launching or resuming a research experiment.
