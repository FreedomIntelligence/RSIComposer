# Source and version identity

Use one canonical source checkout per working environment. Develop on reviewable
branches, and identify every experiment by its actual Git commit plus any dirty
source snapshot. A directory suffix, model name or deployment label is not source
identity. Frozen experiment snapshots are execution evidence, not a second source
of truth.

Before claiming two machines are aligned, compare commit IDs and tracked changes.
Preserve and reconcile local edits before fast-forwarding. Never reset a dirty
checkout merely to make the version match, force-push shared history or silently
replace a source snapshot used by a running experiment.

The repository's default branch is `main`; feature branches and unmerged pull
requests may contain later work. A pushed branch is not a release. Release tags
must identify a checked commit; see [the release guide](releasing.md).

Archived deployments and historical review documents can retain older layouts
and terminology. They do not override the current public API or add another
installation requirement. Local machine aliases, absolute deployment paths,
weights and raw logs belong in operator configuration or ignored `.local/`
records, rather than in the public installation instructions.
