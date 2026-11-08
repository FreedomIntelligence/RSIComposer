# Security and trust boundaries

RSISys 0.1 is intended for trusted research environments. Only the current
development line receives fixes; the project makes no response-time guarantee.

## Reporting

Use the repository's **Security → Report a vulnerability** facility when it is
enabled. If unavailable, open an issue asking for a private reporting channel
without including exploit details, tokens, private datasets or full run logs.
Do not upload sensitive material to public issues or pull requests.

## Execution model

- `--plugin` executes explicitly trusted Python code with the host's privileges.
  A JSON config selects registered names; it does not make plugins untrusted.
- The local process backend and GPU workers run programs as the current user.
  Workspaces, path checks, permissions and process groups manage ownership;
  they do not provide an OS security sandbox. Use a separate user, container
  or stronger external executor for untrusted generated code.
  The optional `IsolatedProcessBackend` applies an explicit Bubblewrap (Linux)
  or Seatbelt (macOS) capability policy and fails when its driver is unavailable;
  it is not enabled by ordinary local-process registration. Validate the chosen
  executor and host configuration for your threat model.
- A custom Lab owns its network access, external jobs, billing and cleanup.
  In-process Python code can bypass host abstractions. Do not treat an operation
  catalog or model prompt as an enforcement boundary for arbitrary plugins.
- Model endpoints and third-party SDKs receive the inputs you configure. Supply
  credentials through environment variables; do not put secrets in JSON configs.
  Saved requests and artifacts can contain prompts, answers, local paths and
  model responses. Share only explicitly reviewed evidence.
- RSIMonitor defaults to loopback and is read-only. It has no authentication or
  multi-tenant isolation. Bind externally only behind suitable access control.
  Public text fields are supplied by your producer; review them before sharing.

The default CPU quickstarts use synthetic fixtures and no model service.
GPU experiments require operator-controlled hardware, dependencies and data.
