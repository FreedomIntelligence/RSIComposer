# Changelog

## 0.1.0 — initial research release (prepared, not yet tagged)

- Public controller/Lab registration and independent JSON configurations, with
  library-external plugins through `rsisys.api`.
- Built-in `single_loop`, `meta_agent`, `rest_em`, `rrsi`, `seal` and `ttrl`
  controllers; reusable mathematical, harness and local-process Labs.
- Operation authorization, structured receipts, provenance, owned cleanup and
  explicitly adapted snapshot/recovery support.
- Optional harness/memory adapters and an independent read-only `rsimonitor`.
- CPU quickstarts, GPU experiment recipes and scoped result reports. SEAL and
  TTRL completed small two-round runs; no paper-level performance gains were
  reproduced.
- Apache-2.0 licensing, contribution/security guidance, CI and clean installed
  wheel validation for the core and monitor packages.
- Registration discovery through `Registry.describe()` / `rsisys catalog`.
- Independent `output_subdir` for multiple SEAL/TTRL Labs. ReMe's pinned VCS
  runtime is installed with `requirements/reme.txt`, separately from the core.
- Close the SQLite connection when schema initialization fails, including
  malformed snapshot databases, before returning ownership to the caller.
- Canonicalize an isolated executable's parent while preserving its virtualenv
  entrypoint, so relative interpreter paths and directory aliases can execute
  against the explicitly mounted runtime.
- Add the SEAL paper profile with native Trainer/SFTTrainer, semantic grading,
  disjoint round datasets and seed-averaged selection. Add registered `ttrl_verl`
  controller/Lab for the pinned official full-parameter training schedule.
  These adapters do not establish paper-level gains.
- Deliver exact typed terminal operation feedback independently of text evidence
  projection, so large native failures retain their matching operation and data.
- Add optional registered FT-Agent and AIDE research controllers, with native
  policy code pinned upstream and execution/feedback through registered Labs.
  FT-Agent uses a local GSM8K/native LLaMA-Factory reference adaptation; AIDE
  uses a GSM8K LLM-program task. Their recipes document scope and differences;
  neither integration establishes the original papers' performance gains.

Current limits: weight-evolution recipes are synchronous and do not support
automatic restart; multi-Lab GPU leasing remains future core work. The optional
verl backend requires pinned upstream source and an isolated environment.
See [compatibility](docs/compatibility.md).
