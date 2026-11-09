# Third-party material and experiment assets

Original RSISys and RSIMonitor code is licensed under Apache-2.0. Existing
third-party notices remain applicable to the corresponding material.

| Material | Origin / revision | Treatment |
| --- | --- | --- |
| SEAL prompts, sequence construction and training recipes | Continual-Intelligence/SEAL, `6d9c9f9ee392c6cc618e771f399d436d190f6ca4` | Adapted code in `rsisys.labs.seal`, `seal_grader` and `seal_worker`; MIT text and attribution are preserved in [SEAL_NOTICE.txt](src/rsisys/labs/SEAL_NOTICE.txt), included in the wheel. |
| TTRL / GRPO reference and native bridge | PRIME-RL/TTRL, `5806e119789132fcb43552beaf2532d7d0447213` | Independent reference backend plus an optional pinned upstream verl/reward adapter; [method notice](experiments/ttrl/NOTICE). No vendored verl distribution. |
| Historical LangMem validation material | `docs/review/repros/langmem` | Historical evidence, not runtime code; retained upstream [license](third_party/LANGMEM-LICENSE). |
| Optional dependencies | DSH, ReMe, LangMem, Math-Verify, PyTorch, Transformers, PEFT, vLLM | Installed separately under their own terms; not bundled in the core or monitor wheels. |

Model weights, downloaded datasets and generated checkpoints are not part of
the release. Experiment preparation scripts identify sources and revisions;
users must obtain assets under their own terms. The project license does not
relicense Qwen weights, SQuAD, GSM8K, MATH or medical benchmarks. The tiny
arithmetic fixtures in `examples/quickstart/data` are repository test fixtures.

Raw experiment evidence and local dependency checkouts belong in ignored
`.local/` directories. Published result summaries identify the method scope
and evidence hashes; they do not grant redistribution rights to external assets.
