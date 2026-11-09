# Third-party material and experiment assets

Original RSISys and RSIMonitor code is licensed under Apache-2.0. Existing
third-party notices remain applicable to the corresponding material.

| Material | Origin / revision | Treatment |
| --- | --- | --- |
| SEAL prompts, sequence construction and training recipes | Continual-Intelligence/SEAL, `6d9c9f9ee392c6cc618e771f399d436d190f6ca4` | Adapted code in `rsisys.labs.seal`, `seal_grader` and `seal_worker`; MIT text and attribution are preserved in [SEAL_NOTICE.txt](src/rsisys/labs/SEAL_NOTICE.txt), included in the wheel. |
| TTRL / GRPO reference and native bridge | PRIME-RL/TTRL, `5806e119789132fcb43552beaf2532d7d0447213` | Independent reference backend plus an optional pinned upstream verl/reward adapter; [method notice](experiments/ttrl/NOTICE). No vendored verl distribution. |
| FT-Agent policy bridge | microsoft/RD-Agent, `484776c211e4fbbeef03e0ec00d6bbee7362a4f4` | Named upstream policy definitions and templates load from a separately obtained clean checkout. Local reference adapters do not bundle upstream code; [MIT notice](src/rsisys/controllers/FT_AGENT_NOTICE.txt) is included in the wheel. |
| AIDE native agent bridge | WecoAI/aideml, `ad73d227af481bea3de06f884f41b2993dd6453a` | Native Agent/Journal loaded from a separately installed checkout, with a registered Lab callback; [MIT notice](src/rsisys/controllers/AIDE_NOTICE.txt) is included in the wheel. No vendored AIDE implementation. |
| Native FT training | hiyouga/LLaMA-Factory v0.9.3, `ca75f1edf3cb50343ed1c98605141c3e22075b5f` | Optional, separately installed Apache-2.0 training backend; no bundled LLaMA-Factory code or weights. |
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
