# Rolling-summary output headroom

Run7 reached a 16k compaction target with only 35 tokens available for the
structured summary. Its three charged generations were all length-truncated.
The format-recovery path worked, but admission treated any positive output
allowance as adequate. No rejected generation executed a Lab action.

`BasicContextCompactor` now reserves the configured `summary_tokens` plus the
existing 64-token wrapper allowance before calling a model summarizer. It prices
the actual source manifest and folds additional eligible whole records first.
If the allowance still cannot fit, it raises `ContextBudgetError` before
inference. Existing adaptive targets may then expand within their configured
bounds. The lion profile remains 16k -> 28k -> 40k, trigger48k, input guard56k.
At the ceiling, failure commits no folded state or summary and makes no model
call. Explicitly smaller summary configurations remain valid; scripted fallback
retains its existing partial-allowance behavior.

This changes fixed-target behavior: a configuration that previously admitted a
shortened model summary may now fail for insufficient headroom. Callers may
configure a smaller summary allowance or bounded adaptive targets explicitly.
No protocol or snapshot field was added. Provider format/identity failures do
not expand compaction targets. The existing bounded format recovery, cumulative
call admission, absolute deadline, protected facts and archive links remain in
force. Reserving the configured allowance does not guarantee semantic accuracy
or prevent every future truncation.

Regression coverage uses a deterministic injected character meter, not a real
provider tokenizer. It reproduces the old35-token admission, checks expansion
before inference, additional folding before expansion, exact headroom and a
one-token deficit, ceiling atomicity, scripted fallback, and no target expansion
after format/identity errors. Existing full-request policy accounting and
provenance tests use the new admission contract. Runtime evidence and local/lion
test results are recorded separately in the campaign control directory.

The new run continues under the user's renewed instruction, with the original
November5 09:00 Shanghai deadline and all1319 prior model calls charged.
Run7's failed result, cleanup and scientific audit remain preserved. Invalid
evaluation scores and bounded LangMem failures are separate unresolved issues;
this fix does not establish training, adoption or medical generalization.
