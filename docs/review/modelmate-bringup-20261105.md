# Modelmate Open Lab setup and feedback routing

The modelmate deployment reuses the existing Qwen3.8-27B service for decision
and independent feedback contexts, and Qwen3.5-9B for candidate inference.
Generated programs receive an explicit Ascend NPU mapping. Controller packages
live in a separate environment; existing serving environments are unchanged.

## Context accumulation repair

Projecting an operation's history row into an evidence card previously lost the
nested receipt digest. A small failed receipt then appeared unrecorded and was
permanently protected, even when repeated. History cards now retain
`operation_result_sha256`; residency checks recognize that exact archived
identity. Legacy or mismatched cards keep the conservative behavior. Original
history, metrics, errors and artifact bytes are unchanged. The summary allowance
and context limits are unchanged.

## Replaceable routing implementation

`RoutedLab` adds an optional `feedback_route` argument to `lab_result`; it removes
the hint before calling the native Lab. `ContextOrMemoryAdmission` selects
organization separately from the memory implementation. `ImmediateFeedbackWindow`
changes only the decision's working view. These are deployment adapters, not
requirements imposed on replacement Labs or memory providers.

- `context` is the default: direct short-term result, raw trajectory archive,
  no LangMem organization. Recent operation receipts and live dependencies stay
  visible; older recoverable receipts expire from the working view.
- `memory` explicitly requests LangMem organization. The exact native result is
  archived and supplied to the organizer. Decision sees a request acknowledgement
  and can check feedback status or retrieve organized memory. Acknowledgement
  does not assert successful organization, scientific validation or adoption.
- Existing memory retrieval remains active on either route. Raw archive reads
  remain available. A read does not automatically create another memory job.

The underlying local Lab still exposes process observations through `lab_poll`
for debugging. Choosing to promote a result later is explicit; an earlier read
does not itself write long-term memory. Ordinary LangMem output failures remain
bounded failures and are not a reason to restart the research loop.

## Deployment configuration

The medical launcher accepts explicit meta and candidate aliases/weight paths,
device family and template family. vLLM uses its `/tokenize` endpoint for complete
request admission, including native tools, rather than applying the SGLang tool
normalizer. Model/provider identity checks, call accounting, fixed evaluation,
deadline and cleanup remain host-owned. NPU visibility is environment routing
for trusted code, not an OS security boundary.

Validation records are under `.local/modelmate-bringup-20261105T044026Z`.
The old source reproduces permanent failure residency; regression tests check
archive recovery, exact result preservation, conservative legacy handling,
exclusive organization admission, transient context, NPU environment injection
and provider token counting. Real service probes and the short medical run are
separate from scientific training or improvement claims.
