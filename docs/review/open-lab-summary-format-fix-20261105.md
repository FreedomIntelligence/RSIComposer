# Rolling-summary format boundary

Run `medical-lion-night-06-20261104T161442Z` stopped after 628.549 seconds,
148 model calls and six committed compactions. Its last rolling-summary request
had 40,580 input tokens, a dynamically computed 217-token output ceiling and no
response schema. The provider returned `finish_reason=length` with an unfinished
`{"summary":"...` string. The transport accepted unstructured text and policy
JSON parsing terminated the controller. Owned cleanup completed without errors;
the GPU compute list was empty and the allocation Pod was terminal before repair.

When structured output is enabled, the policy now declares the summary object
schema as well as the decision schema. This connects malformed/truncated summary
responses to the Open Lab backend's existing bounded format recovery: at most two
extra generations, each charged and admitted against the same call limit and
absolute deadline. Model identity, schema, original records and output ceiling
are preserved. Rejected wire responses remain archived. No Lab action is replayed.

The policy still checks the summary's exact keys and nonempty text. Explicit
`structured_output=False` remains supported. Compactor state commits only after
a valid summary fits the complete-request target. Format recovery exhaustion
does not expand that target; only `ContextBudgetError` uses configured headroom.
Neither token limits nor graph guards were raised.

Local HTTP fixtures reproduce the old JSONDecodeError through the actual policy,
compactor, recording backend and serving adapter. They check successful summary
recovery followed by a decision, three-call exhaustion with unchanged raw records
and compactor state, no target expansion, and the structured-output opt-out.
These CPU fixtures are not research or training results. Live recovery and summary
semantic correctness require separate evidence.

Run6's 148 raw-wire input counts matched provider usage. Five memory jobs completed,
33 failed under the existing validation/graph guards, eight were queue-full and
seven were cancelled at cleanup. No weight artifact or training/adoption chain was
verified. Reference integrity and successful JSON parsing do not establish the
truth of summary prose, evaluation claims or memory.
