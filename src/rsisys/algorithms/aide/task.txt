Improve a fixed language model's accuracy on GSM8K arithmetic word problems by
developing a Python inference program. Model weights stay fixed. You may change
prompting, use examples, generate and aggregate multiple responses, or implement
verification within the installed inference budget. Choose your own strategy.

Public examples with worked answers are in `input/train.csv` (id,prompt,answer).
Current questions without answers are in `input/validation.csv` (id,prompt).
The Lab owns validation labels and will append `HOST_VALIDATION_METRIC` with the
measured accuracy to your execution log. Use exactly that accuracy as the metric;
ignore any metric printed by candidate code. Higher accuracy is better.
Question IDs are opaque. Keep the same data split, metric and output protocol.
The final selected program is evaluated on different, previously unseen questions
after search; do not hardcode question-specific answers.

The only model access is the installed standard-library client:

```python
from llm_client import complete
answer = complete(
    [{"role": "user", "content": "your prompt"}],
    max_tokens=1024, temperature=0.0, seed=73,
)
```

`complete` returns the generated text. It can be called concurrently from threads;
use up to 64 threads to batch independent questions over the available model
replicas. Each call accepts at most 1536 output tokens. The total budget is four
calls per input question, shared across the entire program. The model and its
weights cannot be changed; there is no training or unrestricted network/GPU
access. Use standard-library Python (csv, json, re, collections,
concurrent.futures); do not install packages or download resources.

Write all predictions to `working/submission.csv` with columns exactly
`id,answer`, one row for every input question and no extra rows. `answer` may be
a plain final number or a complete solution ending in `#### <number>` or
`\boxed{<number>}`. Read the current questions at runtime. Do not assume a fixed
number of questions. Include every ID even if model output is malformed; an
empty answer counts as incorrect. The Lab computes correctness independently.

Return a short plan followed by the complete runnable Python program.
