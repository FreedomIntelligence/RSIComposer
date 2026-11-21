# ProfessionalWorkLab

Run SkillsBench v1.0 in Harbor. The first three cohorts use the upstream primary
categories `natural-science`, `office-white-collar`, and `finance-economics`.
The sandbox gives the actor the task brief and materials; the task's original
verifier checks the resulting document, workbook, or other deliverable.

Install `rsicomposer[coding]` and Docker for the Harbor worker. Freeze one manifest
per cohort in the [Harbor task manifest format used by CodingLab](../coding/README.md#connect-a-benchmark).
Use the task directories from SkillsBench v1.0, including the original verifier
and solution files; the candidate runtime receives the task through Harbor.

```toml
[framework.labs.professional_work]
name = "professional_work"

[framework.labs.professional_work.config]
experiment_dir = "/runs/professional-work"
python = "/opt/harbor/bin/python"
endpoints = ["http://127.0.0.1:8000/v1"]
model = "<actor model>"
split_roles = { sealed = "sealed" }
primary_metric = { aggregate = "benchmark-mean" }

[framework.labs.professional_work.config.benchmarks.skillsbench-science]
manifest = "/data/skillsbench/science.json"

[framework.labs.professional_work.config.benchmarks.skillsbench-office]
manifest = "/data/skillsbench/office.json"

[framework.labs.professional_work.config.benchmarks.skillsbench-finance]
manifest = "/data/skillsbench/finance.json"
```

The Lab accepts the same pinned candidate artifact and per-benchmark runtime
configuration as CodingLab. `evaluate` returns the official Harbor rewards by
cohort. A configuration may select one cohort as the primary metric instead of
the explicit equal-weight `benchmark-mean` shown above.
