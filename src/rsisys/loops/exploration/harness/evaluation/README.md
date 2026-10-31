# Evaluation

对候选进行评测，结果返回 [默认 Exploration Harness](../README.md)，不自行接受候选或更新模型。

## 接口

[Evaluator](__init__.py) 提供 `submit(request, *, idempotency_key)` / `poll(job)` / `cancel(job)` / `result(job)`；后端 cancel 返回 None，默认 Harness 的 cancel_evaluation 返回 JobOutcome。EvaluationRequest 包含 candidate 和非空 protocol_ref；EvaluationResult 包含 candidate_id、protocol_ref、metrics、evidence_refs 和 simulated。

通过 `loop.harness.poll_evaluation` / `loop.harness.cancel_evaluation` 获取成功结果时，校验候选 ID 与协议一致；终态以 evaluation 记录到 Experience，同一 job 只记录一次。scope=actor 表示反馈来源，不限制 Decision policy 检索或指定后续优化对象。异步后端约定见[共享协议](../../../contracts/README.md)。

## 实现与替换

[MockEvaluator](mock.py) 读取候选产物并生成 `simulated=True`、`quality_evaluated=False` 的报告，metrics 为空；不执行 protocol_ref 指向的真实评测。实现 Evaluator 后通过 DefaultExplorationHarness 的 evaluator 参数接入真实评测服务。

评测协议定义指标和证据；基线比较、候选接受规则与部署流程需要单独设计。

评测由 Harness 直接提交，不调用 ActorPolicy.act；即使真实 evaluator 需要推理打分，也应调用模型服务而非 Actor 环境动作。当前 Mock 不执行真实推理或质量打分。
