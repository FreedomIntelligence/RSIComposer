# Evaluation

对候选进行评测，结果返回 [默认 Exploration Harness](../README.md)，不自行接受候选或更新模型。

[EvaluationExecution](__init__.py) 封装默认评测流程，接收 JobBook、共享 Experience 和探索状态，负责请求提交、终态身份校验与可重试的证据记录。评分逻辑仍由 Evaluator 后端实现；本地控制与 evaluate 指令共用这一组件。

## 接口

[Evaluator](__init__.py) 提供 `submit(request, *, idempotency_key)` / `poll(job)` / `cancel(job)` / `result(job)`；后端 cancel 返回 None，默认 Harness 的 cancel_evaluation 返回 JobOutcome。EvaluationRequest 包含 candidate、非空 protocol_ref 和 `data_refs: tuple[ArtifactRef, ...] = ()`；新增字段位于末尾，原有 `EvaluationRequest(candidate, protocol_ref)` 构造仍有效。data_refs 必须为 ArtifactRef 元组，空元组表示没有显式输入产物。EvaluationResult 包含 candidate_id、protocol_ref、metrics、evidence_refs 和 simulated。

`evaluate` 系统指令的 `input_refs` 原样传入 EvaluationRequest.data_refs。默认 ModelDecisionPolicy 的 `input_ids` 可选择当前 operation_results 或 last_feedback 中可用的产物作为验证集；未指定 input_ids 时沿用当前 operation_results 的产物。后端通过注入的 ArtifactStore 解析引用，不依赖 Actor 的本地轨迹。提交重试必须保留相同请求（包括 data_refs）和 idempotency_key。

通过 `loop.harness.poll_evaluation` / `loop.harness.cancel_evaluation` 获取成功结果时，校验候选 ID 与协议一致；终态以 evaluation 记录到 Experience，同一 job 只记录一次。记录的 `details.request.data_refs` 保留评测输入引用，Feedback.evidence_refs 指向该系统记录。scope=actor 表示反馈来源，不限制 Decision policy 检索或指定后续优化对象。异步后端约定见[共享协议](../../../contracts/README.md)。

## 实现与替换

[MockEvaluator](mock.py) 读取候选产物及每个 data_refs 输入，通过 ArtifactStore.get 验证引用；缺失或损坏的产物使作业失败。报告包含完整请求（含 data_refs），标记 `simulated=True`、`quality_evaluated=False`，metrics 为空；不执行 protocol_ref 指向的真实评测。实现 Evaluator 后通过 DefaultExplorationHarness 的 evaluator 参数接入真实评测服务。

评测协议定义指标和证据；基线比较、候选接受规则与部署流程需要单独设计。

评测由 Harness 直接提交，不调用 ActorPolicy.act；即使真实 evaluator 需要推理打分，也应调用模型服务而非 Actor 环境动作。当前 Mock 不执行真实推理或质量打分。
