# 共享协议

两个 loop 共用的消息、引用和公开端口；此目录不包含后端实现。可选 ReviewAuthorizingEndpoint 让 Runner 在执行模型取消建议及重试前重新检查当前权限。协同顺序与交付规则见[系统架构](../../../../docs/ARCHITECTURE.md)。

## 实验查询

[QueryProvider / QuerySpec / QueryRequest / QueryResult](queries.py) 是可选的只读查询边界。能力由 provider 显式声明，返回来源、读取时间、内容版本、结构化数据、ArtifactRef 和分页游标。默认实现提供本地状态、指标、任务、日志摘要、候选、模型、产物引用和经验摘要；外部 Harness 可以提供其他查询。列表查询为实时视图，不是跨页事务快照。

[OperationResult](coordination.py) 的可选 data 字段让工具结果直接进入 researcher 上下文；大内容通过引用按需读取。查询权限不要求把所有信息一次性送入模型，也不改变终态 Feedback 的交付职责。

[JobProgress](jobs.py) 可附在 JobOutcome 上，包含 step、total_steps、metrics、elapsed_seconds 和可选的后端样本时间 observed_at（Unix 秒）。默认命令将它暴露在 ExplorationProgress.job_progress 中，非终态 metrics 同步提供最近的后端指标；没有新样本时保留旧样本和原时间，不推算进度。查询结果 observed_at 是读取时间，两者含义不同。

## 指令与反馈

[Directive](directive.py) 包含身份、目标、策略、输入、允许行动、步数上限、模拟标记和 payload；payload 支持 JSON 数据及值相等的 dataclass，构造时校验可深复制、复制前后值相等及 JSON 序列化能力（拒绝 NaN/Infinity）；它是可扩展的数据对象，内置 payload 定义在 [requests.py](requests.py)。

| mode | payload | 语义 |
|---|---|---|
| execute | 无 | 在环境中有界执行 |
| sft | SFTSpec(recipe_ref, base=None, apply_result=False) | input_refs 是数据；缺省 base 在接收时捕获；默认产出候选 |
| rl | RLSpec(recipe_ref, updates=1) | 内部交替采样与更新，每次成功更新用于下一轮采样 |
| evaluate | EvaluationSpec(candidate, protocol_ref) | 独立评测指定候选；input_refs 传入 EvaluationRequest.data_refs |
| configure_prompt | PromptSpec(instructions) | 替换 Actor 指令文本；空字符串清空；绕过 ActorPolicy.act |

可选 `train` 使用 `TrainSpec(recipe_ref, method, base=None, apply_result=False)`，需显式注册 `training_directive()`、`TrainingCommandHandler` 并注入 learner，不属于默认模式。input_refs 作为训练 data_refs，缺省 base 在接收时捕获；整个任务交给一个后端作业，后端可自主采样与训练，默认仅返回候选。默认 `rl` 继续使用本地采样循环。装配见[使用指南](../../../../docs/GUIDE.md#接入整项训练任务)。

具体 [ActorTrainingRequest](../exploration/harness/learner/README.md) 与 [DecisionTrainingRequest](../decision_making/harness/learner/README.md) 是后端中立的请求：校验模型、非空 method / recipe 字符串和引用元组类型，不统一限定算法或要求非空数据。Decision 请求支持外部 data_refs 与自定义 method；mock 的能力限制由各后端负责。

Directive 由 decision policy 或外部系统发给 Exploration Harness，不是 ActorPolicy 动作。sft 直接调用 learner；rl 只有 rollout 使用 Actor 环境动作；推理评测由 evaluator 调用模型服务。默认训练与评测仍是 Mock，未实现真实梯度或推理评分。

[Feedback](feedback.py) 关联原指令，携带状态、质量/训练/执行指标、产物、证据、候选、作业及错误。completed / failed / cancelled / partial 都是终态；缺失指标保持为空。evidence_refs 引用系统反馈记录；Actor RL 请求的 trajectory_refs 引用 Actor 本地 history，两者不能互换。

Feedback 构造时要求 `ExecutionStatus`，拒绝 `JobStatus.RUNNING` 等作业状态；failed 必须含 error，completed 不得含 error。`ExplorationProgress` 表达在途阶段、作业引用、可选的实时指标与耗时，只有包含合法 Feedback 才表示终态。未知指标留空，不推测后端训练进度。

[coordination.py](coordination.py) 定义 DecisionEndpoint、ExplorationEndpoint、ExplorationProgress、LoopStepResult 和 OperationResult。DecisionEndpoint 统一通过 receive_feedback(directive, feedback) 接收反馈；它们是 Runtime 协调的公开边界，结果类型不依赖具体 Harness，可直接用于自定义端点和运行时类型反射。

可选端口 `RecoverableExplorationEndpoint.pending()` 返回当前未 acknowledge 的进度；`ObservableExplorationEndpoint.execution_view()` 返回 `ExecutionView(model, progress)`；`ReviewingDecisionEndpoint` 接收该视图并提供 `review(objective, progress)`。不要求外部 Harness 暴露 policy、memory 或后端实例。

Directive 可指定正数 `review_after_seconds`（默认 None，不复查）。`ReviewDecision` 只允许 continue / cancel，`next_review_seconds` 设定下一次复查间隔；None 停止自动复查。进展是当前执行事实，选定的复查事件进入 Experience，不另建 history/memory 系统。任务最终结果仍只通过 Feedback 交付。

## 作业与版本

模型产物、推理可用性和角色绑定是三个不同概念。`Candidate.changes` 引用产物，`TrainingResult.model` 指定预期启用的版本，不表示服务已加载。两个默认 Harness 接受 `model_activator`；真实结果应用前调用 `ModelActivator.activate(result, idempotency_key=...)`，仅在返回完全匹配的 ModelRef 后更新 Policy 绑定。适配器负责部署或检查已经部署的版本，无需重新部署已有服务。

部署进行中或响应不确定时抛出 `ModelActivationPending`，调用方稍后推进同一任务；键在重试间保持一致，适配器必须兑现幂等性。其他异常表示失败，旧角色绑定保持不变。模拟结果在未配置 activator 时仍可模拟切换；真实结果缺少 activator 时拒绝应用，但仍可保存为候选或提交评测。取消待启用的训练指令不会切换绑定，不负责撤销已在外部启动的部署。

外部 `ArtifactRef` 无需先登记到内置 ArtifactStore；version 与 digest 未知时用 None。传递引用不代表下载、解析或验证了内容；这些由接入实现负责。内置存储仍只读取自身管理的产物，并保留完整性校验。

[jobs.py](jobs.py) 定义 JobRef、JobStatus、JobOutcome。queued / running / unknown 非终态；unknown 不能当作失败。仅 succeeded 携带结果，failed 必须携带错误，cancelled 表示作业已确认取消。训练/评测后端的 result(job) 只读取成功结果，不轮询、不推进任务；内置 Mock 在其他状态下抛出 RuntimeError。

相同幂等键与相同请求应返回同一任务，不同请求复用键应拒绝。取消请求不等于作业取消完成；命令取消时作业仍可能成功，Feedback 的 cancelled 不等同于 JobStatus.CANCELLED。具体后端必须实现作业约定，[Runtime](../../runtime/README.md) 的进程内记账不能替代远端幂等保证。

[TrainingResult](training.py) 关联 base / model / candidate / metrics / simulated。[Candidate](candidate.py) 描述变更产物和受影响角色，不表示已经验收。默认 Harness 校验角色归属和基础版本；应用结果需要 policy 支持模型更新。

[model_service.py](model_service.py) 定义 ModelRef、ModelBackend、ModelUpdatable、请求/响应和 TokenUsage；[ArtifactRef](artifacts.py) 只携带产物身份、URI、版本和摘要，不含内容，解析与校验由产物存储负责。模型引用相同不表示角色同步；真实权重加载归模型后端。

可选并行协议位于 `coordination.py`：`ConcurrentExplorationEndpoint.pending_all()` /
`execution_views()` 和 `ConcurrentDecisionEndpoint.pending_all()` / `update_executions()`。
单条 accept/advance/inspect/cancel/acknowledge 始终使用 directive_id。
`ExplorationProgress.version` 从 0 开始；`DirectiveUpdate(directive_id, expected_version,
idempotency_key, changes)` 是独立控制请求，不替换原始 Directive。
确认后的 `UpdateReceipt` 版本恰好增加 1；同键同请求重复返回原回执，不能用同键改变内容。
支持哪些字段由可选 backend/handler 声明与验证，协议不承诺任意训练超参可热更新。
