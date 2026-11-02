# 共享协议

两个 loop 共用的消息、引用和公开端口；此目录不包含后端实现。可选 ReviewAuthorizingEndpoint 让 Runner 在执行模型取消建议及重试前重新检查当前权限。协同顺序与交付规则见[系统架构](../../../../README.md#系统架构)。

## 实验查询

[QueryProvider / QuerySpec / QueryRequest / QueryResult](queries.py) 是可选的只读查询边界。能力由 provider 显式声明，返回来源、读取时间、内容版本、结构化数据、ArtifactRef 和分页游标。默认实现提供本地状态、指标、任务、日志摘要、候选、模型、产物引用和经验摘要；外部 Harness 可以提供其他查询。列表查询为实时视图，不是跨页事务快照。

[OperationResult](coordination.py) 的可选 data 字段让工具结果直接进入 researcher 上下文；大内容通过引用按需读取。查询权限不要求把所有信息一次性送入模型，也不改变终态 Feedback 的交付职责。

[JobProgress](jobs.py) 可附在 JobOutcome 上，包含 step、total_steps、metrics、elapsed_seconds 和可选的后端样本时间 observed_at（Unix 秒）。默认命令将它暴露在 ExplorationProgress.job_progress 中，非终态 metrics 同步提供最近的后端指标；没有新样本时保留旧样本和原时间，不推算进度。查询结果 observed_at 是读取时间，两者含义不同。

## 决策关系与采纳回执

Decision 使用稳定 `decision_id`，`depends_on_decision_ids` 引用已成功完成的前置决策，`source_decision_ids` 只引用已知信息。多个前置形成汇合；独立分支通过并行 Runner 推进。具体执行及权限仍由原有 Action Space / Harness 负责。

`OperationResult` 和 `Feedback` 的可选 `adopted_candidate_ids` 只由执行器在成功确认启用后填写，失败或取消不能携带采纳声明。决策 Harness 将已知候选映射到真实生产决策，记录 `adopted_decision_ids`；外部候选缺少来源时仍保留成功采纳回执，只省略无法证明的来源边，不把评测高分、模型提出的采纳意图或记录先后当作采纳事实。`LoopStepResult.decision_id` 将异步操作回执对应到原决策。

## 指令与反馈

[Directive](directive.py) 包含身份、目标、策略、输入、允许行动、步数上限、模拟标记和 payload；payload 支持 JSON 数据及值相等的 dataclass，构造时校验可深复制、复制前后值相等及 JSON 序列化能力（拒绝 NaN/Infinity）；它是可扩展的数据对象，内置 payload 定义在 [requests.py](requests.py)。

| mode | payload | 语义 |
|---|---|---|
| execute | 无 | 在环境中有界执行 |
| sft | SFTSpec(recipe_ref, base=None, apply_result=False) | input_refs 是数据；缺省 base 在接收时捕获；默认产出候选 |
| rl | RLSpec(recipe_ref, updates=1) | 内部交替采样与更新，每次成功更新用于下一轮采样 |
| evaluate | EvaluationSpec(candidate, protocol_ref) | 独立评测指定候选；input_refs 传入 EvaluationRequest.data_refs |
| configure_prompt | PromptSpec(instructions) | 替换 Actor 指令文本；空字符串清空；绕过 ActorPolicy.act |

可选 `train` 使用 `TrainSpec(recipe_ref, method, base=None, apply_result=False)`，需显式注册 `training_directive()`、`TrainingCommandHandler` 并注入 learner，不属于默认模式。input_refs 作为训练 data_refs，缺省 base 在接收时捕获；整个任务交给一个后端作业，后端可自主采样与训练，默认仅返回候选。默认 `rl` 继续使用本地采样循环。装配见[使用指南](../../../../README.md#接入整项训练任务)。

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

### 可选训练来源与后端能力

[TrainingOrigin](training.py) 使用 `TrainingOrigin(parent_system, initial_model,
resume_mode=ResumeMode.RESTART, checkpoint=None)` 显式区分实验父版本与初始权重。
`parent_system` 是稳定的系统版本 ID，与 `initial_model.revision` 独立；它不是权重来源。
训练结果进入 SystemRegistry 时使用 `register_training(result, origin=提交来源)`；注册检查实际声明的初始化，应用再检查实验父版本是否仍为 current。直接更新某个 loop 的角色模型没有系统版本上下文，仍要求当前角色模型与 result.base 相同；跨初始化的候选应用走 SystemRegistry。
请求原有 `base` 仍用于训练结果和角色应用的版本检查；传入 origin 时由调用方和后端确认
`initial_model` 与实际训练请求一致，框架不根据父系统隐式替换模型。

| resume_mode | checkpoint | 语义 |
|---|---|---|
| `RESTART` | 必须为 None | 从 initial_model 开始新的训练运行，不恢复运行状态 |
| `WEIGHTS_ONLY` | 可选 ArtifactRef | 有 checkpoint 时仅加载其权重，否则使用 initial_model；优化器、调度器、步数和 RNG 重新初始化 |
| `FULL_STATE` | 必须为 ArtifactRef | 恢复 checkpoint 的权重、优化器、调度器、步数及 RNG 等完整训练状态 |

引用只描述来源；后端负责解析和验证实际 checkpoint 内容，无法兑现请求的恢复语义时必须明确拒绝，
不能把 full_state 静默降级为 weights_only。已有请求无需声明来源，原有 TrainingResult 与
ModelActivator 构造及调用方式保持兼容。

[BackendCapabilities](backend.py) 是可选声明，字段为 `methods`、`resume_modes`、
`update_fields` 三个不可变元组，以及 `supports_checkpoint`、`supports_reattach` 两个布尔值。
`BackendCapabilityProvider.capabilities()` 返回该对象；已有 learner 不需要实现这个端口。
方法名由后端定义，空元组表示不支持任何值。`validate_training_capabilities`、
`validate_update_capabilities`、`require_checkpoint` 和 `require_reattach` 在声明不支持时抛出
`NotImplementedError`，不自动注册或发现后端，不替代 recipe、数据、更新值或版本的校验。
没有 origin 时，训练能力校验按 restart 处理。

checkpoint 标志声明**导出**能力，与消费现有 checkpoint 的 resume_modes 分开。
可选 `CheckpointingTrainingBackend.checkpoint(job, idempotency_key=...)` 返回实际 ArtifactRef；
`ReattachableTrainingBackend.reattach(job)` 返回同一 JobRef 的 JobOutcome，不重新提交训练。
同键重试 checkpoint 必须复用同一操作，不能为重试创建新作业。

### 可选部署生命周期

[DeploymentBackend](backend.py) 提供 `prepare(model, idempotency_key=...)`、
`query(deployment)` 和 `release(deployment, idempotency_key=...)`；可选
`ReattachableDeploymentBackend` 增加 `reattach(deployment)`。它们都返回不可变
`DeploymentState(deployment, model, status, job=None, error=None)`，其中部署使用稳定的
`DeploymentRef(deployment_id, backend)`，外部部署作业使用现有 JobRef。

status 为 preparing / ready / releasing / released / failed / unknown；只有 ready 表示精确的
ModelRef 已可用于推理，`state.ready` 才为 True。unknown 不等于失败，作业成功也不等于部署就绪。
failed 必须含 error，只有 failed / unknown 可以含 error。查询和重连保留同一部署、模型及已有
作业身份，不创建替代作业；同键同模型 prepare 返回同一部署，同键不同内容必须拒绝。
release 可异步停留在 releasing，必须幂等，不表示删除模型产物。未支持的操作明确抛出
NotImplementedError。

这是供接入适配器组合的协议，Harness 内没有部署实现或自动能力探测。
ModelActivator 适配器只有在观察到 ready 且 model 与目标完全一致后才能返回模型；角色绑定仍由
现有应用流程更新。保留外部引用本身不承诺控制器重启后的持久恢复能力。

[model_service.py](model_service.py) 定义 ModelRef、ModelBackend、ModelUpdatable、请求/响应和 TokenUsage；[ArtifactRef](artifacts.py) 只携带产物身份、URI、版本和摘要，不含内容，解析与校验由产物存储负责。模型引用相同不表示角色同步；真实权重加载归模型后端。

可选并行协议位于 `coordination.py`：`ConcurrentExplorationEndpoint.pending_all()` /
`execution_views()` 和 `ConcurrentDecisionEndpoint.pending_all()` / `update_executions()`。
单条 accept/advance/inspect/cancel/acknowledge 始终使用 directive_id。
`ExplorationProgress.version` 从 0 开始；`DirectiveUpdate(directive_id, expected_version,
idempotency_key, changes)` 是独立控制请求，不替换原始 Directive。
确认后的 `UpdateReceipt` 版本恰好增加 1；同键同请求重复返回原回执，不能用同键改变内容。
支持哪些字段由可选 backend/handler 声明与验证，协议不承诺任意训练超参可热更新。

Model requests may carry an optional `GenerationRequest.response_schema` JSON
Schema. This is an output-format constraint, independent of model deployment and
weight identity. Policies opt in only through a supporting backend; unsupported
transports must fail explicitly rather than silently discard a requested schema.
