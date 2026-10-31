# 共享协议

两个 loop 共用的消息、引用和公开端口；此目录不包含后端实现。协同顺序与交付规则见[系统架构](../../../../docs/ARCHITECTURE.md)。

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

[coordination.py](coordination.py) 定义 DecisionEndpoint、ExplorationEndpoint、ExplorationProgress、LoopStepResult 和 OperationResult。DecisionEndpoint 统一通过 receive_feedback(directive, feedback) 接收反馈；它们是 Runtime 协调的公开边界，结果类型不依赖具体 Harness，可直接用于自定义端点和运行时类型反射。

## 作业与版本

模型产物、推理可用性和角色绑定是三个不同概念。`Candidate.changes` 引用产物，`TrainingResult.model` 指定预期启用的版本，不表示服务已加载。两个默认 Harness 接受 `model_activator`；真实结果应用前调用 `ModelActivator.activate(result, idempotency_key=...)`，仅在返回完全匹配的 ModelRef 后更新 Policy 绑定。适配器负责部署或检查已经部署的版本，无需重新部署已有服务。

部署进行中或响应不确定时抛出 `ModelActivationPending`，调用方稍后推进同一任务；键在重试间保持一致，适配器必须兑现幂等性。其他异常表示失败，旧角色绑定保持不变。模拟结果在未配置 activator 时仍可模拟切换；真实结果缺少 activator 时拒绝应用，但仍可保存为候选或提交评测。取消待启用的训练指令不会切换绑定，不负责撤销已在外部启动的部署。

外部 `ArtifactRef` 无需先登记到内置 ArtifactStore；version 与 digest 未知时用 None。传递引用不代表下载、解析或验证了内容；这些由接入实现负责。内置存储仍只读取自身管理的产物，并保留完整性校验。

[jobs.py](jobs.py) 定义 JobRef、JobStatus、JobOutcome。queued / running / unknown 非终态；unknown 不能当作失败。仅 succeeded 携带结果，failed 必须携带错误，cancelled 表示作业已确认取消。训练/评测后端的 result(job) 只读取成功结果，不轮询、不推进任务；内置 Mock 在其他状态下抛出 RuntimeError。

相同幂等键与相同请求应返回同一任务，不同请求复用键应拒绝。取消请求不等于作业取消完成；命令取消时作业仍可能成功，Feedback 的 cancelled 不等同于 JobStatus.CANCELLED。具体后端必须实现作业约定，[Runtime](../../runtime/README.md) 的进程内记账不能替代远端幂等保证。

[TrainingResult](training.py) 关联 base / model / candidate / metrics / simulated。[Candidate](candidate.py) 描述变更产物和受影响角色，不表示已经验收。默认 Harness 校验角色归属和基础版本；应用结果需要 policy 支持模型更新。

[model_service.py](model_service.py) 定义 ModelRef、ModelBackend、ModelUpdatable、请求/响应和 TokenUsage；[ArtifactRef](artifacts.py) 只携带产物身份、URI、版本和摘要，不含内容，解析与校验由产物存储负责。模型引用相同不表示角色同步；真实权重加载归模型后端。
