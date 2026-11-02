# Exploration Harness

默认 Harness 通过 `describe_queries()` / `query(QueryRequest)` 提供显式只读查询，具体实现在 [queries.py](queries.py)。查询只查看当前模型、已有 inspect 进展、任务回执、本地 history、候选及 Experience 摘要和产物引用；不会 poll 或触发外部执行。返回数据有页数和大小限制，远端日志没有显式来源时标记不可用。ExplorationLoop 只转发 QueryProvider 能力，其他 Harness 可不实现；不支持时发现结果为空，调用 query 明确抛 NotImplementedError。完整参数与正文读取入口见 [Decision query tools](../../decision_making/harness/tools/README.md#只读查询)。

[ExplorationHarness](__init__.py) 是系统指令执行边界，提供 `accept(directive)`、`advance(directive_id, action_space)`、`inspect`、`cancel` 和 `acknowledge`，由 [ExplorationLoop](../README.md) 委托。它接收 decision policy 或外部系统的 Directive；协议不要求暴露 Actor policy、memory 或工具。

## 默认实现与分派

[DefaultExplorationHarness](engine.py) 接收 `policy, environment, experience, learner=None, evaluator=None`，可选关键字参数为 `memory=None, memory_limit=32, command_handlers=None, model_activator=None`；默认使用 ExplorationMemory。Actor 上下文包含当前任务 history 和工作记忆，不默认注入共享 Experience。

| 系统指令 | 默认流程 | 调用 ActorPolicy.act |
|---|---|---|
| execute | 环境 episode：观察 → Actor 提案 → Action Space 校验 → 环境动作 | 是 |
| sft | learner 提交、轮询，按配置应用候选 | 否 |
| rl | rollout → 训练 → 应用版本 → 下一轮 rollout | 仅 rollout |
| evaluate | evaluator 提交与轮询 | 否 |
| configure_prompt | PromptSpec(instructions) 替换 Actor 指令文本 | 否 |

`train` 为显式启用的扩展，不在默认映射中。配对注册 `training_directive()` 与 `command_handlers={"train": TrainingCommandHandler()}` 并注入 learner 后，整项任务通过单个后端作业执行，后端可自主 rollout 与训练；Harness 不启动本地采样循环。默认 `rl` 的上述流程不变，详见 [learner](learner/README.md)。

推理评测属于 evaluator 的模型调用流程，不通过 Actor 的环境动作选择。内置 learner/evaluator 仍是 Mock，既不执行真实梯度训练，也不产生真实推理评分；prompt 配置不等于模型权重训练。

configure_prompt 将 instructions 保存到 Harness 状态，空字符串清空；后续 execute / RL rollout 的 ActorContext 使用该文本。配置操作不调用 Actor.act，系统反馈只记录文本摘要哈希、字符数及是否配置，不写入原文。

## 内部组件

| 组件 | 默认职责 |
|---|---|
| [policy](policy/README.md) | Actor 环境动作选择 |
| [environment](environment/README.md) | Environment 执行动作；EpisodeExecution 运行 episode、保存轨迹与反馈、重试证据写入 |
| [memory](memory/README.md) | 本地轨迹与工作记忆 |
| [learner](learner/README.md) | LearningExecution 管理训练作业、校验结果、保留候选与应用版本；算法仍由 learner 后端执行 |
| [evaluation](evaluation/README.md) | EvaluationExecution 管理评测作业、校验结果与记录证据；评分由 evaluator 后端执行 |
| [commands.py](commands.py) | 系统指令接收、校验、分派与交付状态 |
| [rl.py](rl.py) | RLPlan / RLState，以及编排 episode → 训练 → 应用版本的 LocalRLExecution |
| [state.py](state.py) | 执行、训练和待写结果状态 |

Action Space 位于 [../action_space/](../action_space/README.md)，只描述和校验 Actor 环境动作。上述 policy、上下文、记忆、工具/环境、learner/evaluator 是默认 Harness 的内部实现；可整体替换成不含 memory 的 Harness。

engine 负责装配、控制占用和兼容公开入口；具体执行由公开组件 `episodes`、`learning`、`evaluations`、`local_rl` 完成。commands 直接消费组件方法，不回调 Harness 私有执行方法。EpisodeExecution 只接收 policy、environment、memory、experience、state 和 memory_limit；LearningExecution 接收 policy、memory、experience、state、JobBook 和 model_activator；EvaluationExecution 接收 experience、state 和 JobBook；LocalRLExecution 接收 episode、learning 和 state。四者均可独立构造，不持有 Harness。

现有 policy、environment、memory、experience、memory_limit、training、evaluation、learner、model_activator 属性保留转发，保证替换依赖后执行使用新对象。快照恢复赋值 `harness.state` 时，同步四个组件的 state；替换 `episodes` 或 `learning` 时同步 LocalRL 的组件引用。自行装配的组件应共享同一 state、policy、memory 和 experience，并在空闲边界替换。组件方法供执行编排使用；外部本地控制仍使用 Harness 入口以遵守指令占用。快照只支持内置组件和一致的依赖绑定，自定义组件需要显式快照集成。

## 生命周期与本地控制

accept 复制并校验请求；advance 推进阶段，execute 会同步运行一个有界 episode；inspect 不推进；cancel 标记取消，后续 advance 处理；终态 Feedback 交付后 acknowledge 释放占用。进度返回副本，终态由是否携带 Feedback 判断。

`pending()` 返回当前未 acknowledge 的 ExplorationProgress 副本，包括已经终态但尚未交付确认的回执；空闲时返回 None。`execution_view()` 返回当前 policy 的 ModelRef（若有）和 pending 进度，不执行或轮询后端。elapsed_seconds 从首次 advance 开始用进程内单调时钟计时，终态后固定，不承诺跨进程恢复；尚未开始时为 None。metrics 只来自已取得的 Feedback，等待后端时保持空字典，不补做 poll。

内置本地控制通过 `loop.harness` 使用：`run(directive, loop.action_space)`、`submit_learning` / `poll_learning` / `cancel_learning` / `apply_learning`、`submit_evaluation` / `poll_evaluation` / `cancel_evaluation`，以及 `start_rl` / `step_rl(loop.action_space)` / `cancel_rl`。示例见 [loop_internals.py](../../../../../examples/loop_internals.py)。

训练成功后，默认 Harness 按 candidate_id 保存完整 TrainingResult。`candidate_result(id)` 返回副本，`apply_candidate(id)` 在本地空闲边界应用已保存候选，不依赖原训练 job 继续存在。指令路径需显式配对决策侧 `apply_candidate_directive()` 与 `ApplyCandidateHandler()`；验收标准由 policy 或调用方负责，handler 不假定评测通过。应用仍校验 base 是否过时、模型 readiness 和控制占用；同一候选重试不会重复切换绑定。应用成功后的证据写入失败只重试记录。

已应用候选只有在 policy 仍绑定其目标版本时才可幂等返回；切换到其他版本后，新应用请求会被拒绝为 stale，不虚报成功或回滚模型。旧 directive 的回执重放仍返回当时的结果，不重复执行。

指令未 acknowledge 前，禁止本地执行、新提交、模型应用和 RL 控制；已有 job 仍可 poll / cancel。本地 RL 占用执行、训练提交与模型更新，评测可独立提交。SFT/RL 取消与成功竞速时不自动应用本次候选，已应用的版本不回滚。RL 当前顺序采样与训练，不支持并行采样或异步权重广播。

每个执行 episode 写一条 runtime_feedback，Feedback.evidence_refs 引用共享系统记录；RL 的 trajectory_refs 引用默认 Harness 本地 history，不能互换。训练/评测终态写入共享 [Experience](../../../experience/README.md)。执行后记录失败时缓存结果与稳定 ID，再次推进只补写，不重新 rollout；待写期间禁止快照，暂存不支持崩溃恢复。

## 扩展系统指令

`command_handlers={mode: handler}` 在默认映射上增加或覆盖模式；handler 继承 [CommandHandler](commands.py)。映射属于当前 Harness 实例，不通过全局注册或核心 if/elif 分派：

- `prepare(harness, state)`：校验 payload、冻结请求；此时不启动外部副作用。
- `advance(harness, state, action_space)`：推进一个有界阶段；等待时返回 None，终态返回匹配 directive_id 的 Feedback。
- `cancel_before_start(...)`：默认返回 CANCELLED，可覆盖；已启动的任务由 advance 检查 `state.cancel_requested` 并处理取消。

`CommandState` 提供 directive、phase、request、job、data、cancel_requested 和稳定的 idempotency_key。框架在首次调用 advance 前独立标记 started；之后即使 phase 未改变，取消也会交给 handler 处理。handler 用 phase/data 保存自身进度；响应丢失后继续使用原请求与幂等键，后端需保证重复提交不重复执行。advance 抛出异常且未产生终态回执时保留状态，后续 tick 重试；永久失败应返回 FAILED Feedback。

prepare 失败会形成失败回执并重新抛出异常；调用方交付并 acknowledge 后才能接收下一条指令。handler 负责自身详细证据的稳定写入；Runner 交付后，默认 Decision Harness 仍统一记录策略结果。终态交付失败只补交 Feedback，不再调用 handler。框架统一保留原指令的 simulated 标记，handler 不能将模拟来源降为真实结果。全部回执仅在进程内，不提供崩溃恢复或自定义快照。

决策侧需同时注入对应的 [DirectiveDefinition](../../decision_making/action_space/README.md#扩展系统指令)，并将模式加入 directive_modes。自定义 handler 可以调用自身后端；ActorTrainingRequest 允许自定义 method，具体算法与数据处理能力由 learner 实现。整项训练可直接使用 [TrainingCommandHandler](learner/command.py)，装配见[使用指南](../../../../../README.md#接入整项训练任务)。

### 默认 Harness 的公共 handler 支持方法

`harness.commands` 的以下公共 helper 供 `DefaultExplorationHarness` 的 handler 复用，不属于通用 `ExplorationHarness` 协议；其他 Harness 无需提供这些方法：

- `validate_training_payload(payload, expected, *, apply)`：检查 learner 是否存在、payload 是否为 expected 类型，以及 apply 为真时 policy 是否支持模型更新。成功返回 None，失败抛出异常，不启动外部工作；base、数据输入等具体请求约束仍由 handler 校验。
- `advance_job(state, *, training)`：handler 先在 prepare 中冻结 `state.request`，再在 advance 中调用。training 为真使用训练作业，为假使用评测作业；每次推进一个提交或轮询/取消阶段，复用请求和幂等键，记录终态证据，并按 payload.apply_result 应用训练结果。该 helper 返回 None，直接更新 state.phase、state.job 和终态 state.feedback；调用方无需再返回 Feedback。

`advance_job` 的不确定提交、轮询、证据写入失败保留回执供重试。应用阶段抛出 `ModelActivationPending` 时同样保持未终态，下次 advance 对同一 job 重试应用；其他应用异常生成 FAILED 回执后重新抛出。取消后不再应用候选，已发生的外部应用不回滚。开始前的取消仍由框架的 cancel_before_start 处理。这些 helper 应在 handler 生命周期内使用，不替代 Harness 的本地控制入口。

Harness 检测到已缓存终态结果的类型、base、角色、candidate ID 内容或评测 candidate/protocol 不匹配时，抛出 `InvalidJobResult`（ValueError 子类）。命令将该确定的协议错误转为 FAILED 回执后重新抛出，供调用方交付并 acknowledge；不会发布错误候选或伪造有效训练/评测证据。RL 序列也结束为 failed，避免残留控制占用。后端本身抛出的异常和 Experience 写入异常不因此转为终态，仍可按原请求、job 与记录 ID 重试。

`ConcurrentExplorationHarness(factory, max_in_flight=2)` 是可替换的实验容器。
factory 接收完整 Directive，必须为每个 ID 创建独立的 harness、policy、environment、
memory、state、job books 和命令处理组件；容器检查这些可见对象的共享别名并拒绝复用。
嵌套在自定义组件内的资源隔离由 factory 负责，不能靠浅层检查保证。共享的 backend
客户端必须以完整 JobRef 路由异步任务，并支持并发在途请求；GPU/服务容量由 backend
或 factory 分配。容器不会复制模型权重、跨实验修改 prompt 或自动晋升模型。

`pending_all()` 返回未 ack 的回执，`execution_views()` 按 ID 返回模型和任务绑定。
已 ack 子 harness 继续保留，用于幂等回执和候选读取；没有自动淘汰或持久化策略。
共享 Experience 是显式的跨实验反馈来源，局部 memory 和状态不会共享。

候选跨实验读取通过 `candidate_result(id)`，同 ID 不同结果会拒绝。
`apply_candidate(id, target_directive_id=..., expected_model=...)` 要求目标先终态并 ack，
同时检查目标当前完整 ModelRef 和 TrainingResult.base；只更改这个目标。
`apply_candidate` 指令也可由 factory 安装的 ApplyCandidateHandler 执行，容器会向新目标
显式导入候选结果并检查 base。不会把这个绑定隐式传播到其他实验或新 factory 实例。

动态控制使用 `DirectiveUpdate`，标准 CommandHandler 默认拒绝更新。
自定义 handler 必须实现纯验证 `validate_update(harness, state, request)` 和
确认应用的 `update(harness, state, request)`。`TrainingCommandHandler` 可适配显式提供
`backend.validate_update(job, request)` / `backend.update(job, request)` 的训练后端。
后端负责支持的字段、应用时机和请求去重；成功返回表示已确认生效，失败响应保留原请求供显式重试。
未确认更新只阻止另一条更新，不阻止现有任务的状态推进和取消。handler.advance 在这种情况下
须继续轮询/取消已有任务，不重放更新、不把未知响应记成成功。取消发起后拒绝未确认更新的重试，
继续推进直至终态；原 pending_update 留作证据，版本保持最后确认值。
当前内置训练后端没有通用热更新能力；示例仅修改 mock 的剩余轮询次数。
