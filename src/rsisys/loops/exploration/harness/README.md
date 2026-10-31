# Exploration Harness

[ExplorationHarness](__init__.py) 是系统指令执行边界，提供 `accept(directive)`、`advance(directive_id, action_space)`、`inspect`、`cancel` 和 `acknowledge`，由 [ExplorationLoop](../README.md) 委托。它接收 decision policy 或外部系统的 Directive；协议不要求暴露 Actor policy、memory 或工具。

## 默认实现与分派

[DefaultExplorationHarness](engine.py) 接收 `policy, environment, experience, learner=None, evaluator=None`，可选关键字参数为 `memory=None, memory_limit=32, command_handlers=None`；默认使用 ExplorationMemory。Actor 上下文包含当前任务 history 和工作记忆，不默认注入共享 Experience。

| 系统指令 | 默认流程 | 调用 ActorPolicy.act |
|---|---|---|
| execute | 环境 episode：观察 → Actor 提案 → Action Space 校验 → 环境动作 | 是 |
| sft | learner 提交、轮询，按配置应用候选 | 否 |
| rl | rollout → 训练 → 应用版本 → 下一轮 rollout | 仅 rollout |
| evaluate | evaluator 提交与轮询 | 否 |
| configure_prompt | PromptSpec(instructions) 替换 Actor 指令文本 | 否 |

推理评测属于 evaluator 的模型调用流程，不通过 Actor 的环境动作选择。内置 learner/evaluator 仍是 Mock，既不执行真实梯度训练，也不产生真实推理评分；prompt 配置不等于模型权重训练。

configure_prompt 将 instructions 保存到 Harness 状态，空字符串清空；后续 execute / RL rollout 的 ActorContext 使用该文本。配置操作不调用 Actor.act，系统反馈只记录文本摘要哈希、字符数及是否配置，不写入原文。

## 内部组件

| 组件 | 默认职责 |
|---|---|
| [policy](policy/README.md) | Actor 环境动作选择 |
| [environment](environment/README.md) | 执行动作、生成观察与工具副作用 |
| [memory](memory/README.md) | 本地轨迹与工作记忆 |
| [learner](learner/README.md) | SFT/RL 训练后端 |
| [evaluation](evaluation/README.md) | 候选评测后端 |
| [commands.py](commands.py) | 系统指令接收、校验、分派与交付状态 |
| [rl.py](rl.py) | RLPlan / RLState |
| [state.py](state.py) | 执行、训练和待写结果状态 |

Action Space 位于 [../action_space/](../action_space/README.md)，只描述和校验 Actor 环境动作。上述 policy、上下文、记忆、工具/环境、learner/evaluator 是默认 Harness 的内部实现；可整体替换成不含 memory 的 Harness。

## 生命周期与本地控制

accept 复制并校验请求；advance 推进阶段，execute 会同步运行一个有界 episode；inspect 不推进；cancel 标记取消，后续 advance 处理；终态 Feedback 交付后 acknowledge 释放占用。进度返回副本，终态由是否携带 Feedback 判断。

内置本地控制通过 `loop.harness` 使用：`run(directive, loop.action_space)`、`submit_learning` / `poll_learning` / `cancel_learning` / `apply_learning`、`submit_evaluation` / `poll_evaluation` / `cancel_evaluation`，以及 `start_rl` / `step_rl(loop.action_space)` / `cancel_rl`。示例见 [loop_internals.py](../../../../../examples/loop_internals.py)。

指令未 acknowledge 前，禁止本地执行、新提交、模型应用和 RL 控制；已有 job 仍可 poll / cancel。本地 RL 占用执行、训练提交与模型更新，评测可独立提交。SFT/RL 取消与成功竞速时不自动应用本次候选，已应用的版本不回滚。RL 当前顺序采样与训练，不支持并行采样或异步权重广播。

每个执行 episode 写一条 runtime_feedback，Feedback.evidence_refs 引用共享系统记录；RL 的 trajectory_refs 引用默认 Harness 本地 history，不能互换。训练/评测终态写入共享 [Experience](../../../experience/README.md)。执行后记录失败时缓存结果与稳定 ID，再次推进只补写，不重新 rollout；待写期间禁止快照，暂存不支持崩溃恢复。

## 扩展系统指令

`command_handlers={mode: handler}` 在默认映射上增加或覆盖模式；handler 继承 [CommandHandler](commands.py)。映射属于当前 Harness 实例，不通过全局注册或核心 if/elif 分派：

- `prepare(harness, state)`：校验 payload、冻结请求；此时不启动外部副作用。
- `advance(harness, state, action_space)`：推进一个有界阶段；等待时返回 None，终态返回匹配 directive_id 的 Feedback。
- `cancel_before_start(...)`：默认返回 CANCELLED，可覆盖；已启动的任务由 advance 检查 `state.cancel_requested` 并处理取消。

`CommandState` 提供 directive、phase、request、job、data、cancel_requested 和稳定的 idempotency_key。框架在首次调用 advance 前独立标记 started；之后即使 phase 未改变，取消也会交给 handler 处理。handler 用 phase/data 保存自身进度；响应丢失后继续使用原请求与幂等键，后端需保证重复提交不重复执行。advance 抛出异常且未产生终态回执时保留状态，后续 tick 重试；永久失败应返回 FAILED Feedback。

prepare 失败会形成失败回执并重新抛出异常；调用方交付并 acknowledge 后才能接收下一条指令。handler 负责自身详细证据的稳定写入；Runner 交付后，默认 Decision Harness 仍统一记录策略结果。终态交付失败只补交 Feedback，不再调用 handler。框架统一保留原指令的 simulated 标记，handler 不能将模拟来源降为真实结果。全部回执仅在进程内，不提供崩溃恢复或自定义快照。

决策侧需同时注入对应的 [DirectiveDefinition](../../decision_making/action_space/README.md#扩展系统指令)，并将模式加入 directive_modes。自定义 handler 可以调用自身后端；内置 learner 的 SFT/RL 请求类型不会自动支持新的训练算法。
