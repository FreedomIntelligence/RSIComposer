# 系统架构

框架提供两个 loop 的公开端点与协同推进，每个 loop 组合一个 Harness 和一个 Action Space，两个角色共享 Experience。组件入口见[根 README](../README.md)，操作方法见[使用指南](GUIDE.md)。

## 模块关系

```text
src/rsisys/
├── loops/
│   ├── decision_making/
│   │   ├── loop.py        # 薄端点与有界推进
│   │   ├── harness/       # 可替换执行边界；engine.py 为默认实现
│   │   └── action_space/  # research 操作与系统指令的描述/校验
│   ├── exploration/
│   │   ├── loop.py        # 薄端点
│   │   ├── harness/       # 系统指令分派；engine.py 为默认实现
│   │   └── action_space/  # Actor 环境动作的描述/校验
│   └── contracts/         # 消息、引用与公开端口
├── experience/            # 共享系统反馈库
├── runtime/               # Runner、RunPolicy、运行恢复与公共后端
├── snapshot/              # 独立保存与恢复
├── assembly.py            # 共享装配上下文、默认工厂与资源清理
├── application.py         # 本地持久化装配
├── config.py              # 内存演示装配
├── cli.py                 # 命令逻辑
└── __main__.py            # CLI 薄入口
```

`DecisionMakingLoop(harness, action_space)` 与 `ExplorationLoop(harness, action_space)` 是组合边界。`harness/engine.py` 负责默认装配、状态协调和公开入口；`loop.py` 只委托公开端点，Decision 的 `run` 额外提供有界推进。默认实现的策略、上下文、记忆、工具、learner/evaluator 和状态均由 Harness 封装。memory/tools 是默认实现细节，不是框架要求每个 Harness 单独暴露的接口；可整体换成不含 memory 的实现。

Decision／Actor Policy 位于默认 Harness 内部：[DecisionPolicy.decide](../src/rsisys/loops/decision_making/harness/policy/README.md) 选择操作、系统指令、等待或结束；Exploration Policy 即现有 [ActorPolicy.act](../src/rsisys/loops/exploration/harness/policy/README.md)，只选择环境动作。Harness 组织执行与结果处理，Action Space 定义可用操作/动作及约束。

默认 Decision Harness 的 `context.py` 做可替换上下文投影，`research.py` 保存模型维护的长期议程与修订；操作绑定仍在 tools 中。完整记录属于 Harness，本地短摘要与共享 Experience 不混为同一存储。

可选 `harness/dsh.py` 接入持久 DSH 会话，复用默认回执生命周期，异步生成研究提案；`dsh.patch.yml` 保留原生压缩并关闭原生副作用工具，提案通过现有 Action Space 执行。整体替换其他 Harness 仍不要求继承默认实现。

`runtime/recovery.py` 保存运行中协同状态；空闲 `snapshot/` 负责独立可移交快照。前者在同一工作目录继续，后者恢复到新目录。运行恢复按意图 / 回执边界处理，中断且结果不明时要求核对，不假设所有外部副作用都可重放。默认仅串行内置组合，外部组件显式实现恢复契约。

默认执行职责落在现有组件中，不新增平级系统：

| 位置 | 执行职责 | Harness 保留的职责 |
|---|---|---|
| Decision `tools/` | 普通 step 与 review 共用参数校验、授权、执行、结果归一化及固定回执记录 | 上下文、policy 调用、预算、等待 / 继续与反馈接纳 |
| Exploration `environment/` | EpisodeExecution 运行 Actor episode、记录轨迹与系统反馈，失败补写 | 指令占用与本地控制入口 |
| Exploration `learner/`、`evaluation/` | LearningExecution / EvaluationExecution 管理作业、验证结果、记录证据；learning 还保留候选并应用版本 | 选择对应执行组件 |
| Exploration `rl.py` | LocalRLExecution 顺序协调 rollout、训练和应用 | 对外启动 / 推进 / 取消入口 |

这些执行组件只持有所需的 policy、environment、memory、Experience、状态和作业依赖，不反向持有整个 Harness。它们是默认实现内部的职责划分；外部 Harness 不必照搬。Command handler 复用这些组件，普通本地调用与系统指令不各写一套算法。

内置细节通过 `loop.harness` 访问；Action Space 通过 `loop.action_space` 访问，不保留 `loop.policy`、`loop.memory`、`loop.space` 等旧属性别名。内部组件和默认行为分别见 [Decision Harness](../src/rsisys/loops/decision_making/harness/README.md) 与 [Exploration Harness](../src/rsisys/loops/exploration/harness/README.md)。

## 两种 Action Space 与系统指令

| 边界 | 定义 | 执行位置 |
|---|---|---|
| Decision Action Space | 本地 research 操作，以及 execute / sft / rl / evaluate / configure_prompt 系统指令 | 本地操作由 Decision Harness 的 tools 执行；Directive 交给 Exploration Harness |
| Exploration Action Space | Actor 的环境动作，如 echo / execute | Harness 调用 ActorPolicy.act，校验提案，再交给环境执行 |

内置指令定义集中在 `decision_making/action_space/directives.py`；可通过 `DirectiveDefinitions` 注入其他模式。探索侧通过 `DefaultExplorationHarness(command_handlers=...)` 绑定执行器，两个映射在装配时显式配对。新增系统指令不需要扩充 Actor Action Space。

Decision Action Space 只负责 `describe` / `validate` / `validate_directive`，通过 `directive_modes()` 列出系统指令模式；它不执行工具或训练。Directive 来自 decision policy 或外部系统，目标是 Exploration Harness，不是 ActorPolicy 的动作。指令模式 `execute` 与环境动作 `execute` 分属不同层次。

Exploration Harness 按模式分派：execute 运行 Actor episode；SFT、prompt 配置、评测直接进入对应控制流程，绕过 `ActorPolicy.act`；RL 只有 rollout 阶段使用 Actor 动作，训练与版本应用不进入 Actor Action Space。推理评测应由 evaluator 调用所需模型服务，不通过 Actor 的环境动作选择。默认训练/评测仍为 Mock，不提供真实梯度或推理打分；`configure_prompt` 使用 `PromptSpec(instructions)` 更新 Harness 保存的 Actor 指令文本；空字符串清空配置，后续 Actor 上下文读取该文本，见[共享协议](../src/rsisys/loops/contracts/README.md)。

训练请求是后端中立（backend-neutral）的接口，只约束模型、非空 method / recipe 字符串与引用元组类型；不把训练算法限定为 sft / rl，也不要求引用非空。Actor 请求允许外部后端自主采样；Decision 请求保留原位置参数，追加 `data_refs=()` 和 `method="sft"`，支持外部数据。提供本地轨迹或系统经验引用时，默认 Harness 仍验证其归属。数据格式和算法能力由具体 learner 校验。

默认 `rl` 保留本地顺序采样循环。可选 `train` 通过 `training_directive()` + `TrainingCommandHandler` + 注入的 learner 配对启用，不属于默认指令映射：Harness 将整个训练任务作为一个作业提交，后端可独立负责 rollout 与更新，Harness 负责作业生命周期、证据和可选应用。该路径开放外部 RL/SFT 框架接入，但不意味着已有 verl 等框架的具体适配。[backend_training.py](../examples/backend_training.py) 是独立子进程中的真实 CPU 标量回归示例，展示作业与产物边界，不演示 LLM 训练或 RL rollout；装配见[使用指南](GUIDE.md#接入整项训练任务)。

## 双 loop 协同

1. Decision Harness 读取上下文、完成本地准备，policy 发出 Directive；Action Space 校验操作和指令模式。
2. Runner 将 Directive 交给 Exploration 端点，Harness 接收、校验 payload 并推进对应流程。外部系统也可直接通过同一端点提交 Directive。
3. 遇到 waiting，Runner 返回调用方，下次推进继续原任务。指令可指定首次复查间隔；到期时 Decision Harness 可在有限操作预算内按需研究和读取实验信息，再选择继续、取消和下一次复查间隔。
4. Exploration Harness 汇总终态 Feedback；Runner 通过 Decision 的 `receive_feedback` 交付。默认 Decision Harness 将策略结果记为 improvement，供下一次决策使用。
5. 反馈交付成功后 acknowledge，释放探索流程占用。

默认串行装配一次协调一个未完成交付的指令；可选 `Runner(parallel=True)` 与 `ConcurrentExplorationHarness(factory, max_in_flight=...)` 支持多个独立实验。Decision Harness 用 `max_outstanding` 限制未完成指令数，模型从 outstanding / executions 查看各实验；每 tick 公平推进一个实验，未完成时仍可进行研究或提交新实验。一条 RL 指令内部可以多次采样和更新 Actor。后端轮询由调用方驱动，决策复查由 Directive / ReviewDecision 设定频率，两者不绑定。时间/token 自训练触发及共享模型更新事务尚未实现。并行实验由工厂创建隔离的 Actor、environment、memory 和执行状态；资源分配、作业是否物理并行由后端决定。模型可共享或独立绑定，部署在框架外；角色引用同一模型不表示训练结果自动同步。

运行中的 review 保留 continue / cancel。安装 `ExperimentControls` 后，研究者可通过普通获授权工具按 ID inspect / cancel / update / compare。`DirectiveUpdate` 携带 expected_version、幂等键和 changes；原指令不可变，支持更新的 handler/backend 应用控制变更并确认版本，不支持的字段或后端明确拒绝。候选比较要求已知完整训练 base、相同评测协议/输入及有效指标；排序不自动采纳。训练、评测、采纳的业务策略属于可替换的 Decision Harness，不写入 Runner。

Decision-maker 在科研场景中充当 researcher。实验查询由 Exploration Harness 及其后端提供，Decision Action Space 声明查询操作，Decision Harness 的 tools 绑定执行。查询能力可发现、可替换，返回内容带来源与时间，大内容可通过引用读取；可访问的信息不必全部预先注入模型上下文。`WorkspaceTools` 提供挂载工作区内的 list/read/search 基础操作，研究者可自行读日志、数据、配置和代码；实时服务内部状态仍需后端提供读取途径。外部权限配置决定当前可用 Action Space；默认运行中只读，也可开放其他操作或全部已接入能力。Harness 在展示和执行前检查权限，read_only 是操作属性而非固定限制。权限表达“当前允许做什么”，沙箱限制“执行代码实际能接触哪些资源”；例如允许训练但禁止删除候选，由能力权限检查，限制文件、网络和设备访问则由部署的沙箱承担。两者互补；本地进程后端不是宿主安全隔离。原始查询记录留在研究侧 history，最终判断引用这些记录进入 Experience；不新增独立的观察或记忆子系统。

默认 Decision 上下文另含当前 Exploration 模型/进度、Decision 模型、发出指令时的版本归因，以及每个候选最近一次评测的协议、输入引用、结果和证据。这些事实不依赖有限长度的任务 history；完整评测档案仍由 Experience 保存。

## 外部运行控制

运行是否继续由 runtime 的 [RunPolicy](../src/rsisys/runtime/run_policy.py) 决定，接口为 `evaluate(RunContext) -> RunDecision`；Runner 接收 `run_policy=` 并执行决定。它与 Harness 内决定研究步骤的 DecisionPolicy、选择环境动作的 ActorPolicy 分开：默认 researcher 模式接受研究者 finish，time 模式运行到时限，goal 模式在目标未达时继续。可替换 goal callable 根据已交付 outcomes 判断；MetricGoal 提供绑定评测协议的指标阈值判断，默认不接受 Mock 结果，达标不等于部署候选。

研究者 finish 后，继续运行的策略通过显式 `restart_research` 开启新研究轮次，保留 memory、Experience 和候选。轮次间隔由后续 tick 驱动，`run()` 在 waiting 时返回，不阻塞 sleep。策略停止后 Runner 禁止新实验，执行 decision 指定的 `shutdown='drain'` 或 `'cancel'`，继续终态反馈交付。超时默认 cancel，另设 `cleanup_timeout_seconds=30` 的清理上限；未确认任务明确记录，不能视为成功或已取消。收尾异常原样交给调用方，保留状态供下次 step 重试；超时检查仍需调用方持续推进，不自动硬杀后端。接口、上下文和清理语义见 [Runtime](../src/rsisys/runtime/README.md)，装配见[使用指南](GUIDE.md#选择运行策略)。

## Experience 与持久化

| 概念 | 职责与范围 |
|---|---|
| Feedback | 单条 Directive 的终态回执，由 Runner 交付给决策端 |
| ExplorationProgress / ExecutionView | 当前执行阶段、模型绑定与可选指标，不是另一套记忆系统 |
| Experience | 跨轮共享的系统证据，记录执行、训练、评测与策略结果 |
| memory | Harness 内部的任务上下文，默认包含本地 history 与工作记忆；替代 Harness 不必提供 |

Experience 收集 runtime_feedback / training_run / evaluation / improvement / insight。scope 仅标记来源，不限制可见性，也不指定优化对象。默认 Decision Harness 使用可替换 `FeedbackSelector.select(experience, objective, limit=...)` 形成模型的反馈输入；默认 RecentFeedbackSelector 选取最近记录，自定义实现可筛选或派生带 source_refs 的视图，不改写原始库、不自动训练。默认 Decision policy 可检索全部来源并选择策略；Actor policy 默认只接收 Harness 提供的任务上下文。improvement / insight 是记录接口，不表示已实现自动系统自修改。

默认 Harness 将原始 test-time history 留在自身实现中。每个 Actor episode 写一条 runtime_feedback，`Feedback.evidence_refs` 引用系统记录；RL 的 `trajectory_refs` 则引用默认 Exploration Harness 的本地轨迹，不把原始轨迹复制进 Experience。

Runtime 提供运行策略、模型、作业、产物、执行后端与 Runner 协调；实验业务编排属于 Harness。Snapshot 在空闲边界保存内置 Harness 状态、模型绑定、共享 Experience 及产物；运行策略仅兼容默认 ResearcherRunPolicy，其他策略、替代 Harness、并行装配或自定义反馈提取器需显式集成。

## 交付与失败边界

- 串行 Runner.step 最多调用一次普通决策或一次复查流程，并推进一个探索阶段；并行 tick 可调用一次普通决策并推进一个实验，该实验到期时可额外复查。复查内部允许有界的模型/查询往返。run 遇到 waiting 返回，不连续轮询。等待可以没有 JobRef，rounds 统计已交付反馈的指令数，与 RunContext.research_round 的研究轮次不同；一次有界 run 返回不等于运行策略已结束。
- 新决策使用新 ID；同 ID 同内容重复接收返回原进度，内容不同则拒绝。串行默认实现中，指令交付确认前独占探索流程；并行装配中占用对应实验实例。
- 取消只提出请求，后续推进确认终态；取消与训练成功竞速时不自动应用候选。同步环境调用不能由下一 tick 中途抢占。
- 提交响应丢失后保留请求与幂等键，不重新生成任务。后端须兑现幂等提交；非终态查询或记录异常允许后续推进重试。
- Runner 保留待交付反馈；默认 Decision Harness 使用稳定记录 ID，按指令 ID 识别相同交付的重试。写入失败只重试交付，不重跑 Actor。
- 默认 Exploration 端点可发现未确认的指令；重建 Runner 后先恢复交付，避免决策侧已接收、探索侧未确认时错误发出下一条指令。两侧身份冲突时停止推进。
- 后端终态结果违反请求身份/角色/协议时，指令转为失败反馈并可正常确认退出；暂时的网络或记录失败继续保留原任务供重试。
- 默认 Harness 在操作或 episode 已执行但记录失败时保留结果和稳定 ID，下次推进只补写。结果保存前不交付终态，不重复执行或 rollout；执行异常仍抛给调用方，清理异常附加到原异常。
- 未启用 durable 时待写结果和回执仅在进程内；启用后按 Runner 边界保存，重启恢复原预算与任务身份。未确认副作用须核对后继续，不自动重放；空闲 Snapshot 仍禁止待补写状态。

`completed` 表示终态反馈已交付，任务结果看 Feedback；质量提升需真实评测确认。模型绑定、服务部署与候选验收由对应实现或策略负责。

## 尚待定义的边界

- 优化目标可能是 Actor 模型/prompt、决策者自身，或 Harness 配置/代码；当前没有统一的 target 接口。
- 决策者训练可通过默认 Decision Harness 在本地调用，但没有自动自优化指令或触发机制。
- 候选跨轮保存，显式启用 `apply_candidate` 可独立应用已评测候选；验收指标和阈值由具体 policy 决定，没有全局默认验收规则。SFT 与可选 train 的 `apply_result` 默认 false，默认 RL 每轮应用学习结果，evaluate 不自动部署候选。

## 图示

![系统流程](diagrams/system-flow.svg)

![文件架构](diagrams/file-architecture.svg)

运行 `python docs/diagrams/build_diagrams.py` 重生成两个 SVG；同一生成器重复运行应得到相同字节。PNG 仅在临时目录中用于视觉验收，不作为仓库文档副本保存。
