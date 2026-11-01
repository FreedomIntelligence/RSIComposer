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
├── runtime/               # Runner 与公共后端
├── snapshot/              # 独立保存与恢复
├── assembly.py            # 共享装配上下文、默认工厂与资源清理
├── application.py         # 本地持久化装配
├── config.py              # 内存演示装配
├── cli.py                 # 命令逻辑
└── __main__.py            # CLI 薄入口
```

`DecisionMakingLoop(harness, action_space)` 与 `ExplorationLoop(harness, action_space)` 是组合边界。业务编排位于各自 `harness/engine.py`；`loop.py` 只委托公开端点，Decision 的 `run` 额外提供有界推进。默认实现的策略、上下文、记忆、工具、learner/evaluator 和状态均由 Harness 封装。memory/tools 是默认实现细节，不是框架要求每个 Harness 单独暴露的接口；可整体换成不含 memory 的实现。

Policy 位于默认 Harness 内部：[DecisionPolicy.decide](../src/rsisys/loops/decision_making/harness/policy/README.md) 选择操作、系统指令、等待或结束；Exploration Policy 即现有 [ActorPolicy.act](../src/rsisys/loops/exploration/harness/policy/README.md)，只选择环境动作。Harness 组织执行与结果处理，Action Space 定义可用操作/动作及约束。

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
3. 遇到 waiting，Runner 返回调用方，下次推进继续原任务。指令可指定首次复查间隔；到期时 Decision Harness 可在有限查询预算内按需读取实验信息，再选择继续、取消和下一次复查间隔。
4. Exploration Harness 汇总终态 Feedback；Runner 通过 Decision 的 `receive_feedback` 交付。默认 Decision Harness 将策略结果记为 improvement，供下一次决策使用。
5. 反馈交付成功后 acknowledge，释放探索流程占用。

当前一次只有一个未完成交付的指令；一条 RL 指令内部可以多次采样和更新 Actor。后端轮询由调用方驱动，决策复查由 Directive / ReviewDecision 设定频率，两者不绑定。时间/token 自训练触发、多 Actor 并发及共享模型更新事务尚未实现。模型可共享或独立绑定，部署在框架外；角色引用同一模型不表示训练结果自动同步。

运行中的控制先支持 continue / cancel，不修改已提交指令。换策略需确认旧指令终态并完成交付后，再发送新 ID 的指令。训练、评测、采纳的业务策略属于可替换的 Decision Harness，不写入 Runner。

Decision-maker 在科研场景中充当 researcher。实验查询由 Exploration Harness 及其后端提供，Decision Action Space 声明查询操作，Decision Harness 的 tools 绑定执行。查询能力可发现、可替换，返回内容带来源与时间，大内容可通过引用读取；可访问的信息不必全部预先注入模型上下文。运行中 review 只开放声明为 read_only 的操作，普通决策可使用完整 Action Space。原始查询记录留在研究侧 history，最终判断引用这些记录进入 Experience；不新增独立的观察或记忆子系统。

默认 Decision 上下文另含当前 Exploration 模型/进度、Decision 模型、发出指令时的版本归因，以及每个候选最近一次评测的协议、输入引用、结果和证据。这些事实不依赖有限长度的任务 history；完整评测档案仍由 Experience 保存。

## Experience 与持久化

| 概念 | 职责与范围 |
|---|---|
| Feedback | 单条 Directive 的终态回执，由 Runner 交付给决策端 |
| ExplorationProgress / ExecutionView | 当前执行阶段、模型绑定与可选指标，不是另一套记忆系统 |
| Experience | 跨轮共享的系统证据，记录执行、训练、评测与策略结果 |
| memory | Harness 内部的任务上下文，默认包含本地 history 与工作记忆；替代 Harness 不必提供 |

Experience 收集 runtime_feedback / training_run / evaluation / improvement / insight。scope 仅标记来源，不限制可见性，也不指定优化对象。默认 Decision policy 可检索全部来源并选择策略；Actor policy 默认只接收 Harness 提供的任务上下文。improvement / insight 是记录接口，不表示已实现自动系统自修改。

默认 Harness 将原始 test-time history 留在自身实现中。每个 Actor episode 写一条 runtime_feedback，`Feedback.evidence_refs` 引用系统记录；RL 的 `trajectory_refs` 则引用默认 Exploration Harness 的本地轨迹，不把原始轨迹复制进 Experience。

Runtime 提供模型、作业、产物、执行后端与 Runner 协调；业务编排属于 Harness。Snapshot 在空闲边界保存内置 Harness 状态、模型绑定、共享 Experience 及产物，当前不自动适配替代 Harness。

## 交付与失败边界

- Runner.step 最多调用一次普通决策或一次复查流程，并推进一个探索阶段；复查内部允许有界的模型/查询往返。run 遇到 waiting 返回，不连续轮询。等待可以没有 JobRef，rounds 统计已交付反馈的指令数。
- 新决策使用新 ID；同 ID 同内容重复接收返回原进度，内容不同则拒绝。指令交付确认前独占探索流程。
- 取消只提出请求，后续推进确认终态；取消与训练成功竞速时不自动应用候选。同步环境调用不能由下一 tick 中途抢占。
- 提交响应丢失后保留请求与幂等键，不重新生成任务。后端须兑现幂等提交；非终态查询或记录异常允许后续推进重试。
- Runner 保留待交付反馈；默认 Decision Harness 使用稳定记录 ID，仅接受最近一次相同交付的重试。写入失败只重试交付，不重跑 Actor。
- 默认 Exploration 端点可发现未确认的指令；重建 Runner 后先恢复交付，避免决策侧已接收、探索侧未确认时错误发出下一条指令。两侧身份冲突时停止推进。
- 后端终态结果违反请求身份/角色/协议时，指令转为失败反馈并可正常确认退出；暂时的网络或记录失败继续保留原任务供重试。
- 默认 Harness 在操作或 episode 已执行但记录失败时保留结果和稳定 ID，下次推进只补写。结果保存前不交付终态，不重复执行或 rollout；执行异常仍抛给调用方，清理异常附加到原异常。
- 待写结果和回执仅在进程内；待补写期间禁止快照，不提供崩溃恢复。

`completed` 表示终态反馈已交付，任务结果看 Feedback；质量提升需真实评测确认。模型绑定、服务部署与候选验收由对应实现或策略负责。

## 尚待定义的边界

- 优化目标可能是 Actor 模型/prompt、决策者自身，或 Harness 配置/代码；当前没有统一的 target 接口。
- 决策者训练可通过默认 Decision Harness 在本地调用，但没有自动自优化指令或触发机制。
- 候选跨轮保存，显式启用 `apply_candidate` 可独立应用已评测候选；验收指标和阈值由具体 policy 决定，没有全局默认验收规则。SFT 与可选 train 的 `apply_result` 默认 false，默认 RL 每轮应用学习结果，evaluate 不自动部署候选。

## 图示

![系统流程](diagrams/system-flow.svg)

![文件架构](diagrams/file-architecture.svg)

运行 `python docs/diagrams/build_diagrams.py` 重生成两个 SVG；同一生成器重复运行应得到相同字节。PNG 仅在临时目录中用于视觉验收，不作为仓库文档副本保存。
