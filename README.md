# RSISys

Feedback、显式 Memory、reflect 近期证据覆盖以及新 attempt 的成果继承见
[交接与组件边界](docs/attempt-continuation.md)。医疗候选可使用 `medical_package` /
`medical_preflight` 做无模型调用的打包检查；公开评测的输出质量与比较身份见
[评测协议](docs/evaluation-protocol.md)。
单 meta-agent 可复用 ReMe BM25 检索；安装、装配与验证见
[Memory 后端](docs/memory-backends.md)。

Current working source follows the [single-checkout version policy](docs/version-policy.md).
The programmable medical Lab accepts candidate weights **and inference scripts**
through the [public/closed evaluation protocol](docs/evaluation-protocol.md).

可客制化的 RSI（递归自改进）框架。**一个 meta-agent** 在 **Lab** 中选择并执行研究动作，直接读取原始反馈、调试并管理自己的记忆。Harness 内的 **memory 模块**负责存储与检索；默认不另设整理或总结 agent。Host 定义目标、能力、权限与预算；整套 RSI 方法可以整体接入，也可组合模型、harness、memory、训练和推理后端。

- **一个 meta-agent**：行动选择、反馈解释、上下文管理和记忆组织属于同一工作流。Lab 内部可运行 agent 或直接执行程序。
- **共同演化与 reflect**：RSI 优化器决定两侧改进的触发与编排，包含面向 meta-agent 自身的 reflect 策略；Lab 侧改进作用于其中的研究对象。候选由 meta-agent 按需构建，当前接口能力与尚需适配的闭环见[架构说明](docs/architecture.md#共同演化与-reflect)。
- **任务反馈与长期记忆**：原始错误保留到实验/调试任务关闭；meta-agent 显式决定是否写入、更新或删除 memory，完整原文独立归档。
- **轻量协同**：框架负责指令、作业、反馈与版本引用，不统一各训练框架的数据格式；跨框架数据对齐由接入方完成。
- **自主研究**：支持隔离的并行任务、显式决策依赖、多结果采纳与汇合；研究尝试（Attempt）记录围绕同一想法的一组决策。
- **实验监控**：`rsimonitor` 是与 `rsisys` 并列的只读可视化包。一个项目包含多个实验，一个实验对应一次完整 RSI run；进入实验查看尝试、决策理由、改动、反馈和训练指标。

术语和职责以[统一术语与架构](docs/architecture.md)为准；本 README 提供使用入口，各目录 README 说明接口与实现。`decision_maker`、`DecisionMakingLoop`、`ExplorationLoop` 等保留为兼容标识，不另设同名概念角色。

[快速开始](#快速开始) · [实验监控](#实验监控) · [系统架构](#系统架构) · [使用指南](#使用指南) · [组件文档](#组件文档)

## 快速开始

Python 3.12+，在仓库根目录安装：

```sh
python -m pip install -e .
rsisys demo --config configs/demo.json

# 监控单独安装，也可只安装这一项。
python -m pip install -e ./src/rsimonitor
rsimonitor --runs-dir runs
```

监控默认访问 `http://127.0.0.1:8765/`。未产生 `monitor.json` 时显示空列表；监控不会启动研究或训练。

Linux 可直接使用已锁定的 Pixi 环境，无需全局安装：

```sh
pixi run demo
pixi run coordination-demo
pixi run backend-demo
pixi run monitor
pixi run test
```

默认模型、训练和评测为 Mock；版本变化不证明能力提升。真实模型服务、训练作业和部署通过显式适配器接入。CPU 标量训练示例验证作业链路；Qwen、医学领域装配与真实后端要求见下方示例。

## 实验监控

`rsisys` 与 `rsimonitor` 是两个独立 distribution，互不声明安装依赖。只需监控时，在仓库根目录运行 `python -m pip install ./src/rsimonitor`；也可安装独立构建的 `rsimonitor` wheel，无需安装 RSI 核心。

监控以实验过程为中心，直接读取 `runs/<run-id>/monitor.json`，无需启动两个 loop，也不要求安装训练框架或 Node。它负责实验可视化分析，项目的功能介绍与使用说明由本 README 承载。

- 管理页按项目组织实验，按最近更新时间排列；一个实验是完整 RSI 系统 run，不是单个训练 job。
- 实验主图展示系统实际记录的 Attempt，保留并行、依赖和多结果采纳；展开查看具体 decision 与 feedback。历史数据没有分组时不推断关联。
- 展示已报告的决策理由、参数改动、损失/学习率历史和候选评测对比；缺失信息不补造。
- 中英切换与静默自动刷新，保留所选节点、展开状态和画布位置。

```python
from rsimonitor import MonitorServer, RunStore

store = RunStore("runs")
projects = store.projects()
snapshot = store.read("experiment-id")  # 读取已有实验的公开监控投影

# 可在自己的 Python 程序中嵌入服务；关闭由调用方负责。
with MonitorServer(("127.0.0.1", 8765), "runs") as server:
    server.serve_forever()
```

生产端原子写入轻量快照；监控不读取 checkpoint、训练样本或任意日志文件，不执行训练控制。监控接口不等于统一训练数据协议。协议、API、接入方法与部署说明见 [Monitor README](src/rsimonitor/README.md)。现有 SwanLab 导出为可选示例，见 [export_swanlab.py](examples/qwen_toy/export_swanlab.py)。

## 系统架构

### 模块关系

概念主链路为 **meta-agent → Lab → 原始 feedback → 同一个 meta-agent**。
`harness/agent_memory.py` 提供显式记忆工具和任务反馈窗口；`feedback/` 保留原始回执与归档设施，
独立 LangMem 整理仅作为显式兼容工作流。新的 Lab 通过 `LabExecution.describe/execute` 接入。

```text
src/rsisys/
├── loops/
│   ├── decision_making/
│   │   ├── loop.py        # 薄端点与有界推进
│   │   ├── harness/       # meta-agent（decision侧）的可替换 harness
│   │   ├── feedback/      # 原始反馈、归档与显式兼容处理器
│   │   └── action_space/  # research 操作与系统指令的描述/校验
│   ├── exploration/
│   │   ├── loop.py        # 薄端点
│   │   ├── harness/       # 兼容 Lab harness；系统指令分派与内部执行
│   │   └── action_space/  # Actor 环境动作的描述/校验
│   └── contracts/         # 消息、引用与公开端口
├── experience/            # 共享系统反馈库
├── runtime/               # Runner、RunPolicy、SystemRegistry、恢复与公共后端
├── adapters/              # 可替换的具体训练 / serving 接入，位于 Harness 外
├── snapshot/              # 独立保存与恢复
├── assembly.py            # 共享装配上下文、默认工厂与资源清理
├── components.py          # host 显式组件目录、依赖契约与配置组合
├── methods.py             # 整套 RSI 方法的版本化选择与接入协议
├── research.py            # 研究任务的统一组件装配与配置校验
├── application.py         # 本地持久化装配
├── config.py              # 内存演示装配
├── cli.py                 # 命令逻辑
└── __main__.py            # CLI 薄入口
```

`src/rsimonitor/` 与 `src/rsisys/` 并列，包含监控 API 与前端资源，通过公开快照读取实验，不依赖双 loop 的内部实现。

`examples/medical/` 位于源码包外，通过统一装配入口组合领域任务、固定评测协议、两个 Harness 与 Runner。
它使用共享端口和 adapters，不向核心注入医学数据集、验收阈值或训练配方。
`native.py` 显式包装原始数据、tokenize 回调与进程训练，并保留转化证据；MedicalServing 在评测前
准备独立候选路由。装配可替换 evaluator_factory 或注入 prepare_inference；默认评测 poll 每次
至多推进一个样本，就绪等待保留原作业，明确应用拒绝通过终态反馈返回研究者。

`DecisionMakingLoop(harness, action_space)` 与 `ExplorationLoop(harness, action_space)` 是组合边界。`harness/engine.py` 负责默认装配、状态协调和公开入口；`loop.py` 只委托公开端点，Decision 的 `run` 额外提供有界推进。默认实现的策略、上下文、记忆、工具、learner/evaluator 和状态均由 Harness 封装。memory/tools 是默认实现细节，不是框架要求每个 Harness 单独暴露的接口；可整体换成不含 memory 的实现。

Decision／Actor Policy 位于默认 Harness 内部：[DecisionPolicy.decide](src/rsisys/loops/decision_making/harness/policy/README.md) 选择操作、系统指令、等待或结束；Exploration Policy 即现有 [ActorPolicy.act](src/rsisys/loops/exploration/harness/policy/README.md)，只选择环境动作。Harness 组织执行与结果处理，Action Space 定义可用操作/动作及约束。

默认长程上下文、研究议程与可选 DSH 的配置见[默认长程研究能力](#默认长程研究能力)和[使用 DSH Harness](#使用-dsh-harness)。

研究任务通过 `ResearchAssembly()` / `ResearchAssembly.single_meta()` 装配显式 memory、
当前任务原始反馈和默认原始轨迹归档。meta-agent 自己解释结果和写记忆；模块不自动调用整理模型。
训练、正式评测及应用变更通过 Lab 的已安装能力执行。当前边界见[统一架构](docs/architecture.md)。

`runtime/recovery.py` 保存运行中协同状态；空闲 `snapshot/` 负责独立可移交快照。前者在同一工作目录继续，后者恢复到新目录。运行恢复按意图 / 回执边界处理，中断且结果不明时要求核对，不假设所有外部副作用都可重放。默认仅串行内置组合，外部组件显式实现恢复契约。

默认执行职责落在现有组件中，不新增平级系统：

| 位置 | 执行职责 | Harness 保留的职责 |
|---|---|---|
| Decision `tools/` | 普通 step 与 review 共用参数校验、授权、执行、结果归一化及固定回执记录 | 上下文、policy 调用、预算、等待 / 继续与反馈接纳 |
| Exploration `environment/` | EpisodeExecution 运行 Actor episode、记录轨迹与系统反馈，失败补写 | 指令占用与本地控制入口 |
| Exploration `learner/`、`evaluation/` | LearningExecution / EvaluationExecution 管理作业、验证结果、记录证据；learning 还保留候选并应用版本 | 选择对应执行组件 |
| Exploration `rl.py` | LocalRLExecution 顺序协调 rollout、训练和应用 | 对外启动 / 推进 / 取消入口 |

这些执行组件只持有所需的 policy、environment、memory、Experience、状态和作业依赖，不反向持有整个 Harness。它们是默认实现内部的职责划分；外部 Harness 不必照搬。Command handler 复用这些组件，普通本地调用与系统指令不各写一套算法。

内置细节通过 `loop.harness` 访问；Action Space 通过 `loop.action_space` 访问，不保留 `loop.policy`、`loop.memory`、`loop.space` 等旧属性别名。内部组件和默认行为分别见 [Decision Harness](src/rsisys/loops/decision_making/harness/README.md) 与 [Exploration Harness](src/rsisys/loops/exploration/harness/README.md)。

### 两种 Action Space 与系统指令

| 边界 | 定义 | 执行位置 |
|---|---|---|
| Decision Action Space | 本地 research 操作，以及 execute / sft / rl / evaluate / configure_prompt 系统指令 | 本地操作由 Decision Harness 的 tools 执行；Directive 交给 Exploration Harness |
| Exploration Action Space | Actor 的环境动作，如 echo / execute | Harness 调用 ActorPolicy.act，校验提案，再交给环境执行 |

内置指令定义集中在 `decision_making/action_space/directives.py`；可通过 `DirectiveDefinitions` 注入其他模式。探索侧通过 `DefaultExplorationHarness(command_handlers=...)` 绑定执行器，两个映射在装配时显式配对。新增系统指令不需要扩充 Actor Action Space。

Decision Action Space 只负责 `describe` / `validate` / `validate_directive`，通过 `directive_modes()` 列出系统指令模式；它不执行工具或训练。Directive 来自 decision policy 或外部系统，目标是 Exploration Harness，不是 ActorPolicy 的动作。指令模式 `execute` 与环境动作 `execute` 分属不同层次。

能力声明、参数 schema 与执行校验需一致；已知不支持的模式、参数组合不暴露给模型。默认装配依据 backend methods / resume_modes 过滤 SFT 和本地 RL。医疗示例的初始化权重与实验父版本由执行器绑定，模型只选择研究动作与开放参数；Harness 不修写模型参数。JSON Schema 契约测试检查静态边界，候选身份、版本冲突、批量变更总数与资源状态等语义仍在执行前校验。

Exploration Harness 按模式分派：execute 运行 Actor episode；SFT、prompt 配置、评测直接进入对应控制流程，绕过 `ActorPolicy.act`；RL 只有 rollout 阶段使用 Actor 动作，训练与版本应用不进入 Actor Action Space。推理评测应由 evaluator 调用所需模型服务，不通过 Actor 的环境动作选择。默认训练/评测仍为 Mock，不提供真实梯度或推理打分；`configure_prompt` 使用 `PromptSpec(instructions)` 更新 Harness 保存的 Actor 指令文本；空字符串清空配置，后续 Actor 上下文读取该文本，见[共享协议](src/rsisys/loops/contracts/README.md)。

训练请求是后端中立（backend-neutral）的接口，只约束模型、非空 method / recipe 字符串与引用元组类型；不把训练算法限定为 sft / rl，也不要求引用非空。Actor 请求允许外部后端自主采样；Decision 请求保留原位置参数，追加 `data_refs=()` 和 `method="sft"`，支持外部数据。提供本地轨迹或系统经验引用时，默认 Harness 仍验证其归属。数据格式和算法能力由具体 learner 校验。

默认 `rl` 保留本地顺序采样循环。可选 `train` 通过 `training_directive()` + `TrainingCommandHandler` + 注入的 learner 配对启用，不属于默认指令映射：Harness 将整个任务作为一个作业提交，后端独立负责算法与 rollout，Harness 负责作业、证据和可选应用。具体接入放在 [adapters/](src/rsisys/adapters/)，不嵌入 Harness；[backend_training.py](examples/backend_training.py) 仍是独立 CPU 标量回归示例。原生训练框架的配置、数据格式和资源分配由适配器及调用方明确提供，装配见[使用指南](#接入整项训练任务)。

### 系统版本、训练来源与部署

[SystemVersion](src/rsisys/loops/contracts/candidate.py) 是 `version`、`components`、可选 `model` 与 `parent_version` 的组合清单。components 将命名组件映射到 ArtifactRef，格式由消费方决定，prompt、数据或配置变更不必产生新权重。Candidate 的 `base_version` 继续表示训练权重基础版本；可选 `parent_system` 独立表示实验父系统，changes 只包含提议替换的组件。

[SystemRegistry](src/rsisys/runtime/systems.py) 是可选的单写者组合状态，不是后端发现中心。`register(candidate, model=None)` 从已知父版本合并组件并保留候选，`get(version)` 读取清单，`current` 读取当前已启用组合；注册不采纳候选。`activate(candidate_id, idempotency_key=...)` 先检查候选父版本仍是 current，再由注入的 SystemActivator 确认精确清单已准备好，最后更新 current。旧父版本必须显式 rebase；只有 SystemActivationRejected 明确确认无副作用或已完成回滚时可释放 pending，其他异常继续冻结原意图等待核对。ReferenceSystemActivator 仅用于模型保持不变的引用级组合；真实部署和绑定事务须显式实现。

[TrainingOrigin](src/rsisys/loops/contracts/training.py) 显式保存 `parent_system`、`initial_model`、`resume_mode` 和可选 checkpoint。可选 train 接收 origin；base 缺省且有 origin 时采用 origin.initial_model，两者均缺省时捕获当前角色模型；显式 base 必须与 origin.initial_model 一致。restart 不允许运行 checkpoint；weights_only 只加载权重并重置优化器、调度器、步数和 RNG；full_state 必须提供完整状态 checkpoint，后端不能降级恢复语义。成功结果的 candidate.parent_system 必须保留请求来源。系统注册训练结果使用 `register_training(result, origin)`，完整校验初始化 ModelRef 并保存来源；实验父系统可以使用另一轮 LoRA。组件组合不再把训练初始化等同于父模型，应用仍检查 parent_version 与 current，拒绝过期候选。

[backend.py](src/rsisys/loops/contracts/backend.py) 将方法、resume_modes、update_fields、checkpoint 导出及 reattach 能力显式分开声明。声明不支持的能力应在提交或操作前失败；已有无能力声明的后端继续实现原有 learner 端口。具体配方和 checkpoint 内容仍由后端验证。

部署协议独立提供 prepare / query / release 与可选 reattach，返回稳定 DeploymentRef、可选 JobRef 及精确 ModelRef；只有 READY 表示可推理，UNKNOWN 保留不确定性。ModelActivator 适配器在确认目标模型就绪后返回，Harness 才更新角色绑定。SystemActivator 负责更大组件组合的准备和应用，不能用 JobStatus.SUCCEEDED 或新模型 URI 代替服务就绪。registry 与外部回执的持久化需单独集成，不因存在引用或训练后端 reattach 就自动获得 Application 恢复支持。

### 兼容执行路径：两个端点协同

下面说明已有 `DecisionMakingLoop / ExplorationLoop` 接口，不将其内部结构规定为所有 Lab 的架构。
标准开放执行路径和单一 meta-agent 的职责见[统一架构](docs/architecture.md)。

1. Decision Harness 读取上下文、完成本地准备，policy 发出 Directive；Action Space 校验操作和指令模式。
2. Runner 将 Directive 交给 Exploration 端点，Harness 接收、校验 payload 并推进对应流程。外部系统也可直接通过同一端点提交 Directive。
3. 遇到 waiting，Runner 返回调用方，下次推进继续原任务。指令可指定首次复查间隔；到期时 Decision Harness 可在有限操作预算内按需研究和读取实验信息，再选择继续、取消和下一次复查间隔。
4. Exploration Harness 汇总终态 Feedback；Runner 通过 Decision 的 `receive_feedback` 交付。默认 Decision Harness 将策略结果记为 improvement，供下一次决策使用。
5. 反馈交付成功后 acknowledge，释放探索流程占用。

默认串行装配一次协调一个未完成交付的指令；可选 `Runner(parallel=True)` 与 `ConcurrentExplorationHarness(factory, max_in_flight=...)` 支持多个独立实验。Decision Harness 用 `max_outstanding` 限制未完成指令数，模型从 outstanding / executions 查看各实验；每 tick 公平推进一个实验，未完成时仍可研究或提交新实验。一条 RL 指令内部可多次采样和更新 Actor。后端轮询与模型复查频率独立。时间/token 自训练触发及共享模型更新事务尚未实现。并行实验由工厂隔离 Actor、environment、memory 和执行状态；资源分配由后端决定。模型可共享或独立绑定，具体部署由 Harness 外的适配器负责；角色引用同一模型不表示训练结果自动同步。

同一次 RSI run 内的决策关系记录为 DAG。`Decision` 携带稳定身份、多个前置依赖和信息引用；系统校验前置已完成后执行。多个独立指令可并行产生候选，researcher 再显式选择一个或多个结果；`SystemRegistry.compose/adopt` 合并兼容组件并通过 activator 确认启用，执行回执才记录候选来源的采纳边。后续决策可依赖该汇合决策。冲突组件或不兼容权重不自动合并，运行时消费新清单由装配实现负责。关系记录、结果和监控沿用现有状态、memory、Experience 和 monitor，不另设图数据库或调度服务。

研究尝试是决策记录上的可选关联，不增加执行层级。researcher 用标题发起新尝试，由 Harness 分配稳定 ID；后续决策明确引用该 ID，系统校验并保存归属与标题。一次尝试可含多个修改、训练、评测或采纳操作；并行尝试与结果汇合仍由原始决策依赖及成功采纳回执决定。监控仅将已接受的关联投影成主图节点，节点内保留所有原始决策和反馈。历史记录缺少关联时不推断分组，也不按名称或相邻时间补造研究过程。


运行中的 review 保留 continue / cancel。安装 `ExperimentControls` 后，研究者可通过普通获授权工具按 ID inspect / cancel / update / compare。`DirectiveUpdate` 携带 expected_version、幂等键和 changes；原指令不可变，支持更新的 handler/backend 应用控制变更并确认版本，不支持的字段或后端明确拒绝。候选比较要求已知完整训练 base、相同评测协议/输入及有效指标；排序不自动采纳。训练、评测、采纳的业务策略属于可替换的 Decision Harness，不写入 Runner。

meta-agent（decision侧） 在科研场景中充当 researcher。实验查询由 Exploration Harness 及其后端提供，Decision Action Space 声明查询操作，Decision Harness 的 tools 绑定执行。查询能力可发现、可替换，返回内容带来源与时间，大内容可通过引用读取；可访问的信息不必全部预先注入模型上下文。`WorkspaceTools` 提供挂载工作区内的 list/read/search 基础操作，研究者可自行读日志、数据、配置和代码；实时服务内部状态仍需后端提供读取途径。外部权限配置决定当前可用 Action Space；默认运行中只读，也可开放其他操作或全部已接入能力。Harness 在展示和执行前检查权限，read_only 是操作属性而非固定限制。权限表达“当前允许做什么”，沙箱限制“执行代码实际能接触哪些资源”；例如允许训练但禁止删除候选，由能力权限检查，限制文件、网络和设备访问则由部署的沙箱承担。两者互补；本地进程后端不是宿主安全隔离。原始查询记录留在研究侧 history，最终判断引用这些记录进入 Experience；不新增独立的观察或记忆子系统。

默认 Decision 上下文另含当前 Exploration 模型/进度、Decision 模型、发出指令时的版本归因，以及每个候选最近一次评测的协议、输入引用、结果和证据。这些事实不依赖有限长度的任务 history；完整评测档案仍由 Experience 保存。

### 外部运行控制

运行是否继续由可替换 `RunPolicy.evaluate(RunContext) -> RunDecision` 决定，独立于两个角色的 Policy。默认提供 researcher、time、goal 三种模式；time / goal 可在研究者 finish 后开启新研究轮次，保留记忆、经验与候选。停止后禁止新任务，继续处理已有任务的取消或排空及反馈交付。具体配置与清理边界见[选择运行策略](#选择运行策略)。

### Experience 与持久化

| 概念 | 职责与范围 |
|---|---|
| Feedback | 单条 Directive 的终态回执，由 Runner 交付给决策端 |
| ExplorationProgress / ExecutionView | 当前执行阶段、模型绑定与可选指标，不是另一套记忆系统 |
| Experience | 跨轮共享的系统证据，记录执行、训练、评测与策略结果 |
| memory / TaskMemory | Harness 内部的任务上下文、原始 history 与工作记忆；默认 compact 摘要带来源，归档不并入 Experience；替代 Harness 不必提供同一接口 |

Experience 收集 runtime_feedback / training_run / evaluation / improvement / insight。scope 仅标记来源，不限制可见性，也不指定优化对象。默认 Decision Harness 使用可替换 `FeedbackSelector.select(experience, objective, limit=...)` 形成模型的反馈输入；默认 RecentFeedbackSelector 选取最近记录，自定义实现可筛选或派生带 source_refs 的视图，不改写原始库、不自动训练。默认 Decision policy 可检索全部来源并选择策略；Actor policy 默认只接收 Harness 提供的任务上下文。improvement / insight 是记录接口，不表示已实现自动系统自修改。

默认 Harness 将原始 test-time history 留在自身实现中。每个 Actor episode 写一条 runtime_feedback，`Feedback.evidence_refs` 引用系统记录；RL 的 `trajectory_refs` 则引用默认 Exploration Harness 的本地轨迹，不把原始轨迹复制进 Experience。

Runtime 提供运行策略、系统清单注册与应用、作业、产物和 Runner 协调；具体训练与服务适配器独立于 Harness，领域业务留在装配示例。Snapshot 在空闲边界保存内置组合；运行策略仅兼容默认 ResearcherRunPolicy，SystemRegistry、外部部署回执、替代 Harness、并行装配和自定义组件仍需显式集成。

### 交付与失败边界

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

### 尚待定义的边界

- SystemVersion / Candidate 已能描述命名组件变更；各组件的格式、执行校验和原子启用仍由消费方及 SystemActivator 定义，框架不执行任意代码自修改。
- meta-agent（decision侧）训练可通过默认 Decision Harness 在本地调用，但没有自动自优化指令或触发机制。
- 候选跨轮保存，显式启用 `apply_candidate` 可独立应用已评测候选；验收指标和阈值由具体 policy 决定，没有全局默认验收规则。SFT 与可选 train 的 `apply_result` 默认 false，默认 RL 每轮应用学习结果，evaluate 不自动部署候选。

### 图示

当前图示使用统一角色名，并分别标出正文路由、可选归档、宿主与面向人的看板。

![系统流程](docs/diagrams/system-flow.svg)

![文件架构](docs/diagrams/file-architecture.svg)

两张图由 `docs/diagrams/build_diagrams.py` 生成，文字与布局均为可编辑的原生 SVG；修改图示时同步生成脚本并重新核验。PNG 仅在临时目录中用于视觉验收，不作为仓库文档副本保存。

## 使用指南

以下示例在仓库根目录执行；自定义工厂中的变量由调用方提供。

### 配置运行

- [demo.json](configs/demo.json)：内存演示，由 config.py 装配。
- [local.json](configs/local.json)：本地持久化，由 application.py 装配。
- [model-service.example.json](configs/model-service.example.json)：HTTP 模型模板。

backends 定义调用后端，roles 分别绑定 decision_maker / actor；execution 选择 mock / local_process，learning 选择 none / mock。默认 Decision Harness 的 experience_limit 限制系统反馈条数，memory_limit 限制自身 history 与工作记忆条数；默认 Exploration Harness 仅使用 memory_limit。这些限制不计算 token。JSON 配置装配内置组合；Python 可通过下述工厂替换任一 loop。

```bash
PYTHONPATH=src pixi run python -m rsisys run \
  --config configs/local.json --data-dir runs/example \
  --literature path/to/paper.txt
```

配置 data_dir 相对配置文件解析，命令行 --data-dir 相对当前目录解析。已有目录不覆盖，同一目录不能由两个 Application 同时打开。--literature 导入普通文本供 [Decision tools](src/rsisys/loops/decision_making/harness/tools/README.md) 使用。

本地目录包含 run.json、config.json、experience.sqlite3、decision_memory.json、exploration_memory.json、artifacts/、工作目录和 snapshots/。Experience SQLite 使用 schema 2，显式拒绝旧 schema，不自动迁移。

### 连接模型

复制模型服务模板，填写 endpoint、model_id、revision。endpoint 为 API base，后端请求其 /chat/completions；两个角色可选相同或不同后端。

需要鉴权时 credential_env 填环境变量名，启动前设置对应环境变量。不支持 JSON response_format 的服务可将 json_mode 设为 false，但内置模型 policy 仍要求回复合法 JSON。传输能力、版本验证和本地进程限制见 [Runtime](src/rsisys/runtime/README.md)。

#### vLLM 推理与部署

[适配器 README](src/rsisys/adapters/README.md#serving) 提供 VLLMBackend、部署与 LoRA 启用的完整配置。
服务由操作者独立启动，endpoint 包含 `/v1`；默认只观察既有部署，动态 LoRA 需显式开启。
只有 READY 且模型身份匹配才允许绑定；服务别名不证明权重不可变，query / reattach 不重放加载。

### Researcher 查询实验

默认装配会将 Exploration 的可用查询注册到 Decision Action Space，名称为 `query_exploration_<name>`。researcher 可在普通决策或运行中 review 中调用；review 默认权限仅开放 read_only 操作，可通过外部权限配置开放其他操作，默认最多调用 4 次，再作 continue / cancel 判断。训练继续由后端执行，控制器的同步查询应由对应后端设置超时。

```json
{"kind":"invoke","operation":"query_exploration_metrics","arguments":{"filters":{"directive_id":"实际指令ID"}}}
```

模型在下一次调用的 `operation_results` 中读到结果，`data` 包含来源、读取时间、内容版本、正文和引用。默认可查 status、metrics、tasks、logs、candidates、model、artifacts、experience；提供本地 ArtifactStore 时还有 artifact_read，可按已知引用分页读取小型文本。logs 是已记录的本地摘要，不是未接入的远程训练日志。指标来自最近一次后端轮询，查询不会触发训练推进；后端提供时保留原始样本时间，未知时间保持未知。

外部 Harness 可实现 [QueryProvider](src/rsisys/loops/contracts/queries.py) 的 `describe_queries()` 和 `query(request)` 来暴露其他实验信息；未实现时不广告这些能力。自定义装配可使用 [QueryTools](src/rsisys/loops/decision_making/harness/tools/queries.py) 的 definitions 和 bindings 分别注入 Action Space 与 tools。协议只统一查询封装，不转换训练数据格式，也不要求把执行历史搬进 Experience。

Qwen toy 已绑定同样的查询入口。`tests/test_researcher_integration.py` 使用 CPU HTTP fixture 验证「后端训练进度 → researcher 查询 → 模型判断 → 实验继续完成」；该测试不加载真实 Qwen 权重。

### 配置能力与权限

能力可以显式定义，也可以通过自定义 provider 发现后注入；[Capabilities](src/rsisys/loops/decision_making/action_space/capabilities.py) 将操作定义与执行绑定配对，装配时拒绝重复名称和缺失 handler。它不包含网络发现客户端，也不自动开放发现的能力。

```python
from rsisys.application import Application
from rsisys.loops.decision_making.action_space.access import ActionPermissions

with Application(config, action_permissions=ActionPermissions.unrestricted()) as app:
    app.run()
```

完全开放仅针对已接入能力，仍保留参数校验、实验容量和交付一致性等协同规则。默认使用 ActionPermissions()：普通决策允许全部已接入能力，运行中仅允许只读操作。也可指定 operations / directives / review_operations 名称集合，或 check 回调检查实际请求的参数和资源。运行时替换 `app.runner.decision_making.action_space.permissions` 即可更新默认 Harness 的权限；已提交任务不会被自动撤回。静态权限配置会随内置快照恢复；外部权限回调需自定义快照适配。

显式/provider 能力装配和参数级检查见 [Action Space](src/rsisys/loops/decision_making/action_space/README.md)。整体替换 Harness 时，应在操作执行前复用 authorized_space 或提供等效权限检查。

### 使用 Python 接入组件

统一装配配置使用 `ModuleAssembly`，Application、demo、医疗和 Qwen 的循环构造都经过
`assemble_runner`。旧工厂关键字继续兼容；不能与 `modules=` 同时传入，以免覆盖配置。

```python
from rsisys.application import Application
from rsisys.assembly import ModuleAssembly
from rsisys.research import ResearchAssembly

modules = ModuleAssembly(lab_factory=make_lab, research=ResearchAssembly())
with Application(config, modules=modules) as app:
    app.run()
```

需要跨进程恢复时，通过 `state_factory(context, decision, exploration)` 注册具备
`state_id / capture() / restore(state)` 的组件适配器；默认要求覆盖 decision、exploration、
已启用的 lab_execution 和 feedback。它们自行管理内部状态格式与依赖，host 不读取插件私有字段。
同一接口也支持空闲快照，但需额外声明并实现 `validate_snapshot()`。
完整接入方式、支持范围和验证见[统一装配与组件状态](docs/review/unified-module-assembly-20261104.md)。

整套 RSI 方法使用 `MethodCatalog` 和 `MethodDefinition(factory, version)` 注册，
通过 `catalog.select({"method": "my_method", "config": {...}})` 返回上述 `ModuleAssembly`。
方法可以整体提供 harness/Lab，也可以内部组合；memory 属于 harness，正式评测属于 Lab，
不要求把研究思考拆成固定流水线。相同启动入口下整体/组合方法的 CPU 示例：

```sh
PYTHONPATH=src python -m examples.rsi_methods configs/rsi-method-demo.json
```

需要独立组合训练策略、推理策略、训推引擎、RSI 算法与 harness 时，可以用
`ComponentGraph` 声明 host 提供的实现目录、原生参数和依赖，再通过
`ModuleAssembly.from_components(graph, decision="harness", lab="lab")` 接入同一个启动入口。
组件兼容性、修改权限和正式评测协议分别管理；配置不会自动授予模型修改引擎或数据的权限。
完整分类、状态与资源边界见[组件组合说明](docs/component-composition.md)。CPU 组合示例：

```sh
PYTHONPATH=src python examples/modular_research.py configs/modular-demo.json
```

开放代码任务使用新的 `lab_factory(context)` 入口：

```python
from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.application import Application
from rsisys.research import ResearchAssembly

# config.execution='local_process'；roles 只需 decision_maker。
# 模型后端和本机已安装环境由操作者配置。
def lab_factory(context):
    return LocalProcessLab(context.sandbox, context.artifacts)

with Application(config, lab_factory=lab_factory, research=ResearchAssembly()) as app:
    app.run(max_decision_steps=32)
```

Lab 只需实现 `describe() -> OperationSpec[]` 与 `execute(OperationCall)`。
默认 decision 直接读取 Lab 工具，模型可以改写 SFT、RL、evaluate 的程序实现，
无需注册固定研究动作类型。持有后台工作时，Lab 可提供 `shutdown(mode)->bool` 和 `close()`，
由 host 确认退出并清理。上例本地进程没有 OS 隔离；正式资源、数据权限和训练准入由 executor 落实。
`run()` 的步数只限制本次调用；后台工作应持续通过 `step()` 驱动至终态，关闭 Application 时取消其 owned 工作。
CPU 示例：`PYTHONPATH=src python examples/open_lab.py`。
完整接口、后台任务、反馈交付与恢复边界见 [M04 实现说明](docs/review/open-lab-execution-20261104.md)。

未传入 `lab_factory` 时，保留下面的旧 exploration 装配方式。

[Application](src/rsisys/application.py) 和 [build_demo](src/rsisys/config.py) 接受可选的 `exploration_factory(context)` 与 `decision_factory(context, exploration)`，分别返回 ExplorationLoop 与 DecisionMakingLoop。先构造 exploration，再构造 decision；未传入的一侧使用 [assembly.py](src/rsisys/assembly.py) 的 `build_default_exploration` / `build_default_decision`，也可在自定义工厂内直接调用这两个函数。默认 decision 的 allowed_actions 从已构造的 exploration.action_space 读取；directive_modes 根据实际未被覆盖的内置 command handlers，以及 learner/evaluator、模型和 rollout 能力推导，不从 with_learning 猜测。

`AssemblyContext` 提供共享 experience、artifacts、backends、sandbox、roles，以及 data_dir（demo 为 None）、execution、with_learning、experience_limit 和 memory_limit。Application 管理共享资源的创建与关闭，自定义工厂无需重复创建存储或构造后修改属性。例如只替换 exploration：

```python
from rsisys.application import Application
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions, builtin_directives
from rsisys.loops.exploration.loop import ExplorationLoop

# config、MyHarness 和 MyActionSpace 由调用方提供；本例 MyHarness 支持 execute。
def exploration_factory(context):
    harness = MyHarness(context.experience, context.artifacts)
    context.add_cleanup(harness.close)  # 若实现持有自有资源，在取得资源后立即注册。
    context.exploration_directives = DirectiveDefinitions(
        definition for definition in builtin_directives() if definition.spec.mode == "execute"
    )
    return ExplorationLoop(harness, MyActionSpace())

with Application(config, exploration_factory=exploration_factory) as app:
    app.run()
```

未知自定义 Harness 必须显式设置 `context.exploration_directives`，或提供 `decision_factory`。新增或覆盖 handler 的模式不会自动向默认 decision 开放，也需 declaration（`DirectiveDefinitions`）或由自定义 decision_factory 配置。显式 catalog 完整指定可用模式及参数解析器，会同时注入默认 policy 和 Action Space；它不是追加列表。对默认 Harness，装配还会检查声明的 handler 是否存在、内置模式所需能力是否齐备。

工厂自有资源用 `context.add_cleanup(callback)` 登记无参清理函数。回调逆序执行且只执行一次；某个回调失败不跳过其余清理。正常关闭和任一工厂装配失败均会清理；Application 在回调完成后关闭共享存储与执行后端。替代 Harness 不必提供 policy 或 memory。

`build_demo` 保持 `(runner, store)` 返回形式，同样接受两个工厂。返回的 runner 支持 `close()` 和上下文管理；自定义工厂使用资源时须显式关闭：

```python
from rsisys.config import build_demo

runner, store = build_demo(demo_config, exploration_factory=exploration_factory)
with runner:
    runner.run(demo_config.objective, rounds=demo_config.rounds)
```

工厂代码仅通过 Python 注入；可选组件图允许 JSON 选择 host 已注册的实现、参数和依赖，
不从 JSON 导入任意代码。自定义装配仍需显式集成状态与 Snapshot。

先组合 Harness 与独立 Action Space，再交给 loop。默认实现内部的 policy、tools、memory、learner/evaluator 通过 Harness 构造函数注入：

```python
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.action_space import LocalDecisionActionSpace
from rsisys.loops.decision_making.harness.tools.local import LocalDecisionTools
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.action_space import EchoActionSpace

# decision_policy、actor_policy、environment、experience 和 artifacts 由调用方提供。
decision = DecisionMakingLoop(
    DefaultDecisionHarness(decision_policy, LocalDecisionTools(artifacts), experience),
    LocalDecisionActionSpace(),
)
exploration = ExplorationLoop(
    DefaultExplorationHarness(actor_policy, environment, experience),
    EchoActionSpace(),
)
```

内部控制通过 `loop.harness`，动作定义通过 `loop.action_space` 访问。例如 `decision.harness.resume(job, result)`、`exploration.harness.submit_learning(...)`；本地执行使用 `exploration.harness.run(directive, exploration.action_space)`，本地 RL 推进使用 `exploration.harness.step_rl(exploration.action_space)`。loop 不提供旧 policy / memory / space 属性或本地控制方法别名。

默认实现将执行放在现有组件中：Decision 的 tools 统一普通操作与运行中 review 的执行/记录；Exploration 的 environment、learner、evaluation 和 rl.py 分别负责 episode、学习、评测和本地 RL 推进。对外仍使用上述 Harness 入口，以保留占用检查；内部执行组件不负责跨 loop 调度。

也可直接实现 Harness 协议并整体替换，无需提供 memory 或 tools 子接口。示例：

- [loop_internals.py](examples/loop_internals.py)：训练请求、查询、版本应用和独立评测。
- [coordination.py](examples/coordination.py)：自定义 policy 消费 last_feedback 并发出下一条指令。
- [backend_training.py](examples/backend_training.py)：通过可选 train 指令委托独立 CPU 训练进程并回传结果。

两个默认 Harness 内部分别使用 DecisionMemory / ExplorationMemory，可在默认实现中替换。DecisionContext 同时包含全部来源的 experience、自身 history 和 memories；ActorContext 读取当前任务 history 与自身 memories，不默认注入系统经验。scope 仅标记反馈来源，优化对象由 Decision policy 选择。

默认 Harness 的本地 RL 请求使用 trajectory_refs 引用 Actor 本地 history；Feedback.evidence_refs 引用系统记录。两侧训练请求均为后端中立：method / recipe_ref 是非空字符串，base 为 ModelRef，引用要求正确类型的 tuple，但允许为空。DecisionTrainingRequest 保留 `(base, experience_refs, recipe_ref)` 位置参数，追加 `data_refs=()` 与 `method="sft"`；可使用系统反馈 experience_refs、外部数据 data_refs，或由后端根据 recipe 获取数据。

调用 runner.step(objective) 推进一步，或 runner.run(objective, rounds=1) 推进到本次反馈交付额度完成、等待或结束；waiting 后由调用方安排下一次推进。Runner 通过 decision_making.receive_feedback(directive, feedback) 交付反馈。取消用 runner.cancel()，之后继续推进直到终态。协同语义见[系统架构](#系统架构)。

### 选择运行策略

Python 装配的 `Runner`、`build_demo` 和 `Application` 均接受 `run_policy=` 与 `cleanup_timeout_seconds=30`。下例复用前面构造的 decision / exploration；省略参数即使用 `ResearcherRunPolicy()`，研究者 finish 后停止。时间模式允许研究者提前 finish 后再开启一轮，持续到指定时限：

```python
from rsisys.runtime.runner import Runner
from rsisys.runtime.run_policy import TimeRunPolicy

runner = Runner(
    decision, exploration,
    run_policy=TimeRunPolicy(seconds=600, restart_delay_seconds=1),
    cleanup_timeout_seconds=30,
)
result = runner.run("改进 Actor 并评测候选", rounds=1)
```

`rounds=1` 限制本次调用交付的指令数量，不表示研究轮次数或运行策略已完成；`completed` 也不证明目标达成。保留同一 Runner，在 `waiting` 后由事件循环或调度器安排后续 `step()` / `run()`；返回 `finished` 后检查 `shutdown_complete` 与 `run_stop` 中的未确认任务，清理到限不等于全部任务已终结。Runner 不在轮次间 sleep；新轮次通过 `restart_research` 显式开启，保留 memory、经验和候选。自定义 Decision 端点要支持此重启接口，才能接入跨轮次继续研究。

目标模式使用 `GoalRunPolicy(goal, restart_delay_seconds=1, max_seconds=None)`，其中 goal 是可替换的 `goal(context) -> bool` callable，读取 RunContext 中已交付的 outcomes。可用 `MetricGoal` 表达绑定评测协议的指标阈值目标，默认不接受 Mock 评测；复杂目标可自行验证证据。`max_seconds=None` 表示没有目标模式的运行时限；需要时间兜底时显式传入秒数。例如复用已有配置、按验证集 accuracy 目标装配：

```python
from rsisys.config import build_demo
from rsisys.runtime.run_policy import GoalRunPolicy, MetricGoal

# protocol_ref 须对应 evaluator 实际使用的评测协议。
goal = MetricGoal("accuracy", 0.8, protocol_ref="validation-v1")
runner, store = build_demo(
    demo_config,
    run_policy=GoalRunPolicy(goal, max_seconds=600),
    cleanup_timeout_seconds=30,
)
with runner:
    result = runner.run(demo_config.objective, rounds=1)
    # 持续运行时，在此资源作用域内由调用方调度后续 step / run。
```

上例阈值与协议由调用方选择；默认 Mock evaluator 不能满足该目标。损失等越低越好的指标使用 `higher_is_better=False`；`allow_simulated=True` 仅显式允许模拟结果参与判断。`Application(config, run_policy=..., cleanup_timeout_seconds=30)` 使用相同注入方式。

停止后的 drain / cancel 由 policy decision 指定；超时默认 cancel，清理另受 `cleanup_timeout_seconds`（默认 30 秒）限制。停止禁止新实验，但继续确认与交付已有反馈；未确认任务不会被记为成功。收尾异常原样抛给调用方，保留状态供后续 `step()` 重试；时限检查依赖调用方继续推进，不会自动硬杀后端。完整接口、结果与同步调用限制见 [Runtime](src/rsisys/runtime/README.md#运行策略与研究轮次)。

### 接入整项训练任务

默认 `rl` 继续由 Harness 执行本地 rollout → learner 训练 → 应用版本 → 下一轮 rollout。若要由外部后端自主完成整个任务的采样与训练，显式启用 `train`：决策侧加入 `training_directive()`，探索侧注册 `TrainingCommandHandler()` 并注入实现作业协议的 learner。

```python
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinitions, builtin_directives, training_directive,
)
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler

catalog = DirectiveDefinitions((*builtin_directives(), training_directive()))
# 手动装配时，将 catalog 同时传给 Decision policy 与 Action Space，允许 train。
# 工厂装配时，设置 context.exploration_directives = catalog；
# catalog 中其他模式也必须具备对应的 handler 和能力，可按实际能力筛选。
exploration_harness = DefaultExplorationHarness(
    actor_policy, environment, experience,
    learner=external_learner,
    command_handlers={"train": TrainingCommandHandler()},
)
```

上述 actor_policy、environment、experience 与 external_learner 由调用方提供。`train` payload 为 `TrainSpec(recipe_ref, method, base=None, apply_result=False, origin=None)`；handler 将 input_refs 作为 data_refs，以单个 ActorTrainingRequest 提交，不预先生成 trajectory_refs，也不运行本地 RL 循环。后端决定 method、数据与 rollout 的语义，Harness 复用提交重试、轮询、取消、证据记录和可选版本应用。直接向 Exploration 端点提交时仍需注册 handler，调用方负责反馈消费与 acknowledge。

默认 MockActorLearner 只支持 sft / rl，并在提交时分别要求非空 data_refs / trajectory_refs，不能替代自主采样的后端。具体框架通过 Harness 外的适配器接入，原生配置与数据格式仍由调用方明确提供。

#### 具体训练接入与组合版本

[适配器 README](src/rsisys/adapters/README.md#training) 提供 ProcessTrainingBackend 的完整 recipe、
原生命令、数据格式和结果读取接口。默认 worker 支持 Transformers / PEFT SFT，外部依赖自行安装；
LLaMA-Factory 只是原生命令扩展，未验证 verl 适配。原生作业重连保留同一 JobRef，不自动恢复控制器。

[TrainingOrigin](src/rsisys/loops/contracts/README.md#可选训练来源与后端能力) 区分实验父系统与初始权重。
TrainSpec 可传 origin；显式 base 必须等于 initial_model，成功候选须保留 parent_system。
restart / weights_only / full_state 必须按后端声明执行；默认 worker 不支持 full_state，不隐式转换数据。

[SystemRegistry](src/rsisys/runtime/README.md#系统组合与显式应用) 管理组件清单。下例仅演示引用级 prompt
组合；initial_model、prompt_ref 和 next_prompt_ref 由调用方提供：

```python
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.runtime.systems import ReferenceSystemActivator, SystemRegistry

initial = SystemVersion("system-v1", {"prompt": prompt_ref}, initial_model)
systems = SystemRegistry(initial, ReferenceSystemActivator(initial_model))
proposal = Candidate("prompt-v2", initial_model.revision,
                     {"prompt": next_prompt_ref}, ("actor",), parent_system=initial.version)
systems.register(proposal)  # current 仍为 system-v1。
systems.activate(proposal.candidate_id, idempotency_key="apply-prompt-v2")
```

register 不采纳候选；activate 检查父版本等于 current，但不做质量评测。ReferenceSystemActivator
不部署或修改角色；真实组合需显式 SystemActivator，registry 与外部回执的恢复也需单独集成。
仅 SystemActivationRejected 确认无副作用或已回滚时释放 pending；未知异常继续冻结原意图。

运行 `pixi run backend-demo` 查看 [backend_training.py](examples/backend_training.py)。示例使用独立子进程在 CPU 上优化标量回归模型，返回真实权重产物与 MSE；它展示训练作业接入，不是 LLM 训练、RL rollout 或 verl 的现成适配。请求字段及结果要求见 [Actor learner](src/rsisys/loops/exploration/harness/learner/README.md) 和 [Decision learner](src/rsisys/loops/decision_making/harness/learner/README.md)。

### 模型启用与外部引用

真实后端返回训练结果后，若要 apply_learning 或 apply_result，需要在对应 Harness 构造时传入 `model_activator`。该适配器实现 `activate(result, *, idempotency_key)`：解析候选权重引用，完成部署或确认既有部署，并返回与 result.model 相同的 ModelRef。等待状态抛出 ModelActivationPending，由调用方稍后继续推进；不在框架内部持续轮询。角色绑定仅在确认后改变。

`backend-demo` 还展示标量模型的加载与推理：独立进程输出权重，ScalarModelService 加载并确认，Harness 更新绑定，最后通过新版本进行预测。具体推理服务通过独立适配器接入；prepare / query / release / reattach 的共享协议见 [backend.py](src/rsisys/loops/contracts/backend.py)。部署 JobRef 成功不等于 READY，适配器必须核对精确模型身份后才允许绑定。

外部数据可以直接使用 `ArtifactRef("dataset-id", "dataset", "s3://bucket/path")` 等引用，无需先写入框架存储；URI 和数据格式由使用者的适配实现解释。未知的 version/digest 保留 None，不伪造摘要。

### 恢复运行

CLI 在空闲边界生成快照，并在 JSON 输出中给出路径：

```bash
PYTHONPATH=src pixi run python -m rsisys inspect --snapshot <快照目录>
PYTHONPATH=src pixi run python -m rsisys resume \
  --snapshot <快照目录> --data-dir runs/restored --rounds 1
```

恢复目标必须是新目录，原目录不被修改。快照保存内置 Harness 的控制状态、两份 memory 和共享 Experience；内置快照仅兼容默认 ResearcherRunPolicy 和默认清理预算；time / goal、自定义运行策略或清理预算会被明确拒绝，需显式集成保存恢复。整体替换 Harness 或替换持久化组件也需单独集成。旧快照受源码版本精确匹配约束，不提供自动迁移。保存内容、版本一致性要求及自定义组件边界统一见 [Snapshot](src/rsisys/snapshot/README.md)。

CLI 的 `resume --snapshot` 仍只处理空闲快照。持续研究可使用 Python 的运行恢复入口：

```python
from pathlib import Path
from rsisys.application import Application, ApplicationConfig
from rsisys.runtime.run_policy import TimeRunPolicy

config = ApplicationConfig.load(Path("configs/local.json"))
with Application(config, durable=True, run_policy=TimeRunPolicy(3600)) as app:
    app.runner.step(config.objective)

# 重新打开原工作目录；不刷新已经消耗的时间预算。
with Application.resume(config.data_dir) as app:
    app.runner.step(app.config.objective)
```

每个 Runner `step/cancel/update` 前记录意图，结束后原子保存回执、控制状态与运行策略。
离线时间也计入已开始的时间预算。内置恢复覆盖串行默认组合、Mock 训练/评测作业、反馈交付，
不重建活跃 shell、GPU 进程或并行实验。外部组件用 `RecoveryContract.capture/restore` 显式
保存并重新关联真实 job；恢复不能再次提交作业。发生未确认操作时，运行被标记为
`RecoveryRequired`，需根据回执调用 `app.recovery.reconcile(repair, evidence=...)`；
它没有盲目重试或清除未知状态的快捷入口。完整边界见 [Runtime](src/rsisys/runtime/README.md)。

### 自主研究与并行实验

[Qwen toycase](examples/qwen_toy/README.md) 使用模型自主选择的步骤，外部配置能力、权限和预算。提前结束、没有训练或未采纳候选都按实际结果记录，不自动补齐工作流。

[parallel_experiments.py](examples/parallel_experiments.py) 演示两个隔离实验、乱序完成、运行中更新和候选比较。它使用明确标记的 Mock 后端与脚本 policy，仅验证接口；不代表实际 GPU 并行训练。装配使用 `ConcurrentExplorationHarness(factory, max_in_flight=2)`、`DefaultDecisionHarness(max_outstanding=2)` 与 `Runner(parallel=True)`；factory 决定后端资源放置。

`ExperimentControls(exploration_loop, decision_loop)` 的 definitions / bindings 接入普通 Action Space / tools，可供模型调用 inspect、cancel、update、compare，沿用外部权限。动态更新只对实现 validate_update / update 的后端开放；它不是任意训练参数热更新。比较与候选应用仍检查来源、协议和版本，不隐式改动其他实验的模型。

反馈提取可通过 `DefaultDecisionHarness(..., feedback_selector=...)` 替换；接口和默认实现见 [Experience](src/rsisys/experience/README.md)。工作区基础读取见 [tools](src/rsisys/loops/decision_making/harness/tools/README.md)，宿主资源隔离由沙箱/部署层实现。

### 独立医学研究示例

[medical README](examples/medical/README.md) 提供数据格式、prepare / check / run 命令、Python 装配、
固定评测及证据边界。领域代码位于 examples/medical，通过两个默认 Harness 与 Runner 独立装配。
prepare / check 只读；CLI run 支持 prompt 研究，原生训练通过 Python 注入 learner 与显式输入绑定。
缓存数据准备 CLI 与单 GPU 自主 LoRA SFT 入口、预算和恢复限制见 [medical README](examples/medical/README.md#single-gpu-autonomous-experiment)。

医疗 prompt / evaluate / apply_candidate 声明 `accepts_input_refs=False`，省略 input_ids 即为空，
显式非空拒绝并可进入有界输出纠错；固定评测后端仍拒绝非空 data_refs。train 保留输入引用。
原生 worker 的 tokenized JSONL 与医学题目格式不同，转换和消费证据由接入实现提供。
[native.py](examples/medical/native.py) 的 MedicalTrainingBackend / MedicalServing 提供显式
tokenize / 进程训练包装与候选路由准备；完整接线见上述 README。
评测可注入 prepare_inference / evaluator_factory，默认 poll 每次至多推进一个样本；明确应用拒绝返回反馈。

已验证 native-final 的 3 步训练、6 行／48 input tokens 消费及同 job 重连一致。
真实医疗运行仅为 prompt→评测→持平拒绝 apply→finish，current 未变；使用 4 条 train 和
4 条原 source TRAIN 子集作 dev。尚无真实 medical SFT / LoRA 部署或训练→评测→apply 全链路证明，
这些接线验证不代表医学增益。

### 默认长程研究能力

默认 `BasicContextCompactor` 按完整模型请求预算，在 80k tokens 触发后压缩旧历史，
压缩后以 24k 为目标。最新反馈与近期历史保留必要事实和原文引用，旧记录生成带来源的滚动摘要，
原始档案仍可查询。vLLM 使用 `/tokenize`；没有 tokenizer 的后端明确标记 UTF-8 保守估计。
摘要使用同一模型预算，压缩失败不会丢弃反馈或执行动作。单条超长证据先投影为事实与有界预览；必要固定信息仍超出目标时
明确报预算错误，不自动提高预算。`IdentityContextCompactor` 用于 DSH 等自行管理会话的 Harness。
具体配置、计数和 snapshot 约定见 [Decision Harness](src/rsisys/loops/decision_making/harness/README.md)。

`ResearchAgenda` 保存模型维护的 agenda、todo、stage_summary；默认装配注册
`research_read`、`research_patch`、`research_archive`。模型按稳定 ID 增量修改，带 expected_revision
避免覆盖更新，分页读取完整档案；短上下文只显示导航。没有要求先写计划再训练，也不把模型笔记
当作已验证结论。这些能力遵循相同的外部权限设置，组件说明见
[Decision Harness](src/rsisys/loops/decision_making/harness/README.md)。

### 项目目标与资源输入

`ModelDecisionPolicy(project_context=...)` 和 `AssemblyContext.project_context` 接收有界、
版本化的宿主项目配置。医疗示例的启动输入明确区分 target/exploration policy、decision
policy、评测契约和资源声明；可执行能力仍以当轮 operations/directives 与执行权限为准。
未安装的 Decision 训练、harness 或系统修改只能提出建议。资源未声明时显示 unknown，
不从设备可见性猜测可用算力；静态配额不等于当前空闲内存。

长期目标与资源配置不由学习 memory 改写。系统指令规定角色与证据规则；项目配置、memory、
摘要和动态反馈仍是分层的用户上下文。配置改变的默认 snapshot 会显式拒绝无恢复契约的保存，
防止恢复时静默丢失配额。医疗配置与启用方式见 [医疗示例](examples/medical/README.md)。

完整决策账本留在执行端与前端数据源。模型工作视图保留最近节点、活跃任务、候选来源、
attempt 代表及它们的依赖/引用/采纳关系；旧节点通过 `research_archive` 的 `decision_graph`
集合分页或按 ID 读取。只在该只读能力可用时投影，不删除原始节点或推断 attempt 完成。
过大的活跃状态仍明确报预算错误，不以丢弃最新反馈或扩大预算处理。

### 统一研究任务装配

默认架构为一个 meta-agent、Lab 和确定性 memory 模块：

```python
from rsisys.application import Application
from rsisys.research import ResearchAssembly

with Application(config, research=ResearchAssembly.single_meta()) as app:
    app.run()
```

默认安装 `memory_search/read/write/delete` 与 `research_task`：meta-agent 在普通 decision 中选择
读什么、记什么、如何修订或删除。`research_task` 的 open/pin/close 将原始错误保留到当前
实验/调试任务结束。中间读文件、改代码和复测不会替换固定的错误；关闭任务释放驻留窗口，原文仍可归档读取。
较大的反馈和 memory 使用显式原文分页。记忆带来源、状态与修订，但散文解释始终不等于验证事实。

`ResearchAssembly()` 使用相同默认路径。`trajectory_archive=None` 可关闭派生轨迹产品，
`context_memory=None` 被拒绝。旧 `.langmem(...)` 独立抽取与 proposer 必须显式选用；
它们不是默认单 meta-agent 架构。医疗常规入口默认 `--memory-provider agent_managed`，
Open Lab 默认 `--feedback-routing agent_memory`；旧 journal 和 LangMem 选项保留兼容。

装配 schema_version=5，启动即记录组件配置，恢复拒绝隐式更换组件。
`AgentMemoryState` 必须与决策、执行和反馈状态一起注册；医疗 Open Lab 已接线原目录恢复。
恢复保留记忆、活跃任务、调用账本和原截止时间；状态不一致或副作用不明时阻塞核对。
通用可迁移 snapshot 仍由完整的组件适配器兑现。基础 `Application(config)` 不自动安装研究组件。

### 底层反馈接口与兼容路径

以下底层入口及组合解释 prompt 保留给旧调用方；新研究任务使用上一节的统一入口。

```python
from rsisys.loops.decision_making.harness.learning_harness import feedback_learning_decision_factory

with Application(config, decision_factory=feedback_learning_decision_factory(max_calls=64)) as app:
    app.run()
```

反馈学习位于独立的 `decision_making/feedback/` 包，不导入默认 harness 实现。
做决策的模型统一称 meta-agent，decision 是研究动作/任务；固定目标下由 data/model/harness
组成的研究执行环境称 lab。以下独立处理器和组合解释入口属于显式兼容路径；默认
语义整理由唯一 meta-agent 负责，harness 处理装配、计量和容量。当前实现映射、Lab 接口迁移范围和三类产物边界见[反馈架构](src/rsisys/loops/decision_making/feedback/README.md)。
基础 `LabSpec / LabTask / Lab` 和 `ExplorationLab` 已提供；默认 Runner 仍走兼容 exploration
端口。`ExecutionReceipt` 统一本地与 Lab 任务的终态观察，默认 adapter 自动整理本地非只读
操作的实际完成，异步提交与只读导航不会被当作新的完成任务。
由 `FeedbackLearningService` 管理，`FeedbackAdapter` 只负责模型身份、
证据快照、发布结果及生命周期接入。默认适配器读取内置执行账本，也可用于 DSH；
第三方 harness 提供自己的适配器，不需要继承默认 harness 或使用它的 policy/state。
安装保留原 harness、模型策略和 compactor 实例；普通模型策略可选择 memory-prefix 排序。

`feedback_learning_decision_factory(decision_factory=..., backend=..., model=..., adapter_factory=...)`
可组合现有 factory。省略 backend 时复用普通 Decision 后端；会话型策略显式提供独立
反馈后端，例如 `DSHFeedbackBackend(config)`。DSH 反馈采用相同实现与配置，但每份
反馈创建独立会话，不把整理任务写进 Decision 的会话历史。独立后端必须接入宿主的
统一准入与预算；医疗示例已提供这层接线，不因替换 harness 重置预算。

反馈职能共享 meta-agent 当前权重，使用另一份上下文；每次新捕获读取当前 ModelRef，
不会被安装时的 model 参数固定在旧版本。context/memory 整理能力必须激活，具体实现可替换；
轨迹归档默认开启、可关闭或替换，harness 修改按需提出。零改进提案或反馈失败不会抹掉原始执行事实。
装配参数 `context_memory=...` 替换整理器，显式 `None` 会被拒绝；
`trajectory_archive=...` 替换轨迹组件，显式 `None` 关闭研究轨迹产物。执行/交付 journal 仍用于事实查询与恢复。
轨迹后续可通过 skill/整理规则转换为训练材料；现有 input/target 提案仅兼容为未验证草稿。
事实指标直接复制回执，解释带来源且保持 unverified。
新输出用 subject 区分 meta_agent、lab、unspecified，避免混淆双方 harness。
旧输出不补造归属；proposal 视图保留其原始内容、反馈模型及轨迹快照入口。
有候选证据时，建议和 Attempt 整理需提供 `evaluation_claims`（无指标声明则为空）；
每条声明校验候选、评测来源、协议、指标和 simulated 是否与回执一致。
训练完成不代表已评测；缺失评测与输入未包含评测分开标记。发布前再次校验，
包括恢复后尚未发布的旧结果。此校验不证明自由文本正确，也不自动改写已发布历史。
`feedback_learning_read` 可分页查看任务或读取某个任务的建议；指定 `job_id` 和
`evidence_path`（JSON Pointer）可分页读取原始证据，返回内容指纹与下一页位置。
`view=attempts` 读取最新解释，`view=proposals,target=context_memory|harness|weights|system`
分类读取提案；`/proposals` 与 `/attempt_revisions` 单独读取解释输出。后续页用
expected_digest 固定内容版本，不能将变化前后的页拼接为同一条证据。
`view=trajectories` 查看每次捕获的轨迹索引；指定 job 的 `/context_memory`、`/trajectory`
读取已启用的产物；`trajectory.enabled` 标记轨迹是否开启。
`/feedback_request`、`/feedback_response` 读取实际反馈模型交换。当前索引
不声称覆盖完整 meta-agent 会话；原决策模型交换的通用捕获仍需补齐。

兼容解释路径中的 Attempt 可由反馈整理模型把已观察到的 Decision 和回执增量组织成研究想法；
Decision 仍说明具体行动、参数与验证计划，不必填写归属。整理沿用反馈学习调用，
独立保存成员引用、摘要、模型来源、输入摘要哈希和修订历史，不改写执行账本。
`feedback_learning_read` 可查看整理及错误，并按整理后的 `research_attempt_id` 查询。
前端读取 `research_attempts` 解释视图；未知引用、旧版本覆盖和成员冲突会被拒绝，
证据不足的记录可以暂未归组。模型总结不证明实验成功、采纳或事前假设；
没有开启反馈学习时不自动归组，旧显式归属仍兼容，历史记录不会自动补写。

启用反馈学习时，memory provider 通过 `context(query) -> str` 提供下一次决策的
`memory_context` 文本，更新时替换当前视图；`research_state` 保留宿主的证据与依赖保护元数据。
新组件也可直接使用函数，SDK 可通过 `CallbackContextMemory` 接入可选反馈通知；
三方法与内部 Memory 发布只在兼容适配器中保留。详见 [Context / Memory 接入窗口](docs/review/feedback-context-window-20261104.md)。
内置 journal 将最新 Attempt 解释、带适用条件的发现及学习异常编码为知识视图，默认限制编码后 12KB；
同一 Attempt 不重复注入旧 job 长文，完整成员/来源和旧版本按需读取。
Harness 在具备 archive 入口时也选择候选/评测/来源窗口，执行校验仍使用完整账本。
固定项目约束仍由宿主维护；当前模型、候选、预算和
未完成任务仍使用原始执行字段，不由摘要重写。每次调用前（含运行中 review）重新
装配工作视图，普通等待不会重写未变化的研究摘要，也不新增整理模型调用。

只有原始记录可通过已授权 archive 读取时，才从工作视图移出旧的已完成 Decision
过程和指纹匹配的重复学习 memory；被修改的 memory 不会被当作旧副本移除。
容量 compact 继续受 80k/24k 预算约束。旧的已结束证据在具备授权归档入口时可投影为
事实卡片：保留指标（包括零分）、状态、身份、来源和原始内容指纹，完整记录仍在原存储。
有只读归档入口时，容量压缩可移出旧证据卡；每个证据流常驻最近两条测量/失败回执，
另保留最近两条失败回执、最新两条历史、最新工具结果/反馈及未完成任务依赖。
零分和失败不删除，旧事实通过带来源的归档按需读取；没有归档入口时不放宽事实保护。
常驻证据的超长正文仍可投影。
单条证据超过 8KiB 时，默认和 DSH 接入均先保留精确汇总指标、身份、状态及来源，
日志/错误采用明确标记的首尾预览，逐项大表提供原文引用；执行账本不变。
`research_archive` 支持 `evidence_path` 按 JSON Pointer 读取字段，返回页也限制大小。
此输入预处理不计为模型 compact；必要汇总指标/身份本身放不下仍明确失败。工具评测使用 `OperationResult.data.quality_metrics` 与 `data.evidence_refs`。
事实卡片由代码提取，不把模型摘要当成已验证证据。
尚未验证此工作视图比线性历史加 compact 有更好的研究效果或实际 KV cache 命中。
发起实验和整理反馈的 Decision 版本分别记录；混入模拟证据的学习结果仍标记 simulated。
默认共享模型后端时调用串行化；独立反馈后端的并发策略由宿主决定，医疗示例统一串行准入。
反馈输入以有界证据包进入模型，原始输入另行保留；省略/缩略的内容显式标记，不能声称已阅读。
学习队列按持久化 `capture_index` 调度和分页，重载后的 JSON 键顺序不影响预算分配；
旧日志若未记录顺序，仅保留其存储顺序，不能还原原始捕获时间。
`max_calls` 是额外学习请求的保守预约上限（含 token 预检拒绝），不是整个 run 的调用预算。

学习日志保存到 `feedback_learning.json`，重启不自动重发结果不明的请求；它不等于完整
Runner 恢复支持，自定义装配的 durable/snapshot 仍需扩展契约。关闭时等待在途学习，
超时显式报错；后端必须配置有限传输超时，并允许从 worker 线程调用。CPU fixture 示例：
`PYTHONPATH=src python examples/feedback_learning.py`，不调用真实模型或训练。

### 使用 DSH Harness

可选依赖固定为测试过的 SDK / runtime 版本，默认安装不引入 Node 或 DSH：

```bash
python -m pip install -e '.[dsh]'
PYTHONPATH=src python examples/dsh_smoke.py
```

smoke 启动真实 DSH runtime 和本地确定性 HTTP 响应服务，验证会话恢复及 RSISys 指令链路；
不使用真实大模型，不训练真实 Actor。

```python
from pathlib import Path
from rsisys.application import Application, ApplicationConfig
from rsisys.loops.decision_making.harness.dsh import DSHConfig, dsh_decision_factory

config = ApplicationConfig.load(Path("configs/local.json"))
dsh = DSHConfig(
    home=Path("/absolute/path/to/dsh-home"),
    workspace=Path("/absolute/path/to/research-workspace"),
    model="deepseek-v4-flash", session_id="research-001",
    api_key_env="DEEPSEEK_API_KEY",  # 只存环境变量名
)
with Application(config, decision_factory=dsh_decision_factory(dsh), durable=True) as app:
    app.runner.step(config.objective)  # waiting 时由调用者稍后继续 step
```

DSH 维护持久会话、模型调用及原生压缩；适配器复用默认回执协调，通过异步 worker 提出结构化
Decision。真正的操作仍在 Runner 线程执行，每次重新检查 Action Space 权限和 RunPolicy。
DSH 自身 turn 完成只产生一个提案；`finish` 是否结束整个 RSI 由外部 RunPolicy 决定。
原生 shell / 文件操作在附带的 sdk-minimal patch 中关闭，所有研究能力经 RSISys 提交。
这不是操作系统隔离，也尚未提供任意 DSH native tool 与 RSISys 权限之间的自动映射。

`base_url` 由该 DSH provider 解释；验证过的 rc2 `deepseek-official` 请求使用 `/v1/messages`，
不能假定任意只实现 `/v1/chat/completions` 的服务可直接使用。模型身份记录为 provider-managed，
不宣称验证了远端权重版本；未取得的 token 用量保持未知。SDK 官方接口见
[DeepSeek Harness Python SDK](https://github.com/deepseek-ai/deepseek-harness/blob/master/python/sdk/README.md)。

恢复 DSH 装配时需向 `Application.resume(..., decision_factory=同一工厂)` 提供原配置。
已缓存的完整提案可恢复；未完成的推理会保留为待核对状态，不自动重复请求。
`harness.resume_interrupted_inference(evidence=...)` 显式恢复原生会话、记录核对说明并丢弃未确认建议，
不发送新模型输入；若 Runner journal 同时中断，应在 `app.recovery.reconcile` 回调中调用。
`DSHConfig(resume=True)` 单独只恢复 DSH 原生会话，不等于恢复 RSISys 实验或外部进程。

## 示例与验证

| 入口 | 验证内容 |
|---|---|
| `pixi run demo` / `local-demo` | Mock 双 loop、本地进程与持久化 |
| `pixi run loop-demo` / `coordination-demo` | 本地学习、评测与跨 loop 指令链 |
| `pixi run backend-demo` | 独立子进程中的真实 CPU 标量回归，不是 LLM 训练 |
| [Qwen toycase](examples/qwen_toy/README.md) | 模型自主研究、LoRA、评测和候选应用的显式装配 |
| [决策图示例](examples/decision_graph.py) | 脚本策略、CPU 任务；6 条决策组成 3 个 Attempt，验证并行和采纳汇合 |
| [医学装配](examples/medical/README.md) | 领域数据、固定评测、研究模型与原生训练接线 |

```sh
PYTHONPATH=src python examples/decision_graph.py --run-dir runs/decision-graph-demo
pixi run monitor
node --test tests/monitor_frontend/*.test.js
```

示例目录必须未使用。模型没有选择训练或没有采纳候选时，按实际结果记录，不补齐固定工作流。

可选 `test` extra 提供 JSON Schema 契约验证，不增加运行依赖。真实服务参数回归须显式设置 `RSISYS_LIVE_BASE_URL`（含 `/v1`）与 `RSISYS_LIVE_MODEL`，modelmate 另设 `RSISYS_LIVE_SCHEMA_MODE=portable`，再运行 `PYTHONPATH=src:tests python -m unittest test_live_generation -v`。它核对生成参数、解析与请求准备，不提交训练；默认测试不联网，详见 [测试说明](tests/test_live_generation.py)。

## 组件文档

| 组件 | 内容 |
|---|---|
| [Decision-making](src/rsisys/loops/decision_making/README.md) | Decision Harness 与 Action Space |
| [Exploration](src/rsisys/loops/exploration/README.md) | Exploration Harness 与 Actor Action Space |
| [共享协议](src/rsisys/loops/contracts/README.md) | 指令、反馈、候选、模型与后端引用 |
| [Experience](src/rsisys/experience/README.md) | 系统反馈、来源、检索和存储 |
| [Runtime](src/rsisys/runtime/README.md) | Runner、RunPolicy、作业与组合版本 |
| [适配器](src/rsisys/adapters/README.md) | 原生进程训练、vLLM 推理和部署 |
| [Snapshot](src/rsisys/snapshot/README.md) | 内置组合保存恢复及自定义边界 |
| [Monitor](src/rsimonitor/README.md) | 实验可视化、快照协议与只读 API |

全局文档只维护本 README；组件维护各自 README，示例位于 `examples/`，`docs/diagrams/` 仅存架构图与其生成脚本。
