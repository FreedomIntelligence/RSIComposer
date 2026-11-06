# 组件组合与 host 管理边界

更新：2026-11-06。

RSISys 使用 `ModuleAssembly` 作为统一装配入口。可选的 `ComponentGraph` 把组件选择、
原生参数、依赖和版本记录为一个配置；同一启动代码可以组合不同的研究策略、harness、
训练策略、推理策略和执行引擎。已有 Python 工厂继续兼容。

## 顶层归属与整套 RSI 方法

顶层职责是 host、meta-agent/harness 和 Lab。Memory 是 harness 可替换的子模块；正式
评测是 Lab 内的流程或服务入口，由 host 固定协议与权限。反馈解释、实验设计、记忆内容
组织和下一步选择可以由同一个 meta-agent 完成，不要求 host 分别提供研究阶段组件。

一套 RSI 方法包含所选实现、接线，以及由这些实现兑现的迭代规则。它可以整体提供
harness 和 Lab，也可以在内部使用组件图；两种形式都通过 `ModuleAssembly` 接入。
不要求拆出候选生成、训练、推理、评测、选择等固定阶段，也不要求所有组件都能自修改。

整套方法还定义 meta-agent 与 Lab 对象的 **coevolution**；其中改进 meta-agent 自身
harness、工具、memory 机制或训练其模型的过程统一称为 **reflect**。执行可以使用 Lab，
归属按改进对象确定。Reflect 策略属于 RSI 优化器，可整体封装或作为其内部组件。
触发时机由方法决定，不是每轮必经阶段；完整语义见
[共同演化与 reflect](architecture.md#共同演化与-reflect)。

`methods.py` 提供版本化的 `rsisys-method/v1` 接入协议：

```python
from rsisys.methods import MethodCatalog, MethodDefinition

catalog = MethodCatalog({
    "my_method": MethodDefinition(
        factory=make_method,  # make_method(config) -> ModuleAssembly
        version="1",
        validate_config=check_method_config,  # 可选，成功返回 None
    ),
})
modules = catalog.select({"method": "my_method", "config": {}})
# 使用已有 Application(config, modules=modules) / build_demo(config, modules=modules)
```

host 显式注册可信实现；选择文件只有 `method` 和 `config`，不包含导入路径、host 权限、
预算或评测规则。选择时检查协议版本、JSON 参数和返回类型；构造运行时再检查公共端口。
`make_method` 只生成装配描述，不能获取资源或启动任务；资源在返回的装配工厂中构造，
研究从正式 step/run 开始。这是可信实现的接入约定，不是任意 Python 代码的隔离机制。

| 方法提供的边界 | 最小协议 | 可自由组织的内部实现 |
| --- | --- | --- |
| Meta-agent harness | `step(objective, action_space)`、`pending()`、`receive_feedback(...)`，接到现有 `DecisionMakingLoop` | 策略、SDK、memory、状态、prompt、搜索与迭代方法 |
| Lab | `describe()`、`execute(call)` | 数据、训练、推理、正式评测入口及内部编排 |
| 可选兼容执行端点 | 现有 accept/advance/inspect/cancel/acknowledge 协议 | 旧 directive 工作流；仅 Lab 工具运行可使用默认空适配 |
| 可选恢复与资源能力 | 显式 state ports，以及适用时的 shutdown/close | 实现自己的状态格式、任务重连和 owned 资源退出 |

`ModuleAssembly(model_session=...)` 在构造组件之前把同一个 `HostModelSession` 接到
`context.model_session` / `context.backends`，同时检查角色模型属于该会话。
医疗入口强制选择的方法继续使用本轮同一本账，不能换一个 client 重置预算。
不可信研究脚本通过 Unix broker 查询；直接网络访问由隔离执行器阻止。
可信 host Python 插件须使用该会话，不自动截获任意进程内 SDK 调用。
内部组件可以共用已有接口，host 不递归读取实现私有字段。

完整 CPU 示例通过相同启动入口选择 `whole`（整体实现）或 `composed`（内部组件组合）：

```sh
PYTHONPATH=src python -m examples.rsi_methods configs/rsi-method-demo.json
```

将选择文件的 `method` 从 `whole` 改成 `composed` 即可切换。示例中的任务和调度均为
合成 fixture，只证明接入与组合，不代表通用第三方算法已自动适配或真实研究有效。

## 按职责拆分

组件槽位名由方法或领域装配定义。下表列出可选的内部拆分；一个方法可以整体封装其中多项，
不需要把每一行变成独立插件、独立模型或固定研究步骤。

| 模块 | 负责什么 | 模型可修改范围由谁决定 |
| --- | --- | --- |
| 训练数据策略 | 从授权训练分区采样、过滤、构建课程或合成数据 | host 授予指定数据和产物访问权限；数据来源与曝光需要记录 |
| 公开开发集 | 调试问题、检查预测和公开误差分析 | host 固定可比较评测的数据身份；模型可另建探索性测量 |
| 封闭评测数据 | 不对研究环境公开的问题和 gold | host/独立评测服务持有；不作为可写研究组件 |
| 训练策略 | 目标函数、优化器、SFT/RL、超参和训练流程 | 允许的脚本、配置或提案入口由 host 指定 |
| 推理策略 | 加载、tokenizer、prompt、解码、batching、工具使用 | 允许提交的推理实现由 host 指定；属于候选身份 |
| 训练引擎 | 原生框架、分布式执行、设备和作业管理 | host 选择已安装适配器和环境；可开放策略参数而固定引擎 |
| 推理引擎 | 模型加载、服务或离线生成的执行适配 | host 选择实现及资源；不要求训练和推理使用同一个引擎 |
| Meta-agent harness | 输入、记忆接线、工具调用、结果交付 | host 装配；可通过审核后的候选替换 |
| Lab harness | Lab 内部训练/推理任务的编排和执行 | 可作为独立组件；不等于 meta-agent harness |
| 进化触发与编排 | 决定是否、何时以及对哪一侧启动改进，允许跳过或延后 | 可由整套方法内部实现，也可组合独立策略；不要求每轮产出候选 |
| RSI 决策策略 | 如何提出、选择、调度和接受下一次改进 | 公共 `DecisionPolicy.decide(DecisionInput)` 是可选子模块；整套方法通过上述方法协议接入 |
| memory/归档 | harness 内的存储、检索、分页及显式修订；原始归档由 host 保留 | 已有 `ResearchAssembly` 管理这些接线；它不等于整个 RSI 算法 |
| 正式评测协议 | Lab 流程中的数据身份、解析、计分、预算、提交配额 | host 固定；研究代码不能提交自己计算的正式分数 |

组件可被替换与模型有权修改组件是两个独立属性。host 可以固定训练引擎，允许模型探索
训练和推理脚本；也可以另开实验，允许提出引擎或 harness 候选。正式评测规则始终单独管理。
医疗候选允许提交 checkpoint/LoRA **及推理脚本**，公开/封闭划分和每 1800 秒一次的
封闭提交规则见[评测协议](evaluation-protocol.md)。本轮没有部署独立封闭评测服务。

## 共同演化的组件边界

方法可整体提供进化逻辑，也可按需组合触发策略、证据选择、候选构建、验证与版本选择。
这些职责可以由同一个 meta-agent 完成；独立组件不意味着独立模型或独立研究阶段。
触发后可以只分析、收集证据或决定保持当前版本，候选产物是可选的。
Meta-agent reflect 与 Lab 侧改进可以复用这些接口，各自选择内部算法、验证规则和节奏。

候选由 meta-agent 自行构建，包括编写工具/harness 代码、修改配置和训练权重；也可
调用算法提供的生成或训练组件。Host 管理的是执行边界与产物接口，不规定候选内容。
组件组合优先依赖清晰的输入输出、模型/产物格式、配置和生命周期协议；必要时由适配器
解决兼容。多个组件共同变化时保留联合候选和版本关系，不要求每项变化独立提交或审批。

候选存在时，方法集成需明确其目标对象、基线版本、受影响组件、代码/配置/权重产物、
来源和验证协议；验证记录实际使用的两侧版本组合，启用记录实际绑定变化。
这些是接入与留证要求，不是已新增的强制核心消息字段或内置状态机。
多个 Lab agent 时还需定位具体对象，不能只凭 `subject=lab` 判断是哪一个 agent。

当前 `Capabilities` 支持显式组合工具描述、参数校验和执行绑定，默认 harness 可注入
决策策略、上下文压缩、反馈选择、memory 与工具实现，整套 harness 也可替换。
这些子接口尚未全部提供独立 JSON 槽位。`ResearchAssembly` 的反馈 scheduler 选择已有
反馈任务，不自动等价于完整进化触发器；`harness_change` 提案也不自动执行或启用改动。
旧 `policy_training` 提案路径仍限定 meta-agent 目标，Lab 侧训练通过其已安装能力执行；
共同演化的语义对齐不改变旧消息校验或冻结运行。

通用的候选构建、验收、启用/回滚和状态迁移仍需方法适配。若在一次运行中迭代工具或
harness，需由固定执行组件显式管理候选版本、切换边界、任务与 memory 状态、权限和
账本；保持 host 装配身份可追踪，不能绕过固定组件图的恢复检查直接替换运行对象。

## 显式目录与依赖

host 在 Python 中建立有限的实现目录。JSON 只选择目录中的 ID，不支持导入路径、
自动扫描插件或安装依赖。工厂接收 `(assembly_context, dependencies, configuration)`。

```python
from rsisys.components import ComponentDefinition, ComponentGraph

catalog = {
    "training_recipe": ComponentDefinition(
        factory=make_training_command,
        version="recipe-v1",
        provides=("training-command/v1",),
        stateless=True,
    ),
    "process_engine": ComponentDefinition(
        factory=make_process_backend,
        version="process-v1",
        provides=("training-backend/v1",),
        requires={"command": "training-command/v1"},
    ),
}
graph = ComponentGraph(catalog, {
    "training": {"component": "training_recipe", "config": {}},
    "train_engine": {
        "component": "process_engine",
        "dependencies": {"command": "training"},
    },
})
```

`make_training_command` 和 `make_process_backend` 是领域提供的可信工厂。
前者返回原生 command 回调，后者可返回
`ProcessTrainingBackend(context.data_dir / "jobs", command=dependencies["command"])`。
策略与引擎各自保留原生参数，不要求将所有框架参数压成一套通用 schema。

每个定义必须声明非空版本和输出契约；依赖按契约 ID 匹配。构造前检查未知组件、
缺失依赖、循环、契约冲突和非 JSON 参数。可选 `validate_config(config)` 提前检查
参数范围，可选 `validate(instance)` 检查实际对象；二者成功时返回 `None`。
契约 ID 是兼容性声明，具体语义仍需 host 的适配器和验收测试兑现。

## 接到现有循环

整个图通过现有端口进入运行器：

```python
from rsisys.assembly import ModuleAssembly

modules = ModuleAssembly.from_components(
    graph, decision="meta_harness", lab="lab",
)
# Application(config, modules=modules) 或 build_demo(config, modules=modules)
```

这里的完整图还需定义 `meta_harness` 和 `lab` 槽位。它们分别提供
`decision-factory/v1` 和 `lab-execution/v1`。前者返回原有
`factory(context, exploration) -> DecisionMakingLoop`；后者是提供 `describe/execute` 的
Lab 对象。可选 `exploration=` 绑定 `exploration-factory/v1`。没有新建另一套运行循环。

只替换 RSI 决策策略时，harness 工厂可以调用
`build_default_decision(context, exploration, policy=dependencies["rsi"])`。
策略接收公共 `DecisionInput`，默认工具、action space 和权限校验仍然使用现有实现。
策略若调用模型，必须接到宿主已有的计费后端；这个参数本身不会自动拦截任意 SDK 调用。
完整自定义 harness 同样需要显式兑现权限、计费、反馈与恢复接线。

可直接运行的完整例子：

```sh
PYTHONPATH=src python examples/modular_research.py configs/modular-demo.json
```

它用 CPU 线性拟合演示独立的数据、训练、推理、训推运行时、研究调度与 harness 组合。
更换 JSON 中的实现 ID 即可换组合，启动代码相同。它没有 LLM/GPU 调用，不能据此宣称
医疗任务改进或原生 Transformers/vLLM 组合已验收。

## 权限和生命周期

权限设计原则是在已装配能力和分配资源内默认尽量开放，由 host 显式限制敏感操作。
读取、构建、测试和启用可分别配置，分级能力不意味着所有操作默认拒绝，也不意味着
候选必须人工批准。Host 可预设自动启用条件；模型不能自行改写权限或正式评测边界。

图配置由 host 提供并在一次运行内固定。模型只获得实际装配的 Lab 工具，以及
`ActionPermissions` 允许的操作和参数范围。依赖契约不会授予修改目录、数据或引擎的权限。
可通过 `ActionPermissions(check=SensitivePolicy(...))` 配置具名、带版本的显式 deny 规则，
选择 phase/kind/name、具体参数值或绝对路径前缀；其身份与规则可持久化并在恢复时校验。
任意 Python `check(AccessRequest)` 回调仍需自己的状态集成，不静默丢弃回调。其检查力度
取决于领域实现，不会自动推断哪个任意文件属于哪个模块。

现有 `ActionPermissions()` 的普通 decision 已允许全部已接入操作和 directive；review
默认只读是独立的阶段配置。`ActionPermissions.unrestricted()` 可开放 review 的已接入
操作，配合 `check` 回调实现具体敏感限制。此处对齐设计原则，不更改这些兼容默认值，
也不把未安装能力视为已经可用。

工具准入与子进程隔离分别执行。`IsolatedProcessBackend` 用 host 的 `ExecutionPolicy`
限制实际文件、网络和设备可见性；只读运行时不能包含 host 账本、恢复目录或凭据。
允许编辑与运行工作区程序，不会授予改写 host 策略或访问未分配设备的权限。Linux 显式
声明设备节点及运行时挂载，不以 `CUDA_VISIBLE_DEVICES` 等变量作为隔离依据。
隔离设施不可用时拒绝执行；`LocalProcessBackend` 保留其明确的可信本地语义。
封闭数据仍由独立 gateway 持有。可信 Python 工厂不是不可信代码沙箱。

医疗入口的 `--rsi-method '{"method":"open_lab","config":{"reflect":true}}'` 选择默认
RSI 优化器。Host 也可给 `run(..., method_catalog=...)` 注册整体实现或组件装配；配置
不接受候选 Python 导入路径。默认 reflect 可跳过，不要求每轮或每次触发产生候选。

`runtime.evolution.ActivationPort` 由 host 注册 prepare/read/apply 与实现版本。候选由
meta-agent 在工作区构建，再经 `system_register` 冻结到 artifact store。`system_activate`
合并兼容目标、先持久写 intent，再应用并逐目标读回；失败时回滚并确认，结果不确定则保留
pending。`EvolutionState` 与 registry/activation journal 一起恢复，不重建研究任务或
清空 memory/调用账本。默认医疗端口为 `meta.prompts` 和 `meta.harness` 参数；新权重、
新工具和整套 harness 的真实部署需要相应激活实现，不默认执行候选提供的 host 代码。
注册清单可携带 `models: {role: {backend_id, model_id, revision}}` 和
`model_bases: {role: revision}`；目标必须有已安装的模型端口并出现在 `affected_roles`。
格式错误返回可修正的工具反馈。停止或绝对截止后禁止注册/启用，系统状态仍可查询。

工厂只构造对象，不提交训练、推理或研究任务；恢复同样遵守这条约定。资源可声明
`owned=True`，对象必须提供幂等 `close()`。构造失败也保留已经注册的清理；按依赖反序
关闭，失败的 owned close 阻止提前释放其依赖，之后可以重试。一个对象不能有两个 owner。
每个工厂仍须处理自己在返回对象之前获取资源却失败的情况。

`ProcessTrainingBackend` 当前只有作业级 `cancel(job)`，没有整体 `close()`。不要仅为
加入组件图就给它标记 `owned=True`；使用它的 Lab/owner 需跟踪并清理自己的作业。
`VLLMBackend` 等已有原生适配器可由目录工厂装配，但对应模型、产物和设备兼容性需要
各自的真实验收；图只检查已声明的组件契约。

## 状态、恢复和来源

`Application` 将整套方法的协议、ID、版本和配置，以及可选组件图的版本、配置、依赖和
根端口绑定保存到 `module_assembly.json`。原目录恢复时若这些身份变化，会在调用资源
工厂之前拒绝；换方法或组合应创建新运行。方法选择时的纯配置 builder 可以先执行，
不能在其中创建资源或提交任务。
版本是 host 声明，不是自动计算的代码证明。实验仍需记录 canonical commit、代码和
数据/权重 manifest；修改实现时必须更新版本，不能沿用旧标识隐藏变化。

持久恢复需要显式 `state_factory` 或完整自定义 `recovery_contract`。
整体方法不需要为内部结构虚构图槽位；由它的 harness/Lab 状态适配器覆盖真实内部状态，
包括所用 memory 和未完成任务。host 自动加入 `method_manifest` 状态端口，不能覆盖。
使用 `state_factory` 时，除了原有 decision、exploration、Lab、feedback/memory 端口，
每个有状态组件都需 `component:<slot>` 状态端口。host 自动注册组件清单端口，以及
明确声明 `stateless=True` 的空状态端口；不能用显式适配器覆盖它们。
stateless 是 host 对实现的承诺，不是自动证明；带计数器、随机数状态、作业或可变缓存的
组件需要真实状态集成。完整自定义 recovery contract 则自行覆盖整个组合。

状态端口保留原有 `state_id/capture/restore` 契约和配置检查；跨目录空闲快照还需
`validate_snapshot()`。状态适配器必须处理内部依赖、未完成任务和产物身份，不能用
空字典冒充恢复。运行中重新绑定已注册组件会在下一次 durable step 之前被阻止。
新建的便携组件快照同时保存并校验装配清单；恢复时会在资源构造前拒绝方法或组件身份
变化。历史没有清单的快照继续受原有显式状态契约约束，不能补称已通过该提前检查。

## 本次检查与验证

原来已有统一装配、独立 harness/decision 策略和原生训推后端，缺少统一组件清单、依赖
检查和跨运行组合身份。本次补充图配置、根端口绑定、独立 RSI 策略注入、owned 清理接线
和组件状态要求，没有重写原生训练或服务引擎。

`tests/test_component_graph.py` 验证 32 种组合，以及非法图提前拒绝、host 权限拒绝、
owned 清理重试、恢复/快照、版本/参数/接线变化拒绝和运行中组件重绑定拒绝。另用真实
本地子进程检查两种 command 策略复用相同 `ProcessTrainingBackend`、构造时不提交任务、
重复 key 不重启。子进程脚本是测试 fixture，不执行机器学习训练。

这些检查覆盖组合机制。原生训推组合、具体 RSI 算法有效性、医疗分数和隔离评测服务
仍分别需要相应验收，不能由组件接线通过推导。

首轮组件组合验证（`0da61f2`，Python 3.12.14）：15 项新增测试通过（包含上述 32 种组合）；
完整 CPU 回归共 1825 项，44 项跳过，无失败。整套方法测试见 `tests/test_rsi_methods.py`，
覆盖相同入口下整体/组合实现、host 权限、实际端口验证、恢复与快照方法身份。复现命令：

```sh
PYTHONPATH=src:tests python -m unittest discover -s tests -t . -q
```

本轮方法协议验证（2026-11-06，Python 3.12.14）：新增方法测试 11 项及便携快照组件身份
测试 1 项通过；完整 CPU 回归 1837 项，44 项跳过，无失败。未启动 GPU 实验或模型服务。
