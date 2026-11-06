# 组件组合与 host 管理边界

更新：2026-11-06。

RSISys 使用 `ModuleAssembly` 作为统一装配入口。可选的 `ComponentGraph` 把组件选择、
原生参数、依赖和版本记录为一个配置；同一启动代码可以组合不同的研究策略、harness、
训练策略、推理策略和执行引擎。已有 Python 工厂继续兼容。

## 按职责拆分

组件槽位名由领域装配定义，下表是推荐拆分，不是核心强制的算法菜单。

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
| RSI 算法 | 如何提出、选择、调度和接受下一次改进 | 公共 `DecisionPolicy.decide(DecisionInput)` 可独立替换；复杂算法可组合其他组件 |
| memory/归档 | 存储、检索、分页及显式修订 | 已有 `ResearchAssembly` 管理这些接线；它不等于整个 RSI 算法 |
| 正式评测协议 | 数据身份、解析、计分、预算、提交配额 | host 固定；研究代码不能提交自己计算的正式分数 |

组件可被替换与模型有权修改组件是两个独立属性。host 可以固定训练引擎，允许模型探索
训练和推理脚本；也可以另开实验，允许提出引擎或 harness 候选。正式评测规则始终单独管理。
医疗候选允许提交 checkpoint/LoRA **及推理脚本**，公开/封闭划分和每 1800 秒一次的
封闭提交规则见[评测协议](evaluation-protocol.md)。本轮没有部署独立封闭评测服务。

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

图配置由 host 提供并在一次运行内固定。模型只获得实际装配的 Lab 工具，以及
`ActionPermissions` 允许的操作和参数范围。依赖契约不会授予修改目录、数据或引擎的权限。
可通过已有 `ActionPermissions.check(AccessRequest)` 检查具体候选或操作参数；其检查力度
取决于领域实现，不会自动推断哪个任意文件属于哪个模块。

目前没有一个通用的“组件级文件修改沙箱”。允许研究程序执行任意代码时，真正的文件、
进程和网络边界依赖执行后端及 OS 隔离。尤其同 UID 本地执行不能保护研究进程也能访问的
封闭数据。可信 Python 工厂也不是不可信代码沙箱。

工厂只构造对象，不提交训练、推理或研究任务；恢复同样遵守这条约定。资源可声明
`owned=True`，对象必须提供幂等 `close()`。构造失败也保留已经注册的清理；按依赖反序
关闭，失败的 owned close 阻止提前释放其依赖，之后可以重试。一个对象不能有两个 owner。
每个工厂仍须处理自己在返回对象之前获取资源却失败的情况。

`ProcessTrainingBackend` 当前只有作业级 `cancel(job)`，没有整体 `close()`。不要仅为
加入组件图就给它标记 `owned=True`；使用它的 Lab/owner 需跟踪并清理自己的作业。
`VLLMBackend` 等已有原生适配器可由目录工厂装配，但对应模型、产物和设备兼容性需要
各自的真实验收；图只检查已声明的组件契约。

## 状态、恢复和来源

`Application` 将图、组件版本、配置、依赖和根端口绑定保存到 `module_assembly.json`。
恢复时只要这些身份变化，就在调用组件工厂之前拒绝；换组合应创建新运行。
版本是 host 声明，不是自动计算的代码证明。实验仍需记录 canonical commit、代码和
数据/权重 manifest；修改实现时必须更新版本，不能沿用旧标识隐藏变化。

持久恢复需要显式 `state_factory` 或完整自定义 `recovery_contract`。
使用 `state_factory` 时，除了原有 decision、exploration、Lab、feedback/memory 端口，
每个有状态组件都需 `component:<slot>` 状态端口。host 自动注册组件清单端口，以及
明确声明 `stateless=True` 的空状态端口；不能用显式适配器覆盖它们。
stateless 是 host 对实现的承诺，不是自动证明；带计数器、随机数状态、作业或可变缓存的
组件需要真实状态集成。完整自定义 recovery contract 则自行覆盖整个组合。

状态端口保留原有 `state_id/capture/restore` 契约和配置检查；跨目录空闲快照还需
`validate_snapshot()`。状态适配器必须处理内部依赖、未完成任务和产物身份，不能用
空字典冒充恢复。运行中重新绑定已注册组件会在下一次 durable step 之前被阻止。

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

2026-11-06 本地 Python 3.12.14 验证：新增测试 15 项通过（包含上述 32 种组合）；
完整 CPU 回归共 1825 项，44 项跳过，无失败。复现命令：

```sh
PYTHONPATH=src:tests python -m unittest discover -s tests -t . -q
```
