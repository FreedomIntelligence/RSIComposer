# RSISys 架构

顶层只有两个业务模块：**RSI 决策器（controller）与 Labs**。
一条主循环是 **决策器 → Lab 执行 → 反馈 → 决策器**。
固定算法、搜索策略和 LLM meta-agent 都实现相同的决策边界；不要求研究/场景双 loop。

![RSI 决策器、Lab 与公共运行时](diagrams/framework-loop.svg)

## 两个模块

| 模块 | 公开接口 | 负责什么 |
| --- | --- | --- |
| RSI 决策器 | `decide(DecisionInput) -> Decision` | 根据目标、当前状态和真实反馈决定下一步操作、等待或结束。算法决定候选、训练规则、采纳与版本选择。 |
| Lab | `describe()`、`execute(LabRequest) -> LabFeedback` | 描述能力、校验领域参数、执行实验，返回状态、指标、产物及原始证据引用。可以接模型、程序或训练/推理服务。 |

`DecisionInput` 包含目标、最新操作及结构化反馈、可用能力和上下文。
`Decision` 表达 invoke/wait/finish。`LabRequest` 与 `LabFeedback` 是现有操作/回执协议的
公开名称，复用相同的序列化和身份校验。完整类型从 `rsisys.api` 导入，定义见
[API 契约](api-contracts.md)。

Lab 内部可以运行生成、训练、推理、评测或自己的 agent 循环；这些不增加顶层模块。
一次 Lab 操作可以同步返回终态，也可以显式提供作业查询与取消能力。
提交作业、训练完成、模型启用和质量改善分别留证。

## 公共运行时的作用

Host / Runner 装配两个模块，检查能力与授权，路由操作、交付反馈，管理运行生命周期、
执行账本和自有资源。多个 Lab 通过 `alias.operation` 区分，保持统一运行身份和回执来源。
宿主根据已配置规则控制权限与运行预算，算法负责研究选择。

即使 controller、harness、memory 和 Lab 全部由用户实现，框架仍提供上述公共执行语义。
这些保障适用于已接入的端口；受信任插件自行发出的网络调用、后台任务和资源分配，
需要显式适配，注册本身不会自动接管它们。

运行时已有可选并行执行、本地进程作业与显式恢复设施。具体方法是否支持这些能力，
取决于其状态与执行适配器。当前同步 GPU 配方不能凭通用 runtime 的存在自动获得
训练中断、搜索恢复或跨 run GPU 调度。准确范围见[支持矩阵](compatibility.md)。

## 内部组件与库外注册

决策器可组合自己的 harness、memory、研究模型和搜索策略；Lab 可组合数据、验证场景、
scorer、执行后端及训推引擎。库提供 default / DSH harness、memory 后端和数学 Lab
组件等已有实现，接入方可以复用或替换，也可以完全实现自己的内部结构。
不强制所有方法使用一种内部组件图。

常规扩展在项目自己的 `plugin.py` 中实现 `register(registry)`：
注册 controller、注册 Labs，再用独立 JSON 选择名称和参数。
`rsisys run --config experiment.json --plugin plugin.py` 沿同一 Host/Runner 运行。
插件是调用方信任的宿主代码；配置仅选择已注册名称。完整例子见[注册指南](registration.md)
和[库外示例](../examples/registered/README.md)。

| 改动 | 放在哪里 |
| --- | --- |
| 新算法、候选生成/选择、自己的领域任务 | 库外 controller / Lab 注册文件及配置 |
| 替换 harness、memory、校验器或引擎 | controller / Lab 工厂内装配；需要时复用库内组件 |
| 改变并发、反馈交付、恢复、预算中断或资源所有权 | 扩展核心实现与正式端口，并补充行为回归 |

原生 harness、完整组件图或专门恢复逻辑可使用 `MethodDefinition` / `ModuleAssembly`
高级装配。方法是两个模块及其组件的组合，不是第三类运行对象。

## 状态、证据与停止

运行保留配置身份、注册版本、插件 SHA、操作回执和证据引用。
`finished` 表示 host 生命周期结束；`controller_report` 是方法报告，
应再检查完成轮数、失败记录、真实 optimizer steps、权重变化与评测协议。
模拟 Lab、真实训练、质量改善和论文数值复现分别验证。
各方法的来源、已测范围和偏差见[方法支持与证据](methods.md)。

资源释放遵循所有权：组件清理自己创建的进程、连接和服务，不能把共享服务当作自有资源关闭。
插件须显式提供相应 shutdown/close 行为。同步调用的超时由执行端兑现；
Host 的有界推进不能抢占任意阻塞的 Python 或第三方训练调用。

恢复要求 controller、Lab、外部作业和预算具备匹配的状态契约与核对能力。
不确定副作用不能盲目重放；通用注册不自动序列化任意内存状态。
现有恢复机制见[运行时](../src/rsisys/runtime/README.md)和[快照边界](../src/rsisys/snapshot/README.md)。

## 共同演化与 reflect

这是 meta-agent 方法的可选能力。算法可以选择改进 Lab 中的模型/程序，
也可以改进决策器自己的 harness、memory 策略或模型；不要求逐轮同时更新两侧。
同一个 meta-agent 可以负责行动、反馈解释与显式记忆管理，无需另设整理 agent。
候选、验证、启用和收益仍分别记录；研究文本和记忆更新不自动证明能力提升。

默认 meta-agent 的任务窗口、memory、reflect、激活端口与医疗 Open Lab 的详细接线，
以及旧端点名称的兼容解释，集中在[默认实现与高级架构](advanced-architecture.md)。
这些是可选实现细节，不是所有 RSI 方法的必需内部结构。

## 代码与目录

| 位置 | 职责 |
| --- | --- |
| [src/rsisys/api.py](../src/rsisys/api.py)、[loops/contracts](../src/rsisys/loops/contracts/README.md) | 公开导入入口与正式接口 |
| [registry.py](../src/rsisys/registry.py)、[assembly.py](../src/rsisys/assembly.py)、[runtime](../src/rsisys/runtime/README.md) | 注册、装配和公共运行语义 |
| [controllers](../src/rsisys/controllers)、[labs](../src/rsisys/labs/README.md) | 可复用方法、Lab 与内部组件 |
| [examples](../examples/README.md) | CPU quickstart、库外注册与组件教程 |
| [experiments](../experiments/README.md) | 模型/数据配方、环境要求与结果，不复制框架运行器 |
| [tests](../tests/README.md) | 长期维护的行为回归，验证 `src` 中的正式契约 |
| [docs/review](review/README.md) | 有日期的设计与审查记录，不是当前 API 规范 |
| `.local/`、`runs/` | 被 Git 忽略的临时验证、运行日志、数据、模型及其他产物 |

后续核心工作与发布要求见[发布指南](releasing.md)。
