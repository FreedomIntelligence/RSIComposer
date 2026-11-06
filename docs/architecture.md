# RSISys：统一术语与架构

更新：2026-11-06。当前默认架构是 **一个 meta-agent + Lab + memory 模块**。
历史审查材料和冻结实验保留当时的两侧称谓；新运行以本文和实际装配配置为准。

## 一条主链路

**meta-agent → Lab → 原始 feedback → 同一个 meta-agent。**
meta-agent 直接负责行动选择、结果解释、调试、上下文管理以及记忆的写入、更新和删除。
旧“整理侧”的语义判断职责合入这个工作流；默认不再创建独立上下文中的整理模型。

![系统主链路与信息生命周期](diagrams/system-flow.svg)

| 名称 | 定义与职责 |
| --- | --- |
| **meta-agent** | 根据目标、真实状态、当前任务原始反馈和相关记忆选择行动，读写代码、调试和解释结果，并显式管理自己的记忆。`decision侧` 可作为兼容说明，不再另设整理侧。 |
| **Lab** | 任务与实验的执行环境，包含数据、模型或 agent、工具、harness 和授权资源；内部可以直接运行程序，也可以含 agent。 |
| **memory 模块** | 存储、索引、检索和修订记忆；默认 `AgentMemory` 使用本地原子 JSON 和确定性文本检索，不发出模型或 embedding 请求。 |
| **decision** | 一次带身份、参数和来源的行动或任务，不是另一种 agent。 |
| **Meta-agent harness** | 装配输入、执行工具边界、计量和容量管理；与 Lab harness 分开。 |

## 决策时看到什么

1. 宿主给定的任务、约束、权限、资源和截止时间。
2. 当前任务、作业和候选的真实状态；历史记忆不能覆盖它们。
3. 最新原始 Lab 回执，以及仍开放的实验/调试任务所固定的原始证据。
4. 有界检索的既有记忆与归档读取入口；记忆正文可通过 `memory_read` 阅读。

完整原始历史保留在执行归档中，按需读取。单条原文在容量内直接交付；超长内容使用明确的
UTF-8 原文页、完整性标记、SHA 和精确读取地址，不能把摘要或预览当作已读完整原文。

## 当前任务与长期记忆

| 信息 | 谁管理 | 生命周期 |
| --- | --- | --- |
| **原始即时反馈** | 执行设施交付，同一个 meta-agent 使用 | `previous_feedback` 展示最新原生操作；任务固定的证据见 `active_feedback`，中间读代码/日志和编辑不会挤掉它。 |
| **实验/调试任务** | meta-agent 调用 `research_task` 的 open/pin/close | open 可固定已有错误，pin 增补原始回执；任务关闭才释放该驻留窗口。任务完成不是单个工具调用结束。 |
| **可检索 memory** | meta-agent 调用 `memory_write`、`memory_delete` | 跨决策保留，显式修订；更新和删除均记录修订证据，不自动把每条 feedback 写成记忆。 |
| **原始归档** | 宿主持久化 | 独立于任务窗口和记忆；关闭任务或删除记忆不删除实验原文。可选轨迹产品不等于执行账本。 |

默认最多同时开放两个任务，每个可固定四条回执，并保留该任务最近两条 producer 回执。
原始正文使用共享有界窗口，超量时给出原文分页地址。没有授权归档能力时不悄悄丢弃原文。
这些限额是公开工具能力，超限返回可修正反馈，不驱逐别的活跃任务。

`memory_search` 提供文字匹配和近期检索；`memory_read` 可读当前或历史修订，大记录分页。
写入需要明确来源，或以无来源的 unverified 假设保存；supported/refuted 是 meta-agent 的判断，
宿主只校验来源与修订身份，**不把记忆散文升级为已验证评测或事实**。

## 默认接线与兼容范围

`ResearchAssembly()` 和 `ResearchAssembly.single_meta()` 默认安装同一个 meta-agent 的记忆工具与任务窗口。
医疗 Open Lab 默认 `--feedback-routing agent_memory`，没有后台 LangMem 语义抽取或独立提案模型。
默认 `ModelDecisionPolicy` 在该模式下的容量压缩只生成机械归档导航，不额外调用总结模型；
需要语义摘要时，由 meta-agent 在普通 decision 中写研究笔记或 memory。
所有这些普通 decision 仍按宿主统一调用和 token 账本计费。

旧 `ResearchAssembly.langmem()`、journal provider、`context_or_memory` 路由和 proposer 保留为
**显式选择的兼容工作流**，不混入单 meta-agent 默认路径。旧冻结运行不能换装本版恢复；
装配配置升为版本 5，仍拒绝无证据迁移。自定义 harness/provider 自行兑现其已声明能力。

新增 `AgentMemoryState` 保存任务、记忆和修订。医疗原目录恢复已注册该组件，原始历史、
调用账本和绝对截止时间继续沿用；磁盘状态与 checkpoint 不一致时阻塞核对，不清空日志、
回滚记忆或自动重放不明 mutation。可迁移快照仍需所有组件显式提供完整状态接口。

## harness、能力与宿主

始终写清归属：**meta-agent harness** 组织 meta-agent 的模型输入、工具调用和研究状态；
**Lab harness** 组织 Lab 内部的任务执行、推理、训练或评测。修改其中一边不表示另一边也改变。
同一个 meta-agent 负责行动与语义整理；memory provider 负责存储和检索。第三方 harness 可替换内部实现，但须显式接入记忆工具、上下文与恢复状态。

开放执行按已安装工具及权限提供编程能力，SFT / RL / evaluate 是可实现的任务，
不是框架强制给模型的固定算法菜单。模型可以修改 Lab 代码；真正可执行的范围仍由
已安装能力、实际权限、资源和执行后端决定。

| 基础设施 | 职责 |
| --- | --- |
| **Host / 宿主** | 装配模块、固定目标和验收规则、授权、资源准入、统一调用计费、截止时间、生命周期与 owned 资源清理。 |
| **Runner / 调度** | 按公开端口推进任务、等待、交付和停止。调度模块可替换；不自行产生研究结论。 |
| **Sandbox / 执行后端** | 运行代码并兑现其实际隔离边界。Lab 是环境，Sandbox 是可使用的设施，两者不是同义词。 |
| **看板 / Monitor** | 面向人的只读可观测界面，消费公开运行投影；不处于模型输入链路，不作为 agent 的反馈工具。模型读运行状态用 Lab 或显式状态查询端口。 |

组件选择使用统一 `ModuleAssembly`。可选 `ComponentGraph` 将 host 注册的训练/推理策略、
训推引擎、数据、harness 和 RSI 决策策略按显式依赖组合；构造前检查缺失项、契约冲突与
循环，运行身份记录每个组件的版本、参数及根端口绑定。训练引擎和推理引擎可以独立选择，
`ResearchAssembly` 继续负责 memory/feedback 接线，不代表整个 RSI 算法。
模型可修改的脚本和参数仍由 host 的工具、权限与执行隔离决定；组件目录不授予权限。
分类、代码入口、恢复与验证范围见[组件组合说明](component-composition.md)。

## 评测、训练和改进声明

| 名称 | 可以声明什么 |
| --- | --- |
| **执行回执 / receipt** | 某任务的提交、状态、完成、错误和产物引用；成功退出不等于质量提升。`Feedback` 是兼容执行端口中的一种终态消息类型。 |
| **探索性评测** | 模型自行编写或修改的测量结果；须保留代码、数据与协议身份，不能冒充固定正式评测。 |
| **正式评测** | 领域装配或宿主固定的验收协议。可比较结果须绑定精确候选、数据、解析、指标与来源。医疗 Open Lab 的正式评分由 `FixedMedicalLab` 提供。 |
| **训练完成** | 有真实 optimizer steps、数据曝光、loss 或权重产物等适用证据；写过训练代码、提交过程序、测速或 fixture 不能代替训练。 |
| **候选 / candidate** | 待验证的模型或组件变更；可以尚未评测、部署或采纳。 |
| **提案 / proposal** | 未执行的建议，不能写成已构建、已训练或已应用。 |
| **部署 / 应用 / 采纳** | 分别记录服务就绪、实际版本绑定变化与结果接受；三个状态不由摘要、分数或成功回执互相推定。 |

训练数据构建、SFT / RL、harness 构建与验证都可以作为 **Lab 内任务**；不强制每类任务
新建一种 Lab。只有需要独立资源、权限或生命周期时，才单独装配改进 Lab。
原始归档到数据构建、训练和应用之间都需要正式任务及相应证据，不能按轨迹数量自动推定训练质量。
固定正式协议适用于对应领域装配；核心框架不内置通用于所有领域的评分阈值。

## 项目、运行和恢复

- **项目 / project**：共同目标和领域配置下的一组运行。
- **运行 / run**：一次完整系统运行，不是单个训练 job；暂停后在原目录恢复仍是同一 run。
- **任务 / task、作业 / job**：某次实际执行及其生命周期；提交确认不等于终态完成。
- **研究尝试 / Attempt**：围绕一个想法的可选决策关联；执行时的明确归属与历史兼容工作流的事后分组分别记录，不增加调度层级。
- **版本**：区分代码版本、模型权重版本、系统组件版本、评测协议版本和记忆修订，不能用一个 revision 替代所有身份。
- **运行恢复 / durable resume**：在原目录按显式组件状态适配器接续研究状态、调用账本与相同截止时间；未知副作用阻塞核对，不自动重放。
- **可迁移快照 / snapshot**：在支持的空闲边界导出、到新目录恢复；不是任意组件的通用进程快照。Lab 的 optimizer checkpoint 另由具体训练程序实现。

新版医疗 Open Lab 已接通正常暂停与原目录恢复；旧的非 durable 运行不能追溯恢复。
完整边界与命令见[恢复说明](review/open-lab-recovery-20261105.md)。

## 接口与实现映射

![模块与代码位置](diagrams/file-architecture.svg)

新文档、图示和 prompt 使用上面的统一称谓；旧 Python 符号、配置键、日志字段与来源引用
暂保持兼容。以下是映射，不应把兼容名字重新宣称为额外的系统角色。

| 现有标识 | 统一解释 |
| --- | --- |
| `decision_maker` | 历史 meta-agent 模型角色键；默认唯一的 meta-agent 负责行动、调试与记忆管理。 |
| `DecisionMakingLoop` / `DefaultDecisionHarness` / `ModelDecisionPolicy` | 唯一 meta-agent 的默认端点、harness 和策略实现。 |
| `FeedbackLearningService` / `feedback_learning` | 原始回执、归档、交付与兼容处理器管理；默认不调用整理模型，learning 不表示权重训练。 |
| `AgentMemory` | 默认确定性记忆存储、检索、显式写入和任务反馈驻留；不持有模型。 |
| `LangMemContextMemory` | 显式选用的旧版独立抽取工作流；不属于默认单 meta-agent 路径。 |
| `LabExecution` / `LocalProcessLab` / `FixedMedicalLab` | Lab 的公开执行窗口及两种具体实现。 |
| `ExplorationLoop` / `ExplorationLab` | 兼容执行端口及其 Lab 适配；不是所有 Lab 都必须有的内部结构。 |
| `actor` / `ActorPolicy` / `Environment` | 兼容 Lab 的内部执行组件，不能与整个 Lab 或 meta-agent 画等号。 |
| `issuing_decision_model` / `learning_decision_model` | 分别是发出动作时、捕获整理任务时的模型身份，保留各自历史。 |
| `subject=meta_agent|lab|unspecified` | 提案或产物作用对象；它与当次模型的工作职责是两个维度。 |

可替换模块只兑现公开窗口：Lab `describe/execute`、memory `context(query) -> str`、
可选归档 `write/read`、可选提案 `propose`、可选调度 `select`，以及需要恢复时的显式状态端口。
各模块可以自由选择算法和内部结构；宿主保留身份、授权、状态、引用与恢复一致性的必要校验。
窗口的具体协议见[模块窗口规范](review/feedback-module-windows-20261104.md)。

文档和两张 SVG 由同一组术语维护；重新生成图示运行：

```sh
python docs/diagrams/build_diagrams.py
```
