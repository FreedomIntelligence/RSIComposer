# Feedback 模块边界审查

日期：2026-11-04。范围：当前工作区，包括尚未提交的反馈重组与原生 Lab 输出适配。
本次是接口与实现方案审查，没有修改生产逻辑，也不代表 CP3–CP7 已经通过用户验收。

后续进展：用户已确认并授权落实 [Context / Memory 接入窗口](feedback-context-window-20261104.md)。
R1 及相关默认接线已调整；下文保留审查时发现，不表示其他待审接口已经全部更改。

后续实现见 [模块窗口与运行边界](feedback-module-windows-20261104.md)：公开原始事件归档、
原生产物处理/建议、调度选择已接线，旧内部协议隔离在适配器中；产品独立与发布故障隔离已调整。
本页下方保留原始问题证据。完整决策模型交换尚未捕获、真实总结质量未验证等能力边界仍存在。

结论：默认链路已有可运行实现，但“能换一套遵守内部协议的实现”与“只接基本职能窗口即可替换”
仍有差距。Lab 输入已减负；memory、归档、处理器和调度仍混入默认实现的组织方式。
其中发布异常传播到 decision context 是具体运行风险，其余主要是接口与装配边界偏差。

## 审查依据

- Lab 输出保持原生文本、JSON、日志或文件引用；宿主补充任务关联及生命周期。
- context/memory 能力始终存在；不要求使用内置算法、存储、prompt 或某个开源 SDK。
- memory 对外提供可供 meta-agent 使用的 context，不必模拟本项目的 journal 或 Memory 类。
- 原始轨迹归档默认启用但可关闭或替换；提案、扩展调度按需安装。
- 数据集构建与训练由改进 Lab 负责；反馈端不把建议当作正式执行或已应用改进。
- 宿主负责身份、交付、准入、调用计费、停止与恢复。它的执行账本不等于可选轨迹归档产品。

## 四个出口的现状

| 出口 | 当前已具备 | 尚需调整 |
| --- | --- | --- |
| 即时 context | 默认 harness 直接记账、交付实际结果，不等待反馈模型整理 | 新 Lab 的 observe_output 只是观察入口，接入方仍需完成结果交付；不应误称为已自动接通所有 Lab |
| context/memory | 默认 journal 整理器与可选 LangMem；可替换实现能够进入 context | 三方法、内部 job、dict/Memory 类型要求超出基本职能窗口 |
| 原始轨迹 | 默认索引与不可变公开片段；支持关闭 | 插件接口仍是 journal 查询，没有独立原始事件写入窗口；默认未捕获完整 meta-agent 模型交换 |
| 改进提案 | 可关闭；正式 decision 引用后才构建/训练 | 与 memory 共用 proposal 协议、数量限制及成功依赖，输出与失败隔离不够独立 |

## R1：Memory 窗口仍强制三方法和内部类型

证据：

- [components.py](../../src/rsisys/loops/decision_making/feedback/components.py) 15–26：要求 record、working_memory、memory。
- [research.py](../../src/rsisys/research.py) 52–55：装配时逐个检查，不能只提供 context 输出。
- [learning.py](../../src/rsisys/loops/decision_making/feedback/learning.py) 814–825：记录、工作视图必须返回 dict；调用方传入完整 job、attempts。
- [service.py](../../src/rsisys/loops/decision_making/feedback/service.py) 155–159：发布又要求内部 Memory 或 None。
- [langmem.py](../../src/rsisys/loops/decision_making/feedback/langmem.py) 296–321：实际使用检索视图，仍需实现返回 None 的 memory 方法以满足核心要求。

影响：一个已经能生成 context 文本或 messages 的 SDK，仍要实现额外方法、理解 journal 字段和
模拟内部发布方式。当前测试证明“自定义 dict 可以接入”，尚未证明“只接 context 窗口即可”。

调整方案：核心接收可注入 context；SDK 的更新、检索、整理与持久化由其 adapter 对接。
JournalContextMemory 的三方法留在默认适配内部；普通实现不必返回内部 Memory，记录/审计展示
也不成为它的必选方法。当前决策查询信息按实际需要提供，不强制所有实现消费 attempts 或 WorkingMemoryPolicy。
输出究竟选择文本还是宿主支持的 messages，在输出窗口 checkpoint 中确定，不在本次增加一套大 schema。

验收：一个直接生成 context 的最小实现可以进入下一次 decision 的实际请求；不需要假实现
record/memory，也不需要了解 jobs、reserved_calls 或内置存储布局。

## R2：处理器产品协议承担了过多宿主职责

证据：

- [components.py](../../src/rsisys/loops/decision_making/feedback/components.py) 38–48：要求 configuration/process/recover，以及每任务正数调用预留。
- [learning.py](../../src/rsisys/loops/decision_making/feedback/learning.py) 399–427、455–460：自报 charged_calls、固定终态、字段白名单；自然的 context 字段也会被拒绝。
- [pipeline.py](../../src/rsisys/loops/decision_making/feedback/pipeline.py) 51–67：memory 和提案都使用 validate_proposals，memory 必须是 kind=memory 的 proposal。

影响：无模型调用的处理器也不能将 max_calls_per_job 设为 0；普通同步函数也必须提供恢复方法。
Memory 内容要借用提案格式表达。避免插件覆盖 journal 的目的合理，但不必通过限制其所有业务输出实现。

调整方案：产品内容与宿主任务记录分开。插件通过 adapter 返回 context、存储引用或建议等本职结果；
宿主记录任务状态、错误、用量与交付。模型调用继续走受控 gateway，未知调用仍保守计费、不盲目重放。
恢复集成按实际副作用能力提供：简单函数不必虚构 recover；有外部写入或模型调用的实现，由 adapter
明确提供恢复策略，不能因简化接口而取消执行保证。提案消费者所需的结构只在对应 adapter 中校验。

验收：同步纯整理器、使用外部存储的整理器、受控模型整理器均能接入；前两类不因没有模型调用被拒绝。
内置用量/状态不可被业务 payload 覆盖；非法正式动作仍在执行前被具体消费者拒绝。

## R3：可选 Archive 实际仍是 journal 派生查询

证据：

- [components.py](../../src/rsisys/loops/decision_making/feedback/components.py) 29–35：基本方法是 records(jobs)，可选 extract/snapshot 也读取内部 job。
- [learning.py](../../src/rsisys/loops/decision_making/feedback/learning.py) 832–858：导出由完整内部 job 调用插件，外部归档没有独立写入接点。
- [knowledge.py](../../src/rsisys/loops/decision_making/feedback/knowledge.py) 99–115、134–144：默认覆盖公开 decision 记录与执行回执，明确没有完整原始 meta-agent 交换。

影响：替换成文件、对象存储等实现时，仍需理解内部 journal，或者在查询时才进行写入。
当前默认片段有用，但不能作为“已经完整归档 meta-agent 训练轨迹”的证明；这也是已记录的能力缺口。

调整方案：给原始事件保存/读回提供独立接点，内容保持原生；默认 journal 通过 adapter 实现该接点。
目录、索引、分段、压缩与检索是归档实现选择。特定数据构建 Lab 需要不可变引用或完整交换时，
由该消费者及其适配器要求，不推广为所有 archive 的基础格式。
完整轨迹需要另接实际 decision 请求/响应的观察点，不能用 reason 或反馈摘要补造缺失内容。

验收：简单外部归档无需 jobs schema；关闭归档不关闭实际结果交付和 context/memory，也不删除宿主账本。
数据构建 Lab 能识别来源覆盖不足，不能把部分公开片段当完整会话。

## R4：可选提案存在未声明的 Memory 成功依赖

证据：

- [pipeline.py](../../src/rsisys/loops/decision_making/feedback/pipeline.py) 99–111：memory 失败直接将 proposer 标记为 not_started。
- 同文件 74、111：两个产品实际读取同一原始 job，proposer 并不消费本次 memory 结果。
- 同文件 80–97：总体 status 取自 memory；memory 与提案合计超过 6 项会拒绝提案产品。

影响：局部 memory 故障会阻止原本可独立产生的建议；memory 数量也影响提案能否保留。
共享宿主总预算合理，但产品依赖和共享内容上限不应成为所有实现的通用事实。

调整方案：分别记录产品完成与错误，只有实际声明依赖的组件才等待前置结果。
默认策略可以串行运行，但是否依赖成功由装配决定；共享预算仍由 host 准入。
Memory 不再占用提案名额；内置 proposer 的内容限制留在其消费适配中。

验收：独立提案不被 memory 失败自动跳过；有显式依赖的提案保持等待。
可选提案失败不抹掉已经完成的 context，也不被误报为已执行或已应用。

## R5：发布异常会传播到主 Decision 上下文组装

证据：

- [service.py](../../src/rsisys/loops/decision_making/feedback/service.py) 138–164：provider.memory 或 adapter.publish 的异常保留待发布状态，但继续向外抛出。
- [learning_harness.py](../../src/rsisys/loops/decision_making/harness/learning_harness.py) 175–205：project_context 直接 refresh、读取工作视图，没有产品故障边界。
- [test_feedback_components.py](../../tests/feedback/test_feedback_components.py) 150–169：已有测试验证发布失败后重试、不会重复模型调用；本次探针进一步验证异常传播到 _context。

影响：处理已经完成，单次存储/发布异常仍可能让下一次 decision 无法构造输入。
当前恢复避免重复推理是正确行为，需要保留；缺口是产品错误如何影响主循环尚未由明确策略控制。

调整方案：宿主分别记录整理完成、投递成功和重试状态；可选产品故障留在自身通道。
对必需 context 窗口，建议默认允许使用仍有效的上一份 context，并直接交付最新实际结果，显式记录整理失败；
无法提供任何有效 context 时按配置停止或采用事先配置的简单实现。该策略需 checkpoint，
不把临时故障处理变成无声替换用户选择的 provider。持久写入失败、身份冲突和未知副作用仍按执行恢复策略处理。
复用范围只限整理视图；当前执行状态、候选和模型身份仍由 host 提供最新值，不回退整个 DecisionContext。

验收：fixture 中的发布异常成为可观测产品错误；使用允许继续的策略时主 decision 仍能拿到实际结果。
重试只重试已完成产物的投递，不重新执行 Lab 或重复模型请求；严格策略仍可明确停止。

## R6：Executor 可替换，但调度策略尚未独立

证据：

- [learning.py](../../src/rsisys/loops/decision_making/feedback/learning.py) 438–440：支持传入 executor。
- 同文件 745–747、795–797：有在途 future 就不再派发，固定 FIFO；默认单 worker。
- [learning_harness.py](../../src/rsisys/loops/decision_making/harness/learning_harness.py) 175–176：通过组装 decision context 驱动 refresh。

影响：不能说执行器完全不可替换，但换 executor 仍无法替换优先级、合并、批处理或后台 tick 策略。
“每次反馈整理”“累计后整理”等实现选择目前受到固定 journal 调度影响。

调整方案：保留简单默认调度，将刷新时机、选择哪些待处理输入与实际执行器区分。
扩展策略可选安装；模块内部已有批处理/后台管理时由 adapter 对接，不再强迫使用核心 FIFO。
Host 仍控制停止、准入和计费；不因调度扩展可选而取消基本运行驱动。

验收：默认串行策略可独立工作；自带后台管理的实现可接入；不需要伪装默认 future/jobs 状态。

## R7：Prompt 的组件化尚未覆盖旧兼容路径

证据：

- [langmem.py](../../src/rsisys/loops/decision_making/feedback/langmem.py) 172–174：支持资源文件与 additional_instructions 覆盖。
- [proposals.py](../../src/rsisys/loops/decision_making/feedback/proposals.py) 25–34：支持资源文件与 instructions 覆盖。
- [learning.py](../../src/rsisys/loops/decision_making/feedback/learning.py) 610–651：旧解释路径仍将 prompt 拼在核心代码中，并包括旧 training_data 等产品。

判断：LangMem、提案的 prompt 已放在各自实现内，这是应保留的方向。
旧组合解释仍可通过低层 enable_feedback_learning 使用；统一 ResearchAssembly 的新 pipeline
不走这个 prompt，不能将兼容路径的问题误报为所有新装配都会产生训练数据。

调整方案：将旧组合解释作为显式兼容组件保存其 prompt/schema，不再把它当核心默认定义。
新实现可以拥有完全不同的 prompt 或无需 prompt；只审查其消费接口及实际请求。

验收：切换 memory/proposer 不要求修改核心 _request；历史请求仍从已记录交换读回，不用新 prompt 重造历史。

## 建议的最小边界

下面是职责窗口，不是另加一套强制 Python 基类或业务 schema：

| 部件 | 输入窗口 | 输出窗口 | 归实现所有 |
| --- | --- | --- | --- |
| 即时反馈 | 原生执行结果与宿主关联 | 交给当前 decision 的实际结果 context | 领域结果的呈现和必要适配 |
| Memory/context | 反馈、按需提供的当前决策查询 | 可注入的 context | 抽取、总结、更新、检索、存储、prompt、多阶段模型调用 |
| 可选原始归档 | 要保存的原始事件 | 可读回记录或位置 | 文件/数据库、索引、分段、压缩、目录 |
| 可选提案 | 反馈及实现需要的来源 | 可供正式 decision 考虑的建议 | 筛选规则、频率、prompt、建议的内部结构 |
| 可选调度扩展 | 待处理输入与运行事件 | 待执行的整理工作 | 优先级、批次、触发时机 |

运行元数据由宿主或 adapter 维护。只有具体消费者要自动评测、执行或训练时才增加它需要的契约，
不要求所有插件先生成候选、协议、指标声明或训练样本。

## 后续实现顺序

1. 先确认并落实 context 输出窗口及到实际 decision 请求的接线；将三方法与内部 job 隔离在 adapter 内。
2. 按相同原则拆原始归档和可选提案接口，产品独立记录，具体执行仍走正式 decision/Lab。
3. 明确发布故障与 context 必需能力的处理策略，再拆默认调度和兼容 prompt。

身份、预算、原始执行事实、幂等和未知调用不重放继续由 host 保证。
上述顺序是调整建议；本次没有把尚未 checkpoint 的取舍写成已实施架构。

## 验证与材料

本次离线探针验证当前限制，未调用真实模型或训练：

- 只提供 context 的实现被装配检查拒绝。
- processor 返回 context 字段被白名单拒绝。
- memory 失败后 proposer 的执行次数为 0，状态 not_started。
- 发布异常传播到默认 harness._context；再次组装成功，fixture 模型请求仍只有 1 次。

脚本：[feedback-contract-probe.py](repros/feedback-contract-probe.py)。
结果：[feedback-contract-probe.json](repros/feedback-contract-probe.json)。
该结果保留审查时快照；脚本在后续版本允许 context-only 组件通过，重跑写入 feedback-contract-probe-current.json，避免覆盖原证据。
从仓库根目录运行：`PYTHONPATH=src:. python3.12 docs/review/repros/feedback-contract-probe.py`。
探针用于复现当前行为，输出不表示这些限制已经符合目标架构。

本次未修改生产代码，未重跑整仓测试；前序验证为 1,482 项、1,470 通过、12 跳过、0 失败/错误，
详见 [前序验证摘要](repros/feedback-thin-validation.json)。该回归不能替代本次接口职责审查。
