# Lab 执行请求与生命周期接口审查

审查结论：`describe / execute` 的开放入口可以保留。默认执行器与 host 的生命周期接线有
4 项已复现缺陷，其中 2 项涉及 owned 程序的清理。本轮完成审查和复现，以下缺陷尚未修复。

基线为 `cb67628151cdb855b30f657e37cbf0af801644a8` 加现有工作树。
审查前 121 个已有差异文件的 SHA 与补丁保存在
`.local/lab-execution-contract-review-20261104/`。此前的执行修复和交付语义记录保持为历史证据。

## 当前调用链与最小协议

| 阶段 | 当前实现与职责 | 审查结果 |
| --- | --- | --- |
| 发现能力 | [LabExecution](../../src/rsisys/loops/contracts/execution.py) 的 `describe() -> tuple[OperationSpec, ...]`，提供名称、description、input_schema、read_only | 目录来自 Lab，未固定研究算法；装配时捕获一次 |
| 提出请求 | [OperationCall](../../src/rsisys/loops/contracts/decision.py) 的 `name / arguments` | meta-agent 可提交具体修改或完整程序任务；JSON 结构通用，业务参数由 Lab 校验 |
| 授权与调用 | [LabExecutionTools](../../src/rsisys/adapters/execution.py)、[执行器](../../src/rsisys/loops/decision_making/harness/tools/__init__.py)、[权限窗口](../../src/rsisys/loops/decision_making/action_space/access.py) | 执行前再次授权；schema 描述不冒充权限或完整 JSON Schema 校验 |
| 同步结果 | `execute(call) -> OperationResult` | 表示此次工具调用的结果；业务输出放在 data，提交调用完成不能证明后台程序完成 |
| 异步提交 | `execute(call) -> JobRef` | 等待 host 显式 resume；此两方法接口本身不提供通用 poll/cancel/resume 桥接 |
| 非阻塞工具路径 | [LocalProcessLab](../../src/rsisys/adapters/local_lab.py) 的 start / poll / result / cancel | 当前内置实现采用普通工具回执和原生 job_id，decision 可以继续选择其他动作 |
| 停止 | 可选 `shutdown(mode) -> bool` | 后台资源持有者需要兑现：停止准入、cancel 或 drain；True 只证明 owned 工作退出 |
| 释放资源 | 可选 `close()`，由 assembly / Application 注册 | 释放 workspace 等资源；失败后的上层重试链路存在 F01 |

原始 Lab 文本/JSON 也可由 [ExecutionReceipt.from_output](../../src/rsisys/loops/contracts/receipts.py)
进入反馈端；执行状态与任务关联由执行设施补充，领域输出不必转成统一训练/评测字段。

这里有三种不同的完成：工具调用完成、后台程序终态、资源清理完成。
LocalProcessLab 的普通 OperationResult.status 表示工具调用结果，程序状态另见 data.state。
Runner 的 execution shutdown 确认程序退出后，workspace 的释放仍由 close 负责。

## F01 · P1：上层 close 失败后丢失 owned 清理的重试机会

位置：[Application.close](../../src/rsisys/application.py)、
[AssemblyContext.close](../../src/rsisys/assembly.py)。

二者在清理前就置 `_closed=True`；ExitStack 执行失败 callback 后也将其弹出。
因此 LocalProcessBackend 虽然保留了失败 session，Application / demo 的再次 close
已经不会调用 Lab.close。Application 还会释放 workspace 锁、store 等共享资源。

复现使用一个持有自己后端的合法 Lab 插件和真实 CPU 程序：首次关闭注入临时信号失败，
故障消失后再次调用公开 owner.close，Lab.close 累计仍只有 1 次，程序仍活着、session 为 1。
Application 与 demo 两条装配路径均复现。直接调用保存下来的 Lab 引用才成功清理到 session=0。

建议实现：单独跟踪尚未完成的 owned 资源清理。关闭开始后禁止新工作；成功的清理不重复，
失败的幂等 Lab 清理保留为 pending，后续 close 可继续。owned 退出与资源释放确认前保持必要的
共享依赖和 workspace 锁。普通自定义 callback 不能因结果未知而一律重放。
需要补上层生命周期回归，已有 backend.close 重试测试不能覆盖这个问题。

## F02 · P1：一个任务的 poll 异常阻断其他任务取消及停止超时报告

位置：[LocalProcessLab.shutdown](../../src/rsisys/adapters/local_lab.py)、
[Runner._stop_step](../../src/rsisys/runtime/runner.py)。

shutdown 按顺序 poll，首个异常立即退出，后续任务没有收到 cancel。
Runner 在调用 shutdown 后才计算 cleanup timeout；shutdown 每次抛错时，
无法进入超时分支生成 unresolved_execution 报告。

复现同时运行两个 owned 程序，首个任务发生 exit-observer 故障且第一次 kill 失败。
恢复信号能力后，首个 poll 仍报告 cleanup incomplete，随后任务的 cancellation_requested=False，
程序继续运行。在实际 Runner 中，清理上限设为 1 秒，推进到已清理 8 秒时仍连续抛 RuntimeError，
没有 terminal stop report。最后通过直接 close 的 finally 后端清理使两个程序退出。

建议实现：各 owned job 独立推进取消/清理，遍历后再汇总异常；单个 poll 失败不能跳过其他任务。
host 保留异常和未退出任务，在后续 tick 继续收尾；即使 shutdown 抛错也要检查清理期限，
到期明确报告未完成清理。基础设施错误仍由 host 处理，不能伪报 shutdown_complete=True。
额外边界：任意阻塞的同步 provider 需要自己的超时，tick 层的计时不能抢占 Python 方法。

## F03 · P2：监视设施故障被降为普通程序失败

位置：[LocalProcessBackend._record_error / _finish](../../src/rsisys/runtime/local_process.py)、
[LocalProcessLab._observe](../../src/rsisys/adapters/local_lab.py)。

watcher / reader 的内部异常先写成 entry.error，最终只剩 FAILED 与 stderr 文本。
LocalProcessLab 无法区分它与用户程序自身非零退出，于是生成普通失败 OperationResult。

复现 exit observer 抛 OSError：owned 程序已被正确终止，但 runner.step 返回 progressed，
没有错误抛给 host，记录为 Program failed。它突破了已确认的“基础设施故障交给 host”语义。
此前 LabToolError 修复覆盖 provider 抛出异常的边界，此处异常在更底层已经被转成了结果。

建议实现：进程结果保留机器可识别的故障来源，例如程序退出、超时、取消、监督设施故障。
适配器将监督设施故障交给 host，并保持原始失败与清理证据可读。
不依赖匹配 stderr；程序自己打印相同文字不应被升级为基础设施故障。
这属于执行状态元数据，不要求领域业务输出符合新 schema。

## F04 · P2：常见路径输入错误仍中断当前 host 调用

位置：[LocalProcessLab._parent / _file](../../src/rsisys/adapters/local_lab.py)。

“父路径实际上是普通文件”和“把文件内容写到已有目录名”分别抛 NotADirectoryError /
IsADirectoryError，没有进入 LabToolError 路径。实际 runner.step 因此抛异常，虽然已经记录失败回执，
仍需要 host 捕获才能继续 decision。

建议实现：只将 workspace 内这两个明确的路径类型冲突归为模型可修正工具错误。
workspace 根目录异常、权限/符号链接拒绝、空间或存储错误继续交 host；
不宽泛吞掉 OSError。补“错误进入下一次 decision 的输入，修正路径后成功”的回归。

## 工具目录与 prompt 核对

[工具 description/schema](../../src/rsisys/adapters/local_lab.py) 在 describe 中提供。
现有模型请求回归确认它们实际进入 context.operations，以下均是内置实现的能力，不是通用 Lab 的必填工具。

| 工具 | 最小参数 | 当前承诺 |
| --- | --- | --- |
| lab_write_file | path, content | 写代码/配置；SFT、RL、evaluate 可自行实现 |
| lab_read_file | path；可选 offset/max_bytes | 有界 UTF-8 字节窗口，分片边界可能显示替换字符 |
| lab_export | path | 在 workspace 清理前显式保存文件产物 |
| lab_start | key, argv；可选 timeout_seconds | 同 key/同 ExecRequest 幂等；只确认提交，不冻结代码树 |
| lab_poll | job_id | 查询状态和日志；只读归档 |
| lab_result | job_id；可选 reprocess | 收集终态进入反馈；同事件默认整理一次，显式重整不重跑程序 |
| lab_cancel | job_id | 仅请求取消，后续查询或 host 确认退出 |

[decision.txt](../../src/rsisys/loops/decision_making/harness/policy/prompts/decision.txt)
实际说明了开放程序入口、提交不等于执行完成、仅使用已安装能力；
[research.txt](../../src/rsisys/loops/decision_making/harness/policy/prompts/research.txt)
要求记录代码/协议/数据引用，并维持同协议比较与 host 验收权限。
这两处 prompt 的方向可保留，四个缺陷都需要执行实现和 host 状态管理来修复。

## 已明确的实现缺口

- 当前 LocalProcessBackend 为受信本机执行：文件工具的路径校验与 argv 的进程组管理不提供 OS 隔离。
  任意程序的设备、数据、网络、磁盘和内存权限需要部署 executor 实际限制。
- 现有每命令超时、并发数、job 数和文件字节上限，不自动计量通用程序内部的训练 steps、样本曝光或模型调用。
- JobRef 的自动 poll/cancel/resume 适配未包含在两方法入口；当前应选择已装配的外部桥接或非阻塞工具路径。
- 开放 Lab 的 durable snapshot / restore 仍被显式拒绝；内存内幂等 key 和反馈 journal 去重不能替代进程恢复。
- 终态 artifact 保存 argv、状态、日志；完整代码/权重/评测身份仍需 executor 或任务主动保存引用。
  这四项生命周期修复不会自动补齐真实训练、部署和验收能力。

## 验证与后续实现顺序

[可重复 probe](repros/execution-lifecycle-contract-probe.py) 使用临时 CPU 程序、确定性模型响应和故障注入。
所有试验程序最终由 probe 的显式清理退出，session=0；没有启动远端任务、模型服务或训练。
模型响应仅为接口 fixture，不能当成自主研究质量验证。

现有 8 个相关测试模块共 108 项通过，耗时 1.672 秒。
这些测试通过与上述反例同时成立：当前回归未覆盖这些分支。
本轮未改生产代码和测试文件，没有重复运行上一轮的全量 1607 项。
原始观察、测试日志与来源 SHA 见 [validation](repros/execution-contract-review-validation.json)。

建议先兑现 F01/F02 的清理与停止，再补 F03/F04 的错误来源分类；维持开放执行入口。
实现方案按上述职责拆分，后续修复应逐项通过新的反例回归，再运行全量测试。
