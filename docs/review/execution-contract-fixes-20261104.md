# Lab 执行生命周期修复

> 历史记录：原始探针、日志和请求已转入[归档说明](README.md#archived-evidence)；本文命令描述当时的工作，不作为当前可执行教程。

[执行请求与生命周期审查](execution-contract-review-20261104.md) 的 F01–F04 已修复。
保持 `LabExecution.describe/execute` 的开放入口，调整 owned 清理与错误交付，不增加领域训练算法约束。
原审查、失败 probe 和验证文件保持原样，作为修复前证据。

基线为 `cb67628151cdb855b30f657e37cbf0af801644a8` 加本地已有模块化改动。
修复前 124 个差异文件的 SHA 与补丁保存在 `.local/execution-contract-fixes-20261104/`。
本轮未提交、推送、连接远端或启动研究任务。

## 修复与行为

| 问题 | 修复后行为 | 实现 |
| --- | --- | --- |
| F01：owner.close 失败后不再清理 Lab，并提前释放锁/store | 关闭开始即拒绝新工作；成功和普通回调不重放；幂等 owned close 失败则保留该项及其依赖，后续公开 close 重试；Application 保持 workspace 锁和共享资源，直到 pending 清理完成 | [assembly.py](../../src/rsisys/assembly.py)、[application.py](../../src/rsisys/application.py) |
| F02：首个 poll 异常跳过其他取消，Runner 无法报告清理到期 | 每个 owned job 独立 poll/cancel；包括 poll 失败的同一 job；全部取消请求发出后逐个保存终态证据，再汇总异常；Runner 即使收到异常也检查期限 | [local_lab.py](../../src/rsisys/adapters/local_lab.py)、[runner.py](../../src/rsisys/runtime/runner.py) |
| F03：监督设施故障被当成普通程序失败 | ProcessState 增加可选 failure_origin；监督/reader 故障为 infrastructure，超时为 timeout，程序非零退出为 program；Lab 抛携带已归档 OperationResult 的 LabInfrastructureError，记录原始证据后继续抛给 host | [sandbox.py](../../src/rsisys/runtime/sandbox.py)、[local_process.py](../../src/rsisys/runtime/local_process.py)、[execution.py](../../src/rsisys/loops/contracts/execution.py)、[共享执行器](../../src/rsisys/loops/decision_making/harness/tools/__init__.py) |
| F04：普通路径类型错误抛给 host | workspace 内普通文件被用作父目录、文件写到已有目录、读取/导出目录，均为 LabToolError；下一次真实 policy 请求包含失败回执，修正路径后继续 | [local_lab.py](../../src/rsisys/adapters/local_lab.py) |

### 关闭与依赖

`AssemblyContext.add_cleanup` 仍是 LIFO、每项最多执行一次；普通回调失败也不会跳过后续普通回调。
`add_owned_cleanup` 显式标识幂等 owned close：失败后保留自身，并暂停释放注册在它之前的依赖。
Lab factory 返回的可选 close 自动按 owned close 注册。已经执行的普通回调，即使结果未知，也不重放。
存在多项清理错误时保留为 exception group，避免覆盖较早的失败。

Application 和 demo 都使用 AssemblyRunner 的关闭准入检查。Application.run、直接 runner.step
及重新进入已关闭的 owner 都拒绝新决策。首次 owned 清理失败期间，工作区锁和 store 保持可用；
再次 owner.close 成功后释放。该行为不要求捕获私有 Lab 引用手工清理。

### 停止报告

shutdown 有错误时，在清理期限内仍抛给 host；Runner 保留最后一份结构化错误，exception group
逐层保留子错误类型与消息。到期生成 `run_stop`，明确 `shutdown_complete=False`、
`cleanup_timed_out=True`、`unresolved_execution=True`，不把取消请求当成退出确认。
暂时错误恢复后可确认退出，同时保留错误及 `resolved=True`，便于追查。
非 bool 的 shutdown 返回值仍是 TypeError，不能冒充完成。

`finished` 表示研究调度已停止；存在 unresolved_execution 时，host 仍需执行 owner.close，
重试失败的 owned 清理并确认退出。workspace 释放与 shutdown 的进程退出确认分别验收。
同步 provider 必须自行限制方法调用时长；cleanup timeout 无法抢占一个阻塞不返回的 Python 方法。

### 基础设施失败与证据

失败来源由进程后端填写结构化字段，不匹配 stderr。程序打印 “Process supervision failed …”
并非基础设施故障的证据。字段默认 None，保持旧 ProcessState 构造兼容；成功、取消仍有自己的状态。
LocalProcessLab 将该字段、原始 stdout/stderr/exit_code、request 和 process 引用保存到 artifact。
poll/result 遇到 infrastructure 则抛 LabInfrastructureError，异常本身携带同一份失败结果，
共享执行器和 decision history 保留其 artifact 引用后抛回 host。

lab_result 的终态事件 ID 在抛异常前仍被保留，重复收集使用同一事件；lab_poll 仍为只读查询。
内部 shutdown/close 关注退出与归档，不会因同一个已退出且已记录的设施故障永久抛错，
因此“执行因设施失败”和“资源已清理”可以同时成立。

### 路径分类

父目录 no-follow 打开失败后，只在 descriptor-relative、follow_symlinks=False 的 stat
确认该组件为普通文件时转成 LabToolError。符号链接可能同样报 ENOTDIR，仍交给 host。
workspace 根打开错误、权限拒绝、空间不足及其他存储异常没有宽泛转成可恢复错误。
临时写文件与读取 descriptor 在上述失败时仍释放。

## 回归与证据

[新增回归](../../tests/regressions/test_execution_contract_fixes.py) 共 14 项，覆盖真实 owned CPU 程序、
故障注入和实际 ModelDecisionPolicy 请求；[底层生命周期测试](../../tests/runtime/test_local_process_lifecycle.py)
补上 watcher / reader 故障来源断言。模型响应为确定性 fixture，没有真实模型推理或训练。

- 定向 9 模块 122 项通过，涵盖新反例、默认装配、权限、重复交付、操作记录和 run policy。
- 全量最终结果与用时见 [validation](README.md#archived-evidence)。
- 使用原封不动的 [历史 probe](README.md#archived-evidence) 在最终代码重新执行；
  原始输出保持不变，新输出为 `.local/execution-contract-fixes-20261104/boundary-probe-verified.json`。
- Application/demo 公开 owner.close 重试后 owned 程序退出、session=0；后续任务收到取消请求；
  Runner 到期存在明确终态报告；路径错误不抛 host，监督设施错误抛 LabInfrastructureError。
- 修复涉及文件、历史证据 SHA、已有改动保留检查、文档链接检查及日志 SHA 统一记录在 validation。

## 仍需单独验收

LocalProcessLab 仍是受信本机程序执行；不提供 OS 级设备、文件系统或网络隔离。
任意程序内部训练步数、样本曝光及模型调用的计量属于部署执行器，不能从 argv 自动推断。
开放 Lab 的 durable 恢复和通用 JobRef poll/cancel/resume 桥接仍需专门接入。
本轮只验证执行接口、故障交付和 owned 清理，没有验证真实训练、部署、评测采纳或生成 memory 的语义质量。
