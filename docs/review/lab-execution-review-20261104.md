# M04 / M10：Lab 执行与结果交付审查

本次沿开放动作出口审查实际进程、异步结果、任务所有权和反馈交付。
已修复三个进程层缺陷；两个交付语义问题已复现，方案待用户 checkpoint。
这不是整个 Lab 或真实训练部署的验收。

代码基线为 `cb67628151cdb855b30f657e37cbf0af801644a8` 加现有未提交工作树。
审查前 100 个差异文件的 SHA 与补丁保存在 `.local/lab-execution-review-20261104/`，
此前 [开放动作出口报告](open-lab-execution-20261104.md) 及其验证数据保持为历史记录。

## 已修复的实际缺陷

实现：[LocalProcessBackend](../../src/rsisys/runtime/local_process.py)。
回归：[test_local_process_lifecycle.py](../../tests/runtime/test_local_process_lifecycle.py)。

| 级别 | 触发与原行为 | 修复后的行为 |
| --- | --- | --- |
| P1 | 进程监视异常时，watcher 的 done 已置位，state 为空且程序仍活着；close 跳过程序并删除所有权 | 监视异常生成失败证据，先杀 owned 进程组、再回收 leader；清理失败保留所有权，close 可重试，未确认退出不释放 workspace |
| P2 | 删除 workspace 前先移除 session；删除异常后 active_count=0，再次 close 无法清理 | 删除成功后才移除 session；失败保留根目录与归属；目录已被删除可正常完成清理 |
| P2 | pipe 使用 read(8192)，短日志即使已 flush 也可能一直等到退出才显示 | 使用 read1 读取当前可用字节，运行中 stdout/stderr 可见，保持原输出字节上限 |

watcher / reader 线程启动失败也进入失败与清理路径，避免 Popen 已成功但后续注册流程失败而遗留程序。
杀进程组仍在 leader 未被 wait 回收时执行，kill / wait 使用同一把锁；不向已回收 leader 的 PGID 发信号。
原有正常退出清理后台子进程、取消、超时、其他 session 隔离测试保持有效。

修复前后 CPU 复现：

| 观察 | 修复前 | 修复后 |
| --- | --- | --- |
| 运行中短 stdout | 空 | `READY`，程序仍为 running |
| workspace 删除失败后活跃 session 数 | 0 | 1，之后重试可删除 |
| watcher 异常后 close | 程序活着但 session 为 0 | 程序已回收，保留 failed 终态证据，session 为 0 |

原始修复前记录：`.local/lab-execution-review-20261104/runtime-before.json`。
修复后与交付问题的可重复脚本：[lab-execution-boundary-probe.py](repros/lab-execution-boundary-probe.py)。
该脚本只运行临时 CPU 程序和确定性 decision / memory fixture，不调用模型服务。

## 待 checkpoint：工具错误如何继续研究

当前 [LabExecution.execute](../../src/rsisys/loops/contracts/execution.py) 可以返回失败 OperationResult，
也可能抛出异常。[execute_operation](../../src/rsisys/loops/decision_making/harness/tools/__init__.py)
会把异常记成失败结果，但 [默认 harness](../../src/rsisys/loops/decision_making/harness/engine.py)
在记录后仍向 host 重抛。读取不存在的文件已实测触发 FileNotFoundError。
host 若显式捕获异常，下一步仍可执行；普通驱动循环若未捕获，则当前运行被中断。
程序自身非零退出已经以失败结果返回，不存在这层重抛。

建议语义：模型可以修正的输入或程序调用错误，以失败结果进入 context 与反馈，再由 decision 决定下一步；
身份冲突、权限 / host 准入拒绝、产物存储及基础设施故障仍交给 host。
实现上应由 provider 明确区分可恢复工具错误，避免框架把所有 ValueError / OSError 都当成可继续错误，
也不依靠匹配异常文字或硬编码某个工具名称。

这是错误分类与驱动策略的接口选择，本次没有改变已有异常契约或吞掉 host 错误。

## 待 checkpoint：重复收集与原始生产任务身份

[LocalProcessLab](../../src/rsisys/adapters/local_lab.py) 的 lab_poll 是只读查询，只归档；
lab_result 是非只读交付，每次调用都成为新的 decision 回执。
进程结果 artifact 会缓存，同 key 的 start 也不会重复启动；但不同 collection decision
具有不同 task_id，因此默认 feedback 将其视为不同观察。
当前 `_collected` 集合虽记录已收集 job，但未参与路由，不能防止重复整理。

实际复现：**1 个原生程序、2 次 lab_result，产生 2 个完成的 feedback job、2 次 memory processor 调用**。
两次内容指向同一个原生 job，却具有不同反馈 event_id。默认 journal 不调用模型；
若替换为调用模型的 memory processor，重复交付也会重复消耗处理调用。
尚未验证真实模型在此路径的调用数或记忆质量。

建议语义：每次工具调用保留独立原始轨迹；同一生产任务的同一终态事件默认只整理一次 memory，
需要重新整理时有显式入口。收集动作 ID 与生产任务 / 结果事件 ID 分开。
这需要可选的生产事件身份或等价 adapter 路由，不能让通用 feedback 硬编码 lab_result、data.job_id，
不能按输出文字去重，也不能把相同输出的两个真实实验合成一个。

同时，后续收集回执里的 issuing_model 是收集 decision 的模型；它不自动证明原始启动模型、
执行代码、训练权重、评测协议或采纳身份。确切归因应由执行器关联原始提交记录。

## 本次验证范围

新增 7 个真实 CPU 进程回归，覆盖实时日志、删除重试、目录已删除、监视失败、清理失败后重试、
watcher 启动失败和部分 reader 启动失败。
与 foundation / execution audit / open Lab 的 74 项定向测试通过。
全量 1585 项：1573 通过、12 跳过，零失败；运行耗时 87.864 秒。
结果和来源 SHA 见 [validation](repros/lab-execution-review-validation.json)。

仍未覆盖真实 SFT / RL、设备隔离、训练用量观测、权重与部署身份、指标有效性和医学泛化。
LocalProcessBackend 继续是受信本机执行，未提供 OS 隔离；通用程序的长期持久恢复仍明确不支持。
本次保持 Lab 的 describe / execute 两方法入口，不新增训练类别或限制模型自行实现算法。
