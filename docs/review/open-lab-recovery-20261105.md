# Medical Open Lab 原目录恢复接口

`experiments.medical.open_lab_experiment` 默认使用 `Application(..., durable=True)`，
通过 `ModuleAssembly.state_factory` 注册 decision、exploration、Lab、feedback、
模型调用账本五个状态适配器。替换模块仍自行提供 `ComponentState`；没有通用对象图序列化。

## 操作

正常新建仍使用原启动参数。向**所属 controller** 发送 SIGINT 或 SIGTERM 后，
本轮 decision 完成再暂停；原绝对 deadline/总调用限额仍持续生效。暂停取消所属 Lab
程序并确认退出，收集真实终态回执，等待已提交 feedback 收敛，保留未提交的 queued
memory 工作，随后提交 `pause` checkpoint。`result.status=paused` 才表示完成暂停，
信号发送成功不代表已经可恢复。须给正在进行的模型请求和清理留出时间。

```bash
PYTHONPATH=src:. python -m experiments.medical.open_lab_experiment --root /absolute/run --resume
PYTHONPATH=src:. python -m experiments.medical.open_lab_experiment --root /absolute/run --inspect-recovery
```

`--resume` 仅接受原目录；参数、source、数据摘要、模型文件清单、设备环境均沿用原配置，
不接收新的预算。停机时间计入原 deadline；已到期则仅恢复核对与收尾，不做新推理。
终态运行拒绝续跑。`--inspect-recovery` 不装配模块、不调用模型，读取 checkpoint、
journal 及作业所有权/回执存在情况；这不是资源已释放证明。

`controller.lock` 和 Application workspace lock 防止双 controller。
`controller.claim` 留存 PID/UID/本次启动记录，其历史存在不等于活锁，不通过删除 claim 绕过锁。

## 保存范围

- DecisionState、原始决策 memory 的一致性摘要、research agenda、滚动摘要与 folded fingerprints。
- 固定模型绑定、预算上限、已尝试的全部模型调用计数，覆盖 decision/feedback/summary/recovery。
- LangMem 原 SQLite/namespace、feedback journal、生产者结果去重、已完成/排队状态与 reservation。
  构造时依照既有 LangMem committed exchange 恢复；未知调用不自动重试。
- 原 Lab 工作目录及修改过的代码/产物；相同 session、job ID、提交 key、argv 绑定。
- 原 supervisor 写下的终态进程回执、日志内容摘要、固定评测绑定/已验证结果与 artifact 引用。
  原提交 key 重读只返回原提交；恢复本身从不运行程序。

`RetainedLocalProcessBackend` 沿用当前 owned-process 监督和取消流程，
关闭 session 后保留目录。恢复只重建终态进程的读取视图，不把 PID 强行变成 Popen，
不向恢复出的 PID 发信号。PID/UID/创建信息只作为核对证据。

## 异常中断

checkpoint/journal 不一致、memory 比 checkpoint 超前、未知/新增进程身份、缺少原监督者的
终态回执、日志损坏都会阻塞执行。不能因 PID 已消失就伪造完成，也不能清 journal 重试。
当前 CLI 没有自动核销不确定副作用的入口；这类状态需使用框架的显式
`recovery.reconcile(repair, evidence=...)` 与对应模块的证据修复，或保留证据后另起已授权运行。
若装配因不完整的模块状态无法安全恢复，需先由该模块的 host 适配器修复；本实现不声称可
透明恢复任意硬崩溃窗口或任意外部进程。

这不是内存镜像、可搬迁 snapshot、模型权重备份或 optimizer step 恢复。
训练从某一步续跑由 Lab 程序自己保存并加载 checkpoint。原未启用 durable 的历史运行
没有可恢复状态，不能事后补成同 run 的续跑。

## 验证

CPU fixture 覆盖真实子进程与日志恢复、提交去重、固定评测重收集、不完整/损坏回执拒绝、
controller 互斥，以及真实启动器收到信号后的暂停/恢复。实际 LangMem SQLite 内容、
decision ledger、压缩状态、调用账本与 deadline 跨暂停保留；在途程序确认取消；
注入未提交 mutation 后不推理、不重放、不清 checkpoint；到期不产生新调用。
这些是状态恢复验证，不是新的医疗研究或训练成果。

主开发代码汇总到 `main`；历史实验分支对应冻结部署版本，合并后仍保留提交可追溯性。
