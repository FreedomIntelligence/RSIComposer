# Snapshot

统一口径见[系统术语与架构](../../../docs/architecture.md)。区分原目录运行恢复、可迁移快照与 Lab 训练 checkpoint；各自按显式状态适配器兑现。

独立于两个 loop 的状态捕获与恢复。内置默认组合使用明确的类型校验；替换组件通过显式状态适配器保存各自内部状态。可搬迁快照只接受空闲、可迁移边界；原目录持续研究重启使用 durable 路径。

## 接口与当前实现

[capture.py](capture.py)、[restore.py](restore.py) 提供协议与抛出 NotImplementedError 的占位类，[storage.py](storage.py) 仅定义清单存取协议。[local.py](local.py) 实现 LocalSnapshotCapture、FileSnapshotStore、load_manifest、restore_snapshot；[manifest.py](manifest.py) 定义清单，[codec.py](codec.py) 仅编解码明确允许的状态类型。

[components.py](components.py) 提供 `component_quiescent` 快照，复用同一校验清单、摘要、原子发布和目标锁所有权检查。
在 `ModuleAssembly.state_factory` 注册 [ComponentState](../loops/contracts/component_state.py) 后，
`LocalSnapshotCapture(app)` 自动选择 `AssemblyRecovery`；每个适配器必须额外实现
`validate_snapshot()`，确认无在途任务、外部副作用已确认、必要私有数据已导出或有稳定可解析引用。
恢复调用 `restore_snapshot(directory, new_target, modules=原装配)`；恢复不提交工作。
全组合自定义契约也可显式传入 `LocalSnapshotCapture(app, contract=...)` 与
`restore_snapshot(..., recovery_contract=...)`，但需实现相同的空闲验证。

协议中的 restore(manifest) 是抽象边界；当前本地入口使用 restore_snapshot(directory, target)，返回恢复后的 Application。操作命令见[使用指南](../../../docs/usage.md#使用指南)。

## 保存与恢复范围

在空闲边界保存两个默认 Harness 的状态、模型绑定、配置、共享 Experience SQLite、本地产物及父快照关系。Decision memory 与 Actor memory 分别保存为 decision_memory.json 和 exploration_memory.json，包含各自 history 与工作记忆，不混入 Experience。Decision agenda 的完整 revision archive 另存为 decision_research.json；BasicContextCompactor 的配置及 IdentityContextCompactor opt-out 也保留，恢复到同一 agenda 对象以保持工具绑定。待交付指令、待补写的操作/执行反馈、在途训练/评测、活跃 RL 或执行 session 都禁止捕获。

候选最近评测摘要、指令版本归因与当前执行视图随内置状态保存；待补写复查记录也禁止捕获。复查计时和消息交付的进程内状态不是 Snapshot 的恢复承诺。运行策略仅兼容默认 `ResearcherRunPolicy()` 与 `cleanup_timeout_seconds=30`；time / goal、自定义 RunPolicy、改变默认策略参数或清理预算都会被内置捕获明确拒绝，不会静默恢复成 researcher 策略。

恢复要求目标目录不存在，默认保留原 run_id、parent_run_id 和任务目标；检查文件清单、摘要、框架源码指纹及 Python 主/次版本。源码版本必须精确匹配；旧快照不自动迁移。Experience SQLite schema 2 显式拒绝旧 schema。已完成准备结果可继续使用，训练与评测终态反馈从 Experience 读取，准备历史与执行轨迹从各自 memory 读取，旧 JobHandle 不重建，不能据旧 JobRef 继续轮询或调用后端 result。

显式分叉为新运行使用 `restore_snapshot(snapshot, Path('runs/run6'), run_id='run6')`，
或 `rsisys resume --snapshot ... --data-dir runs/run6 --run-id run6`。
parent_run_id 自动取快照中的直接来源 run_id（例如 run5），不能手动覆盖为其他来源；
禁止用相同 ID 创建自指父子。新身份与来源写入目标 `config.json` / `run.json`。
已有 Experience 的 ID、run_id 和内容不重写，新增记录使用新 run_id，原目录不受影响。
不传 `run_id` 的同 run 恢复不产生新的父子边。
接入示例 monitor 时，它从目标 `run.json` 读取来源；目录名必须等于 run_id，
父运行同样需以其 ID 命名目录。来源字段不会扩大上述状态与外部权重恢复范围。

不捕获外部模型权重、优化器、在途作业、随机数生成器状态，也不保证逐位重放。保存 ModelRef 不代表已保存权重或验证远端版本可恢复；外部服务仍须提供绑定的模型版本。

## 自定义边界

配置了 model_activator 的组合也需显式快照集成；内置捕获不会丢弃部署适配器配置后以默认组合恢复。外部产物引用的保存不代表外部内容已被快照备份。

自定义集成需定义空闲边界、状态保存与恢复，以及组件间引用一致性。`state_factory` 返回具名适配器，框架核对完整 inventory、state_id 和可选 configuration 后才恢复；组件自己的内部依赖和格式仍由适配器负责。组件快照保存 Runner 计时、outcomes 和内置 RunPolicy，停机时间计入原预算；自定义 RunPolicy 需再次显式提供并由适配器保存其依赖状态。整体替换 Harness（包括无默认 memory 的实现）无需继承内置类，但不能仅靠存在接口就自动获得恢复支持。

没有状态适配器时，通过工厂参数装配的 Application、自定义 DirectiveDefinitions，以及新增或覆盖 command_handlers 仍由内置捕获器在写文件前拒绝；即使工厂返回内置组件，也不能据 JSON 配置推断重建方式。内置 Exploration 的 episode / learning / evaluation / local RL 执行组件继续检查类型和共享引用；组件使用不同 state 或脱离原绑定时拒绝捕获。恢复时统一重绑新状态，后续执行使用恢复后的计数、候选和版本信息。

并行 Runner、max_outstanding 不为 1，以及自定义 FeedbackSelector 暂不由本地快照重建，捕获时明确拒绝；默认 RecentFeedbackSelector 可正常恢复。

## 与运行恢复的区别

`LocalSnapshotCapture` / `restore_snapshot(directory, target)` 是可搬迁的空闲副本：
要求新的目标目录，拒绝在途任务和并行；默认内置路径仍拒绝非默认运行策略。
组件快照可通过显式适配器保存非默认运行策略，保留同一 run 身份；
通用组件状态不能自动改写内部来源，因此组件快照暂不提供 `run_id=` 分叉。
新增 [runtime/recovery.py](../runtime/recovery.py) 的
`Application(config, durable=True)` / `Application.resume(data_dir)` 用于原目录重启，
可保存仍在研究、等待内置 Mock 训练/评测、待交付反馈的控制边界。
两者不能互换；CLI 原有 `resume --snapshot ... --data-dir ...` 仍走空闲快照，
durable 原目录恢复使用 Python API。

Durable checkpoint + 单条 journal 意图保留运行预算、研究轮次、outcomes、
command receipts/outbox、Mock 作业标识及取消状态。已有 memory、Experience、artifacts
和 agenda 文件必须与原 workspace 一同保留；不要只复制 recovery 目录当作完整快照。
停机时间计入运行/清理/复查/重启等待；不重置 deadline。

未提交边界意味着效果不确定，重新打开可检查状态，但 Runner 拒绝继续执行，
直到显式 `app.recovery.reconcile(repair, evidence=...)` 确认并修复状态。
当前进程有机会保存时保留中断回执；硬崩溃时保留最后提交状态和意图，并保留磁盘上较新的证据。
既不把外部 JobRef 当作完整作业快照，也不盲目重试提交。
有 intervention 的 workspace 禁止生成可绕过恢复检查的空闲快照。

扩展 Harness 的 `recovery_id` / `export_recovery_state` / `restore_recovery_state`
或全组合 `RecoveryContract` 只用于明确声明的恢复路径；要支持可迁移快照还需显式 `validate_snapshot(app)`。
恢复要求显式传回同一工厂/契约；DSH 原生 session 恢复另行核实。
源码指纹、Python 主/次版本和原配置必须匹配；不承诺恢复任意本地/远端进程或权重。

默认上下文管理器的 trigger/target/summary token 配置、滚动摘要、已折叠记录指纹随 snapshot 保存。原始 history 与工具回执继续独立完整存储；恢复不会把已归纳的旧 payload 再次全部灌回模型。自定义 token counter/compactor 仍需显式恢复集成。
