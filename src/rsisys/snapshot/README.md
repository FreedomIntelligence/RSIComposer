# Snapshot

独立于两个 loop 的状态捕获与恢复；通过 loop.harness 访问内置实现的状态。不同组件实现需要保存的内容可能不同；通用组件快照适配尚未落地，当前仅支持内置 Application。

## 接口与当前实现

[capture.py](capture.py)、[restore.py](restore.py) 提供协议与抛出 NotImplementedError 的占位类，[storage.py](storage.py) 仅定义清单存取协议。[local.py](local.py) 实现 LocalSnapshotCapture、FileSnapshotStore、load_manifest、restore_snapshot；[manifest.py](manifest.py) 定义清单，[codec.py](codec.py) 仅编解码明确允许的状态类型。

协议中的 restore(manifest) 是抽象边界；当前本地入口使用 restore_snapshot(directory, target)，返回恢复后的 Application。操作命令见[使用指南](../../../docs/GUIDE.md)。

## 保存与恢复范围

在空闲边界保存两个默认 Harness 的状态、模型绑定、配置、共享 Experience SQLite、本地产物及父快照关系。Decision memory 与 Actor memory 分别保存为 decision_memory.json 和 exploration_memory.json，包含各自 history 与工作记忆，不混入 Experience。待交付指令、待补写的操作/执行反馈、在途训练/评测、活跃 RL 或执行 session 都禁止捕获。

候选最近评测摘要、指令版本归因与当前执行视图随内置状态保存；待补写复查记录也禁止捕获。复查计时和消息交付的进程内状态不是 Snapshot 的恢复承诺。

恢复要求目标目录不存在，保留原 run_id 和任务目标；检查文件清单、摘要、框架源码指纹及 Python 主/次版本。源码版本必须精确匹配；旧快照不自动迁移。Experience SQLite schema 2 显式拒绝旧 schema。已完成准备结果可继续使用，训练与评测终态反馈从 Experience 读取，准备历史与执行轨迹从各自 memory 读取，旧 JobHandle 不重建，不能据旧 JobRef 继续轮询或调用后端 result。

不捕获外部模型权重、优化器、在途作业、随机数生成器状态，也不保证逐位重放。保存 ModelRef 不代表已保存权重或验证远端版本可恢复；外部服务仍须提供绑定的模型版本。

## 自定义边界

配置了 model_activator 的组合也需显式快照集成；内置捕获不会丢弃部署适配器配置后以默认组合恢复。外部产物引用的保存不代表外部内容已被快照备份。

自定义集成需定义空闲边界、状态保存与恢复，以及组件间引用一致性。目前没有通用适配器注册接口；本地捕获显式检查内置组件类型和共享实例关系，恢复则按配置重建内置 Application，只支持内置 Harness 及其内部组件，不能自动恢复自定义组件。整体替换 Harness（包括无 memory 的实现）或替换内部 MemoryStore 都不会自动获得快照支持，需另行实现保存、恢复与引用校验。

通过工厂参数装配的 Application、自定义 DirectiveDefinitions，以及新增或覆盖 command_handlers 都由内置捕获器在写文件前拒绝；即使工厂返回内置组件，也不能据 JSON 配置推断其重建方式。恢复支持需要单独集成，不会静默还原为默认实现。内置 Exploration 的 episode / learning / evaluation / local RL 执行组件也检查类型和共享引用；组件使用不同 state 或脱离原绑定时拒绝捕获。恢复时统一重绑新状态，后续执行使用恢复后的计数、候选和版本信息。

并行 Runner、max_outstanding 不为 1，以及自定义 FeedbackSelector 暂不由本地快照重建，捕获时明确拒绝；默认 RecentFeedbackSelector 可正常恢复。
