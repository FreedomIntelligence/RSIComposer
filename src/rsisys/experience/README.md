# Experience

两个角色 harness 共享的系统反馈库，提供记录、检索与存储。默认 Harness 将原始 test-time history、research 准备过程和 Actor 轨迹保存在各自内部 memory；大文件交给 [Runtime 产物存储](../runtime/README.md)，本模块保留反馈摘要、结构化 details 和关联引用。

## 内容与接口

[records.py](records.py) 定义 runtime_feedback / training_run / evaluation / improvement / insight。scope 的 decision_maker / actor / shared 仅标记来源，不是访问权限或优化对象；Decision policy 可检索全部来源，优化目标由策略独立选择。尚无指令的反馈可省略 directive_id。

[ExperienceSystem](retrieval.py) 提供 record / retrieve；BasicExperienceSystem 接收 ExperienceStore，可附加 run_id。系统反馈可以引用本地 history 或产物，不复制原始轨迹；Feedback.evidence_refs 引用系统记录。Actor policy 默认读取自身 memory/history，不注入 Experience。

默认 Decision Harness 的 receive_feedback 将策略结果记为 improvement；insight 可由调用方记录归纳结论。这些记录接口不代表自动提炼、自动系统自修改或质量提升已经实现。

## 决策反馈投影

[FeedbackSelector](retrieval.py) 是从系统证据形成决策输入的替换点，接口为
`select(experience: ExperienceSystem, objective: str, *, limit: int) -> tuple[ExperienceRecord, ...]`。
调用方传入当前任务目标和正整数条数上限，结果可直接用于现有 `DecisionContext.experience`，不扩展 state schema。默认 `RecentFeedbackSelector` 忽略 objective，仅返回全部来源中最近 N 条原始记录，保持存储插入顺序、record_id、source_refs 和内容；不按来源或任务额外过滤。

自定义 selector 可按 objective 筛选或聚合，也可由外部训练的模型实现。返回值应不超过 limit；原始记录保留身份和来源，派生记录使用独立 record_id，并用 source_refs 引用输入证据的 record_id。派生记录仅是本次决策的临时输入，不自动写入 Experience，也不能直接当作库中已存在的训练证据 ID；显式持久化和训练输入选择由调用方负责。selector 不应调用 record 或修改原记录；这是实现契约，不是对任意插件的写权限隔离或来源自动校验。

此接口不提供自动摘要、提炼、训练循环或固定学习策略。当前最小输入是 objective；其他执行状态仍由 Harness 构建决策上下文，memory/history 仍属于 Harness，不搬入 Experience 或 selector。limit 限制返回条数，不保证 token、字节或自定义检索计算量有界。

## 实现与替换

[storage.py](storage.py) 包含存储协议和内存实现，[sqlite.py](sqlite.py) 提供 SQLite 持久化。记录同 ID 同内容写入幂等，不同内容复用 ID 被拒绝；支持来源、kind、summary 文本和数量过滤。省略 scope 可检索全部来源，指定来源时也包含 shared。SQLite 使用 schema 2，显式拒绝旧 schema，不自动迁移。

limit 仅限制条数，不限制字节数或 token。SQLite 在查询中限量；内存实现先复制全部记录再筛选，不保证查询内存有界。BasicExperienceSystem 的 run_id 约束写入，不隔离查询。

可替换整个 ExperienceSystem，或向 BasicExperienceSystem 注入另一种 store。内置 memory 是默认 Harness 的实现细节，见 [Decision memory](../loops/decision_making/harness/memory/README.md) 和 [Actor memory](../loops/exploration/harness/memory/README.md)。自定义持久存储的[快照集成](../snapshot/README.md)需另行实现，尚无通用适配机制。

[provenance.py](provenance.py) 仅定义来源关系类型；source_refs、evidence_refs 等保存关联标识，不自动解析或校验完整引用链。语义检索和完整来源图尚未实现。
