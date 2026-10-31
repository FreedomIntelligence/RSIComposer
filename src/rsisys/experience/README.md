# Experience

两个角色 harness 共享的系统反馈库，提供记录、检索与存储。原始 test-time history、research 准备过程和 Actor 轨迹保存在各自 memory；大文件交给 [Runtime 产物存储](../runtime/README.md)，本模块保留反馈摘要、结构化 details 和关联引用。

## 内容与接口

[records.py](records.py) 定义 runtime_feedback / training_run / evaluation / improvement / insight。scope 的 decision_maker / actor / shared 仅标记来源，不是访问权限或优化对象；Decision policy 可检索全部来源，优化目标由策略独立选择。尚无指令的反馈可省略 directive_id。

[ExperienceSystem](retrieval.py) 提供 record / retrieve；BasicExperienceSystem 接收 ExperienceStore，可附加 run_id。系统反馈可以引用本地 history 或产物，不复制原始轨迹；Feedback.evidence_refs 引用系统记录。Actor policy 默认读取自身 memory/history，不注入 Experience。

receive_feedback 将策略结果记为 improvement；insight 可由调用方记录归纳结论。这些记录接口不代表自动提炼、自动系统自修改或质量提升已经实现。

## 实现与替换

[storage.py](storage.py) 包含存储协议和内存实现，[sqlite.py](sqlite.py) 提供 SQLite 持久化。记录同 ID 同内容写入幂等，不同内容复用 ID 被拒绝；支持来源、kind、summary 文本和数量过滤。省略 scope 可检索全部来源，指定来源时也包含 shared。SQLite 使用 schema 2，显式拒绝旧 schema，不自动迁移。

limit 仅限制条数，不限制字节数或 token。SQLite 在查询中限量；内存实现先复制全部记录再筛选，不保证查询内存有界。BasicExperienceSystem 的 run_id 约束写入，不隔离查询。

可替换整个 ExperienceSystem，或向 BasicExperienceSystem 注入另一种 store。memory 属于各自 harness，见 [Decision memory](../loops/decision_making/memory/README.md) 和 [Actor memory](../loops/exploration/memory/README.md)。自定义持久存储的[快照集成](../snapshot/README.md)需另行实现，尚无通用适配机制。

[provenance.py](provenance.py) 仅定义来源关系类型；source_refs、evidence_refs 等保存关联标识，不自动解析或校验完整引用链。语义检索和完整来源图尚未实现。
