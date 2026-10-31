# Experience

两个 loop 共用的记录、memory、检索与存储系统；大文件应交给 [Runtime 产物存储](../runtime/README.md)，本模块保留摘要、结构化 details 和关联引用，不会自动把大字段转成产物。

## 内容与接口

[records.py](records.py) 定义 actor_trajectory、decision_episode、training_run、evaluation 四类记录；scope 区分 decision_maker / actor / shared，按角色查询也包含 shared。独立准备操作尚无指令时 directive_id 可以为空。

[memory.py](memory.py) 定义 episodic / semantic / procedural 三类记忆，包含来源、适用条件和 unverified / supported / refuted 状态。内置存储要求 source_record_ids 非空且全部存在；同 memory_id 写入会更新内容。状态由调用方维护，查询不会按状态自动过滤。

[ExperienceSystem](retrieval.py) 提供 record / retrieve / remember / memories。BasicExperienceSystem 接收 ExperienceStore，可附加 run_id；loop 按角色检索并把结果放进 policy 上下文。

## 实现与替换

[storage.py](storage.py) 包含存储协议和内存实现，[sqlite.py](sqlite.py) 提供 SQLite 持久化。经验记录同 ID 同内容写入幂等，不同内容复用 ID 被拒绝；记录支持角色、kind、summary 文本和数量过滤，memory 支持角色、content / applicability 文本和数量过滤。

上下文 limit 仅限制条数，不限制字节数或 token。SQLite 在查询中限量；内存实现先复制全部经验再筛选，不保证查询内存有界。BasicExperienceSystem 的 run_id 约束写入，不隔离查询。

可替换整个 ExperienceSystem，或向 BasicExperienceSystem 注入另一种 store；两个 loop 可共享同一个实例。自定义持久存储的[快照集成](../snapshot/README.md)需另行实现，尚无通用适配机制。

[provenance.py](provenance.py) 仅定义来源关系类型；source_refs、evidence_refs 等保存关联标识，不自动解析或校验完整引用链。自动 memory 提炼、语义检索和完整来源图尚未实现。
