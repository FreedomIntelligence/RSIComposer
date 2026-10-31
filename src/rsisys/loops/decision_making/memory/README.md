# Decision memory

Decision harness 的可替换子模块，保存 research 准备历史、工作记忆与决策记录。原始 test-time history 留在本模块，共享 [Experience](../../../experience/README.md) 只保存系统反馈及历史引用。

## 接口与实现

[LoopMemory](../../contracts/memory.py) 定义 append / history 与 remember / recall。HistoryEntry 保存本地历史；history 支持 task_id / kind / limit 筛选。Memory 无 scope，归属由持有它的 harness 决定。

[DecisionMemory(path=None)](__init__.py) 使用通用 [MemoryBuffer](../../_memory.py)，支持 decision / operation / feedback 历史。history 返回最近匹配记录并按旧到新排列；recall 按新到旧返回工作记忆，可用 text 筛选 content / applicability。同 ID 同内容历史写入幂等，冲突内容被拒绝；remember 更新同 memory_id 的工作记忆，不自动归纳或验证来源。

省略 path 时仅存内存；指定 path 时加载已有 JSON，修改以原子写入保存。文件仅支持单写入实例，Application 使用 decision_memory.json。

## 接入与边界

通过 DecisionMakingLoop 的 memory 参数注入，默认使用 DecisionMemory；memory_limit=32 分别限制 DecisionContext 的 history 与 memories 条数。上下文同时包含全部来源的系统 experience，受独立的 experience_limit 控制。

实现 LoopMemory 即可替换；自定义实现的持久化和[快照集成](../../../snapshot/README.md)需另行提供，当前快照仅支持内置实现。
