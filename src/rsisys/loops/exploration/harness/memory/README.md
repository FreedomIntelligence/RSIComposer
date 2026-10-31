# Actor memory

默认 Exploration Harness 的内部组件，保存任务轨迹与工作记忆。原始观察、行动、转换结果及模型调用信息留在本地 history，不进入共享 [Experience](../../../../experience/README.md)。

## 接口与实现

[MemoryStore](../../../_memory.py) 定义 append / history 与 remember / recall。HistoryEntry 保存本地历史；history 支持 task_id / kind / limit 筛选。Memory 无 scope，归属由持有它的 harness 决定。

[ExplorationMemory(path=None)](__init__.py) 使用通用 [MemoryBuffer](../../../_memory.py)，保存 trajectory 历史。history 返回最近匹配记录并按旧到新排列；recall 按新到旧返回工作记忆，可用 text 筛选 content / applicability。同 ID 同内容历史写入幂等，冲突内容被拒绝；remember 更新同 memory_id 的工作记忆，不自动归纳或验证来源。

省略 path 时仅存内存；指定 path 时加载已有 JSON，修改以原子写入保存。文件仅支持单写入实例，Application 使用 exploration_memory.json。

## 接入与边界

通过 DefaultExplorationHarness 的 memory 参数注入，默认使用 ExplorationMemory；memory_limit=32 分别限制 ActorContext 中当前任务 history 与自身 memories 的条数。Actor policy 不默认接收系统 experience。

RL 请求的 trajectory_refs 从本地 history 获取；Feedback.evidence_refs 引用系统记录，不能用作轨迹引用。训练终态反馈仍写入 Experience。

实现 MemoryStore 即可替换；自定义实现的持久化和[快照集成](../../../../snapshot/README.md)需另行提供，当前快照仅支持内置实现。

本模块用于定制默认 Harness；框架不要求替代 Harness 实现 MemoryStore，也不要求其包含独立 memory。
