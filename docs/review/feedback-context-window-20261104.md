# Context / Memory 接入窗口

日期：2026-11-04。本文落实用户确认的文本 context 窗口；替代此前“三方法是所有插件要求”的描述。
接口位于 [feedback_context.py](../../src/rsisys/loops/contracts/feedback_context.py)，
实现适配位于 [context_memory.py](../../src/rsisys/loops/decision_making/feedback/context_memory.py)。

## 公开职责

必需能力只有：提供可供下一次 decision 使用的 context。

```python
class MyMemory:
    def context(self, query: ContextQuery) -> str:
        return my_store.retrieve(query.objective)
```

也可直接配置一个 `query -> str` 函数。空字符串合法，表示这次没有补充内容。
`ContextQuery` 提供当前 objective 和可选 latest_feedback，不包含内部 jobs、Attempt 结构、
模型调用预留、发布状态或 Memory 对象。组件可以忽略不需要的输入。

可选 `observe(event)` 在后续 tick 获得新反馈：

```python
class MyMemory:
    def observe(self, event: FeedbackContextEvent):
        my_store.enqueue(event.output, source_key=event.event_id)

    def context(self, query: ContextQuery) -> str:
        return my_store.retrieve(query.objective)
```

该通知只是更新机会。何时抽取、总结、合并、检索、存储以及使用什么 prompt，由组件决定。
模块不必实现 observe；可直接使用 query.latest_feedback 或通过适配器接入自己的 SDK。
模型整理继续使用部署已计费的依赖或现有 processor/gateway，配置 renderer 不增加模型调用额度。

## SDK 适配与装配

SDK 本身无须实现本项目的方法：

```python
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.loops.decision_making.feedback import CallbackContextMemory

memory = CallbackContextMemory(
    lambda query: sdk.retrieve_as_text(query.objective),
    on_feedback=lambda event: sdk.enqueue(event.output, key=event.event_id),
    configuration={'store_namespace': 'project/run'},
)
service = assemble_research(
    loop,
    assembly=ResearchAssembly(context_memory=memory),
    backend=host_accounted_backend,
    path=run_dir / 'feedback.json',
)
```

`configuration` 供装配身份与恢复比对；外部数据库与任意 Python 对象的恢复由组件/集成方负责。
CallbackContextMemory 的 on_feedback 可省略；context provider 的 configuration 方法也不是必需方法。
需要原生 messages 等扩展时由相应 harness adapter 处理；text-v1 不规定这些消息的内部格式。

## 输入事件

宿主将原始回执适配为 `FeedbackContextEvent`：

| 字段 | 来源 |
| --- | --- |
| event_id / task_id | 稳定宿主事件身份与原任务关联 |
| status | 执行器确认的终态 |
| output | 原生 Lab output；旧 operation/exploration 保留原始结果对象 |
| request / references | 原请求与原生产者的引用格式 |
| source / identity | 宿主已知的来源、Lab/run/decision/发起模型等关联 |

这些字段由适配器从现有捕获生成，Lab 无须增加输出 schema。原生 output 内同名字段不覆盖宿主状态。
组件收到独立副本，不能通过修改输入改变原始回执或执行账本。

## 实际执行顺序

1. 执行器交付真实结果，反馈服务持久捕获；此时不调用 renderer 或 observe。
2. 后续 `service.refresh()` 给有 observe 的组件发送新反馈，并推进已有后台处理。
3. Harness 调用 `service.context(ContextQuery(objective=...))` 获取当前文本。
4. 默认 Harness 用该文本替换 `DecisionContext.memory_context`，进入实际 decision/review 请求的用户数据部分。

最新真实结果仍在 last_feedback / operation_results；整理视图不替代它们。
更新当前 memory_context 不将上一份视图追加为新的 episodic Memory，不改写原始记录。
host 的实时状态与证据保护元数据继续在 research_state 等字段维护，插件文本不能覆盖这些字段。
默认 MemoryPrefixDecisionPolicy 将 memory_context 放在稳定前缀；完整请求计量包含该字段。
超出完整请求容量会明确失败，接口不承诺任意大小文本都能进入固定窗口，也不静默截断文本。

即时结果交付仍由执行器负责；`observe_output` 单独调用只是观察，不代替一个新 Lab 的结果路由。
默认只读查询仅归档，不触发 observe，也不替换 query.latest_feedback 指向的实际研究回执。

## 通知与恢复

- 新反馈通知在后续 tick 执行，hook 应快速返回，可将耗时工作交给组件自己的后台设施。
- 宿主先持久记录 running，再调用 hook；成功记 delivered，异常记 failed。
- 同事件重送、重复 refresh 不重复通知。进程中断留下 running 时，恢复标记 interrupted。
- 对失败或未知副作用不自动重放；错误在 job.context_delivery 及默认 context 的导航状态中可见。
  外部更新应按 event_id 去重，由组件明确处理自己的重试/恢复。此约束不要求普通纯 renderer 实现 recover。
- 服务关闭后不再发送更新通知；关闭不保证最后一份原始反馈已经完成整理。
- renderer 返回非 str 会在窗口处报错；renderer 异常保持显式失败，没有悄悄改用另一套 provider。

## 兼容边界

内置 JournalContextMemory、LangMem 与已有三方法组件，经 ContextMemoryBinding 适配。
它们的 record / working_memory / memory 保留为实现细节，旧 Memory 发布、引用/指标校验与恢复保持有效。
其结构化工作视图编码为文本后通过同一个 memory_context 出口消费，不再复制到 research_state。

新组件不需要这些方法。FeedbackAdapter 只要求 model/install；旧 evidence、decision_ids、publish
为按需集成，其中 publish 只用于实际产生旧 Memory 对象的兼容实现。
`service.working_memory()` 和 `/context_memory` 保留为兼容检查入口；新组件的后者只显示通知记录，
不将当前检索结果伪装为某个旧事件生成的历史记忆。当前文本以实际 decision 请求及 last_working_view 为准。

该窗口首次将 ResearchAssembly 配置升级为 schema_version=3，记录 context_protocol=text-v1 和 replace_current_view。
后续 [模块接口化](feedback-module-windows-20261104.md) 已升级为 schema_version=4，补充归档、提案与调度窗口。
与旧冻结配置不符时拒绝恢复，不静默迁移历史 run。

本窗口最初未重组归档、提案或调度扩展接口；后续实现见上述模块接口化说明。未实现训练数据构建或自动应用。
前次审查中的其他待确认项仍见 [边界审查](feedback-module-contract-review-20261104.md)。

## 验证

[16 项专门测试](../../tests/test_feedback_context_window.py) 覆盖最小组件/函数/SDK 接入、
实际请求、真实结果先交付、当前视图替换、原生输入与副本隔离、归档查询排除、通知失败与恢复、
无旧 publish 的外部 harness、关闭后不派发及完整请求容量计量。
已有 Journal、LangMem、主循环、工作状态与前缀策略回归同步检查。
测试均使用离线模型 fixture，不代表真实服务推理或 memory 总结质量已得到验证。

最终整仓回归 1,498 项：1,486 通过、12 跳过、0 失败/错误；296 个 source/example/test 文件
运行前后指纹一致。跳过的是 6 项外部 DSH SDK 与 6 项真实推理检查。
验证摘要见 [feedback-context-window-validation.json](repros/feedback-context-window-validation.json)，
完整本地日志在 `.local/context-window-20261104/full-python.log`。
未调用真实模型、训练或重启旧实验；改动当前保存在本地工作区。
