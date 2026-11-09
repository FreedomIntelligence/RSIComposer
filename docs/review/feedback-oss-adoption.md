# Feedback Memory：LangMem 原生接入与验收

> 历史记录：原始探针、日志和请求已转入[归档说明](README.md#archived-evidence)；本文命令描述当时的工作，不作为当前可执行教程。

日期：2026-11-03。**已实现可选接入并验证真实 tool calling；尚未通过默认替换的语义验收。**

采用 LangMem 原生 MemoryStoreManager、Trustcall 抽取/JSONPatch/校验重试和 LangGraph SQLite
store，不实现另一套记忆归并算法。[自建 recipe / 更新协议建议](feedback-memory-design.md) 继续暂停。

## 实际调用链

```text
ExecutionReceipt -> FeedbackLearningService.observe
  -> 原始回执 journal + 唯一 job
  -> 可选 FeedbackProcessor（替代旧解释 prompt）
  -> LangMem 原生检索、抽取、修订、consolidation phase
  -> 事务提交到上游 SQLite store
  -> ContextMemoryOrganizer 的有界视图 -> meta-agent harness
```

反馈完成即触发，不等待上下文触顶。同一 active meta-agent 的 ModelRef 在捕获时固定；
上下文独立，后续反馈跟随当前版本。第三方 harness 通过 FeedbackAdapter 接入。
旧大 prompt 不会先为同一个 memory 任务再运行一次。轨迹归档保持默认开启且可关闭；
原生 memory 组件不生成 harness 修改或 researcher 权重训练材料。
后续组合入口允许显式接入独立 harness proposer；数据先做轨迹提取，见[链路装配](feedback-workflow.md)。

| 组件 | 本次实现 |
| --- | --- |
| 模型桥接 | [BackendChatModel](../../src/rsisys/adapters/langchain_model.py)：标准 LangChain 接口转 GenerationRequest；不另选云模型、不执行工具、不隐式重试 |
| 调用管理 | [FeedbackModelGateway](../../src/rsisys/loops/decision_making/feedback/model_gateway.py)：先保存完整 messages/tools，再经共享 host backend；生成的工具 schema 参与输入计量 |
| 原生整理与视图 | [LangMemContextMemory](../../src/rsisys/loops/decision_making/feedback/langmem.py)：同时充当 processor 和 context provider；记忆算法由上游实现 |
| 业务约束 | [附加 prompt](../../src/rsisys/loops/decision_making/feedback/prompts/langmem_research.txt) 及来源/候选/协议/指标 schema；保留上游 prompt |
| 调度与恢复 | [FeedbackLearning](../../src/rsisys/loops/decision_making/feedback/learning.py)：单 worker、保守预留调用、实际结算、未知请求不重放 |

## 配置与兼容范围

依赖为可选 extra，验证使用 `.local/langmem-validation` 隔离环境；没有在共享研究服务器安装包。
直接固定 LangMem 0.0.30、Trustcall 0.0.39、LangGraph 1.2.19、langchain-core 1.7.4、
langgraph-checkpoint-sqlite 3.1.3。[完整验证依赖](README.md#archived-evidence)。

```python
from rsisys.loops.decision_making.feedback import enable_langmem_feedback

enable_langmem_feedback(
    loop,
    backend=host_accounted_backend,
    path=run_dir / 'feedback_learning.json',
    memory_path=run_dir / 'feedback_memory.sqlite',
    namespace=('lab', lab_id, run_id),
    max_calls=64,  # host 管理，模型不管理预算
    max_input_tokens=16000,
    max_input_bytes=24000,
    memory_options={'max_calls_per_job': 6, 'query_limit': 8},
)
```

自定义 harness 可另传 `adapter=`。每个数据库/namespace 由一个宿主 journal 管理，目录和
配置随 journal 保留；恢复不能偷偷更换 processor。以上为显式启用示例，**默认工厂未切换**。

当前每阶段 max_steps=1，加上原生 consolidation phase；后者读取当前反馈并复用业务约束。
通常两次生成，校验修复额外计数。job 先保守预留最多 6 次，完成后释放未用额度；恢复未知
任务时保留预留，不重置 host 累计预算。共享 backend 的全局准入仍在最内层。

SQLite 此配置**按最近更新检索，不是语义向量检索**，没有暗中接 embedding 或 query model。
常驻视图检查总 UTF-8 字节，完整回执与派生记忆分开。视图有界不代表整个 harness 请求有界。

## 已执行验证

[完整证据与 SHA](README.md#archived-evidence)、[人工语义审查](README.md#archived-evidence)。
各系列采用不同配置，不是冻结版本的连续训练实验。共记录 **33 次真实模型生成调用**；
没有训练、研究 campaign 或服务配置修改。

| 验证 | 实际结果 |
| --- | --- |
| 上游原生默认流程，3 次生成 | 写入和纠错可执行；重复反馈曾使记忆从 1 条增至 2 条，默认去重不能视为可靠 |
| 上游 max_steps=2，3 次生成 | 带旧记忆后出现 dict.__repr_name__ 错误；本插件拒绝此配置，未重写其内部循环 |
| 上游原生 consolidation phase，6 次生成 | 三次输入均保持 1 条，纠错更新同一记录；仅为小范围行为验证 |
| 首次集成，4 次生成 | phase 对未改动临时记录回退为 Memory kind，与自定义 ResearchMemory 名称不兼容，更新被阻止；后续保留上游 Memory 名称，只增加证据字段 |
| 业务约束调整前，9 次生成 | 纠错改分数却仍引用旧评测，被校验拦截；历史整理还把 incumbent 写成 initial baseline |
| 最新配置，8 次生成 | 4 个反馈 job 完成；重复反馈不新增 ID，纠错引用新评测，未评测候选没有借用其他候选分数；仍有下述语义缺口 |

[12 项集成回归](../../tests/memory/test_langmem_feedback.py) 使用真实上游库和 SQLite、fixture 推理，
与真实 endpoint 调用分开。覆盖主动触发、原生 PatchDoc、内部修复计费、限额、身份、完整请求
准入、默认 harness 消费、active 版本跟随、未改动临时记录兼容、持久化、提交后崩溃恢复、
未知调用不重放、事务回滚、原文保留及视图上限。

原生接入验收时的隔离环境全量：**1326 项，12 项跳过，其余通过，83.270 秒**。
[完整日志](README.md#archived-evidence)。依赖检查通过，基础环境导入不加载 LangMem。
这些测试不等于真实自主研究，也不证明自由文字语义正确。

## 适配器的持久化边界

以下为已复现的兼容问题，不是另造记忆算法：

- LangMem 0.0.30 不向调用方传播 store 写入 futures 的异常。适配器暂存上游 PutOp，检查
  错误后，把记忆变更和 job 完成凭据一起提交，失败不泄漏部分更新。
- SQLite store 3.1.3 的 cursor 在异常时仍 COMMIT。同批先删除、后写入失败，会保留先前
  删除；小型事务适配器改为异常回滚。[实际复现](README.md#archived-evidence)。
- Trustcall 0.0.39 校验耗尽后可能丢弃无效调用，只返回通过部分。观察其 extraction 输出，
  整份 staged 更新记为失败，避免误报“成功但无变化”。
- schema 保留上游 Memory kind，避免原生 phase 中未改动的 dictionary 临时记录发生类型漂移。

成功提交含完成凭据。若 owner journal 在提交后崩溃，读回已提交结果并结算，不再生成；
没有完成凭据则标 interrupted，不根据猜测重复调用。推理请求的身份检查不因恢复而放宽。

## 尚未通过的语义验收

1. 最新流程收到另一份历史反馈后，实际 RemoveDoc 删除了先前两条无关 A/B 记忆。
   原始回执还在，但研究知识的保留/删除策略没有通过验收。
2. candidate_evidence.current_candidate_ids 只指当前回执涉及的候选，模型仍可能写成全局
   current candidate；它不等于当前部署的 incumbent。
3. evaluation_claims 的已列条目可以精确校验，正文中的历史分数、比较、否定句却仍可能漏列，
   也可能保留 strategy 提到的分数。结构化通过不能称为摘要语义验证。
4. 没有执行长时间自主研究，也没有用新 provider 完整重放原 62,815-token 活跃状态失败。
5. 真实服务核对了 alias/协议，revision label 不是不可变权重证明。版本跟随、崩溃恢复在
   fixture 中验证，不能冒充真实训练或部署实验。

因此交付为**可插拔、可审查、可回放的原生接入**，不是生产默认替换完成。
继续审查应先看实际 prompt 和失败输出，再定义来源范围、删除权限及是否需要其他上游实现；
不能靠一次小回放通过就宣称所有 memory bug 已解决。

## Prompt 审查入口

上游 prompt 仍由 LangMem 生成；业务增量为独立 txt，可通过 additional_instructions 替换。
实际请求保存 messages、tools、tool_choice，无需靠模板猜最终 prompt。

- [第一份反馈：抽取请求](README.md#archived-evidence)
- [第一份反馈：consolidation 请求](README.md#archived-evidence)
- [纠错：实际请求](README.md#archived-evidence)
- [历史 call48：实际请求](README.md#archived-evidence)
- [历史 call48：实际结果](README.md#archived-evidence)

运行时可用 feedback_learning_read 的 `/context_memory/model_exchanges` 分页审查。
原生路径 `/feedback_request` 是 workflow 输入/配置和实际交换入口描述，不冒充唯一的
GenerationRequest；`/feedback_response` 是处理结果。轨迹仍是公开片段，不是完整训练样本。

## 上游来源

- [LangMem 概念](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [MemoryStoreManager 源码](https://github.com/langchain-ai/langmem/blob/main/src/langmem/knowledge/extraction.py)
- [Trustcall 源码](https://github.com/hinthornw/trustcall/blob/main/trustcall/_base.py)
- [LangGraph SQLite store](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-sqlite)

运行证据以锁定安装版本为准，main 链接用于导航。DSH 仍是独立 harness/session compact
选项，不能与长期记忆整理等同。
