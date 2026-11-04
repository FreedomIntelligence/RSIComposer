# Feedback 接入减负自查

日期：2026-11-04。审查原则来自用户确认的 CP1 / CP2：
Lab 原样输出，执行设施补最小身份/状态；反馈端负责理解和整理。
本记录区分已修正内容、保留的运行保证和仍需用户 checkpoint 的实现取舍。

用户进一步确认“只规定基本职能接入窗口、内部实现自由”后的逐项复审见
[Feedback 模块边界审查](feedback-module-contract-review-20261104.md)，含具体代码证据、离线复现与调整建议；
该复审没有直接变更尚未 checkpoint 的生产接口。

后续用户已确认并授权落实 [Context / Memory 文本窗口](feedback-context-window-20261104.md)：
新组件只需 context 或函数；三方法与内部 job 留在兼容适配。其余可选模块仍按下表待审。

## 发现与处理

| 检查项 | 当前问题 / 原因 | 处理 |
| --- | --- | --- |
| Lab 输入 | 原统一入口要求 ExecutionReceipt 的 result 符合旧业务结构，新 Lab 需转换 summary/error、metrics 等 | 已新增原生 output 薄入口；旧类型适配继续有效 |
| 关联证据 | 默认 adapter 会收集默认 harness 的 history、评测及分组；新 Lab 接入不应依赖这些目录 | 新原生入口默认仅捕获自身输出，FeedbackEvidence 可选 |
| 未知字段和大日志 | 旧投影保护名称含 metric / *_id / *_refs 的字段，原生内容可能被意外钉为不可裁剪事实 | 原生 payload 按不透明内容做有界预览，保存精确原文和读回路径 |
| 文件和产物 | 接收原始内容时若要求统一 ArtifactRef、version/digest，会给所有 Lab 增加转换成本 | 原生 references 自由格式；特定消费者确需校验时再做适配 |
| Memory 插件 | 原三方法与内部 job 曾是固定要求 | 已按用户确认改为 context(query) -> str，可选新反馈通知；默认与旧组件经内部适配 |
| 归档插件 | 默认接口是对内部 job 的 records/snapshot，外部存储仍需理解 journal 结构 | 待 CP4：审查原始事件写入/读回接口，考虑让默认 journal 适配到更薄的独立存储接口 |
| 提案产品 | 固定字段、类别和长度限制属于当前提案消费者，不能要求 Lab 提前提供 | 待 CP6：审查消费者必需字段、可配置策略及外部 proposer 转换责任 |
| 阶段依赖 | memory 失败会跳过 proposer；两者实际读取同一原始快照，并未消费对方输出 | 待 CP7：独立失败/恢复或串行依赖应由装配策略明确 |
| 恢复与身份 | 同任务换结果、来源指纹不匹配、未知在途请求重放会造成重复副作用或混淆 | 保留执行设施的幂等、身份和恢复校验，不转嫁为 Lab 业务 schema |
| 数据构建/训练 | 明确来源、数据集版本、base 和目标角色属于这些消费者的请求契约 | 保留在已安装的改进 Lab 能力内，所有其他 Lab 无须实现 |

自查结论：减轻 Lab 内容接入已落实；memory、归档、提案、调度边界仍有值得简化的地方，
按 CP4–CP7 逐一讨论，不将当前实现约束视为已批准的最终架构。

## 最小接入方案

```python
job_id = service.observe_output(
    native_output,
    task_id=executor_task_id,
    status=executor_terminal_status,
    lab_id=executor_lab_id,
    references=native_references,
)
```

只有 task_id、status 是这次调用必需的宿主元数据；其余均按已知信息提供。
output 可以是原生文本、JSON object/list/scalar/null，也可以只交 references。
Lab 原生字段即使与宿主同名也不被解释成生命周期、metrics 或采纳。
失败 Lab 的错误可以留在原生输出里，无须额外生成 summary/error。
二进制等不可 JSON 持久化的内容保持文件引用；不自动读取或下载远端文件。

实现入口：

- [ExecutionReceipt.from_output](../../src/rsisys/loops/contracts/receipts.py)。
- [FeedbackLearningService.observe_output](../../src/rsisys/loops/decision_making/feedback/service.py)。
- [原生证据投影](../../src/rsisys/loops/decision_making/feedback/learning.py)：只裁剪模型视图，原始正文可按 JSON Pointer 读回。
- [10 项原生接入回归](../../tests/test_feedback_native_output.py)：任意业务键、文本/JSON/文件引用、失败输出、可选证据、大内容、重送、来源隔离与原类型校验。

原生入口仍观察终态输出；长任务进度使用执行器的 progress 通道，进度路由在 CP7 审查。
接收/归档成功不意味着已经把结果交付主 decision，外部执行器仍需履行自己的交付职责。
从原始日志中抽出的分数是解释；若用于自动比较/应用，相关消费者再验证必要事实与来源。

## 验证

52 项定向测试通过。整仓 1,482 项：1,470 通过、12 跳过、0 失败/错误；
293 个 source/example/test 文件运行前后指纹一致。
跳过的是 6 项外部 DSH SDK 和 6 项真实推理检查；未调用真实模型、训练或启动历史实验。
详细验证见 [本次摘要](repros/feedback-thin-validation.json)，本地日志在 `.local/feedback-thin-20261104/`。
