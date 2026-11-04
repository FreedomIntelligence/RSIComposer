# 装配、决策账本、上下文与监控审查

日期：2026-11-04，Asia/Shanghai。基于本地未提交工作树；不修改或恢复历史冻结实验。

本组覆盖 M01、M03、M05、M06、M14、M15。审查方式为源码边界检查、现有用例核对和可复现故障注入。
本组没有请求真实模型、启动训练、安装依赖或操作远端服务。单元测试中的 token meter 是确定性合成计量，不能称为真实 tokenizer 测量。

## 本次修复

### C01：旧摘要被算成不可压缩常驻事实（M06）

入口：`BasicContextCompactor._compact`。

第二次压缩时，最小视图和记录选择都携带上一次完整摘要。旧摘要本来要作为输入重新整理，却先使最小视图超限。
复现使用 22,000 触发、8,500 目标、1,800 摘要额度；第一次产生 1,700 字符摘要后加入新记录和固定状态，旧实现直接报 `ContextBudgetError`，没有调用新的 summarizer。

现在最小值和记录选择排除可替换的旧摘要；完整旧摘要仍交给 summarizer。新摘要、来源清单和所有保留字段一起计量，全部合格后才提交折叠状态。
显式空摘要不再被 `summary or context.context_summary` 意外替换回旧摘要。低压力时仍保留当前摘要。

回归：`RollingSummaryAuditTests.test_prior_summary_is_replaceable_not_an_irreducible_fact`，检查两次压缩、旧摘要完整传入、最新原文保护、原始历史不变和恢复视图一致。
现有 protected-oversize、摘要失败回滚、自适应上限及重复回执用例继续保留。

这不是对此前 62,815-token 失败的完整重放，也不保证所有保护状态都能塞入目标。真正超过上限的必要事实仍明确失败。

### C02：最终 monitor 写入失败阻止正式结果落盘（M14/M15）

入口：`examples.medical.experiment.run` 的 finally。

即使 learner 清理和 serving 释放已经执行，最后一次 `monitor.stage(...)` 抛错仍会跳过 `result.json`。
现在单独捕获该错误，记录 `cleanup_or_persistence_failed`，继续写正式结果，并通过 `execution_outcome` 保留收尾故障之前的执行结果。
如果正式结果文件本身也无法写入，异常仍会向外报告，不伪装落盘成功。

回归：`FinalResultAuditTests.test_terminal_monitor_failure_still_preserves_authoritative_result`。本地 CPU fixture 注入终态监控存储错误，核对 result 文件、失败原因、原始执行结果、训练清理确认和 serving 释放。
监控文件无法写入时，网页仍可能显示旧状态；正式结果可供 operator 判断，不能把此时网页状态当权威终态。

### C03：一个清理回调失败，后续资源不释放（M01/M07）

入口：`DefaultDecisionHarness.close`。

原实现顺序调用回调，遇到首个异常立即停止。现在按逆序尝试全部回调，成功项只执行一次，失败项保留供显式重试；多个错误通过异常 note 留存。
这允许超时的反馈 worker 继续收尾，同时避免重复释放已经关闭的资源。

回归：`HarnessCleanupAuditTests.test_failed_callback_does_not_skip_other_releases_and_can_be_retried`，覆盖中间回调超时、其他资源继续释放、重试和重复 close。

### C04：公开 Attempt 历史窗口遗漏当前修订（M15）

入口：`rsimonitor.server.attempt_interpretations`。

修订列表按 Attempt ID 和 revision 排序。直接取前 300 条，会在同一想法长期更新后遗漏其当前修订。
现在先保留可见 Attempt 的精确当前修订，再用其余历史填充 300 条窗口；原始列表保持不变。
当前 Attempt 列表仍是有上限的投影，不能称为完整历史或按更新时间排列的“最近研究”。

回归：`MonitorRevisionAuditTests.test_bounded_history_keeps_current_revisions_for_each_visible_attempt`，两个 Attempt、超过 300 次修订，核对两个当前修订都在历史窗口里。

### C05：decision learner 的非法指标被缓存为成功（M03/M11）

入口：`DefaultDecisionHarness.poll_learning/cancel_learning`。

与 Actor learner 的同类问题一致，外部 meta-agent learner 返回的 NaN、布尔、字符串或空指标名，可能直接进入成功 job 缓存和经验记录。
现在在 JobBook 终态缓存之前校验，非法指标成为稳定失败回执；重 poll/cancel 不再调用后端，不能进入应用链。
模型 base、结果类型及目标角色不符仍沿原来的严格身份校验失败路径，不借指标归一化绕过身份检查。

回归：`DecisionTrainingAuditTests.test_invalid_training_metrics_become_a_stable_failed_job_before_publication`，分别覆盖 poll 与 cancel、四类非法指标、simulation 标记、幂等记录和未应用状态。
这是已有 decision learner 接口的故障 fixture，不表示实现了轨迹转训练材料或进行了真实 meta-agent 训练。

## 模块边界核对

| 模块 | 已检查的主要边界 | 明天需要用户进一步调整的内容 |
| --- | --- | --- |
| M01 共享契约/装配 | `assembly`、`application`、`LabSpec/LabTask`、`ExecutionReceipt`；本地/exploration 区分，模型身份、task 身份、可替换 harness 和资源所有权 | Lab 契约已具备，但默认 Runner 尚未统一为多 Lab 调度；适配器索引仍是进程内状态，持久恢复须显式契约 |
| M03 decision policy | 九个可替换 prompt、schema/parser、完整上下文与工作视图分离、拒绝动作前校验、模型身份、摘要请求和 usage；医疗模式附加说明 | 逐句评审仍按 [M03](../m03.md) 与 [prompt 地图](../prompts.md)；本轮不重新定义研究策略 |
| M05 账本/归因 | decision dependency/source/adoption、候选创建来源、终态反馈幂等、候选 full base、评测归属、同协议比较、全量 parser 入口 | `candidate_evaluations` 是每候选最新评测索引；更早评测在历史回执中，不是多协议数据库；多协议产品语义可再设计 |
| M06 context/compact | 证据卡与原文、archive 身份/分页/JSON pointer、live 闭包、常驻候选窗口、两次压缩和状态恢复 | 活跃依赖真正巨大时仍不能任意删；完整账本/原始存储和遍历成本仍可增长；C01 不代表自由文字摘要语义已验证 |
| M14 领域装配 | 医疗 settings/run_policy、固定数据准备、initializer 与 parent 分离、收尾、Qwen toy 适配器/worker 的独立示例边界 | 医疗入口仍用兼容 feedback_learning；新组件 pipeline 需显式 opt-in；toy 模式的 prompt/预算不能冒称医疗 time 模式 |
| M15 监控 | 只读 API、受限文件读取、公开字段投影、指标数值、Attempt 解释与执行事实分离、当前模型 revision、前端轮询/隔离缓存/图布局/多语言用例 | 300 条公开窗口、4 MiB 快照读取上限；需要完整历史的产品能力应另定义，不从有界网页推断数据丢失 |

## 已知语义和功能边界

- context/memory 能力必须激活，实现可替换；不要求必须采用本仓库的压缩器。
- LangMem 的整理结果仍是模型解释。保留来源和结构化 claims，不等于解决无关旧记忆删除、局部 incumbent 当全局、自由文字夹带错误分数等问题。
- 公开轨迹是可追溯的执行/反馈片段，缺失范围明确；尚未形成可直接训练 meta-agent 的完整材料。
- harness 输出目前是提案，正式 decision 可引用并选择构建；构建成功不自动等于测试通过、已激活或已改善系统。
- 本地进程执行有工作目录和权限约束，不能把名称中的 Sandbox 理解为强 OS 安全隔离。
- 没有重放真实模型摘要、验证远端部署、跑真实 SFT/RL 或重新测医疗分数；这些均不能由本次测试通过推定。

## 验证入口

- 本组新增回归：`tests/test_audit_context_monitor.py`。
- 上下文定向：`test_context_compaction`、`test_adaptive_context_budget`、`test_repeated_receipt_compaction`。
- 收尾/装配定向：`test_feedback_learning_integration`、`test_assembly`。
- 监控定向：`test_monitor`、`test_monitor_lineage`、`test_monitor_projects`、`test_medical_monitor_identity`。
- 前端：`node --test tests/monitor_frontend/*.test.js`，140 项通过。
- 最终全量验证及文件指纹见本目录总报告；不把前几轮的数字冒充当前最终工作树结果。
