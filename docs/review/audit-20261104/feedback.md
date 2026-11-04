# M07–M09 反馈审查与修复

日期：2026-11-04，Asia/Shanghai。基于当前未提交工作树重新阅读源码，保留已有 feedback pipeline、LangMem、M03 prompt 及其他并行修改。本轮只使用 CPU、本地固定响应、临时 JSON/SQLite 和故障注入；没有真实模型请求、实验、训练、安装依赖或提交。

## 范围和文件

本轮生产代码修改：

- `src/rsisys/loops/decision_making/feedback/learning.py`
- `src/rsisys/loops/decision_making/feedback/attempts.py`
- `src/rsisys/loops/decision_making/feedback/pipeline.py`
- `src/rsisys/loops/decision_making/feedback/proposals.py`
- `src/rsisys/loops/decision_making/feedback/langmem.py`
- `src/rsisys/loops/decision_making/harness/learning_harness.py`
- `src/rsisys/adapters/langchain_model.py`

新增回归：`tests/test_audit_feedback_lifecycle.py`，25 项测试方法，包含额外参数化负例。新增本文档。`service.py`、`components.py`、`knowledge.py`、`evidence.py`、`model_gateway.py`、`research_state.py`、导出和两份组件 prompt 已审查，但本轮没有修改。父任务的 context、engine、monitor、领域收尾修复不在本组修改内。

## 已确认并修复的问题

下表中的测试均位于新回归文件。故障由当前接口的本地替身或真实失败响应 fixture 复现，不推断真实模型质量。

| 编号 | 复现与影响 | 修复 | 回归测试（`test_` 后缀） |
| --- | --- | --- | --- |
| F01 | native 已完成但宿主尚未收取结果时重启；`recover` 返回的提案未做发布验证，可把未知引用带入工作记忆；有效 Attempt 更新也未生成修订 | 正常收取和恢复共用 `_finalize_job`；先验证证据，再按其他 job 的最新修订应用 Attempt，生成实际输入 digest | `recovery_validates_native_outputs_before_any_read_or_working_view`；`recovery_applies_valid_attempt_updates_once_with_exact_provenance` |
| F02 | processor 返回 `None`、非终态、布尔计费或 `job_id/reserved_calls` 等宿主字段，可导致异常、改变身份或错误释放预约 | 共用 `checked_processor_result`：有限 JSON、终态、字段白名单、整数计费范围；无效结果失败并保留全预约，清除输出。pipeline 的单阶段也使用此验证 | `malformed_processor_results_fail_closed_with_reservation_retained`；既有 pipeline 产品隔离用例 |
| F03 | 恢复存储抛 `OSError` 直接中断 journal 构造，其他观察也无法读取 | 按 job 隔离恢复异常，标记 interrupted，保留预约，不重发未知调用 | `recovery_failure_isolated_without_replaying_uncertain_work` |
| F04 | legacy token counter 返回负数、布尔或非整数时可以通过预算判断 | serialized backend 校验非负整数；异常在学习准入层隔离，不发起推理 | `legacy_token_counter_rejects_negative_boolean_and_noninteger_before_inference` |
| F05 | legacy 返回语法完整的 JSON，但 `finish_reason=length`，仍被接受为完成解释 | 显式拒绝截断输出并保留原始响应 | `truncated_legacy_json_is_not_accepted_as_completed_learning` |
| F06 | processor 返回 `simulated=False` 可清除捕获来源；真实后续更新可清除旧 Attempt 的模拟祖先标志 | job 合并对捕获标志做 OR；Attempt 新修订继承旧修订的 simulation，同时保留旧成员/来源与不可变修订 | `processor_cannot_remove_captured_simulation_provenance`；`attempt_extension_keeps_simulated_ancestry_and_does_not_mutate_previous_revision` |
| F07 | 原 durable component receipt 只检查回执和 model；同身份但修改指令/组件配置或实际证据快照仍可复用旧结果 | proposer JSON 和 LangMem 原子提交收据增加 `configuration`、`input_digest`；恢复逐项匹配 | `proposer_recovery_rejects_changed_instructions_even_with_same_receipt_and_model`；`proposer_recovery_does_not_reuse_result_for_different_evidence_snapshot`；`langmem_recovery_rejects_changed_component_configuration`；`langmem_recovery_requires_matching_input_digest_and_returns_detached_result` |
| F08 | required memory 已提交，可选 proposal 的恢复读取抛错却使整个任务 interrupted | 可选阶段按未知结果处理，保留 required memory；可选产物 interrupted，保留剩余未知预约，不重放 | `optional_recovery_exception_preserves_committed_memory_and_unknown_call_reservation` |
| F09 | Pydantic 校验正确推导 `simulated=True`，staging 却写回原始 `simulated=False`；另有已有未知 exchange 导致 gateway 构造失败、被计为 0 次并退款 | staged write 保存校验后的规范化内容且不修改传入对象；已有不确定 exchange 时保留 native 全预约 | `staged_native_write_keeps_host_derived_simulation_flag_without_mutating_caller`；`native_existing_uncertain_exchanges_cannot_refund_the_reservation` |
| F10 | 标准 nested EvaluationSpec 的 `payload.candidate.candidate_id` 未用于寻找生产者；生产者落在最近 24 条之外时消失 | bounded grouping 同时读取 nested candidate，沿现有 candidate_origins 加入旧生产者，仍遵守原有窗口限制 | `default_adapter_includes_nested_evaluation_candidate_producer` |
| F11 | LangChain bridge 在没有声明工具时，仍将响应中的虚构 tool call 转为有效 AIMessage 调用 | 对未声明工具的调用抛 `ModelResponseError`，保留宿主调用记录，桥接层不执行工具 | `bridge_rejects_undeclared_tool_call_without_executing_or_dropping_it` |

F02 的结果校验不会证明插件报告的 `charged_calls` 真实；它限制类型、范围和 journal 写权限。内建 native 组件的实际调用通过 gateway 预约和持久记录。外部 processor 仍负有履行其宿主计量契约的责任。

## 源码覆盖、边界和测试证据

| 模块/文件 | 实际检查的边界与负例 | 本轮验证 |
| --- | --- | --- |
| M07 `learning.py` | capture 去重、不可变原始回执、队列/全局调用预算、完整请求准入、持久预约先于 submit、终态收取、恢复、close、分页 | 新回归 F01–F06；`test_feedback_integration`、`test_feedback_learning_integration`、`test_feedback_learning_evidence`、`test_feedback_required_outputs` |
| M07 `service.py` | 独立 adapter 的证据类型、先验证后调用 organizer、可选 Memory 发布、发布后 journal 持久失败 | 新增 `publication_journal_failure_retries_same_memory_without_second_model_call`；确认重复交付对象和 ID 一致、模型只调用一次；`test_feedback_components` 覆盖 organizer/delivery 异常 |
| M08 `evidence.py` | 来源只来自准入评测；候选/协议/状态/simulation/精确数值绑定；原文与 admitted catalog 重新比对；训练不能支持评测分数 | `test_feedback_candidate_evidence` 的真实 call48、错误绑定、篡改、旧 journal 恢复；新增 `candidate_binding_cannot_be_changed_by_an_organizer_snapshot` 验证可替换组件无法通过修改副本污染绑定 |
| M08 `attempts.py` | supplied member/source 白名单、base revision、所有更新原子冲突处理、不按标题合并、传承成员/来源/simulation、provenance | 新增恢复有效修订和模拟祖先回归；`test_attempt_learning` 的 stale/conflict/有界增量用例 |
| M08 `learning_harness.py` | 当前模型随 capture 解析、issuing/learning model 分离、只观察写操作且避免反馈读递归、bounded grouping、候选生产者、proposal 普通决策链接 | 新增 nested 生产者回归；`test_feedback_decoupling`、`test_feedback_pipeline` 的正式 decision 引用与未知 proposal 拒绝 |
| M08 `research_state.py` | per-collection/read-only archive 能力、原始事实卡和 digest、活跃依赖闭包、旧失败常驻窗口、精确已发布 Memory 去重 | 新增 `research_projection_requires_read_only_per_collection_archive_and_retains_source`；`test_research_working_state`、`test_research_evidence_projection`、`test_resident_evidence_window`，含无 archive、不可恢复 tool result、巨大活跃事实失败 |
| M09 `components.py`/`__init__.py` | 必须存在且可替换的 memory；trajectory 可禁用/替换；detached snapshot；公开导出不强制导入 optional 依赖 | 新增 organizer 隔离与发布持久失败用例；`test_feedback_components`、`test_feedback_decoupling` |
| M09 `knowledge.py` | bounded resident selection、事实与解释权限、proposal targets、稳定引用/digest 分页、failed/queue-full 的轨迹、原始 issuing model | 新增 `trajectory_preserves_failed_learning_receipt_and_identity_without_generated_targets`；`proposal_resolution_uses_ordinary_decision_links_without_applying_or_training`；`test_feedback_knowledge`、`test_feedback_projection` |
| M09 `model_gateway.py` | 捕获模型身份、完整 native messages/tools 的计数、写入 reserved 后才 dispatch、token/call 限制、不可重放、detached exchanges | 新增 `native_gateway_admits_complete_request_and_persists_before_dispatch`；`native_gateway_rejects_identity_token_overflow_and_journal_failure_without_dispatch` |
| M09 `pipeline.py`/`proposals.py`/proposal prompt | required 与 optional 顺序、预算与产品类型、部分失败、恢复身份、工具提案只生成不执行、默认 extraction | F02/F07/F08；`test_feedback_pipeline`，含未知 optional 结果保留预约、required 失败不调用 optional、合并输出限额/类型拒绝 |
| M09 `langmem.py`/memory prompt | 上游 LangMem/Trustcall 整理、源 catalog、schema、staging、SQLite 原子回滚、未观察写错误/提取丢弃检查、reservation/配置恢复、recency view | F07/F09；`test_langmem_feedback` 的真实上游本地运行配固定 backend，含校正调用、超预算、不完整提取、SQLite 批量错误和无推理恢复 |
| M09 `adapters/langchain_model.py` | native 消息和工具透传、精确 model identity、不支持选项拒绝、返回调用校验、不执行工具 | F11；新增 `bridge_rejects_identity_mismatch_and_unsupported_options`；`test_tool_calling_transport` |

## 发布契约、兼容性和语义限制

- 本轮没有增加前端公开 Attempt 字段或更改公开 schema version；已有 `simulated` 的值现在保留祖先来源，恢复后的有效更新会正常生成修订。此前引入的候选相关 `evaluation_claims` 契约继续保留：每条包含 `candidate_id`、`evaluation_ref`、`protocol_ref`、`metrics`、`simulated`；候选证据存在时，proposal 与 Attempt update 必须带这个数组，可为空。本轮没有再新增强制 claims 字段。
- 新增的 `configuration` 和 `input_digest` 属于组件内部 durable receipt。旧组件收据缺少这些字段时恢复拒绝，宿主任务保留预约并报告 interrupted，不自动迁移或重新推理。已经提交到 native store 的内容不会因恢复拒绝自动回滚；需要 operator 显式评审旧存储后处理。
- 真实 host120 call48 的冻结 raw receipt/响应已在既有 `test_feedback_candidate_evidence` 中作为 fixture 运行：原响应没有新 claims 契约，拒绝且不发布；恢复后的未发布旧结果同样阻断并清除本任务 proposal/Attempt 传播，原响应留存。正确结构化 claims 可以发布。没有修改历史 run，也没有声称自动撤回此前已写入外部存储的污染记忆。
- 绑定验证只保证声明的结构化指标与准入证据一致。模型可以提交空 claims 却在 prose 夹带错误分数、复述已过时 open question、遗漏负面结果或作出错误因果解释；既有 `test_empty_claims_do_not_certify_prose_semantics_or_completeness` 明确验证这个未解决边界。没有用自由文本正则猜测语义，也没有声称消灭语义幻觉。
- 默认兼容 combined legacy 路径仍存在；新 pipeline/LangMem 都是显式选择。memory 能力必须启用、实现可替换；没有自建记忆调和算法替代 LangMem。上游整理是否合理仍待语义审查，recency retrieval 也不是语义检索。
- trajectory 默认是已捕获公开片段的提取，包含 history、research context、执行回执、评测和反馈交换。它不包含完整原始 meta-agent 对话，也没有独立展开全部 raw grouping/dispatch catalog；可通过 source snapshot 和精确 evidence read 取回已存材料。`complete_conversation=false`、`training_ready=false`。这属于覆盖/未实现功能边界，未制造训练 target 补齐。
- harness proposal 可由后续普通 decision 引用；自动构建、测试、激活、回滚或系统改善没有由反馈层承诺。宿主仍拥有预算和执行权。
- publication 是至少一次交付：adapter 已接受 Memory、后续 journal 落盘失败时会再次交付同一个稳定 ID。adapter 应按 ID 幂等；本轮没有引入跨存储分布式事务。
- grouping 和 research view 保留原有有界选取；没有扩成全账本语义推断。没有对真实硬件、真实 tokenizer、模型语义质量或医疗分数作本轮验证声明。

## 定向验证结果

使用已有 optional 环境 `.local/langmem-validation/bin/python`，没有安装依赖。

```sh
PYTHONPATH=src:tests .local/langmem-validation/bin/python -m unittest \
  tests.test_audit_feedback_lifecycle \
  tests.test_feedback_candidate_evidence tests.test_feedback_components \
  tests.test_feedback_decoupling tests.test_feedback_integration \
  tests.test_feedback_knowledge tests.test_feedback_learning_evidence \
  tests.test_feedback_learning_integration tests.test_feedback_pipeline \
  tests.test_feedback_projection tests.test_feedback_required_outputs \
  tests.test_attempt_learning tests.test_langmem_feedback \
  tests.test_research_working_state tests.test_research_evidence_projection \
  tests.test_resident_evidence_window tests.test_tool_calling_transport
```

最终结果：**171 tests，OK，无失败、无跳过**；其中新增审计文件 **25 tests，OK**。非法内容的 Pydantic 诊断和上游 `Send` 弃用警告是预期测试输出，不代表外部模型调用。相关 tracked 文件 `git diff --check` 通过；新增文件另做空白检查。

这是受影响的定向测试组，不是全仓库最终验收。父任务负责全部修复合并后的 optional-env suite 和最终指纹；本文数字不替代其结果。
