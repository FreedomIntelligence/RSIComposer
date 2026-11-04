# RSISys Prompt 与模型输入审查地图

整理日期：2026-11-03，基于当前本地工作树。配套：[模块审查地图](modules.md)。
这里为每个指令构造点编号，完整正文以链接的代码为准；动态拼接部分必须结合生效配置、输入和 wire 日志审查。
P 编号覆盖独立调用、指令片段、工具描述、数据投影及传输包装，不代表存在同样数量的模型调用。

2026-11-04：M03 的九个通用文本已迁至 [policy/prompts](../../src/rsisys/loops/decision_making/harness/policy/prompts)，
可通过 `DecisionPrompts` 分别覆盖。完整逻辑和八份渲染请求见 [M03 报告](m03.md)。
下方历史行号可能移动，以方法名与当前模板为准；医疗领域和 DSH 的附加包装仍分别审查。
Feedback 旧组合解释已迁入 [LegacyFeedbackInterpreter](../../src/rsisys/loops/decision_making/feedback/legacy.py)
和 [legacy_feedback.txt](../../src/rsisys/loops/decision_making/feedback/prompts/legacy_feedback.txt)。
这只是兼容组件；公开产物协议不要求调用该 prompt，其他实现可使用自己的 prompt 或无需模型。
接口与故障策略见 [模块窗口](feedback-module-windows-20261104.md)。

## 审查一个 prompt 的完整单位

```text
角色 / 实际 ModelRef / 调用目的
+ 生效配置和继承覆盖链
+ system 指令及追加指令
+ user 中的 objective / project_context / 结构化状态 / 原始与派生证据
+ operations / directives 的 descriptions 和 input_schema
+ canonical response_schema / 实际 wire schema
+ chat template / generation options / 输出限制
+ parser / 业务校验 / 授权 / 执行回执
```

不能只审某个 `system=` 字符串：objective 中的追加说明、工具描述、错误恢复提示、DSH persona
以及训练 chat template 都可能改变模型行为。模板审查与某次真实请求审查分别记录。

## 总目录

| ID | 审查对象 | 属于模块 | 何时生效 |
| --- | --- | --- | --- |
| P01 | Meta-agent 身份、研究范围与证据原则 | M03 | 基础 decision / review 共享片段 |
| P02 | 主 decision 指令与结构化上下文 | M03/M06 | 默认 policy 的决策调用；可被领域覆盖 |
| P03 | Attempt / lineage 的模型侧指引 | M03/M05/M08 | decision；review invoke 也引用 |
| P04 | 运行中实验 review | M02/M03 | 默认模型 review 路径 |
| P05 | Compact 滚动摘要 | M06 | 默认 compactor 实际触发模型摘要时 |
| P06 | 基础 JSON/schema 校验纠正 | M03 | max_validation_attempts > 1 时 |
| P07 | 医疗模式控制：无 wait/finish、host 停止 | M02/M03/M14 | 医疗 ResearchPolicy，按 run_mode 分支 |
| P08 | 医疗 envelope 指引与三次动作校验纠正 | M03/M14 | 医疗每次 decision；失败时加纠正 |
| P09 | 输出截断/格式恢复 | M03/M11 | 医疗 DecisionBackend 的有限恢复 |
| P10 | 医疗 objective 与 project_context | M03/M14 | 医疗主实验装配 |
| P11 | Feedback 分身的整理与路由主指令 | M07/M08/M09 | 选择模型整理器时；observe 本身不调用模型 |
| P12 | Feedback 的 evaluation_claims 指引 | M08/M12 | 输入带 candidate_evidence 时追加 |
| P13 | Feedback / Attempt / claims 输出 schema | M07/M08/M09 | 依 backend schema 能力和输入字段 |
| P14 | 工作知识、证据卡、摘要导航中的说明 | M06/M09 | 不新增模型调用，进入后续 decision 数据 |
| P15 | 工具/指令描述与 schema | M01/M04/M06 | 每轮按授权与当前能力动态选择 |
| P16 | DSH 原生 persona 与 compact 配置 | M03/M06 | 选择 DSH harness 且加载相应 patch 时 |
| P17 | DSH decision/review 增量消息 | M03/M06 | DSH 原生会话每轮包装 |
| P18 | DSH feedback 独立会话包装 | M07 | 选择 DSH feedback backend 时 |
| P19 | Actor 动作选择及可配置 instructions | M10 | execute / RL rollout 等 Actor 路径 |
| P20 | 医疗候选评测问答 | M12/M14 | evaluator 对每个开发样本调用推理 |
| P21 | 医疗 SFT 数据的 system 与 chat template | M11/M14 | 转换训练样本；属于训练输入 |
| P22 | 医疗简化入口的 objective 追加说明 | M14 | MedicalAssembly.run，与主实验入口分开 |
| P23 | Qwen toy 的 researcher 说明与预算提示 | M03/M14 | Qwen toy 入口，行为不同于医疗 time 模式 |
| P24 | Qwen toy 的 Actor、部署探针与 NLL 输入 | M10/M11/M12/M14 | toy Actor、激活探针、训练/评测样本 |
| P25 | 最终模型消息与 provider schema/template | M11 | transport/backend/worker 的真实发送与渲染 |
| P26 | 可选改进任务提案 | M09/M10 | 安装 ModelImprovementProposer 时，memory 阶段之后 |

## P01：Meta-agent 身份与研究范围

代码：[ModelDecisionPolicy._research_instructions](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L66)。

- [ ] 逐句核对 meta-agent、decision、lab、两个 harness 的定义与 M01 一致。
- [ ] 对 target policy / decision policy / research system 的优化范围是否与真实安装能力一致。
- [ ] “自主探索、无需固定轮换或最少实验数”与证据/权限/采纳限制是否相容。
- [ ] memory、summary、proposal 与原始 receipts 的证据等级是否明确；project_context 是否被错误当作能力授权。

关联：`test_medical_project_context.py`、`test_researcher_integration.py`。完整共享正文在该静态方法中。

## P02：主 decision 请求

代码：[ModelDecisionPolicy._decision_request / _context_inputs / decision_schema](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L247)、
[DecisionContext](../../src/rsisys/loops/decision_making/harness/state.py)、
[MemoryPrefixDecisionPolicy._ordered_request](../../src/rsisys/loops/decision_making/harness/learning_policy.py)。

user 输入为结构化 JSON；主要字段包括 objective、project_context、operations、directives、operation_results、
history、experience、memories、last_feedback、candidates、candidate_evaluations、decision_graph、
candidate_origins、directive_provenance、outstanding/executions、run_control、context_summary、research_agenda、research_state、memory_context。
memory_context 为当前 provider 的文本视图；research_state 留给宿主证据/依赖元数据。
详见 [Context / Memory 接入窗口](feedback-context-window-20261104.md)。

- [ ] invoke/explore 命名空间、arguments/input_ids、reason/strategy 是否明确且不会互相误导。
- [ ] 动态裁剪后的输入是否仍能描述当前研究状态；省略字段/历史条目是否有明确导航与范围说明。
- [ ] system 继承片段、工具 schema、输出 schema、输入排序与最终 tokenizer 计数一起核对。
- [ ] parser 仍使用完整输入账本；测试读回旧候选并真正构造合法动作。

关联：`test_structured_decisions.py`、`test_learning_policy.py`、`test_ledger_context.py`。

## P03：Attempt 与 lineage 指引

代码：[ModelDecisionPolicy._attempt_instructions / _envelope / _attempt_branches](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L376)、
[AttemptContract](../../src/rsisys/loops/decision_making/harness/attempts.py)。

- [ ] optional 与 required 两个分支逐句审；实际 run 使用哪一个必须从配置确认。
- [ ] 明确执行时 membership 与反馈事后 interpretation ID 的差别。
- [ ] source / depends_on / candidate_origins 的权限和状态要求与 parser 相同。
- [ ] Attempt title/ID 不代替具体计划、参数与评测方案，也不证明研究成功或结束。

关联：`test_required_research_attempts.py`、`test_research_attempts.py`、`test_decision_lineage.py`。

## P04：实验 review

代码：[ModelDecisionPolicy._review_request / review_schema / review](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L498)、
[DefaultDecisionHarness._review_one](../../src/rsisys/loops/decision_making/harness/engine.py)。

- [ ] 输入 execution/progress 的身份、时点与阶段；允许的操作和 review_actions 来自当前授权。
- [ ] invoke 查询与最终 continue/cancel 的 envelope 是否互斥；不能借 review 发新 directive。
- [ ] next_review_seconds/null、工具次数、记录与在途终态先到的行为。
- [ ] 医疗 ResearchPolicy.review 当前是 host 的静态 continue，不发模型 review 请求；不能把通用 P04 误记为医疗实际调用。

关联：`test_researcher_review.py`、`test_research_queries.py`、`test_action_permissions.py`。

## P05：Compact 摘要

代码：[ModelDecisionPolicy._summarize_context / _summary_current_state](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L320)、
[BasicContextCompactor](../../src/rsisys/loops/decision_making/harness/context.py)。

本次模型输入是 `previous_summary + archived_records + current_state`，输出必须是 `{"summary":"..."}`。

- [ ] archived_records 只是本次归档子集；未出现评测/采纳不能被写成从未发生。
- [ ] 真实工具结果、失败、科学负结果、假设、来源 ID 与限制是否保留且分清。
- [ ] current_state 对旧摘要的优先级；工作选择不完整时的断言范围。
- [ ] 不让摘要决定下一动作或停止；模型身份、输出限制、完整请求重计量和失败回滚。

关联：`test_context_compaction.py`、`test_adaptive_context_budget.py`；还需选真实摘要按候选/协议/采纳回执核语义。

## P06：基础校验纠正

代码：[ModelDecisionPolicy.decide](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L225)。

- [ ] 追加到 objective 的错误文本按 data 引用；固定纠正指令与原目标不冲突。
- [ ] max_validation_attempts 的生效值、截断长度和每次 usage；传输/身份失败不进入此重试。
- [ ] 被拒绝的动作没有执行；这层重试和 P08/P09 不混算。

关联：`test_decision_validation_retry.py`。医疗 ResearchPolicy 的循环另看 P08。

## P07：医疗模式控制

代码：[ResearchPolicy._model_context / _control_instructions / _scheduling_instructions / decision_schema](../../examples/medical/experiment.py#L907)。

- [ ] time 模式让 host 负责停止，模型只选 invoke/explore；system、schema、parser 三者都一致。
- [ ] 传入的 run_control 是否避免要求模型管理倒计时、训练余量和研究轮次；initial_training_data 的角色明确。
- [ ] researcher 模式允许 finish 的分支单独验收；等待是串行 executor 的调度职责。
- [ ] 自主探索不被解释成自动授予能力、取消执行限额或强制产生正结果。

关联：`test_medical_experiment.py`、`test_qwen_run_policy.py`。

## P08：医疗 envelope 与动作纠正

代码：[ResearchPolicy.ENVELOPE_GUIDANCE](../../examples/medical/experiment.py#L882)、
[decide 的校验与纠正](../../examples/medical/experiment.py#L1017)、
[_parse_decision](../../examples/medical/experiment.py#L955)。

- [ ] unique keys、train input_ids、origin 不由模型填写、固定 initializer 与父系统绑定等说明逐句核对。
- [ ] 非规范 action 名称何时允许归一化；不能改变模型具体参数或绕过校验。
- [ ] 每次失败的 previous_output / validation_error / available_candidate_ids 以何种 data 形式追加、可见范围多大。
- [ ] 最多三次动作校验；失败、权限拒绝与真实副作用的边界；与 P09 格式恢复的调用总数。

关联：`test_decision_validation_retry.py`、`test_medical_experiment.py`、`test_training_origin_flow.py`。

## P09：输出截断与格式恢复

代码：[DecisionBackend.generate / _generate_serial](../../examples/medical/experiment.py#L402)。

- [ ] length/格式错误分别追加哪条固定 correction；保持原始 request、schema 和模型身份。
- [ ] 部分输出不得升级为 system 指令；每个响应保留并计费。
- [ ] 最多额外几次恢复，何种错误直接抛出；达到 deadline 的行为。
- [ ] 用实际 model_requests / model_responses / model_errors / wire 记录核对，不能只数最终通过的 decision。

关联：`test_medical_response_recovery.py`、`test_structured_transport.py`。

## P10：医疗 objective 与 project_context

代码：[build_project_context](../../examples/medical/experiment.py#L218)、[experiment.run 的 objective 构造段](../../examples/medical/experiment.py#L1322)。

- [ ] system 外的这两段完整正文也逐句审：固定目标、标签、baseline、候选生成/评测/采纳、训练/数据与资源说明。
- [ ] `proposal_only_unless_installed`、固定 initializer、Pareto 与 current incumbent 的真实执行一致。
- [ ] host 显式资源声明与容量未知，不从设备可见性或外部服务推导训练权限。
- [ ] train/dev/protected、固定 protocol、默认 recipe、anchor 数据是否来源正确；开发选择不被表述为独立泛化证据。

关联：`test_medical_project_context.py`、`test_medical_preparation.py`、`test_medical_experiment.py`。

## P11：Feedback 整理与路由主指令

逐句审查材料：[当前实现与完整 prompt](feedback-implementation.md)，含由当前代码导出的原文、
条件追加指令、完整合成输入/schema/输出，以及尚未随整理器插件化的解释策略边界。

兼容解释代码：[LegacyFeedbackInterpreter.request](../../src/rsisys/loops/decision_making/feedback/legacy.py)、
[主模板](../../src/rsisys/loops/decision_making/feedback/prompts/legacy_feedback.txt)、
[FeedbackLearning._request](../../src/rsisys/loops/decision_making/feedback/learning.py)、
[FeedbackLearning.capture / _bound_input / _evidence_packet](../../src/rsisys/loops/decision_making/feedback/learning.py)、
[FeedbackAdapter](../../src/rsisys/loops/decision_making/feedback/service.py)。

输入包括 receipt、相关 history/evaluations、source_refs、research_context、grouping_context、candidate_evidence
与 evidence_packaging；以本次实际捕获和准入后的字段为准。该调用共享 meta-agent 当前模型
版本、使用独立反馈上下文；实际 system/prompt/schema 在提交前归档为 `/feedback_request`。
这些是旧组合解释器的输入，不是新处理器的基础协议。新处理器只接公开事件和受控模型能力，
自行组织 prompt，实际交换由 host gateway 归档。

- [ ] 整理者只能解释与路由，不替研究 meta-agent 选择下一实验、停止或直接应用改进。
- [ ] context/memory 整理能力必须激活，实现可以来自外部；轨迹归档是默认开启的可选组件。真实结果直接返回 decision，不等待整理；查询归档不触发本 prompt。
- [ ] `training_data` 仅保留旧入口的兼容草稿；新 pipeline 不生成训练样本。筛选、标注、格式转换、训练由正式 Lab 任务及 recipe 负责。
- [ ] coverage、摘录、未提供正文、旧 Attempt 解释与原始事实的区别是否清楚。
- [ ] 反馈跟随当前 meta-agent 版本，历史 decision 发出版本分别归因；排队快照和已完成请求不追改版本。
- [ ] 有限输入是否仍足以整理研究轨迹；捕获的本地操作是否覆盖代码修复等任务。

关联：`test_feedback_learning_evidence.py`、`test_feedback_decoupling.py`、`test_feedback_learning_integration.py`、`test_feedback_required_outputs.py`、`test_feedback_lab_boundaries.py`。
当前四分支与旧组合入口的差异见 [工作流](feedback-workflow.md)；旧完整 prompt 导出只代表当时配置。

## P12：候选指标 claim 指令

代码：[_CANDIDATE_INSTRUCTIONS](../../src/rsisys/loops/decision_making/feedback/learning.py#L29)、
[candidate_evidence / validate_claims](../../src/rsisys/loops/decision_making/feedback/evidence.py)。

- [ ] 每项 factual metric claim 的 candidate_id / evaluation_ref / protocol_ref / metrics / simulated 与来源一致。
- [ ] not_observed_in_snapshot 与 not_supplied 的区别，未评测不能借旧候选、父系统或邻接事件的分数。
- [ ] content / Attempt summary 的自然语言是否遗漏 claim 或与 claim 矛盾；结构校验不能证明这一点。

关联：`test_feedback_candidate_evidence.py`。审查时将 prompt、原始评测、结构化 claims、自由文字四者并排。

## P13：Feedback、Attempt 和 claims 输出 schema

代码：[_schema / validate_proposals / validate_publication](../../src/rsisys/loops/decision_making/feedback/learning.py#L262)、
[attempt_schema / validate_attempt_updates](../../src/rsisys/loops/decision_making/feedback/attempts.py)、
[claim_schema](../../src/rsisys/loops/decision_making/feedback/evidence.py)。

- [ ] proposals 数量、kind、subject、source_refs、validation_plan、training_example 和 unverified 标记。
- [ ] attempt_updates 的 ID/base_revision/member refs，字段约束与真实接受/发布校验一致。
- [ ] canonical schema、portable wire schema 和 legacy 无 subject 兼容的差异明确记录。
- [ ] 通过校验仍只得到 proposal；不出现 applied 或 training_ready 的无来源升级。

关联：`test_feedback_knowledge.py`、`test_attempt_learning.py`、`test_structured_transport.py`。

## P14：派生知识、证据卡与导航文本

代码：[knowledge.working_memory / proposal_records](../../src/rsisys/loops/decision_making/feedback/knowledge.py#L82)、
[research_state.py](../../src/rsisys/loops/decision_making/harness/research_state.py)、
[ledger_context.py](../../src/rsisys/loops/decision_making/harness/ledger_context.py)、
[ResearchAgenda.context](../../src/rsisys/loops/decision_making/harness/research.py)。

- [ ] authority、unverified、excerpt、omitted、source count 和 read_path 等描述能否防止把视图当完整事实。
- [ ] 最新 Attempt、未归组观察、活跃引用与负结果的选择，以及 subject=lab memory 的处理。
- [ ] 重复副本去重、字节上限和精确读回是否保留必要来源，失效 digest 是否变成可恢复回执。
- [ ] 原始归档不变；以上文本是模型数据中的导航说明，没有单独新增“memory 整理调用”。

关联：`test_feedback_knowledge.py`、`test_research_working_state.py`、`test_latest_evidence_view.py`、`test_ledger_context.py`。

## P15：工具与 directive 的完整模型侧接口

代码：[directive 定义](../../src/rsisys/loops/decision_making/action_space/directives.py#L110)、
[ResearchTools](../../src/rsisys/loops/decision_making/harness/tools/research.py)、
[WorkspaceTools](../../src/rsisys/loops/decision_making/harness/tools/workspace.py)、
[QueryTools](../../src/rsisys/loops/decision_making/harness/tools/queries.py)、
[feedback_learning_read spec](../../src/rsisys/loops/decision_making/harness/learning_harness.py)、
[ExperimentOperations](../../examples/medical/experiment.py)。

- [ ] 对本次实际 `context.operations`、`directives` 每个工具逐个审 name/description/input_schema/read_only。
- [ ] read/archive/patch、反馈 jobs/attempts/proposals 分页、旧记录 ID/JSON Pointer 读回的语义一致。
- [ ] 数据预览、数据选择、recipe、train/evaluate/apply_candidate 的输入与真实 executor 限制一致。
- [ ] 参数默认值、optional/required、字段顺序、不可用能力的隐藏，与 parser/validator/permissions/binding 对照。

关联：`test_capability_composition.py`、`test_action_permissions.py`、`test_schema_contracts.py`、`test_workspace_tools.py`、`test_evidence_path_recovery.py`。

## P16：DSH persona 与原生 compact 配置

代码：[dsh.patch.yml](../../src/rsisys/loops/decision_making/harness/dsh.patch.yml)、
[DSHConfig / _runtime_patch / _sdk_runtime](../../src/rsisys/loops/decision_making/harness/dsh.py)。

- [ ] 实际加载的 patch、personaPrefix、外部 SDK 默认系统指令与版本，不只看仓库模板。
- [ ] 原生 shell/file tools 的禁用与 RSISys 授权边界；整体运行时长由外部 RunPolicy 控制。
- [ ] thresholdRatio/headroomTokens/retainRatio/maxTokens 与默认 BasicContextCompactor 的不同归属。

关联：`test_dsh_runtime.py`、`test_dsh_harness.py`、`test_medical_dsh.py`。
仓库只持有接入层和 patch；外部 SDK 完整 prompt/compactor 正文需在对应安装版本另取证。

## P17：DSH 增量 decision/review 消息

代码：[_SessionBackend._increment / generate](../../src/rsisys/loops/decision_making/harness/dsh.py#L146)。

- [ ] request.system 被包装进本轮 user observation 后，与原生 persona 的实际指令优先级。
- [ ] append/replace/absent=unchanged 的说明与 transport delta 实现一致；最新能力每轮重新声明。
- [ ] 相同内容不同发生次数的回执、已归档数组、会话恢复和 native compact 后仍能正确解释。
- [ ] 上游 P02/P04 的 JSON 上下文经过哪些投影；原生 SDK 保留的旧会话内容不能由本次 delta 大小推断。

关联：`test_dsh_harness.py`、`test_dsh_runtime.py`；[examples/dsh_smoke.py](../../examples/dsh_smoke.py) 是独立验证入口，不属于研究流程的生产 prompt。

## P18：DSH feedback 独立上下文

代码：[_FeedbackSessionBackend._increment / DSHFeedbackBackend.generate](../../src/rsisys/loops/decision_making/harness/dsh.py#L236)。

- [ ] 原始 feedback system 指令和 evidence 以何种角色发送；独立短会话不混入 decision 原生会话。
- [ ] 模型与原生通知计费、输出 schema/格式、失败关闭与超时。

关联：`test_dsh_harness.py`、`test_feedback_decoupling.py`、`test_medical_dsh.py`。

## P19：Actor prompt 与可配置 instructions

代码：[ModelActorPolicy.act](../../src/rsisys/loops/exploration/harness/policy/__init__.py#L30)、
[configure_prompt handler](../../src/rsisys/loops/exploration/harness/commands.py)、
[PromptSpec](../../src/rsisys/loops/contracts/directive.py)。

- [ ] ActorContext 的 observation/directive/actions/history/memories；允许动作来自当前 action space。
- [ ] `context.instructions` 追加到 system 的来源、权限、替换/清空和版本语义；这是 Lab/Actor harness 配置。
- [ ] Actor 输出 name/arguments、模型身份、解析、动作校验与 episode 回执。
- [ ] SFT、评测等直接执行路径不应被算作 Actor 动作选择调用。

关联：`test_loop_internals.py`、`test_system_directives.py`、`test_exploration_execution.py`。

## P20：医疗评测 prompt

代码：[MedicalEvaluator._measure](../../examples/medical/evaluation.py#L90)、[Sample.prompt](../../examples/medical/task.py#L31)、
[医疗 prompt candidate](../../examples/medical/run.py)。

system 为候选 instructions 加精确合法标签约束，user 为 question/options；逐样本直接调用推理。

- [ ] MedQA/MedXpertQA/PubMedQA 的合法标签和选项顺序；不把 gold 加入推理输入。
- [ ] 候选 prompt、模型 revision、max_output_tokens、thinking/template 与评测缓存 key 的一致性。
- [ ] 精确单标签评分、非法格式和推理失败；结果来源可与 feedback claims 对齐。

关联：`test_medical_example.py`、`test_evaluation_inputs.py`、`test_medical_transformers.py`。

## P21：医疗 SFT 的训练输入

代码：[_MedicalTokenizer._feature / metadata](../../examples/medical/transformers_runtime.py#L111)、
[_messages / _render](../../examples/medical/transformers_runtime.py#L66)、
[Sample.prompt](../../examples/medical/task.py)、[MedicalTrainingBackend](../../examples/medical/native.py)。

- [ ] 显式 system_prompt 与默认单标签说明的优先级；训练/评测 prompt 何时一致、何时故意不同。
- [ ] chat template、enable_thinking、generation prefix、assistant gold 与 completion-only mask。
- [ ] 超长样本整体排除还是截断、序列长度、实际样本/输入 token/优化步曝光以及转换证据。

关联：`test_medical_transformers.py`、`test_medical_native.py`、`test_training_progress.py`。
这是训练样本格式审查，不能把构造了样本视为已发生训练。

## P22：医疗简化装配入口

代码：[MedicalAssembly.run](../../examples/medical/run.py#L266)。

- [ ] settings.objective 后附 fixed task/protocol/anchor 与操作提示的完整正文。
- [ ] 与 P10 主实验入口的差别：不能把简化入口的研究控制和提示默认套到夜间/三小时实验。
- [ ] 指令 input_ids、fixed initializer、候选和采纳顺序说明与实际 handlers 对应。

关联：`test_medical_example.py`、`test_medical_experiment.py`。

## P23：Qwen toy researcher

代码：[Qwen toy ResearchDecisionPolicy.decide](../../examples/qwen_toy/run.py#L121)。

- [ ] 该示例仍把 remaining budget 放入 objective，并允许 finish/research cycle；这是单独路径，需与用户想要的架构分别核对。
- [ ] 固定 recipe/protocol、NLL 方向、输入数据由后端采样、review_hint 的可选语义。
- [ ] 它自己的重试、能力过滤和结果记录，不与医疗路径的限制混用。

关联：`test_qwen_research_defaults.py`、`test_qwen_run_policy.py`、`test_qwen_toy.py`。

## P24：Qwen toy Actor、部署探针与 NLL 输入

代码：[ParityActorPolicy / EXECUTION_PROMPT](../../examples/qwen_toy/run.py#L221)、
[WorkerActivator 的部署探针](../../examples/qwen_toy/adapters.py#L334)、
[prepare 的训练/评测 prompt](../../examples/qwen_toy/prepare.py#L7)、
[worker 的 encode/evaluate](../../examples/qwen_toy/worker.py#L239)。

- [ ] parity 任务的输入数字、严格 JSON 输出、Actor instructions 与部署 JSON smoke 探针分开。
- [ ] heldout NLL 直接对 eval.jsonl 的 prompt/answer 计算损失，不是一次生成式问答评测。
- [ ] 训练/推理数据与指标的真实范围；toy 任务不证明医疗或 researcher 研究能力。

关联：`test_qwen_toy.py`、`test_qwen_service.py`、`test_qwen_worker.py`。

## P25：模型最终收到的消息和 schema

代码：[RemoteModelBackend](../../src/rsisys/runtime/model.py)、[VLLMBackend](../../src/rsisys/adapters/serving.py)、
[医疗 DecisionBackend](../../examples/medical/experiment.py)、[LocalTransformerServing](../../examples/medical/transformers_runtime.py)、
[toy 服务](../../examples/qwen_toy/serve.py)、[toy transport](../../examples/qwen_toy/adapters.py)。

- [ ] GenerationRequest 的 role/model/system/prompt/schema/options 实际如何变成 messages；是否额外插入 JSON system 指令。
- [ ] canonical response schema 与 portable wire schema 的真实差异、字段顺序和 provider 限制；不能把解码约束当业务校验。
- [ ] tokenizer chat template、thinking、输出上限、context length、计数方式与实际生成一致。
- [ ] readiness 的固定 `Reply OK.` 等探针与研究调用分开；wire 日志仍须记录对应身份和用途。

关联：`test_structured_transport.py`、`test_framework_adapters.py`、`test_live_generation.py`、`test_medical_response_recovery.py`。

## P26：可选改进任务提案

当前完整模板：[improvement_proposals.txt](../../src/rsisys/loops/decision_making/feedback/prompts/improvement_proposals.txt)。
构造与持久化：[ModelImprovementProposer](../../src/rsisys/loops/decision_making/feedback/proposals.py)；
装配与阶段：[ResearchAssembly](../../src/rsisys/research.py)、[FeedbackPipeline](../../src/rsisys/loops/decision_making/feedback/pipeline.py)。

输入是该回执实际捕获并准入的结构化证据，使用捕获的 active meta-agent ModelRef 和独立上下文。
后端支持 native tool calling 时使用单个 `propose_improvements` 函数，否则按能力使用 response_schema
或 JSON 文本；实际交换仍由 FeedbackModelGateway 计费、保存和校验。

- [ ] 只允许 harness_change（subject=meta_agent/lab）或 policy_training（subject=meta_agent）；可返回空提案。
- [ ] 不生成 memory、训练样本、标签、数据集或 system_change；计数只能提示调查，不能自动授权训练。
- [ ] 保留候选/协议/指标/来源及不确定性；缺少快照证据不等于全局未发生。
- [ ] 提案回到主 decision，由 feedback_proposal_ids 关联正式动作；不得直接执行、训练、停止或激活。
- [ ] memory 与 proposer 各阶段的调用、失败、恢复及预算准确，真实结果交付不等待它们。
- [ ] Lab 能力的描述和 schema 与 parser/执行校验一致，数据构建和训练分别下发；只公布已安装能力。

Lab 模型侧接口：[improvement_directives](../../src/rsisys/adapters/improvement.py)。
数据构建算法/recipe 和真实训练材料没有通用生产 prompt，本轮只提供可替换协议与 fixture。
旧 harness-only 选项仍使用 [harness_proposals.txt](../../src/rsisys/loops/decision_making/feedback/prompts/harness_proposals.txt)，
工具名为 propose_harness_changes，与新选项互斥。

关联：`test_feedback_lab_boundaries.py`、`test_improvement_lab.py`、`test_feedback_pipeline.py`。

## 每个 prompt 的验收记录

```text
Prompt ID / 生效装配 / 模型身份 / source revision：
静态构造点与继承覆盖：
实际 system / user 字段 / 工具和 directive schema / canonical 与 wire 输出 schema：
逐句意图：研究目标 | 输出格式 | 证据等级 | 操作权限 | 调度 | 数据约束
重复、冲突、含糊或过期说明：
与 parser / permissions / executor 的不一致：
最小正常请求 / 超长与缺失证据 / 负结果 / 格式错误恢复的输入输出：
真实请求/响应证据及人工语义判断：
结论与后续修改：
```

实际请求通常从相应 run 的 `model_requests.jsonl`、`model_responses.jsonl`、`model_errors.jsonl`
和存在时的 wire 日志取得；反馈任务另核 `feedback_learning.json` 的原始/准入输入与原始响应。
这些文件名以装配和 run 为准。缺失正文、外部 SDK 默认 prompt 或实际服务 chat template 时，记录“待取证”，不据代码模板断言线上已这样执行。
