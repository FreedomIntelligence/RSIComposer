# Feedback 四分支与 Meta-agent 改进 Lab 调整

日期：2026-11-04。基础 revision：`cb67628151cdb855b30f657e37cbf0af801644a8`。
本记录对应当前工作树，实现用户确认的职责调整，供后续逐模块和逐 prompt 审查。
后续 CP2 已确认薄封装与自由格式输出；进一步修正及验证见 [适配自查](feedback-adaptation-review-20261104.md)。

## 已实现的边界

| 路线 | 本次行为 | 主要接口 |
| --- | --- | --- |
| 即时 context | 真实结果先记账和持久捕获，然后返回 decision；交付不等待模型整理 | ExecutionReceipt、service.observe |
| Memory | 必须启用、实现可替换；后续 tick 推进整理和发布 | ContextMemoryOrganizer、FeedbackProcessor、service.refresh |
| 原始轨迹 | 默认归档；保留原始捕获和覆盖范围，不产生标签或训练样本 | TrajectoryArchive、trajectory_snapshot、TrajectoryRef / TrajectorySource |
| 改进任务提案 | 可选 harness_change / policy_training；主 decision 选择正式任务 | ModelImprovementProposer、feedback_proposal_ids |
| Lab 数据与训练 | 数据构建和训练是独立任务，分别替换后端，返回数据集/候选 | DecisionDatasetBuilder、DecisionLearner、improvement_exploration_factory |

完整图和装配示例见 [feedback-workflow.md](feedback-workflow.md)。

`observe()` 只捕获并排队；`archive()` 只归档，不进入推理队列。
默认 harness 在下一次 context 组装时 refresh；独立接入者自行提供 tick。
只读查询（包括反馈导航及被拒绝的读取）保存原始轨迹，不递归调用反馈模型，
也不作为待处理 memory 错误或常驻知识注入。
反馈模型仍使用捕获时 active meta-agent 的版本和独立上下文。

快速交付以原始捕获成功为条件；落盘失败应重试交付，不重新执行任务。
关闭时未提交的整理任务取消，但原始事实不删除，因此最后一份结果未必已完成模型整理。

## 固定来源和 Lab 协议

- [improvement.py 契约](../../src/rsisys/loops/contracts/improvement.py)：TrajectoryRef、TrajectorySource、数据构建请求/结果和 submit/poll/cancel。
- [trajectory.py 适配器](../../src/rsisys/adapters/trajectory.py)：按 archive/id/digest 读取不可变原始捕获；不匹配即拒绝。
- [improvement.py 能力接入](../../src/rsisys/adapters/improvement.py)：两个普通 explore 指令与现有命令生命周期。

`/trajectory_snapshot` 不随 feedback 解释完成而改变；旧 `/trajectory_segment` / extract_trajectory
继续提供含后续反馈交换的组合查看，不能充当固定训练来源。外部归档没有 snapshot 能力时，
固定来源导出明确失败；外部 Lab 可实现自己的 TrajectorySource。

数据构建结果必须与请求的 trajectory_refs / recipe 一致，并包含有版本/指纹的数据集与报告引用。
训练请求必须明确 base 和有版本/指纹的数据集；成功结果必须匹配 base，仅影响 decision_maker，
并提供有效指标。无效终态结果变为失败并保留后端原始 outcome，不重新提交。
提交响应丢失时复用同一请求/幂等键；poll 身份不符严格拒绝；取消请求不代表确认取消。

factory 仅开放已安装能力，可通过 ExplorationLab 暴露为 Lab 端口；没有新增跨 Lab scheduler。
构建数据不会自动训练，训练不会自动评测或激活 meta-agent。
训练算法、数据 recipe、资源准入与完整持久恢复仍由后端/部署装配提供。

## 组件与 prompt

[ResearchAssembly](../../src/rsisys/research.py) 是统一装配入口。新配置 schema_version=2 记录：

```json
{
  "trajectory_mode": "archive_only",
  "dataset_construction_owner": "lab",
  "receipt_delivery": "direct_before_processing",
  "budget_owner": "host"
}
```

默认 journal memory 不新增模型推理，LangMem 可替换整理流程与工作视图。
可选 `improvement_proposals=ModelImprovementProposer(...)` 与旧 harness-only 选项互斥。
旧医疗 CLI 选项维持原来的 harness-only 含义，旧底层组合入口保留兼容；
不静默重开配置不同的旧 journal。

完整新模板：[improvement_proposals.txt](../../src/rsisys/loops/decision_making/feedback/prompts/improvement_proposals.txt)。
只允许提出任务，禁止生成训练样本、标签、数据集、执行代码、直接训练或激活。
支持时使用原生 `propose_improvements` tool calling，否则按后端能力返回 schema/JSON。
提案的候选/协议/指标/来源继续校验，调用继续计入原宿主预算。
主 decision 从有界导航读取提案，并用 feedback_proposal_ids 关联正式任务及来源摘要。

## 验证

最终整仓回归：**1,472 项，1,460 通过，12 跳过，0 失败/错误**。
12 项分别为 6 项外部 DSH SDK 路径及 6 项真实服务推理检查；本次没有安装依赖、
调用真实模型或进行训练。运行前后对 292 个 source/example/test 文件取指纹，无漂移。

```bash
PYTHONPATH=src:tests .local/langmem-validation/bin/python -m unittest discover -s tests -q
PYTHONPATH=src .local/langmem-validation/bin/python examples/feedback_pipeline.py
PYTHONPATH=src .local/langmem-validation/bin/python examples/decision_policy_lab.py
```

实际整仓运行使用本地 validate.py 包装 unittest discover，并保存开始/结束时间、逐项 skip、
source manifest 和输出日志。可提交的 [验证摘要](repros/feedback-lab-validation.json)
记录这些文件的 SHA256 和 fixture 结果；详细日志在 `.local/feedback-lab-20261104/`。

新增 22 项测试：

- [test_feedback_lab_boundaries.py](../../tests/test_feedback_lab_boundaries.py)：慢整理不挡真实结果、查询只归档、固定来源、模型提案/原生工具、正式 decision 引用及非法产物拒绝。
- [test_improvement_lab.py](../../tests/test_improvement_lab.py)：独立构建与训练任务、端到端归档来源、幂等键、取消、缺失能力、来源/recipe/base/角色校验及异常轮询身份。

本轮首次整仓运行暴露 2 failures / 1 error，均已修正后重新整仓通过：

1. 新 kind 扩大旧兼容 prompt 的默认 schema，超过保守 token 准入。修为默认仍用旧四类 schema，新 proposer 显式选择新两类；没有放宽上限。
2. 原测试按交付期间立即整理计调用，修为验证交付后尚未推理、refresh 后才推进。
3. 新测试替换已被 JobBook 绑定的 poll 方法，导致模拟身份错误反复出现；修为稳定回调只注入一次错误，再核验同一作业可继续轮询。

两个 CPU fixture 的结果：

| 示例 | 结果 | 验证边界 |
| --- | --- | --- |
| feedback_pipeline | 2 份回执，1 个关联提案的正式 decision | 提案链路，不证明 harness 已激活 |
| decision_policy_lab | 3 份归档，构建任务完成，显式训练返回 mock 候选 | real_training=false，live_policy_changed=false |

## 尚未验证与后续审查

当前 observer 捕获公开决策/执行片段，**没有原始 meta-agent 完整 prompt/response**。
快照明确 complete_conversation=false、training_ready=false；Lab recipe 需核对 coverage，
不能把缺失输入补写为完整会话，或把反馈文字当正确标签。

具体数据筛选/标签算法、真实训练、独立 policy evaluator、harness 构建/验证/激活器和
跨 Lab 调度没有在本轮验收。LangMem 自由文字语义、候选改进质量仍需真实证据审查，
结构校验和 CPU fixture 不替代这些验收。
原 host 限额、停止所有权、历史冻结 run 和模型部署未调整。

下一步逐项讨论可从 [M01.5 / M07 / M09 / M10.5](modules.md) 和 [P26](prompts.md#p26可选改进任务提案) 开始。
