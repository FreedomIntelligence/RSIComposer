# 单 meta-agent 的反馈与记忆链路

更新：2026-11-06。当前术语以[统一架构](../architecture.md)为准。

1. Lab 执行授权任务并持久化原始回执。
2. 同一个 meta-agent 直接看原始反馈、当前运行状态与相关记忆。
3. 调试任务用 `research_task` open/pin 固定原始错误，读代码、改代码和复测后才 close。
4. meta-agent 通过普通 decision 调用 `memory_write` 更新长期笔记，或 `memory_delete` 撤下旧笔记。
5. 模块仅存储、检索和记录来源/修订；大原文按 SHA 和稳定地址分页读取。

`ResearchAssembly()` / `.single_meta()` 无独立抽取模型，也不因压缩自动请求模型总结。
默认最多两个开放任务、每任务四个固定回执，另外保留最近两条 producer 回执；
关任务不删原文、不自动写 memory。记忆的 supported/refuted 是模型判断，散文始终未验证。

医疗 Open Lab 默认 `--feedback-routing agent_memory`，恢复状态包含 `AgentMemoryState`。
正常恢复保留任务、记忆、来源、调用账本与原始截止时间。未知 mutation 阻塞核对，不重放。
旧 LangMem/proposer 工作流仍可显式装配，完整旧实验记录保留其当时配置。

下方轨迹与 Lab 接口也供显式兼容工作流使用，不表示默认有第二个整理 agent。

## 原始轨迹到 Lab 的接口

```python
from rsisys.adapters.trajectory import FeedbackTrajectorySource
from rsisys.loops.contracts.improvement import DecisionDatasetBuildRequest

source = FeedbackTrajectorySource(service, archive_id='project/run')
ref = source.reference(job_id)
capture = source.read(ref)
request = DecisionDatasetBuildRequest((ref,), recipe_ref='reviewed-data-recipe:v1')
```

`TrajectoryRef` 包含 archive_id、trajectory_id、SHA256 digest。读回必须匹配同一 archive、
轨迹身份和摘要指纹。`/trajectory_snapshot` 是不可变原始捕获，不包括之后生成的 memory/proposal。
整理任务由 queued 变为 completed，不会改变这个快照及引用。

`/trajectory_segment` 和 extract_trajectory 保留为旧的组合查看接口，会附带后来完成的 feedback
交换信息，不应作为固定训练来源。view=trajectories 仍提供索引。
新外部 archive 只需 write(event)/read(reference)，host 校验读回后导出固定捕获。
旧兼容 archive 可以只提供索引；没有 snapshot 的旧实现会明确拒绝导出固定来源。
外部 Lab 也可实现自己的 TrajectorySource，不必依赖 FeedbackLearningService。

当前 observer 捕获的是公开 decision 记录与执行片段，**未捕获原始 meta-agent 的完整模型交换**。
快照明确标记 coverage、complete_conversation=false、training_ready=false。数据构建 recipe
必须检查覆盖范围与所需原始输入；不能由 action/理由伪造完整训练对，也不能把反馈摘要当正确标签。

## Lab 内的可替换能力

| 能力 | 请求 | 结果 |
| --- | --- | --- |
| build_decision_dataset | DecisionDatasetBuildRequest：固定轨迹引用 + recipe_ref | DecisionDatasetBuildResult：数据集引用、构建报告、原始来源/recipe、样本数与 simulated |
| train_decision_policy | 既有 DecisionTrainingRequest：明确 base、data_refs、recipe_ref、method | 既有 TrainingResult：base、新模型、candidate、真实后端提供的指标与 simulated |

数据构建 backend 实现 submit/poll/cancel，主机用稳定 idempotency key 绑定请求。
构建报告应记录筛选/排除、标签与格式规则及样本来源；协议校验来源和 recipe 身份，
不声称验证了自然语言标签质量。训练只接受明确且有版本/指纹的数据集，不将 archive 直接当数据集。
两个任务独立，构建完成不会自动训练；达到某个轨迹数量也不会自动提交训练。

```python
from rsisys.adapters.improvement import improvement_exploration_factory
from rsisys.config import build_demo

runner, _ = build_demo(
    config,
    exploration_factory=improvement_exploration_factory(
        dataset_builder=external_dataset_builder,
        decision_learner=external_decision_learner),
    research=ResearchAssembly(),
)
```

该 factory 只公布已安装的能力，由普通 explore decision 经现有 Runner/命令生命周期执行。
可以用 ExplorationLab 将该 endpoint 暴露为 LabSpec/LabTask/Lab；本次不新增跨 Lab scheduler。
backend 的具体构建算法、训练框架、资源准入和外部恢复由部署方提供，factory 不安装或自动训练。
持久化完整 JobBook/自定义 handler 状态仍需要显式恢复契约。

训练结果只能作用于 decision_maker，返回候选，不绑定或激活当前 meta-agent。
评测/部署/应用由正式 decision 选择已安装能力执行；本次没有新增通用 policy evaluator、
harness 构建器或版本激活器。Harness 提案仍通过现有代码能力构建和验证。

## 改进提案回到 meta-agent（decision侧）

可选 ModelImprovementProposer 的 prompt 位于 feedback/prompts/improvement_proposals.txt。
它只允许 harness_change 或 policy_training，拒绝 training_example 和不匹配的目标。
路由不增加模型调用。内置 pipeline 按准入和宿主容量推进 memory 与可选 proposer；
提案独立可选，只有显式设置 `proposal_requires_memory` 才依赖 memory 成功。

meta-agent（decision侧）在 research_agenda.feedback_learning.improvement_proposals 中获得有界入口，
通过 feedback_learning_read(view='proposals') 查看完整内容，使用 feedback_proposal_ids 关联正式任务。
宿主在执行前检查 ID、目标和来源，记录 proposal digest；提案、选中、构建、训练、评测、采纳分别记录。
原 harness_proposals 导航保留兼容。提案永远不直接执行代码、生成训练标签或修改权重。

## 可运行验证与审查入口

- [接口与迁移审查](m01-feedback-lab-20261104.md)。
- [数据构建契约](../../src/rsisys/loops/contracts/improvement.py)。
- [Lab capability / factory](../../src/rsisys/adapters/improvement.py)。
- [归档来源适配器](../../src/rsisys/adapters/trajectory.py)。
- [提案 prompt](../../src/rsisys/loops/decision_making/feedback/prompts/improvement_proposals.txt)。
- `PYTHONPATH=src python examples/feedback_pipeline.py`：提案进入正式 decision。
- `PYTHONPATH=src python examples/decision_policy_lab.py`：原始归档到 Lab 构建数据，再显式训练候选。

两个示例都是 CPU fixture，不验证真实自主研究、训练、数据标签、memory 语义或质量提升。
当前 LangMem 的语义限制仍见[原验收记录](feedback-oss-adoption.md#尚未通过的语义验收)。
