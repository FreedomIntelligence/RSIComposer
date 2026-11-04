# Decision-making loop

选择优化目标与改进策略，对外发布 Directive 并接收 Feedback。构造方式为 `DecisionMakingLoop(harness, action_space)`；跨 loop 规则见[系统架构](../../../../README.md#系统架构)。

概念上做决策的模型称为 **meta-agent**，decision 是其动作或任务；带固定研究目标的
data/model/harness 执行环境称为 **lab**。反馈角色与 meta-agent 共享当前权重，使用独立
上下文；context/memory 整理能力必须激活且实现可替换；轨迹归档默认开启、可关闭或替换，harness 修改按需提出，
轨迹转训练材料由后续 skill/规则定义。基础 Lab 契约和共享 ExecutionReceipt
已提供，默认 Runner 仍走兼容 exploration 端口；实现映射与迁移边界见[反馈架构](feedback/README.md)。

研究装配入口为 `ResearchAssembly / assemble_research`：context/memory 保持激活，数据先提取公开轨迹，
harness 先给提案。主 decision 通过 `feedback_proposal_ids` 引用提案后选择构建/验证，执行回执
继续进入 feedback；[workflow 与装配](../../../../docs/review/feedback-workflow.md) 描述实际接通范围。

[loop.py](loop.py) 只委托 `step(objective)`、`pending()` 与 `receive_feedback(directive, feedback)`；`run(objective, max_steps=16)` 有界推进，在下发、等待、结束或步数上限处返回。策略、准备操作、训练和状态编排属于 Harness。

## 组合边界

- [Harness](harness/README.md)：可整体替换的决策执行边界；默认实现封装 policy、上下文、记忆、工具与 learner。
- [Action Space](action_space/README.md)：描述并校验本地 research 操作及系统指令模式，不执行操作。
- [Feedback](feedback/README.md)：独立整理与路由研究反馈，派生按 Attempt 更新的知识；不依赖默认 harness，也不代替其检索、上下文选择和 compact。

Directive 由 decision policy 发出，或由外部系统直接提交给 Exploration Harness。execute / sft / rl / evaluate / configure_prompt 属于系统指令，不是 Actor 的环境动作。

内置细节通过 `loop.harness` 访问，例如 `loop.harness.policy`、`loop.harness.resume(...)`、`loop.harness.submit_learning(...)`；不提供旧属性或方法别名。自定义 Harness 只需满足其公开协议，无需暴露 memory/tools。
