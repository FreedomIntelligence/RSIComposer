# meta-agent（decision侧）端点

选择优化目标与改进策略，对外发布 Directive 并接收 Feedback。构造方式为 `DecisionMakingLoop(harness, action_space)`；跨 loop 规则见[系统架构](../../../../README.md#系统架构)。

统一口径为 **meta-agent（decision侧）→ Lab → meta-agent（整理侧）→ context**，
见[系统术语与架构](../../../../docs/architecture.md)。decision 是动作或任务；整理侧使用捕获的
同一权重身份与独立上下文。即时 context 不必触发模型整理，memory 模块保持可用并按需检索。
原始轨迹与提案独立可选，数据构建和训练在 Lab 执行。`DecisionMakingLoop` 是本端点的
兼容类型名；Lab 可以使用开放程序路径，不必有 Actor loop。具体映射见[整理侧](feedback/README.md)。

研究装配入口为 `ResearchAssembly / assemble_research`：context/memory 接入窗口保持可用，
原始轨迹默认归档，提案按需启用。meta-agent（decision侧）通过 `feedback_proposal_ids` 引用提案后选择构建/验证，执行回执
继续进入 feedback；[workflow 与装配](../../../../docs/review/feedback-workflow.md) 描述实际接通范围。

[loop.py](loop.py) 只委托 `step(objective)`、`pending()` 与 `receive_feedback(directive, feedback)`；`run(objective, max_steps=16)` 有界推进，在下发、等待、结束或步数上限处返回。策略、准备操作、训练和状态编排属于 Harness。

## 组合边界

- [Harness](harness/README.md)：可整体替换的决策执行边界；默认实现封装 policy、上下文、记忆、工具与 learner。
- [Action Space](action_space/README.md)：描述并校验本地 research 操作及系统指令模式，不执行操作。
- [Feedback](feedback/README.md)：独立整理与路由研究反馈，派生按 Attempt 更新的知识；不依赖默认 harness，也不代替其检索、上下文选择和 compact。

Directive 由 decision policy 发出，或由外部系统直接提交给 Exploration Harness。execute / sft / rl / evaluate / configure_prompt 属于系统指令，不是 Actor 的环境动作。

内置细节通过 `loop.harness` 访问，例如 `loop.harness.policy`、`loop.harness.resume(...)`、`loop.harness.submit_learning(...)`；不提供旧属性或方法别名。自定义 Harness 只需满足其公开协议，无需暴露 memory/tools。
