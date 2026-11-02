# Decision-making loop

选择优化目标与改进策略，对外发布 Directive 并接收 Feedback。构造方式为 `DecisionMakingLoop(harness, action_space)`；跨 loop 规则见[系统架构](../../../../README.md#系统架构)。

[loop.py](loop.py) 只委托 `step(objective)`、`pending()` 与 `receive_feedback(directive, feedback)`；`run(objective, max_steps=16)` 有界推进，在下发、等待、结束或步数上限处返回。策略、准备操作、训练和状态编排属于 Harness。

## 组合边界

- [Harness](harness/README.md)：可整体替换的决策执行边界；默认实现封装 policy、上下文、记忆、工具与 learner。
- [Action Space](action_space/README.md)：描述并校验本地 research 操作及系统指令模式，不执行操作。

Directive 由 decision policy 发出，或由外部系统直接提交给 Exploration Harness。execute / sft / rl / evaluate / configure_prompt 属于系统指令，不是 Actor 的环境动作。

内置细节通过 `loop.harness` 访问，例如 `loop.harness.policy`、`loop.harness.resume(...)`、`loop.harness.submit_learning(...)`；不提供旧属性或方法别名。自定义 Harness 只需满足其公开协议，无需暴露 memory/tools。
