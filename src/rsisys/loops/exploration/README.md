# Lab 执行端点（Exploration 兼容实现）

统一定义见[系统术语与架构](../../../../docs/architecture.md)。Lab 是 agent 执行任务的环境；
本目录是既有指令式实现，不要求所有 Lab 具有这里的 Actor、environment 或 loop 结构。
开放 Lab 可直接实现 `LabExecution.describe/execute`。动作由 meta-agent（decision侧）选择，
返回结果按用途进入即时 context 或 meta-agent（整理侧）的 memory 路径。

接收系统 Directive，并通过 Harness 推进执行、学习、评测或 prompt 配置。构造方式为 `ExplorationLoop(harness, action_space)`；跨 loop 规则见[系统架构](../../../../README.md#系统架构)。

[loop.py](loop.py) 是薄端点，提供 `accept` / `advance` / `inspect` / `cancel` / `acknowledge`。`advance` 将独立的 Action Space 传给 Harness；指令分派、状态和业务流程由 Harness 实现。

## 组合边界

- [Harness](harness/README.md)：可整体替换的系统指令执行边界，默认封装 Actor policy、上下文、环境、记忆、learner 与 evaluator。
- [Action Space](action_space/README.md)：仅描述和校验 Actor 环境动作，如 echo / execute。

Directive 来自 decision policy 或外部系统。SFT、prompt 配置与评测绕过 `ActorPolicy.act`；RL 仅 rollout 使用 Actor 动作。`configure_prompt` 使用 `PromptSpec(instructions)` 替换 Actor 指令文本，空字符串清空，详见[共享协议](../contracts/README.md)。

内置训练、评测与 RL 本地控制通过 `loop.harness` 访问，例如 `loop.harness.submit_learning(...)`；执行 episode 使用 `loop.harness.run(directive, loop.action_space)`，推进本地 RL 使用 `loop.harness.step_rl(loop.action_space)`。不提供 `loop.policy`、`loop.memory`、`loop.space` 或旧本地控制方法别名。替代 Harness 可不含 Actor policy 或 memory，只需满足公开协议。
