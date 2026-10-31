# Actor policy

根据观察选择行动，交给 [Action space](../../action_space/README.md) 校验，再由环境执行。流程归属 [默认 Exploration Harness](../README.md)。

## 接口

[ActorPolicy](__init__.py) 的 `act(ActorContext) → ActionProposal` 接收 observation、directive、actions、history、memories 和 instructions。history 自动注入当前任务的本地轨迹，memories 来自自身 memory；默认不注入共享系统经验。`memory_limit=32` 限制本地 history 和工作记忆条数，必须为正整数。ActionProposal 包含 name、arguments、simulated，以及可选模型身份和 token 用量。

## 实现与替换

默认 Harness 将配置的 instructions 放入 ActorContext；ModelActorPolicy 将非空 instructions 追加到系统提示词。ModelActorPolicy 调用注入的 ModelBackend，解析 JSON 的 name / arguments，并校验返回模型身份。自定义 policy 通过 DefaultExplorationHarness 的 policy 参数注入，实现 act 并自行决定如何消费上下文。

需要应用训练结果时，实现 [ModelUpdatable](../../../contracts/model_service.py) 的 `model: ModelRef` 与 `update_model(model)`。内置 update_model 只修改 Actor 的模型引用，并拒绝直接跨 backend 切换；部署和权重加载由后端解决，默认 Harness 的应用条件见 [learner](../learner/README.md)。

ActorPolicy.act 仅参与 execute episode 和 RL rollout。SFT、prompt 配置和评测由 Harness 直接调度；评测中的模型推理由 evaluator 实现，不经过 Actor 环境动作选择。
