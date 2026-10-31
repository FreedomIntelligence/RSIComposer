# Actor policy

根据观察选择行动，交给 [Action space](../action_space/README.md) 校验，再由环境执行。流程归属 [Exploration loop](../README.md)。

## 接口

[ActorPolicy](__init__.py) 的 `act(ActorContext) → ActionProposal` 接收 observation、directive、actions、experience 和 memories。ActionProposal 包含 name、arguments、simulated，以及可选模型身份和 token 用量。

## 实现与替换

ModelActorPolicy 调用注入的 ModelBackend，解析 JSON 的 name / arguments，并校验返回模型身份。自定义 policy 只需实现 act，可以采用不同的推理或决策方式。

需要应用训练结果时，实现 [ModelUpdatable](../../contracts/model_service.py) 的 `model: ModelRef` 与 `update_model(model)`。内置 update_model 只修改 Actor 的模型引用，并拒绝直接跨 backend 切换；部署和权重加载由后端解决，loop 的应用条件见 [learner](../learner/README.md)。
