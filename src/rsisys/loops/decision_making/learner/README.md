# Decision learner

负责决策者自身的学习后端，任务控制和版本应用仍由 [Decision-making loop](../README.md) 管理。

## 接口与输入

[DecisionLearner](__init__.py) 提供 `submit(request, *, idempotency_key) → JobRef`、`poll(job) → JobOutcome[TrainingResult]`、`cancel(job) → None`、`result(job) → TrainingResult`。DecisionTrainingRequest 包含 base: ModelRef、非空 experience_refs 元组和非空 recipe_ref；loop 校验引用的是 decision_maker 可见的经验记录 ID。

提交和终态的公共约定见[共享协议](../../contracts/README.md)。成功结果必须是 TrainingResult，base 与提交请求一致，candidate.affected_roles 包含 decision_maker。

## 任务与应用

通过 loop 的 `submit_learning(request, *, idempotency_key)` 提交：键不能为空，同键同请求复用任务，同键不同请求报错；同一 loop 最多一个未终结训练任务。未配置 learner 时不能提交。

`poll_learning(job)` 和 `cancel_learning(job)` 返回 JobOutcome，并在终态记录一次训练经验；取消请求后仍会轮询确认状态，不保证立即终结。loop 从 poll 的结果读取训练产物，不调用后端 result。

`apply_learning(job) → ModelRef` 显式轮询并应用成功结果；首次应用要求 policy 支持 [ModelUpdatable](../../contracts/model_service.py)、当前模型等于 base，且无其他未终结训练任务。仅更新当前 policy 绑定，同一任务重复应用直接返回原结果模型。不会自动评测质量或更新 Actor。

## 实现与替换

[MockDecisionLearner](backends/mock.py) 的构造参数为 artifacts 和仅限关键字的 pending_polls=1（非负整数）：前 pending_polls 次轮询返回 RUNNING，下一次生成 simulated=True 的版本清单，无后台线程或梯度训练。result 只读取已成功结果，不推进任务。

真实实现可放在 backends/ 并注入 loop 的 learner 参数；UnimplementedDecisionLearner 的所有方法均抛 NotImplementedError。

训练样本筛选、优化算法与质量评测由后续实现提供；周期触发不属于 learner 后端。
