# Decision learner

负责决策者自身的学习后端，任务控制和版本应用仍由 [默认 Decision Harness](../README.md) 管理。

## 接口与输入

[DecisionLearner](__init__.py) 提供 `submit(request, *, idempotency_key) → JobRef`、`poll(job) → JobOutcome[TrainingResult]`、`cancel(job) → None`、`result(job) → TrainingResult`。DecisionTrainingRequest 保留原位置参数 `(base, experience_refs, recipe_ref)`，末尾追加 `data_refs=()` 与 `method="sft"`。

请求是后端中立的：base 必须为 ModelRef，recipe_ref / method 为非空字符串，experience_refs 为字符串元组，data_refs 为 ArtifactRef 元组。两类引用均允许为空，method 可自定义，后端决定算法和数据要求。外部数据可通过 data_refs 提交，无需先生成系统反馈。提供 experience_refs 时，默认 Harness 校验它们是共享 Experience 中的系统反馈记录 ID，来源不限于 decision_maker；scope 不指定训练或优化目标。

提交和终态的公共约定见[共享协议](../../../contracts/README.md)。成功结果必须是 TrainingResult，base 与提交请求一致，candidate.affected_roles 包含 decision_maker。

## 任务与应用

DefaultDecisionHarness 接受可选 `model_activator`。应用真实训练结果时必须由它确认 `TrainingResult.model` 已可用于推理，之后才能更新决策者绑定；部署进行中可用 `ModelActivationPending` 保持原绑定并在后续重试。具体接口见[共享协议](../../../contracts/README.md#作业与版本)。模拟后端可继续模拟版本切换。

通过默认 Harness 的 `loop.harness.submit_learning(request, *, idempotency_key)` 提交：键不能为空，同键同请求复用任务，同键不同请求报错；同一 Harness 最多一个未终结训练任务。未配置 learner 时不能提交。

`loop.harness.poll_learning(job)` 和 `loop.harness.cancel_learning(job)` 返回 JobOutcome，并在终态向 Experience 记录一次 training_run；取消请求后仍会轮询确认状态，不保证立即终结。默认 Harness 从 poll 的结果读取训练产物，不调用后端 result。

`loop.harness.apply_learning(job) → ModelRef` 显式轮询并应用成功结果；首次应用要求 policy 支持 [ModelUpdatable](../../../contracts/model_service.py)、当前模型等于 base，且无其他未终结训练任务。仅更新当前 policy 绑定，同一任务重复应用直接返回原结果模型。不会自动评测质量或更新 Actor。

## 实现与替换

[MockDecisionLearner](backends/mock.py) 的构造参数为 artifacts 和仅限关键字的 pending_polls=1（非负整数）：前 pending_polls 次轮询返回 RUNNING，下一次读取并校验每个 data_refs 产物，再生成 simulated=True 的版本清单；引用不存在或摘要不匹配时任务失败。它没有后台线程或梯度训练，method 字段不意味着实际执行相应算法。result 只读取已成功结果，不推进任务。

真实实现可放在 backends/ 并注入默认 Harness 的 learner 参数；UnimplementedDecisionLearner 的所有方法均抛 NotImplementedError。

训练样本筛选、优化算法与质量评测由后续实现提供；周期触发不属于 learner 后端。
