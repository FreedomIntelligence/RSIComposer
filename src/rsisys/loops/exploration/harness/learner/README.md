# Actor learner

提供 SFT/RL 训练后端，采样、任务控制与版本应用由 [默认 Exploration Harness](../README.md) 管理。

## 接口与输入

[ActorLearner](__init__.py) 提供 `submit(request, *, idempotency_key)`、`poll(job)`、`cancel(job)`、`result(job)`；后端 cancel 返回 None，默认 Harness 的 cancel_learning 返回 JobOutcome。ActorTrainingRequest 包含 base、recipe_ref、method、data_refs 和 trajectory_refs。

method=sft 要求非空 data_refs；method=rl 要求非空 trajectory_refs，默认 Harness 校验它们引用 Actor 本地 memory/history 中的轨迹。这些引用不指向 Experience，也不能用 Feedback.evidence_refs 替代。训练终态仍以 training_run 写入共享 Experience。成功的 TrainingResult.base 必须匹配请求，candidate.affected_roles 必须包含 actor。幂等、终态与版本约定见[共享协议](../../../contracts/README.md)。

## 实现与替换

[MockActorLearner](backends/mock.py) 读取数据产物并生成模拟版本清单，metrics 为空，不优化真实权重或计算能力提升。真实后端可放在 backends/，通过 DefaultExplorationHarness 的 learner 参数注入；可自行选择训练框架或外部服务。

成功结果默认形成候选。显式 `apply_learning`、`SFTSpec(apply_result=True)` 或 RL 内部更新改变 Actor 绑定；应用要求 policy 支持 ModelUpdatable、当前模型与训练 base 一致，且无环境交互或其他活跃训练。同一 job 重复应用不再次更新；这些操作不验收质量，也不自动更新 Decision-maker。

真实后端须让推理侧能够解析新版本，并兑现重试提交的幂等性；采样节奏由默认 Harness 编排。

SFT 和 RL 的训练阶段由 Harness 直接调用 learner，绕过 ActorPolicy.act；只有 RL rollout 通过 Actor Action Space 采样。
