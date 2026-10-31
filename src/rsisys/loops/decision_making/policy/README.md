# Decision policy

负责选择优化目标与决策，不执行准备操作或 Actor 任务。所属流程见 [Decision-making loop](../README.md)。

## 接口

[DecisionPolicy](__init__.py) 的同步接口 `decide(context: DecisionContext) → Decision` 接收目标、全部来源的 experience、自身 history 和 memories、操作定义、当前准备结果和最新反馈。loop 的 `experience_limit=32` 限制系统反馈条数，`memory_limit=32` 分别限制本地 history 和工作记忆条数，两者必须为正整数。scope 仅标记反馈来源，policy 独立选择优化对象与策略。

[Decision](../decision.py) 的 invoke / explore / wait 分别只能携带 operation / directive / job；finish 不携带这三项，可附 reason。wait 的任务由外部按 [准备操作回填方式](../decision_space/README.md) 完成。

## 实现与替换

同文件的 `ModelDecisionPolicy(backend, model, *, allowed_actions=("echo",))` 同步调用 ModelBackend.generate，校验返回的 ModelRef 与请求一致，再解析 JSON。allowed_actions 写入探索指令。模型输出参数为：

- invoke：operation 和 arguments；具体名称与参数由 decision_space 校验。
- explore：必填 strategy，mode 默认 execute。sft 需 recipe_ref，apply_result 默认 false；rl 需 recipe_ref，updates 默认 1，必须为正整数；evaluate 需 protocol_ref 和来自最新反馈 candidates 的 candidate_id。
- wait：job 对象含 job_id、backend；finish：可选 reason。

explore 未提供 input_ids 时使用全部当前准备产物；提供时必须是产物 ID 列表，只能引用当前准备结果或最新反馈的 artifact_refs，空列表表示不传输入。

自定义策略实现 `decide` 并通过 loop 的 policy 参数注入，不要求使用 JSON。需要应用学习结果时，另实现 [ModelUpdatable](../../contracts/model_service.py) 的 model 与 update_model。

内置 update_model 只更改当前策略的模型引用，拒绝切换 backend_id。explore 的 sft / rl / evaluate 交给探索 loop，仍需配置对应后端；决策者自身训练见 [learner](../learner/README.md)，指令结构见[共享协议](../../contracts/README.md)。
