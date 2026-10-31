# Decision policy

位于 [默认 Decision Harness](../README.md) 内部。`DecisionPolicy.decide` 选择操作、系统指令、等待或结束；Harness 组织执行与结果处理，[Decision Action Space](../../action_space/README.md) 定义操作参数和指令模式约束。

## 接口

[DecisionPolicy](__init__.py) 的同步接口 `decide(context: DecisionContext) → Decision` 接收目标、全部来源的 experience、自身 history 和 memories、操作定义、当前准备结果、最新反馈和 directive_modes。默认 Harness 的 `experience_limit=32` 限制系统证据条数，`memory_limit=32` 分别限制本地 history 和工作记忆条数，两者必须为正整数。scope 仅标记证据来源，不是优化目标接口。

[Decision](../decision.py) 的 invoke / explore / wait 分别只能携带 operation / directive / job；finish 不携带这三项，可附 reason。wait 的任务由外部通过 `loop.harness.resume(job, result)` 回填，见 [Harness](../README.md)。

## 实现与替换

同文件的 `ModelDecisionPolicy(backend, model, *, allowed_actions=("echo",), directives=None)` 同步调用 ModelBackend.generate，校验返回的 ModelRef 与请求一致，再解析 JSON。allowed_actions 写入探索指令，限制 Actor 环境动作；directive_modes 描述系统指令模式。模型输出参数为：

- invoke：operation 和 arguments；具体名称与参数由 Decision Action Space 校验。
- explore：必填 strategy，mode 默认 execute。sft 需 recipe_ref，apply_result 默认 false；rl 需 recipe_ref，updates 默认 1，必须为正整数；evaluate 需 protocol_ref 和来自最新反馈 candidates 的 candidate_id；configure_prompt 需字符串 instructions。
- wait：job 对象含 job_id、backend；finish：可选 reason。

系统指令的描述、schema 与参数解析由注入的 `DirectiveDefinitions` 提供；ModelDecisionPolicy 和 Decision Action Space 应使用同一个 catalog。请求只包含当前 directive_modes 允许的定义。模型将模式参数放入 `arguments`，例如：

```json
{"kind":"explore","mode":"configure_prompt","strategy":"short answers","arguments":{"instructions":"Be concise."}}
```

仍接受旧的顶层参数形式，但同一响应不能混用顶层模式参数和 arguments。新增模式不需要修改 policy 的 JSON 分派代码，详见 [Action Space](../../action_space/README.md#扩展系统指令)。

explore 未提供 input_ids 时使用全部当前准备产物；提供时必须是产物 ID 列表，只能引用当前准备结果或最新反馈的 artifact_refs，空列表表示不传输入。

自定义策略实现 `decide` 并通过默认 Harness 的 policy 参数注入，不要求使用 JSON。需要应用学习结果时，另实现 [ModelUpdatable](../../../contracts/model_service.py) 的 model 与 update_model。

内置 update_model 只更改当前策略的模型引用，拒绝切换 backend_id。explore 的系统指令交给 Exploration Harness；训练/评测仍需配置对应后端。`configure_prompt` 使用 PromptSpec(instructions)，空字符串清空 Actor 指令配置，结构见[共享协议](../../../contracts/README.md)。

决策者自身训练可通过 Harness 本地调用，见 [learner](../learner/README.md)；当前没有自动自优化指令或触发机制，也没有统一接口覆盖 Actor 模型/prompt、决策者自身及 Harness 配置/代码这些优化目标。
