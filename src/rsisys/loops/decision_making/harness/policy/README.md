# Decision policy

位于 [默认 Decision Harness](../README.md) 内部。`DecisionPolicy.decide` 选择操作、系统指令、等待或结束；Harness 组织执行与结果处理，[Decision Action Space](../../action_space/README.md) 定义操作参数和指令模式约束。

## 接口

[DecisionPolicy](__init__.py) 的同步接口 `decide(context: DecisionContext) → Decision` 接收目标、全部来源的 experience、自身 history 和 memories、操作定义、当前准备结果、最新反馈、directive_modes 和跨轮 candidates。默认 Harness 的 `experience_limit=32` 限制系统证据条数，`memory_limit=32` 分别限制本地 history 和工作记忆条数，两者必须为正整数；候选集合不受这两个 limit 截断。scope 仅标记证据来源，不是优化目标接口。

[Decision](../decision.py) 的 invoke / explore / wait 分别只能携带 operation / directive / job；finish 不携带这三项，可附 reason。wait 的任务由外部通过 `loop.harness.resume(job, result)` 回填，见 [Harness](../README.md)。

## 实现与替换

同文件的 `ModelDecisionPolicy(backend, model, *, allowed_actions=("echo",), directives=None)` 同步调用 ModelBackend.generate，校验返回的 ModelRef 与请求一致，再解析 JSON。allowed_actions 写入探索指令，限制 Actor 环境动作；directive_modes 描述系统指令模式。模型输出参数为：

- invoke：operation 和 arguments；具体名称与参数由 Decision Action Space 校验。
- explore：必填 strategy，mode 默认 execute。sft 需 recipe_ref，apply_result 默认 false；rl 需 recipe_ref，updates 默认 1，必须为正整数；evaluate 需 protocol_ref 和来自 context.candidates 的 candidate_id，也兼容 last_feedback.candidates；configure_prompt 需字符串 instructions。
- wait：job 对象含 job_id、backend；finish：可选 reason。

系统指令的描述、schema 与参数解析由注入的 `DirectiveDefinitions` 提供；ModelDecisionPolicy 和 Decision Action Space 应使用同一个 catalog。请求只包含当前 directive_modes 允许的定义。模型将模式参数放入 `arguments`，例如：

```json
{"kind":"explore","mode":"configure_prompt","strategy":"short answers","arguments":{"instructions":"Be concise."}}
```

仍接受旧的顶层参数形式，但同一响应不能混用顶层模式参数和 arguments。新增模式不需要修改 policy 的 JSON 分派代码，详见 [Action Space](../../action_space/README.md#扩展系统指令)。

parser 的候选按 candidate_id 合并去重，以 context.candidates 为准，最新反馈补齐缺失项。早期训练候选在后续训练或不携带候选的评测反馈之后仍可引用；未知 ID 仍被拒绝。

explore 未提供 input_ids 时使用全部当前准备产物；提供时必须是产物 ID 列表，只能引用当前准备结果或最新反馈的 artifact_refs，空列表表示不传输入。

指令的 simulated 标记合并模型响应与实际选中输入的来源标记；未选中的模拟准备产物不会污染真实输入，模拟反馈中的已选产物也不会丢失标记。

explore 可附顶层 `review_after_seconds`，不放入模式 arguments。`ModelDecisionPolicy.review(context)` 使用独立请求读取当前执行视图，解析 `{"action":"continue","next_review_seconds":30,"reason":"..."}`；仅允许 continue / cancel，None 停止自动复查。自定义 policy 可选择实现 review，也可只保留 decide。调用时机、反馈归因和写入重试由 [Harness](../README.md) / [Runner](../../../../runtime/README.md) 管理。

自定义策略实现 `decide` 并通过默认 Harness 的 policy 参数注入，不要求使用 JSON。需要应用学习结果时，另实现 [ModelUpdatable](../../../contracts/model_service.py) 的 model 与 update_model。

内置 update_model 只更改当前策略的模型引用，拒绝切换 backend_id。explore 的系统指令交给 Exploration Harness；训练/评测仍需配置对应后端。`configure_prompt` 使用 PromptSpec(instructions)，空字符串清空 Actor 指令配置，结构见[共享协议](../../../contracts/README.md)。

决策者自身训练可通过 Harness 本地调用，见 [learner](../learner/README.md)；当前没有自动自优化指令或触发机制，也没有统一接口覆盖 Actor 模型/prompt、决策者自身及 Harness 配置/代码这些优化目标。
