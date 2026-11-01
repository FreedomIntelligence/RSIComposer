# Decision policy

位于 [默认 Decision Harness](../README.md) 内部。`DecisionPolicy.decide` 选择操作、系统指令、等待或结束；Harness 组织执行与结果处理，[Decision Action Space](../../action_space/README.md) 定义操作参数和指令模式约束。

## 接口

[DecisionPolicy](__init__.py) 的同步接口 `decide(context: DecisionContext) → Decision` 接收目标、全部来源的 experience、自身 history 和 memories、操作定义、当前准备结果、最新反馈、directive_modes 和跨轮 candidates。默认 Harness 的 `experience_limit=32` 限制系统证据条数，`memory_limit=32` 分别限制本地 history 和工作记忆条数，两者必须为正整数；候选集合不受这两个 limit 截断。scope 仅标记证据来源，不是优化目标接口。

[Decision](../decision.py) 的 invoke / explore / wait 分别只能携带 operation / directive / job；finish 不携带这三项。四种决策都可附顶层 `reason` 字符串，省略时保留既有空字符串默认值；显式提供 null、数字、布尔、数组或对象均拒绝。wait 的任务由外部通过 `loop.harness.resume(job, result)` 回填，见 [Harness](../README.md)。

## 实现与替换

同文件的 `ModelDecisionPolicy(backend, model, *, allowed_actions=("echo",), directives=None, max_validation_attempts=1, structured_output=None)` 同步调用 ModelBackend.generate，校验返回的 ModelRef 与请求一致，再解析 JSON。allowed_actions 写入探索指令，限制 Actor 环境动作；directive_modes 描述系统指令模式。模型输出参数为：

- invoke：operation 和 arguments；具体名称与参数由 Decision Action Space 校验。
- explore：必填 strategy，mode 默认 execute。sft 需 recipe_ref，apply_result 默认 false；rl 需 recipe_ref，updates 默认 1，必须为正整数；evaluate 需 protocol_ref 和来自 context.candidates 的 candidate_id，也兼容 last_feedback.candidates；configure_prompt 需字符串 instructions。
- wait：可选 job 对象含 job_id、backend，省略时让出一个调度轮次；finish：无额外必填参数。

所有决策的可选顶层 `reason` 都保存在 `Decision.reason`，不会合并到 operation / directive 的 `arguments`。提示模型给出简短、可公开展示的决策依据：观察到的问题、为什么选择该动作、预期检查什么；这不是隐藏思维链，不读取或记录供应商原始 `reasoning_content`。`reason` 是模型的公开说明，不代表已经验证的事实。

`structured_output=None` 在后端明确声明 `supports_response_schema=True` 时自动启用原生 JSON Schema 约束；`False` 显式关闭，`True` 在不支持的后端上于构造时明确报错。`policy.structured_output` 保存解析后的布尔值。启用后，普通决策和复查请求分别携带 `GenerationRequest.response_schema=policy.decision_schema(context)` 和 `policy.review_schema(context)`，不改变原有提示、模型身份检查或执行权限。

两种 schema 均按当前上下文生成根 `oneOf`：每个可用操作各有一个 `kind="invoke"`、固定 `operation` 名称的分支；普通决策还为当前 `directive_modes` 中的每个 catalog 定义生成固定 `mode` 的 explore 分支，并保留 wait / finish。复查只包含当前操作和 `review_actions` 控制分支，不下发新指令。外层分支均设置 `additionalProperties=false`，操作和模式参数只放在嵌套的 `arguments`，不会把参数中的 `kind` 当成决策类型。自定义 `input_schema` 深拷贝后完整嵌入，保留其内部约束，不强制内部对象关闭额外属性，也不规定研究步骤或顺序。`reason` 仍可选，explore 保留必填 `strategy`、可选 `input_ids` 和正数或 null 的 `review_after_seconds`，wait 保留可选 job，复查保留可选正数或 null 的 `next_review_seconds`；不接受输入引用的模式只允许省略或空 `input_ids`。

嵌入会改变 schema 的文档根，因此当前实现对输入 schema 中的文档内 `$ref`、`$dynamicRef`、`$recursiveRef`（空引用或以 `#` 开头，包括 JSON Pointer 和锚点）在请求模型前明确报错，不静默生成错误引用。可提供不使用文档内引用的 schema，或显式关闭 structured output；schema 的 `const` / `enum` / `default` / `examples` 中的引用字样属于数据，不受此限制。原生约束只负责生成格式；Action Space 校验、候选与输入引用检查、后端约束和权限检查仍按原边界执行。

普通决策和复查共用严格 JSON 解析：同一对象中的重复键一律拒绝，包括 `arguments`、数组内对象等任意嵌套层级；不会选择最后一个值或修复歧义。不同对象可分别拥有同名键，例如外层 `kind="invoke"` 与 `arguments.kind="agenda"`。关闭 structured output 时也保留此检查。

`max_validation_attempts` 为 1–8 的整数，表示普通 decide 的输出校验尝试总数，默认 1。
仅 JSON 解析、提案结构或参数 schema 校验错误会带错误提示重新请求模型；传输失败、模型身份
错误直接抛出，不重试工具执行或外部副作用。medical 示例显式使用 3 次；它是领域装配的选择。
非默认预算需显式快照/恢复契约，内置 snapshot / durable 装配不会静默丢弃此配置。

系统指令的描述、schema 与参数解析由注入的 `DirectiveDefinitions` 提供；ModelDecisionPolicy 和 Decision Action Space 应使用同一个 catalog。请求只包含当前 directive_modes 允许的定义。模型将模式参数放入 `arguments`，例如：

```json
{"kind":"explore","mode":"configure_prompt","strategy":"short answers","arguments":{"instructions":"Be concise."},"reason":"回答过长；调整提示，检查答案长度与质量。"}
```

仍接受旧的顶层参数形式，但同一响应不能混用顶层模式参数和 arguments。顶层 `reason` 是公共元数据，不视为模式参数，两种参数形式都可与其共存；不带 reason 的旧输出继续兼容。新增模式不需要修改 policy 的 JSON 分派代码，详见 [Action Space](../../action_space/README.md#扩展系统指令)。

parser 的候选按 candidate_id 合并去重，以 context.candidates 为准，最新反馈补齐缺失项。早期训练候选在后续训练或不携带候选的评测反馈之后仍可引用；未知 ID 仍被拒绝。

explore 的输入行为由 [DirectiveSpec](../../action_space/directives.py) 的 `accepts_input_refs: bool = True` 声明。默认 True 时，省略 input_ids 会使用全部当前准备产物；显式提供时必须是产物 ID 列表，只能引用当前准备结果或最新反馈的 artifact_refs，空列表表示不传输入。声明 False 时，省略或 `input_ids=[]` 都产生空 input_refs，显式非空输入会被拒绝；已配置输出纠错预算时，该校验错误可进入有界纠错。执行端仍须校验实际输入，声明不替代后端约束。

指令的 simulated 标记合并模型响应与实际选中输入的来源标记；未选中的模拟准备产物不会污染真实输入，模拟反馈中的已选产物也不会丢失标记。

explore 可附顶层 `review_after_seconds`，不放入模式 arguments。`ModelDecisionPolicy.review(context)` 使用独立请求读取当前执行视图，可返回 `{"kind":"invoke","operation":name,"arguments":{...},"reason":"检查当前指标，确认是否出现退化。"}` 调用 context.operations 中当前获准的操作。复查 invoke 同样保留可选顶层 `reason` 到返回的 `Decision.reason`，不改变工具权限或参数约束。Harness 将结果追加到 operation_results，其中 data 是结构化内容、artifacts 是引用，然后再次调用 policy。工具为空时必须直接判断，不允许编造工具或下发新指令。

最终判断为 `{"action":"continue","next_review_seconds":30,"reason":"..."}`，其可选 reason 同样必须为字符串，省略时为空字符串；可选控制由 context.review_actions 提供（continue 始终可选，cancel 可被外部权限关闭），None 停止自动复查。自定义 policy 返回相应的 Decision 或 ReviewDecision，也可只保留 decide。查询次数预算、调用时机、反馈归因和写入重试由 [Harness](../README.md) / [Runner](../../../../runtime/README.md) 管理。

自定义策略实现 `decide` 并通过默认 Harness 的 policy 参数注入，不要求使用 JSON。需要应用学习结果时，另实现 [ModelUpdatable](../../../contracts/model_service.py) 的 model 与 update_model。

内置 update_model 只更改当前策略的模型引用，拒绝切换 backend_id。explore 的系统指令交给 Exploration Harness；训练/评测仍需配置对应后端。`configure_prompt` 使用 PromptSpec(instructions)，空字符串清空 Actor 指令配置，结构见[共享协议](../../../contracts/README.md)。

决策者自身训练可通过 Harness 本地调用，见 [learner](../learner/README.md)；当前没有自动自优化指令或触发机制。SystemVersion / Candidate 可描述命名组件变更，具体组件校验和启用仍由消费方及 SystemActivator 实现，不自动执行任意代码修改。
