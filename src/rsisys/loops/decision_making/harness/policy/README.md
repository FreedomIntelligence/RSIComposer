# Decision policy

位于 [默认 Decision Harness](../README.md) 内部。`DecisionPolicy.decide` 选择操作、系统指令、等待或结束；Harness 组织执行与结果处理，[Decision Action Space](../../action_space/README.md) 定义操作参数和指令模式约束。

## 接口

通用策略协议位于 [contracts/decision.py](../../../contracts/decision.py)：
`decide(DecisionInput) -> Decision`，输入为目标、context 文本、能力描述和可选引用。
使用 [ContextDecisionPolicy](../../../../adapters/decision.py) 接入默认 harness，
可以是普通函数，也可以使用自己的 prompt、原生 tool calling 和 SDK。
默认 `DecisionContext` 及下文的字段属于内置实现；第三方策略无须照搬其组织方式。

[DecisionPolicy](__init__.py) 的同步接口 `decide(context: DecisionContext) → Decision` 接收目标、全部来源的 experience、自身 history 和 memories、操作定义、当前准备结果、最新反馈、directive_modes 和跨轮 candidates。默认 Harness 的 `experience_limit=32` 限制系统证据条数，`memory_limit=32` 限制工作记忆 recall 条数，本地 history 由 token 预算管理，两者必须为正整数；候选集合不受这两个 limit 截断。scope 仅标记证据来源，不是优化目标接口。

[Decision](../decision.py) 的 invoke / explore / wait 分别只能携带 operation / directive / job；finish 不携带这三项。四种决策都可附顶层 `reason` 字符串，省略时保留既有空字符串默认值；显式提供 null、数字、布尔、数组或对象均拒绝。wait 的任务由外部通过 `loop.harness.resume(job, result)` 回填，见 [Harness](../README.md)。

### 决策身份、依赖与采纳

`Decision` 尾部增加系统默认生成 UUID 的 `decision_id`，以及 `depends_on_decision_ids: tuple[str, ...]`、`source_decision_ids: tuple[str, ...]`。模型仅可在 JSON 顶层请求后两项（字符串数组，省略为空）；原生 schema 保持 `additionalProperties=false`，不接受模型提供 decision_id、adopted_candidate_ids、adopted_decision_ids 或 confirmed_edges。复查 invoke 使用相同规则。

`DecisionContext.decision_graph` 是轻量身份/状态及非空关系投影，不复制 reason、错误文本或完整边列表；`candidate_origins` 映射候选 ID 到其 producer decision。它们不依赖截断后的 history 顺序。执行前，依赖必须指向已知且 completed 的决策；引用可以指向已知的 pending/failed 决策。未知、前向、自指和重复引用会被拒绝，未满足依赖时不调用执行器，失败验证保留在 history。没有隐式队列：模型可以 wait 后重新提出决策。现有并行 Runner 和 max_outstanding 不变。

普通决策和 review invoke 还支持可选顶层 `research_attempt_id` / `research_attempt_title`。
仅给 title 请求新 attempt，由 Harness 创建 UUID；给 decision_graph 中已知的 ID 继续同一想法，
省略 title 时仍记录 canonical title，提供时不能改名。ID 最长 200 字符，title 最长 500 字符，
默认分别为 None / 空字符串；省略两项不阻止执行。默认提示要求具体行动、参数和验证计划，
不要求模型维护分组；启用 feedback learning 时，由反馈整理模型增量构建独立的 Attempt 解释视图。
显式归属仍用于兼容：同一运行内可复用已有 ID，竞争方案可独立分组。解释视图的 ID 不写入
原始执行归属字段。解析器不分配 ID，
不把归属字段传入 arguments；Harness 接纳后在 ledger 与 history 顶层记录解析后的归属，
原始 proposal 不变。新旧 snapshot、重试及并行细节见 [Harness](../README.md)。

`OperationResult` 和 `Feedback` 尾部的 `adopted_candidate_ids` 是执行器回执，必须是去重的非空字符串 tuple，且仅 COMPLETED 回执可以非空。成功 producer feedback 建立 origin；evaluate/apply_candidate 不建立或覆盖 origin。Harness 保留真实成功的采纳回执，并只从已知成功 origin 派生 `adopted_decision_ids` 及 confirmed adopted 边；旧快照或外部候选缺少 origin 时记入 `unresolved_adopted_candidate_ids`，不猜来源，也不把实际成功改成失败。一次决策可以采纳多个来源。

`DecisionState.decision_records` 保存完整 ledger，`directive_decisions` 关联指令，`candidate_origins` 保存 producer 索引，`pending_operation_decision_id` 关联异步操作；`LoopStepResult.decision_id` 将步骤与同一决策关联。普通 decision history 的 record_id 使用 decision_id，operation/feedback history 的 details 显式包含 decision_id 和确认后的关系。操作执行与写账分开，写入重试复用冻结回执，JSON 恢复不改变判等语义；相同身份和内容的决策不重复执行，不同内容复用身份被拒绝。外部 JobRef 复用按 `(job, decision_id)` 区分；resume 的写账重试安全，完成后重复 resume 继续按既有接口报错，不重新执行。复查写操作同样注册身份、验证依赖并保存关联回执。新快照保存这些字段，旧快照只对明确新增字段使用空默认值，不从历史猜测缺失来源。

复查异步 job 的提交回执独立保存在对应 ledger 记录中，不占用主 `pending_job`，也不阻止后续普通决策或替换当前实验。提交仅标记 pending；只有显式 `resume(job, result, decision_id=...)` 完成后才能满足依赖或确认采纳。无歧义的独立 review job 可以省略 decision_id；同一 JobRef 对应多个 review 决策时必须显式指定。主 pending job 的既有无关键字 resume 语义不变，无后台轮询或额外 Runner 调度。

## 实现与替换

开放式 Lab 通过 `lab_factory` 把原生工具描述传入 policy 的 operations；
directive_modes 为空时仍能通过 invoke 编写、修改、执行研究代码。
内置 research / decision prompt 明确允许在已授予权限内修改 SFT、RL、evaluate 实现；
不同评测协议的分数不能直接比较，host 的目标与验收权限不由模型改写。
见 [M04 接线与验证](../../../../../../docs/review/open-lab-execution-20261104.md)。

通用模型提示文本通过 [DecisionPrompts](prompts.py) 装配。九个内置 [prompts](prompts)
分别包含 research、decision、review、summary、control、scheduling、可选/必需 Attempt 与
validation_correction。可传入 `prompts=DecisionPrompts({'research': custom_text})` 替换单项；
`$name` 是模板变量，JSON 花括号无需转义。构造时捕获文本、拒绝未知模板/变量，
`manifest()` 返回各项 SHA256。最终 request 是实际生效内容的记录。
MemoryPrefix policy 继承此入口，医疗 ResearchPolicy 支持同一参数并保留领域控制 hook。
覆盖 prompt 不修改动作 schema、身份检查、权限或执行准入；自定义文本不能借此安装能力。
默认本地 snapshot 对非默认 prompt bundle 要求显式恢复契约，避免静默丢失指令。
完整逻辑、模式矩阵及渲染示例见 [M03 审查](../../../../../../docs/review/m03.md)。

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

`context_compactor` 默认使用 80k→24k 的 `BasicContextCompactor`，在最终决策/review请求构造后
计量，包含协议和能力定义。摘要调用也经过 backend 并记录 usage；最新完整工具结果不被删除。
`context_scope(...)` 在一次调用中选择显式配置的 policy compactor，否则使用 harness 默认；
不覆盖配置对象，不在 harness 预先重复压缩，`recent_limit` 与宿主 memory_limit 一致。
DSH 使用 Identity，由原生会话管理上下文。
`track_usage(...)` 让宿主观察每份响应，透明包装转发该方法后保留相同计数。
`share_backend(wrap)` 显式更新实际持有的 backend，供可选反馈共享串行模型窗口；
这些能力都不属于最小 decide 接口。独立配置的压缩器需要显式 snapshot/recovery 适配。
