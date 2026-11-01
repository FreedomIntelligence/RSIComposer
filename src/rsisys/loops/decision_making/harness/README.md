# Decision Harness

[DecisionHarness](__init__.py) 是可整体替换的决策执行边界，通过 `step(objective, action_space)`、`pending()` 和 `receive_feedback(directive, feedback)` 服务 [DecisionMakingLoop](../README.md)。协议不要求 memory、tools 或特定 policy 属性。

## 默认实现

[DefaultDecisionHarness](engine.py) 接收 `policy, tools, experience, learner=None`，可选关键字参数包括 `experience_limit=32, memory=None, memory_limit=32, review_tool_limit=4`。缺省 memory 使用 DecisionMemory；前两个 limit 为正整数，限制上下文条数而非 token。review_tool_limit 是单次复查最多执行的操作数，允许为 0。

默认流程为读取目标、全部来源的系统反馈与本地上下文 → policy 选择 Decision → Action Space 校验 → tools 执行本地操作或发布 Directive → 接纳反馈。普通决策与运行中 review 共用 tools 中的操作执行入口：参数校验与授权 → 工具调用 → 结果归一化 → 固定回执记录。Harness 只决定后续等待、继续研究或交付，不再各维护一套调用逻辑。同步准备后可继续推进；发布指令后等待反馈，不能在交付前生成下一条指令。`receive_feedback` 清空本轮准备结果、保留最新反馈，最近一次完全相同的交付可幂等重试。

`DecisionState.candidates` 按 candidate_id 保存历轮已接收候选，`DecisionContext.candidates` 提供其深拷贝 tuple；该字段追加在原上下文字段之后，默认空 tuple，兼容旧位置参数。候选不受 experience_limit / memory_limit 截断，评测等不含候选的反馈不会清空它们。同 ID 同内容去重；同 ID 不同内容（含单次反馈内冲突）在写入 history / Experience 前拒绝。两份反馈记录成功后才更新候选和交付状态，写入失败可用原反馈重试。

`DecisionContext.execution` 是探索侧当前模型与进度，`decision_model` 是当前决策模型；`directive_provenance` 保留每条已发出指令当时观察到的两侧模型引用，后续角色更新不改写旧归因。`candidate_evaluations` 按候选提供最近一次评测的协议、输入引用、指标、状态与证据；失败或另一协议的评测也会替换该摘要，完整记录留在 Experience。它们不受短 history 截断，也不推断未知模型版本。

运行中，DecisionMakingLoop 优先调用可选的 `review_with_tools(objective, progress, action_space)`；外部 Harness 仍可只实现原来的 `review(objective, progress)`；已有子类覆盖该旧方法时仍优先保留其行为，显式覆盖 review_with_tools 可选择新能力。默认实现根据外部 ActionPermissions 提供当前可用操作，支持「执行操作 → 读取结果 → 再操作或判断」。默认权限只开放只读操作，可改为允许列表或全部能力，Harness 不硬编码科研步骤。每次操作在执行前重新检查权限；错误作为操作结果供模型判断。调用预算用完后再给一次不含工具的最终判断机会。这个预算限制次数，不限制外部调用耗时，后端应自行设置超时。

`policy.review(context)` 可以返回 `Decision(kind="invoke", ...)` 请求操作，或返回最终 ReviewDecision。最终仍只允许继续或取消，不发布新指令；取消由 Runner 执行。read_only 仅是能力属性，具体权限属于 Action Space 外部配置；操作权限不能隔离工具内部的实际副作用。查询正文留在本地 history，最终复查摘要和查询记录引用进入 Experience。内置 ModelDecisionPolicy 在收到响应时记录 usage，包含解析失败和内部重试；自定义 policy 按返回结果统计 usage。查询结果、记录 ID 和剩余预算暂存于 review session，查询记录或最终复查写入失败只补写，不重新执行已完成的查询；这不提供崩溃恢复。

未实现 review 的 policy 返回不再复查的 continue。任务终态先到时补齐待写证据后接收结果，不执行过时的取消建议。直接调用 `harness.review()` 不传工具，需使用 loop.review 或 review_with_tools 才能启用查询。频率设置见 [Runtime](../../../runtime/README.md)。

## 内部组件

| 组件 | 默认职责 |
|---|---|
| [policy](policy/README.md) | 构建模型请求，选择操作或系统指令 |
| [memory](memory/README.md) | research 历史、工作记忆与决策记录 |
| [tools](tools/README.md) | 操作绑定，以及普通决策 / review 共用的执行与回执记录 |
| [learner](learner/README.md) | 决策者自身训练后端 |
| [decision.py](decision.py) | Decision 与推进结果 |
| [state.py](state.py) | 上下文、生命周期状态与待写结果 |

这些子组件用于定制默认实现，不构成框架必须独立提供的接口。Action Space 位于 [../action_space/](../action_space/README.md)，与 Harness 平级；可保留 Action Space 并替换整个 Harness，替代实现可以没有 memory。

## 本地控制与记录

异步 tools 返回 JobRef；外部通过 `loop.harness.resume(job, result)` 回填终态 OperationResult。job 必须匹配当前等待任务，不自动轮询或重做。操作已执行但 history/Experience 写入失败时保留结果与稳定 ID，后续 step 或相同回填只补写，待补写期间拒绝不同结果并禁止快照；暂存不支持崩溃恢复。

`loop.harness.submit_learning` / `poll_learning` / `cancel_learning` / `apply_learning` 控制决策者自身训练，不自动执行 Actor 任务或触发周期优化。训练任务与模型应用约束见 learner README。默认训练为 Mock，未实现真实梯度更新。

系统反馈写入共享 [Experience](../../../experience/README.md)，原始准备过程留在默认 Harness 的本地 history。自定义 Harness 的[快照](../../../snapshot/README.md)需显式集成。

运行中操作返回 JobRef 时，Harness 将它作为 submitted 回执连同作业引用交给模型并记录 history，不宣称后台任务已完成，也不替换当前实验指令。后续查询/取消由该能力的 provider 或调用方负责；框架没有因此获得多实验调度器。普通决策阶段的异步操作仍使用原有 wait/resume 路径。

`max_outstanding=1` 默认保持等待单条反馈的行为；配置为大于 1 后，未终态时仍可研究、
调用工具或提交下一实验。`DecisionContext.outstanding` 和 `executions` 按指令保留
独立意图与执行视图，反馈、review 重试和候选来源使用各自 ID。满容量时不再向模型
展示 directive_modes，执行端仍检查容量。`Decision('wait')` / 模型 JSON
`{"kind":"wait"}` 只让出一次调度机会，不设置外部 pending_job；已有带 job 的 wait
仍保留显式 resume 的外部操作语义。finish 停止新决策，runner 继续收齐在途实验。
`can_step()` 向 runner 提供可选的决策就绪信号；max_outstanding=1 的在途反馈保护
不会阻止并行 runner 推进或取消已提交实验。直接调用 step 的默认串行保护保持不变。

`compare_candidates(ids, metric=..., higher_is_better=True)` 显式排序评测结果。
要求已完成、有有限数值指标、相同协议/输入/模拟标记，以及已知相同完整训练 ModelRef。
仅 Candidate.base_version 字符串相同不够，未知完整训练来源会拒绝比较。
当前索引保留每候选最新评测；历史原始证据仍在 memory/Experience。

`rsisys.loops.exploration.control.ExperimentControls(exploration_loop, decision_loop)`
提供 `definitions` / `bindings` / `capabilities`，注册到现有 Action Space 和 Tools 即可
供 researcher 调用 `experiment_inspect/cancel/update/compare`。取消和更新是写操作，
按已有 operations / review_operations 权限检查，默认 review 只开放只读工具。
这些能力不自动安装，也不替代 backend 的版本/字段验证。

`feedback_selector=None` 默认使用 `RecentFeedbackSelector`。
`_context` 调用 `select(experience, objective, limit=...)` 并复制输出，避免自定义 selector
把可变记录引用直接暴露给 policy。

外部运行控制通过 `update_run_control(context)` 提供预算、模式与轮次提示，模型从
`DecisionContext.run_control` 读取；它不改写研究目标。`restart_research(context)`
只在研究者已 finish 且全部指令、工具作业和待写反馈已收尾时开启下一轮，保留
历史、记忆、模型、候选和累计计数。以 `research_round` 和完整 context 做幂等校验，
响应丢失时重试同一请求。`pending_research_job()` 只报告等待外部 resume 的工具作业，
不声称能够取消任意工具。是否续跑及停止条件由 [RunPolicy](../../../runtime/README.md) 决定。
