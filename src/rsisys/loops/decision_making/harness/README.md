# Decision Harness

[DecisionHarness](__init__.py) 是可整体替换的决策执行边界，通过 `step(objective, action_space)`、`pending()` 和 `receive_feedback(directive, feedback)` 服务 [DecisionMakingLoop](../README.md)。协议不要求 memory、tools 或特定 policy 属性。

## 默认实现

[DefaultDecisionHarness](engine.py) 接收 `policy, tools, experience, learner=None`，可选关键字参数包括 `experience_limit=32, memory=None, memory_limit=32, review_tool_limit=4`。缺省 memory 使用 DecisionMemory；前两个 limit 为正整数，限制上下文条数而非 token。review_tool_limit 是单次复查最多执行的查询数，允许为 0。

默认流程为读取目标、全部来源的系统反馈与本地上下文 → policy 选择 Decision → Action Space 校验 → tools 执行本地操作或发布 Directive → 接纳反馈。同步准备后可继续推进；发布指令后等待反馈，不能在交付前生成下一条指令。`receive_feedback` 清空本轮准备结果、保留最新反馈，最近一次完全相同的交付可幂等重试。

`DecisionState.candidates` 按 candidate_id 保存历轮已接收候选，`DecisionContext.candidates` 提供其深拷贝 tuple；该字段追加在原上下文字段之后，默认空 tuple，兼容旧位置参数。候选不受 experience_limit / memory_limit 截断，评测等不含候选的反馈不会清空它们。同 ID 同内容去重；同 ID 不同内容（含单次反馈内冲突）在写入 history / Experience 前拒绝。两份反馈记录成功后才更新候选和交付状态，写入失败可用原反馈重试。

`DecisionContext.execution` 是探索侧当前模型与进度，`decision_model` 是当前决策模型；`directive_provenance` 保留每条已发出指令当时观察到的两侧模型引用，后续角色更新不改写旧归因。`candidate_evaluations` 按候选提供最近一次评测的协议、输入引用、指标、状态与证据；失败或另一协议的评测也会替换该摘要，完整记录留在 Experience。它们不受短 history 截断，也不推断未知模型版本。

运行中，DecisionMakingLoop 优先调用可选的 `review_with_tools(objective, progress, action_space)`；外部 Harness 仍可只实现原来的 `review(objective, progress)`；已有子类覆盖该旧方法时仍优先保留其行为，显式覆盖 review_with_tools 可选择新能力。默认实现向 policy 提供标记为 read_only 的操作，支持「查询实验 → 读取结果 → 再查询或判断」。每次查询使用现有 tools 绑定，必须立即返回 OperationResult；错误作为查询结果供模型判断。查询预算用完后再给一次不含工具的最终判断机会。这个预算限制次数，不限制外部调用耗时，后端应自行设置超时。

`policy.review(context)` 可以返回 `Decision(kind="invoke", ...)` 请求查询，或返回最终 ReviewDecision。最终仍只允许继续或取消，不发布新指令；取消由 Runner 执行。read_only 是接入者对工具行为的声明，不能隔离一个实际会写入的自定义工具。查询正文留在本地 history，最终复查摘要和查询记录引用进入 Experience。内置 ModelDecisionPolicy 在收到响应时记录 usage，包含解析失败和内部重试；自定义 policy 按返回结果统计 usage。查询结果、记录 ID 和剩余预算暂存于 review session，查询记录或最终复查写入失败只补写，不重新执行已完成的查询；这不提供崩溃恢复。

未实现 review 的 policy 返回不再复查的 continue。任务终态先到时补齐待写证据后接收结果，不执行过时的取消建议。直接调用 `harness.review()` 不传工具，需使用 loop.review 或 review_with_tools 才能启用查询。频率设置见 [Runtime](../../../runtime/README.md)。

## 内部组件

| 组件 | 默认职责 |
|---|---|
| [policy](policy/README.md) | 构建模型请求，选择操作或系统指令 |
| [memory](memory/README.md) | research 历史、工作记忆与决策记录 |
| [tools](tools/README.md) | 执行已校验的本地 research 操作 |
| [learner](learner/README.md) | 决策者自身训练后端 |
| [decision.py](decision.py) | Decision 与推进结果 |
| [state.py](state.py) | 上下文、生命周期状态与待写结果 |

这些子组件用于定制默认实现，不构成框架必须独立提供的接口。Action Space 位于 [../action_space/](../action_space/README.md)，与 Harness 平级；可保留 Action Space 并替换整个 Harness，替代实现可以没有 memory。

## 本地控制与记录

异步 tools 返回 JobRef；外部通过 `loop.harness.resume(job, result)` 回填终态 OperationResult。job 必须匹配当前等待任务，不自动轮询或重做。操作已执行但 history/Experience 写入失败时保留结果与稳定 ID，后续 step 或相同回填只补写，待补写期间拒绝不同结果并禁止快照；暂存不支持崩溃恢复。

`loop.harness.submit_learning` / `poll_learning` / `cancel_learning` / `apply_learning` 控制决策者自身训练，不自动执行 Actor 任务或触发周期优化。训练任务与模型应用约束见 learner README。默认训练为 Mock，未实现真实梯度更新。

系统反馈写入共享 [Experience](../../../experience/README.md)，原始准备过程留在默认 Harness 的本地 history。自定义 Harness 的[快照](../../../snapshot/README.md)需显式集成。
