# Decision Harness

[DecisionHarness](__init__.py) 是可整体替换的决策执行边界，通过 `step(objective, action_space)`、`pending()` 和 `receive_feedback(directive, feedback)` 服务 [DecisionMakingLoop](../README.md)。协议不要求 memory、tools 或特定 policy 属性。

## 默认实现

[DefaultDecisionHarness](engine.py) 接收 `policy, tools, experience, learner=None`，可选关键字参数包括 `experience_limit=32, memory=None, memory_limit=32, review_tool_limit=4`。缺省 memory 使用 DecisionMemory；前两个 limit 为正整数，限制上下文条数而非 token。review_tool_limit 是单次复查最多执行的操作数，允许为 0。

另有 `context_compactor=None, research_agenda=None`，缺省分别创建
`BasicContextCompactor()` 与 `ResearchAgenda()`。它们是默认 Harness 的内部组件，
不扩展 `DecisionHarness` 协议，也不要求外部实现采用固定科研流程。

## 短上下文与完整档案

[BasicContextCompactor](context.py) 的 `compact(context, *, recent_limit=32)` 是纯投影：
保留原始 memory/history、工具回执及研究笔记，不写回压缩结果，不修改候选或评测。
默认 `recent_tail=8, stream_chars=16000, summary_chars=6000`。history、memories、
experience、operation_results 各自最多保留 8 条近期完整记录，且每个序列的紧凑 JSON
长度最多 16000 个字符；history/memories 还受 `memory_limit` 限制。memory 按 recall
的新到旧顺序保留，其他序列按旧到新保留连续尾部。

超过条数或字符预算时自动生成 `DecisionContext.context_summary`：包含覆盖条数、
首尾来源标签、输入记录 SHA-256 及少量明确标记为不完整的摘录。history 来源覆盖
完整历史，其他来源覆盖本次选中的上下文记录。单条记录过大时也进入摘要，不能把
截断后的工具数据伪装成完整结果。来源标签过长时显式标记截短；完整 ID 仍在原档案。
摘要只用于导航，不是科学事实、证据验证或自动形成的结论。读原始证据后再判断。
每个来源带 `read_path`（操作名、collection 和 ID 取法）；operation_results 使用
内容 SHA-256 作为稳定 ID，映射回完整工具结果和 history_refs，experience 使用
原始 record_id。自定义 selector 生成的派生 experience 可能不在原始 store 中，
这时 archive 返回 `found=false`，应继续读取摘录保留的 source_refs。

普通决策和每次 review 模型调用都会应用投影。`IdentityContextCompactor()` 可显式
关闭压缩；也可注入任意实现同一 `compact` 方法的对象。摘要是可重建的，没有持久化
或额外 LLM 调用。默认实现会读取并扫描完整 history，因此限制的是模型看到的历史
投影，不是档案存储或读取成本，也不是总 prompt/token 预算：目标、工具 schema、
live execution、候选等结构不受这里截断。JSON 外层和提供商编码还会增加请求长度。

## 模型自管的议程、待办和阶段总结

[ResearchAgenda](research.py) 保存模型选择的稳定 ID；kind 为 `agenda`、`todo` 或
`stage_summary`，status 是自由文本，框架不推进阶段、不替模型选择实验。
`DecisionContext.research_agenda` 只提供版本、总数和最多 6 条近期变更的导航视图，
正文最多 400 字符、证据引用最多 2 条；完整内容和所有修订在 store 中保留。

`patch(expected_revision=..., upserts=[...], remove_ids=[...], reason=..., evidence_refs=[...])`
要求显式版本、修改理由和证据引用（空列表表示未提供证据，不表示验证成功）。新 ID
必须给 kind/text；已有 ID 只改所给字段，未列出的 ID 和字段保留。每次最多 32 项
修改，重复 ID、未知删除项、过时 revision 或后续任一项校验失败均拒绝整次更新。
更新后的 item 带 revision，完整 patch log 保留旧文本、删除记录、理由和证据。
失败的持久化写入不提交内存状态；这与现有文件存储一样是单 writer 协议，不协调
多个进程。`ResearchAgenda(path)` 可使用现有原子 JSON 写入，并在启动时校验回放。

`read(view='items'|'revisions', kind=None, limit=20, cursor=None)` 返回页、total、revision
和 next_cursor；cursor 绑定版本及筛选，状态变动后需重新开始分页。返回值与内部
状态分离。`export()` 返回完整 JSON-compatible 档案，`restore(payload)` 先回放并
验证一致性，成功后替换内存状态；restore 不写文件，供 recovery/snapshot 显式接入。

[ResearchTools](tools/research.py) 复用已有 Capability/ActionSpace/BoundDecisionTools：

- `research_read`：读取笔记或修订，默认 4 条，最多 32 条。
- `research_patch`：显式写操作，遵守现有 operation/review 权限。
- `research_archive`：只读完整 history、memories、research_items 或 research_revisions，
  可按 record_id 查询，返回每块最多 1000 字符的 JSON 文本，默认每页 8 块、最多 32 块。
  同一 record 的块连接后再解析；research_revisions 的 record_id 为版本数字字符串。
  历史分页固定最初的记录前缀和摘要，所以读取操作自己的新增回执不打断分页；原有
  记录或可变 memory 改动会使 cursor 失效。大笔记/修订可用小块读取，避免单个大页
  再次被上下文投影省略。工具回执继续写入原始 history，不覆盖短导航视图。

archive 另支持 `operation_results`，从原始 operation history 回执中按内容摘要
读取结果；当构造 `ResearchTools(..., experience=experience_system)` 时还开放
`experience` collection，通过注入的公开 retrieve API 读取，不检查 provider 内部。

默认 review 权限只允许 read/archive；只有外部显式允许写操作时才能在 review 中
patch。模型仍通过现有 `Decision(kind='invoke', operation=...)` 更新笔记，不新增
policy 解析格式。一次 patch 成功而回执写入失败时，Harness 的既有待写机制只补写
同一回执，不重新执行 patch。
若调用方丢失 patch 响应后自行重发旧 revision，store 会明确拒绝而不是二次应用；
应先读当前 revision/log 判断结果。这不是 request-ID 响应去重接口，跨进程不确定
执行仍交给外部 durable recovery 的 intervention/reconciliation 机制。

### Assembly 接线示例

构建默认 Harness 前创建共享的 memory/agenda，将下面的 definitions/bindings 与
既有能力合并；不要另外创建一份同名 memory。此组件不自动发现或注册工具。

```python
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.tools.research import ResearchTools

memory = DecisionMemory(memory_path)
agenda_path = context.data_dir / 'decision_research.json' if context.data_dir is not None else None
agenda = ResearchAgenda(agenda_path)
research_tools = ResearchTools(agenda, memory, experience=context.experience)
definitions = (*definitions, *research_tools.definitions)
bindings = {**bindings, **research_tools.bindings}
# Existing ActionSpace receives extra_definitions=definitions;
# existing Tools receives extra_bindings=bindings.
# DefaultDecisionHarness(..., memory=memory, research_agenda=agenda)
```

也可使用 `Capabilities.from_provider(research_tools)`。恢复/快照集成应保存
`harness.research_agenda.export()` 和 compactor 配置，恢复到绑定给工具的同一 agenda
实例；不要替换实例而留下过时的 tool binding。DSH 等替代 Harness 可共享这两实例，
或显式选择自己的上下文管理。

默认流程为读取目标、全部来源的系统反馈与本地上下文 → policy 选择 Decision → Action Space 校验 → tools 执行本地操作或发布 Directive → 接纳反馈。普通决策与运行中 review 共用 tools 中的操作执行入口：参数校验与授权 → 工具调用 → 结果归一化 → 固定回执记录。Harness 只决定后续等待、继续研究或交付，不再各维护一套调用逻辑。同步准备后可继续推进；发布指令后等待反馈，不能在交付前生成下一条指令。`receive_feedback` 清空本轮准备结果、保留最新反馈，最近一次完全相同的交付可幂等重试。

`DecisionState.candidates` 按 candidate_id 保存历轮已接收候选，`DecisionContext.candidates` 提供其深拷贝 tuple；该字段追加在原上下文字段之后，默认空 tuple，兼容旧位置参数。候选不受 experience_limit / memory_limit 截断，评测等不含候选的反馈不会清空它们。同 ID 同内容去重；同 ID 不同内容（含单次反馈内冲突）在写入 history / Experience 前拒绝。两份反馈记录成功后才更新候选和交付状态，写入失败可用原反馈重试。

`DecisionContext.execution` 是探索侧当前模型与进度，`decision_model` 是当前决策模型；`directive_provenance` 保留每条已发出指令当时观察到的两侧模型引用，后续角色更新不改写旧归因。`candidate_evaluations` 按候选提供最近一次评测的协议、输入引用、指标、状态与证据；失败或另一协议的评测也会替换该摘要，完整记录留在 Experience。它们不受短 history 截断，也不推断未知模型版本。

运行中，DecisionMakingLoop 优先调用可选的 `review_with_tools(objective, progress, action_space)`；外部 Harness 仍可只实现原来的 `review(objective, progress)`；已有子类覆盖该旧方法时仍优先保留其行为，显式覆盖 review_with_tools 可选择新能力。默认实现根据外部 ActionPermissions 提供当前可用操作，支持「执行操作 → 读取结果 → 再操作或判断」。默认权限只开放只读操作，可改为允许列表或全部能力，Harness 不硬编码科研步骤。每次操作在执行前重新检查权限；错误作为操作结果供模型判断。调用预算用完后再给一次不含工具的最终判断机会。这个预算限制次数，不限制外部调用耗时，后端应自行设置超时。

`policy.review(context)` 可以返回 `Decision(kind="invoke", ...)` 请求操作，或返回最终 ReviewDecision。最终仍只允许继续或取消，不发布新指令；取消由 Runner 执行。read_only 仅是能力属性，具体权限属于 Action Space 外部配置；操作权限不能隔离工具内部的实际副作用。查询正文留在本地 history，最终复查摘要和查询记录引用进入 Experience。内置 ModelDecisionPolicy 在收到响应时记录 usage，包含解析失败和内部重试；自定义 policy 按返回结果统计 usage。查询结果、记录 ID 和剩余预算暂存于 review session，查询记录或最终复查写入失败只补写，不重新执行已完成的查询；单独使用 Harness 不提供崩溃恢复；可在 Application 中启用 durable 运行边界。

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

异步 tools 返回 JobRef；外部通过 `loop.harness.resume(job, result)` 回填终态 OperationResult。job 必须匹配当前等待任务，不自动轮询或重做。操作已执行但 history/Experience 写入失败时保留结果与稳定 ID，后续 step 或相同回填只补写，待补写期间拒绝不同结果并禁止快照；单独使用 Harness 时暂存仅在进程内；durable Application 可保存回执并在中断后要求核对。

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


## 可选 DSH Harness

[dsh.py](dsh.py) 的 `DeepSeekDecisionHarness` 使用 DSH 持久会话和原生上下文压缩，
复用默认操作回执、反馈和候选协调，不为科研步骤增加固定顺序。
`dsh_decision_factory(DSHConfig(...))` 可直接传给 Application / build_demo 的 decision_factory。
可选依赖与实际使用示例见[使用指南](../../../../../docs/GUIDE.md#使用-dsh-harness)。

模型推理在单独 worker 中进行；step 在等待期间返回 waiting，收到提案后在 Runner 线程
执行并重新校验当前权限。运行中 review 与决策共用一个会话，不并发写同一原生历史。
结束的 DSH turn 不等于整个研究结束；RunPolicy 可要求续研。
`cancel_research()` 只关闭自己拥有的 DSH 推理 runtime，不宣称取消其他异步 tools。

附带 [dsh.patch.yml](dsh.patch.yml) 针对固定 SDK 版本关闭 native shell / 文件工具，
研究动作通过 RSISys 的操作 / 指令协议执行；这是能力边界，不是宿主沙箱。
这些约束对应附带的 profile；自行修改 DSH home/profile 的插件须另外对齐权限。
原生压缩保留，但任意 DSH native tools 自动映射、DSH 自身权重训练/切换暂未实现。
SDK native 事件可通过 `on_notification` 交给监控，回调不应直接操作 loop。
用量未确认时不伪造 token 计数；RSISys 接收到的 turn 响应数量不等于压缩等内部模型请求总数。

持续运行恢复包含显式 `recovery_id`、`export_recovery_state` 和 `restore_recovery_state`。
完整缓存提案可恢复；未确认推理保持阻塞，通过 `resume_interrupted_inference(evidence=...)`
恢复原生会话、记录核对依据并丢弃未确认建议后再研究。它不会重提训练或恢复外部进程。
整个 Harness 也可由其他系统替换，DSH 的内部属性不是公开 Harness 协议的要求。
