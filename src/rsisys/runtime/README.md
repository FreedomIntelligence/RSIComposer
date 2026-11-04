# Runtime

提供跨 loop 推进、外部运行策略与公共基础能力；研究步骤和领域执行流程由各自 Harness 负责。配置通过构造函数注入，不使用全局服务注册表。

## 运行来源

[provenance.py](provenance.py) 的 `RunSource(run_id, parent_run_id=None)` 仅表达身份和来源，
不恢复状态。启动器可用 `parent_source.child('run6')` 自动取得直接父运行，传给 monitor；
创建 Application 时将同样的 `run_id` / `parent_run_id` 传入 `ApplicationConfig`。
Application 将这两个字段保存到 `config.json` / `run.json`，旧文件缺失 parent 时保持未知，
不会根据目录顺序或历史记录推断。CLI `run --parent-run-id run5` 是操作者的来源声明。
同目录 `Application.resume()` 保留原身份与 parent，不创建父子关系。

快照默认延续原运行身份；显式 `restore_snapshot(snapshot, target, run_id='run6')`
才建立新运行，并自动将快照的原 run_id 作为 parent。父运行原有 Experience 记录原样保留，
副本中的新记录归属 run6，写入不影响父目录。详见[快照边界](../snapshot/README.md)。

示例 monitor 默认读取当前 `run.json`，也接受 `source=RunSource(...)`；两者冲突时拒绝。
Web 以目录名标识运行，因此受监控运行必须满足 `RunSource.run_id == run_dir.name`，
monitor 在写入前校验此约束；受监控的父目录也必须与 parent_run_id 对齐。
核心仍允许任意 run_id 与目录组合，但这些组合不能直接接入此 monitor。
例如 run5 → run6 应使用目录 `runs/run5`、`runs/run6` 和同名 ID。
来源声明本身不证明父运行存在或已经恢复其模型、记忆或优化器，也没有通用子进程启动器或历史补全。

## Runner 与作业

[runner.py](runner.py) 提供 step / run / cancel，依赖共享端口，通过 DecisionEndpoint.receive_feedback 交付反馈。step 每次至多推进一个决策和一个探索阶段/轮询，但同步模型或环境调用仍可能阻塞；run 遇到 waiting 返回。cancel 只请求取消待处理的探索指令，须继续 step / run 确认并交付反馈；交付规则见[系统架构](../../../README.md#系统架构)。

支持可选的探索侧 `pending()` 发现未确认指令，Runner 重建时先核对两侧所有权、补交反馈和确认，再允许新决策。默认 Harness 实现此接口；只实现基础端口的外部 Harness 仍能运行，但不具备该恢复能力。单独重建 Runner 的此机制依赖仍存活的端点状态。跨进程恢复使用下述显式启用的 durable Application。

Directive 的 `review_after_seconds` 开启首次复查。Runner 在后续 tick 到期时调用可选的 Decision `review`，由其返回 `ReviewDecision('continue' | 'cancel', next_review_seconds=...)`；None 停止后续自动复查。模型调用次数与后端轮询次数独立；串行模式每 tick 最多一次普通决策或一次复查流程；并行模式一次普通决策后，还可执行当前轮询实验的一次到期复查。复查内部可执行有限次权限允许的工具查询和模型调用。Runner 的 decision_steps 按流程计数，模型 token usage 由 Harness 逐次累计。Runner 不 sleep、不自行创建后台线程，实际复查发生在截止时间之后的下一次 tick。调用方负责轮询间隔，单独重建 Runner 后从恢复时刻重新计算首次复查时间；durable Application 则恢复原截止时间并计入停机时间。

取消响应不确定时保留复查结果，通常重试取消不重新问模型。若 Decision 端点提供 authorize_review，Runner 在实际取消及每次重试前重新授权；权限撤销后丢弃旧模型意图，下一次复查读取新权限。已被后端接收的请求不会被撤回。外部调用 Runner.cancel 属于操作者控制，不受模型的取消权限配置限制。默认串行模式只有一个在途指令；可选并行模式隔离管理多个实验，并通过显式版本化 update 请求调用后端支持的控制。不支持替换原始 Directive、通用 pause/resume 或并发修改同一模型绑定。决策者按时间/token 自训练的触发器仍未实现。

[jobs.py](jobs.py) 的 JobBook 管理所有权、幂等键，每个实例至多有一个未完成任务；JobHandle 缓存终态，成功发出一次取消请求后只继续轮询。MockJobQueue 在 poll 时推进模拟任务，result 只读成功结果；JobBackend 是预留集群接口。独立使用时记账仅在进程内；durable Application 显式保存内置 Mock 队列、JobBook、幂等键、取消标记及终态缓存，外部后端需恢复契约；状态与幂等约定见[共享协议](../loops/contracts/README.md)。

## 运行策略与研究轮次

策略与目标判定应有界、无副作用；同一 tick 可在执行前后多次检查。计时从当前运行第一次 `step()` 开始，outcomes 只包含已经完成交付的实验结果；durable 恢复延续原计时与 outcomes。

[run_policy.py](run_policy.py) 定义 `RunPolicy.evaluate(RunContext) -> RunDecision`，通过 `Runner(..., run_policy=...)` 注入；Python 装配入口 `build_demo` / `Application` 同样接受 `run_policy=` 和 `cleanup_timeout_seconds=30`。RunPolicy 管运行是否继续、何时重启研究及如何停止；Harness 内的 DecisionPolicy 仍选择研究操作和实验，ActorPolicy 仍选择环境动作。替换运行规则无需把时间或目标控制塞进 Harness。

| 内置策略 | 继续与停止条件 |
|---|---|
| `ResearcherRunPolicy()` | 默认策略，研究者 finish 后停止，默认 drain |
| `TimeRunPolicy(seconds, restart_delay_seconds=1)` | 研究者 finish 后继续新一轮，直到运行时限，默认 cancel |
| `GoalRunPolicy(goal, restart_delay_seconds=1, max_seconds=None)` | 目标未达时继续；达标默认 drain，可选时限到达则 cancel，默认没有运行时限 |

RunContext 包含 `objective`、`elapsed_seconds`、`research_round`、`researcher_finished` 与已交付 `outcomes`；每个 `ExperimentOutcome` 包含对应的 directive 和 feedback。RunDecision 的 action 为 `continue` / `restart` / `stop`，并携带 reason、shutdown、restart_delay_seconds 与 evidence_refs。`research_round` 是研究轮次，区别于 `run(rounds=...)` 的本次反馈交付额度。持续驱动同一个 Runner 时，时间策略覆盖跨调用的运行时间与轮次间等待；单次 run 因 waiting、调用额度或决策步数上限返回，不代表 RunPolicy 已经结束。

GoalRunPolicy 的 goal 是可替换 `goal(context) -> bool` callable。内置 `MetricGoal(metric, threshold, protocol_ref, higher_is_better=True, allow_simulated=False)` 要求绑定评测协议；默认按指标大于等于阈值判断，反向指标用 `higher_is_better=False`。它仅读取已交付、成功完成的评测结果，拒绝缺失或非有限指标，默认不接受 Mock 等 simulated 结果。同一候选、协议与输入的较新评测覆盖旧结果；不同验证集应使用不同协议标识。训练进度和未确认指标不参与达标判断。目标达成只控制运行停止，候选采纳和模型部署仍由相应业务策略负责。字段和构造参数以 [run_policy.py](run_policy.py) 为准。

默认 Decision loop 的 `step_guarded` / `review_guarded` 在 Action Space 校验边界再次检查运行条件，避免模型返回时预算已耗尽却仍启动工具或实验。整体替换端点时，需实现等效执行前检查；纯传输端点的同步调用仍只能在调用前后检查。`pending_records` / `flush_pending_records` 支持停止时补写既有回执，不能重跑工具或发起新研究。

研究者 finish 后，如果策略要求继续，Runner 通过 Decision 端点的显式 `restart_research` 开启新轮次；默认实现保留 memory、共享经验和候选，不重新装配 Harness。自定义端点需实现此能力。`restart_delay_seconds` 默认 1 秒，到期前返回 waiting，由调用方后续驱动；Runner 不阻塞 sleep、不后台轮询。接入示例见[使用指南](../../../README.md#选择运行策略)。

RunDecision 指定停止时的 `shutdown='drain'` 或 `'cancel'`：drain 继续推进已有任务到终态；cancel 请求取消并继续确认。两者都禁止新实验，保留已有反馈的交付与 acknowledge；已请求取消不等于后端已确认取消。运行超时默认 cancel，清理另由 Runner 的 `cleanup_timeout_seconds=30` 限制，不把运行时限当成无限清理预算。清理到限仍未确认的任务必须明确记录，不报告清理成功或已完成取消；待交付反馈也不能被伪装成交付成功。收尾中的异常保留原异常并交给调用方，状态保留供后续 `step()` 重试。

结果中的 `stop_reason` 区分 `researcher_finished`、`budget_exhausted`、`goal_reached` 等停止原因，`shutdown_complete` 单独表示收尾是否确认完成。事件的 `run_stop` 保留 `cleanup_timed_out`、`unresolved_directives`、`unresolved_research_job` 与 `unresolved_records`；即使返回 finished，也须检查这些字段。研究者 finish、时间耗尽和目标达成是不同结论，目标已达也不证明清理已完成。

运行时限与清理上限在调用方驱动的 tick 中检查，停止调用就不会自动触发超时或硬杀后端。同步模型、环境或后端调用仍需自身设置超时，Runner 不能中途抢占这些调用。可搬迁的空闲快照仅兼容默认 researcher 策略及默认清理预算；其 time / goal、自定义策略或预算仍明确拒绝。原目录 durable 恢复支持内置运行策略与清理预算，区别见下一节与[快照说明](../snapshot/README.md)。

## 原目录重启恢复

[recovery.py](recovery.py) 提供显式启用的 `Application(config, durable=True)` 与
`Application.resume(data_dir)`。恢复使用原目录和原 run_id，不要求研究已经空闲；
不创建调度线程，调用方继续驱动 `runner.step()` / `run()`。

```python
from rsisys.application import Application
from rsisys.runtime.run_policy import TimeRunPolicy

with Application(config, durable=True, run_policy=TimeRunPolicy(3600)) as app:
    result = app.run()  # 可以在 waiting 时退出进程

with Application.resume(config.data_dir) as app:
    result = app.run()  # 继续原运行与剩余预算
```

每次 `step`、`cancel`、`update` 前原子写入并 fsync 一个 journal 意图，成功后用新一代
checkpoint 提交完整本地控制状态。checkpoint 是提交点；不会重放 journal 中的调用。
串行 Runner 的内嵌 cancel 共用外层边界。Application 保持原有目录排他锁。
仅通过这些 Runner 入口执行受保护的动作；直接调用 Harness、后端或修改对象不会自动成为持久边界。

保存内容包括两个 Harness 的状态、模型绑定、操作/反馈回执、探索 command receipts、
Runner outbox、RunPolicy、已交付 outcomes、研究轮次、结束/停止状态、复查与重启等待。
内置 Mock 训练/评测保留原 JobRef、请求、幂等键、剩余轮询数、取消意图及终态结果；
重启后轮询原任务，不重复执行已提交任务。重新构造的是明确支持的模拟队列，不能据此恢复远端进程。
Decision agenda 全量 revision archive 与 context compactor 配置单独保存；memory、Experience
SQLite 和 artifacts 保留原目录中的持久记录，不会用旧 checkpoint 回滚这些证据。

内置 `ResearcherRunPolicy`、`TimeRunPolicy` 和 `GoalRunPolicy(MetricGoal(...))` 可直接恢复。
时间偏移重绑到新进程的 monotonic clock，并计入 checkpoint 后的墙钟停机时间；
运行时限、清理截止、首次 tick 中断、restart delay 和 review deadline 都不会重新获得预算。
墙钟回拨的停机增量按零处理；机器时钟应可靠。自定义策略/goal callable 需显式恢复契约。
恢复要求相同源码指纹、Python 主/次版本和配置；不动态导入或反序列化任意 Python 对象。

进程若在动作中途退出、异常后状态写入失败或效果无法确认，
`app.recovery.intervention` 保留边界描述，后续 Runner 写操作抛出 `RecoveryRequired`。
仍存活时尽量保存当前回执和 outbox；硬崩溃只能保留最后 checkpoint 与未提交意图，
不声称掌握未返回的外部结果。即使工具可能尚未执行，也不会自动重试。
恢复不重建本地子进程、任意远端任务、模型权重或优化器。

操作者/适配器先检查已有作业、外部幂等键、SQLite/产物及回执，再调用：

```python
app.recovery.reconcile(repair, evidence="经核实的任务/回执标识与处理依据")
# repair(app, intervention) 修复控制状态并返回 None：挂接已有结果/任务，或明确标记失败。
```

reconcile 自身也写前置意图并保存 evidence；中断或校验失败继续阻止执行。
没有通用的“清空标记并重跑”功能。已有终态反馈/ack 可在确认回执后恢复幂等交付。
处于 intervention 的运行不能通过空闲快照绕过检查。

外部组合通过 `recovery_contract=` 显式提供 `contract_id`、`capture(app)`、
`restore(app, state)`；重新打开时再次传入同一 contract、原工厂以及需要的 custom RunPolicy。
capture 必须观察性地导出 codec 支持的数据，restore 只挂接已有状态，不执行工具或提交作业。
契约负责自定义策略配置/状态、外部稳定 ID、幂等与不确定结果处理；不支持的状态必须拒绝。
并行 Runner 暂不支持此本地恢复路径。

`DefaultDecisionHarness` 子类也可通过稳定 `recovery_id`、`export_recovery_state()`、
`restore_recovery_state(state)` 保存扩展状态；继承状态、训练记账、memory/agenda 绑定由内置恢复负责。
需通过原 `decision_factory` 重建，探索侧仍为内置组合；未声明的替换明确拒绝。
DSH 用这些 hook 保存 session 身份和已完成提案；未完成推理恢复为需要核对的标记。
DSH 原生 session 的恢复与 RSI checkpoint 是独立契约，不能把 session_id 当成推理已恢复。
见 [DSH Harness](../loops/decision_making/harness/README.md)。

停止策略为 cancel 时，可选 Decision `cancel_research()` 请求取消其自己拥有的研究推理。
Runner 每轮重新观察 `pending_research_job()`，仍有引用就继续等待或到清理截止；
不根据 hook 的返回值声称已取消，也不假定它能取消任意工具/训练任务。

## 模型调用

[model.py](model.py) 提供 MockModelBackend 和 Chat Completions HTTP RemoteModelBackend。policy 决定调用内容与时机；配置方法见[使用指南](../../../README.md#使用指南)。

HTTP 支持非流式 JSON 文本和原生 function tool calling、超时、响应大小和模型身份检查。
`GenerationRequest.tools` 配置标准工具声明，`messages` 支持 assistant/tool 多轮历史；
提供 messages 时将 prompt/system 留空。工具请求不附加旧的 JSON object 输出模式。
`GenerationResponse.tool_calls` 保留标准调用 ID、函数名和原始 arguments 字符串，
工具执行由消费方的库/harness 负责；transport 不执行。工具参数 schema 由消费方校验。
未声明工具的旧 JSON 请求继续拒绝意外 tool_calls；`response_schema` 与原生 tools 互斥。
尚不支持多模态或流式生成。不跟随重定向，也不自动重试推理请求。
密钥仅在调用时从指定环境变量读取，不写入配置或快照。

响应缺少 model_revision 时 revision_verified=false；匹配该字段也不能证明权重不可变。TokenUsage 缺失保持未知，缺失次数由各默认 Harness 记录。

## 系统组合与显式应用

[systems.py](systems.py) 提供可选 `SystemRegistry(initial, activator)`。清单类型
`SystemVersion(version, components, model=None, parent_version=None)` 定义在
[candidate.py](../loops/contracts/candidate.py)；components 是命名 ArtifactRef，不限定组件格式。
`Candidate.parent_system` 表示实验父版本，`base_version` 继续表示训练权重基础 revision。

`register(candidate, model=None)` 根据已知父版本合并变化，保留新清单但不改变 `current`；
父版本有模型时，candidate 的 base_version 必须等于该模型的 revision，不能把系统版本 ID
当成权重 revision。训练输出使用 `register_training(result, origin=提交时的 TrainingOrigin)`：
完整校验 result.base 与 origin.initial_model，以及 candidate.parent_system 与 origin.parent_system，
不要求初始化模型等于实验父系统的模型。因此可以先应用一轮 LoRA，再从原始权重训练另一个候选。
`training_origin(candidate_id)` 返回保留的来源副本；同一候选 ID 不能更换训练来源。
`get(version)` 与 `current` 返回独立副本。

`compose(candidate_ids, *, version) -> SystemVersion` 接受至少一个、不重复的已注册候选 ID，
要求共享同一个显式、已知 parent_system；普通组件候选沿用父模型基准，训练候选的初始化
通过 register_training 单独校验。它们可在同一父系统下组合，无需把训练初始化 revision
改写成父模型 revision。只合并各候选的 changes，
再继承 parent 中未变化的组件。不同组件键直接 union，重叠键的 ArtifactRef 完全相同可接受，
值不同则拒绝，不按候选顺序覆盖。没有修改模型的候选保留 parent 的模型；显式注册的模型
变化可与组件变化组合，但多个不同的目标 ModelRef 会拒绝，不混合权重。组合单独保存来源
ID 集合（顺序无关）；即使清单相同，也不能给既有组合换来源，或冒用普通候选已有的版本 ID。
组合不是新的原始候选，不能再作为 compose 的来源。compose 可基于已知旧 parent，只登记清单。

`activate(candidate_id, idempotency_key=...)` 同时接受已登记的候选和组合版本，
检查父版本等于 current，调用 `SystemActivator.activate(system, idempotency_key=...)`，
只在返回精确相同清单后更新 current。相同应用键返回原回执，键与候选冲突会拒绝；
只有 `SystemActivationRejected` 明确确认无副作用或已完成回滚时才释放 pending，current 不变；
其他异常仍保留同一意图，不能跳到另一候选。不同父版本的候选需显式 rebase。

`adopt(candidate_ids, *, version, idempotency_key) -> SystemVersion` 组合后只调用一次 activate，
返回确认应用的实际 manifest；不会逐个激活来源候选。相同 key、version 和来源集合重试
返回原成功回执，即使 current 已继续前进；新 key 对旧 parent 的采纳仍被 stale 检查拒绝。
同 key 换版本或来源集合会拒绝；响应丢失等不确定失败不会产生成功采纳，pending 阻止其他
应用请求，原请求须用同 key 重试，由 activator 根据自身幂等回执确认结果。

`system_adoption_capability(registry, candidates, *, simulated=False)` 返回名为 `system_adopt`
的可选写能力，参数仅有 candidate_ids、version、idempotency_key。candidates 是每次验证和
执行时读取的 `Callable[[], Iterable[Candidate]]`，选择必须来自该已知集合。验证不注册或激活；
执行时按需注册组件候选，已有注册内容必须与 callback 一致。需要改变模型时，调用方必须
提前 `register(candidate, model=明确的ModelRef)`；callback 不会从产物引用猜权重或推断模型。
成功返回 `OperationResult(adopted_candidate_ids=tuple(ids), data={'system': asdict(actual)})`，
simulated 明确使用装配参数（默认 False），只有 registry.adopt 确认后才报告采纳。下一步 factory
可读取 `registry.current` 构建实际组件绑定；重试得到的旧回执仍描述原应用，不代替当前状态。
装配沿用已有接口，并继续受 ActionPermissions / Harness 的权限校验约束：

```python
from rsisys.runtime.systems import system_adoption_capability
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.capabilities import Capabilities
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools

capabilities = Capabilities((system_adoption_capability(registry, known_candidates),))
space = ModularDecisionActionSpace(capabilities.definitions, permissions=permissions)
tools = BoundDecisionTools(capabilities.bindings)
```

此能力只通过显式装配启用，不会自动安装到 engine 或 Application，也不扩展权限。
所有组件变化保持 ArtifactRef 轻量引用，不下载内容、不转换数据格式。

`ReferenceSystemActivator(model=None)` 只接受模型与构造时一致的清单，适合引用级组件变化，
它确认后会真实更新 registry 的 manifest，但不代表模型部署或产物内容已验证。
自定义 SystemActivator 负责实际组件准备、部署、外部效果和绑定原子性；registry 不替它回滚。
该 registry 无全局实例或隐式存储，不会自动加入 Application 的 durable 状态；自定义恢复必须
同时保存候选、训练来源（training_origins）、组合来源、版本、current、应用意图与 activator 回执。

## Harness 外的具体后端

[adapters/training.py](../adapters/training.py) 的 `ProcessTrainingBackend` 提供可信 POSIX
主机上的持久作业目录、幂等提交、子进程监督、轮询、取消及同 JobRef 重连。注入的
`command(request, job_dir)` 返回 TrainingCommand，原生配置和结果读取由适配器明确解释。
默认 worker 使用可选 Transformers / PEFT 的 SFT；原生 CLI 也可通过命令和 result_reader 接入。
导入模块不加载训练依赖，核心安装不自动安装这些框架。

训练协议与具体执行分开：[BackendCapabilities](../loops/contracts/backend.py) 声明方法、
恢复模式、更新字段、checkpoint 导出与重连能力；默认 worker 不承诺 full_state、checkpoint
导出或热更新。丢失监督进程身份时保留 UNKNOWN，不重新启动不确定的训练。
后端重连只关联作业，不等于恢复 Runner / SystemRegistry / 外部部署的完整控制状态。
进程执行继承宿主权限，不是安全隔离；资源、网络和设备限制由调用方部署承担。

同一协议文件还定义独立的 DeploymentBackend（prepare / query / release）及可选 reattach。
稳定 DeploymentRef 与 JobRef 用于核对已有外部工作，只有 READY 表示精确目标模型可推理；
ModelActivator 将它与现有 Harness 的角色更新入口连接，SystemActivator 则处理完整组合。
实际适配器配置和数据边界见[使用指南](../../../README.md#接入整项训练任务)。

[adapters/serving.py](../adapters/serving.py) 的 VLLMBackend 接入独立运行的 vLLM，
显式映射模型 revision 与服务别名；VLLMDeploymentBackend 维护稳定部署引用及本地回执，
通过模型列表和实际推理探测确认就绪。动态 LoRA 需要 allow_runtime_lora 显式开启和服务端
可访问的原生路径；默认只观察既有静态服务。query / reattach 不加载模型，release 只允许
卸载自己管理的动态 LoRA，拒绝静态/共享部署。VLLMLoRAActivator 将就绪结果接入现有
model_activator 入口；它们不负责服务启动、权重上传或完整 SystemVersion 应用。
缺失服务端 revision 时仍为 revision_verified=False，不把别名当作权重证明。

## 执行后端

[sandbox.py](sandbox.py) 定义创建、执行、查询、取消与释放协议及 Mock；[local_process.py](local_process.py) 提供真实 argv 进程执行、超时、输出限额、独立工作目录与进程组清理。它使用宿主用户权限，不是安全隔离沙箱。

任务语义、输入准备和输出收集属于 [Environment](../loops/exploration/harness/environment/README.md)。替换执行后端时需匹配环境对同步/异步完成的要求。

## 产物与文件

ArtifactRef 是纯数据引用：artifact_id、kind、uri 必须为非空文本；version、digest 默认 None，提供时必须为非空文本。仍兼容原有五个位置参数。契约不限定 URI scheme，也不规定外部摘要格式；外部路径、对象存储地址或后端自定义标识均由消费方解释。引用可无需本地注册，通过 Directive、训练和评估请求传递；构造、传递与 codec 往返不会自动解析、下载或将产物入库，None 会原样保留。

[artifacts.py](artifacts.py) 的 ArtifactStore 协议与 InMemoryArtifactStore 仅提供 put / get；内存对象随实例丢失，引用不能用于跨进程恢复。FileArtifactStore 另有 put_file / resolve / refs；artifact:// 引用须配套存储目录，搬迁时需保留内容与元数据。

两个内置存储仅接受自身管理的引用：InMemoryArtifactStore 依赖当前实例，FileArtifactStore 依赖当前根目录中的内容与元数据。它们要求各自生成的 URI、版本和非缺失的 SHA-256 摘要，读取时仍严格校验完整引用元数据及内容哈希；可选 digest 不意味着跳过内部校验。外部引用、缺失摘要或篡改的引用不会被自动接纳，使用内置存储的 Mock learner/evaluator 同样受此限制。

FileArtifactStore 的 put_file 分块写入，resolve 校验后返回路径，get 会完整载入内存；引用本身不会加载文件，也不保证文件永久可用。[files.py](files.py) 提供原子写入辅助。

远程对象存储和自动回收尚未接入；自定义持久化的[快照集成](../snapshot/README.md)需另行实现。

可选并行实验：`Runner(decision, exploration, parallel=True)` 搭配
`DefaultDecisionHarness(max_outstanding=N)` 和
`ConcurrentExplorationHarness(factory, max_in_flight=N)`。默认 `parallel=False`、
`max_outstanding=1` 保留单实验流程。每个 `step()` 最多做一次普通决策，然后公平轮询
一个实验；review 仍可执行其原有有界工具流程。backend jobs 本身异步，不启动训练线程。
`run()` 仍是有界调用，遇到 waiting 返回，由调用方稍后继续轮询。
`parallel=True, max_outstanding=1` 也可使用：Decision 可选的 `can_step()` 就绪接口
使 runner 在串行反馈保护生效时跳过普通决策，继续推进、复查或取消当前实验；交付后恢复决策。

并行端点通过 `pending_all()` 恢复所有在途指令，按 `directive_id` 路由生命周期；
`cancel(directive_id)` 只取消目标，多任务时省略 ID 会拒绝。每个实验保留独立 outbox、
review 定时器和取消意图，终态反馈或 ack 丢失只重试交付，不重新提交或执行。
恢复 Runner 是同进程端点接管，不是可持久化并行 checkpoint；本地 snapshot 明确拒绝并行配置。RunPolicy 的计时、轮次、目标反馈视图与收尾状态属于当前 Runner，不会随这种端点接管恢复；重建 Runner 会开启新的运行预算，跨进程保留预算可用上述串行 durable Application；并行恢复仍需单独实现。

`runner.update(DirectiveUpdate(id, expected_version, idempotency_key, changes))` 返回
`UpdateReceipt(request, version)`。版本冲突、幂等键内容冲突和不支持的 backend 都会明确报错。
后端响应不确定时保留相同更新意图和原版本，拒绝不同请求；只有完全相同请求可以重试更新。
目标实验的 inspect、advance/status 与取消不被这条更新阻塞，handler 必须继续操作已有任务，
不得从状态推进路径重放更新或把未知结果当作成功。取消请求优先：随后不再发出任何未确认更新，
继续取消/轮询直到终态，保留未确认意图作为证据且不增加版本。已成功确认的旧回执仍可幂等读取。
其他实验可继续。原始 Directive 不改写，更新是单独的 backend 控制回执。

可运行的 mock 示例：
`PYTHONPATH=src python3.12 examples/parallel_experiments.py`。
它展示两个实验在途、模型同款 capability 调用路径上的更新、乱序反馈和评测比较，
不声称真实 GPU 并行训练或真实模型自主决策。
