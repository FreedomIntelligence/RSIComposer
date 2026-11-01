# Runtime

提供跨 loop 推进与公共基础能力，策略和领域执行流程由各自 Harness 负责。配置通过构造函数注入，不使用全局服务注册表。

## Runner 与作业

[runner.py](runner.py) 提供 step / run / cancel，依赖共享端口，通过 DecisionEndpoint.receive_feedback 交付反馈。step 每次至多推进一个决策和一个探索阶段/轮询，但同步模型或环境调用仍可能阻塞；run 遇到 waiting 返回。cancel 只请求取消待处理的探索指令，须继续 step / run 确认并交付反馈；交付规则见[系统架构](../../../docs/ARCHITECTURE.md)。

支持可选的探索侧 `pending()` 发现未确认指令，Runner 重建时先核对两侧所有权、补交反馈和确认，再允许新决策。默认 Harness 实现此接口；只实现基础端口的外部 Harness 仍能运行，但不具备该恢复能力。此机制依赖仍存活的端点状态，不是跨进程崩溃恢复。

Directive 的 `review_after_seconds` 开启首次复查。Runner 在后续 tick 到期时调用可选的 Decision `review`，由其返回 `ReviewDecision('continue' | 'cancel', next_review_seconds=...)`；None 停止后续自动复查。模型调用次数与后端轮询次数独立；串行模式每 tick 最多一次普通决策或一次复查流程；并行模式一次普通决策后，还可执行当前轮询实验的一次到期复查。复查内部可执行有限次权限允许的工具查询和模型调用。Runner 的 decision_steps 按流程计数，模型 token usage 由 Harness 逐次累计。Runner 不 sleep、不自行创建后台线程，实际复查发生在截止时间之后的下一次 tick。调用方负责轮询间隔，重建 Runner 后从恢复时刻重新计算首次复查时间。

取消响应不确定时保留复查结果，通常重试取消不重新问模型。若 Decision 端点提供 authorize_review，Runner 在实际取消及每次重试前重新授权；权限撤销后丢弃旧模型意图，下一次复查读取新权限。已被后端接收的请求不会被撤回。外部调用 Runner.cancel 属于操作者控制，不受模型的取消权限配置限制。默认串行模式只有一个在途指令；可选并行模式隔离管理多个实验，并通过显式版本化 update 请求调用后端支持的控制。不支持替换原始 Directive、通用 pause/resume 或并发修改同一模型绑定。决策者按时间/token 自训练的触发器仍未实现。

[jobs.py](jobs.py) 的 JobBook 管理所有权、幂等键，每个实例至多有一个未完成任务；JobHandle 缓存终态，成功发出一次取消请求后只继续轮询。MockJobQueue 在 poll 时推进模拟任务，result 只读成功结果；JobBackend 是预留集群接口。记账仅在进程内，不保证重启恢复；状态与幂等约定见[共享协议](../loops/contracts/README.md)。

## 模型调用

[model.py](model.py) 提供 MockModelBackend 和 Chat Completions HTTP RemoteModelBackend。policy 决定调用内容与时机；配置方法见[使用指南](../../../docs/GUIDE.md)。

HTTP 支持非流式 JSON 文本、超时、响应大小和模型身份检查；不支持原生 tool_calls、多模态或流式生成。不跟随重定向，也不自动重试推理请求。密钥仅在调用时从指定环境变量读取，不写入配置或快照。

响应缺少 model_revision 时 revision_verified=false；匹配该字段也不能证明权重不可变。TokenUsage 缺失保持未知，缺失次数由各默认 Harness 记录。

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
恢复 Runner 是同进程端点接管，不是可持久化并行 checkpoint；本地 snapshot 明确拒绝并行配置。

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
