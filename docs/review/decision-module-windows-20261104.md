# Decision Harness / Policy 接口修复

> 历史记录：原始探针、日志和请求已转入[归档说明](README.md#archived-evidence)；本文命令描述当时的工作，不作为当前可执行教程。

日期：2026-11-04。对应[修复前审查](decision-module-contract-review-20261104.md)。
范围是本地 Decision 两层替换边界及必要的 Feedback/装配接线，保留工作树已有的反馈模块调整。
本次未 push、部署或启动研究任务，未改研究预算和研究策略 prompt。

## 公开边界

| 边界 | 必要输入 / 输出 | 由实现选择 |
| --- | --- | --- |
| 整体 DecisionHarness | step(objective, action_space) → LoopStepResult；pending；receive_feedback | 内部 policy、状态、存储、prompt、SDK |
| 通用 DecisionPolicy | decide(DecisionInput) → Decision | 模型调用、tool calling 解析、算法与 context 内部格式 |
| 默认 harness 适配器 | DecisionContext → DecisionInput | 文本 renderer、引用组织、能力描述 |
| 可选调用能力 | 响应用量观察、最终请求上下文作用域、共享后端 | 按需声明，支持透明包装转发 |

公开类型在 [contracts/decision.py](../../src/rsisys/loops/contracts/decision.py)，
默认接入在 [adapters/decision.py](../../src/rsisys/adapters/decision.py)。
`DecisionInput` 包含 `objective: str`、`context: str`、`capabilities: tuple[dict, ...]`
和可选 JSON `references`；引用格式保持不透明。输入在交付前复制，不能经共享引用改写宿主状态。
能力描述用于发现，不能自行授权动作。默认 renderer 保留丰富视图的观察字段；
自定义 renderer 决定呈现形式，不能据此声称它已正确保留全部语义。

原 `harness.policy.DecisionPolicy` 继续表示 `decide(DecisionContext)` 的兼容接口。
默认策略和旧接入继续工作；新策略经 `ContextDecisionPolicy` 使用公开窗口。
`Decision`、`OperationCall`、`OperationSpec` 旧路径重导出同一类型；旧快照类型标签不变。
review、异步、模型升级、持久恢复仍是独立能力，不强加到最小策略接口。

## 已兑现的修复

| 审查项 | 修改 | 可观察结果 |
| --- | --- | --- |
| R1 权限 | 普通/定时路径统一给授权视图；dispatch 与 review 控制交付时再核验 | 最小替代 harness 在两种模式下都不能交付被禁指令；撤销权限与省略插件内部检查也不能绕过交付检查 |
| R2 用量 | 按 track_usage 能力观察响应，不再按内置类判断 | 直接/包装策略两次 fixture 响应均记 22 input / 6 output；全失败仍计入，成功不重复加最终 usage |
| R3 反馈装配 | 宿主保留模型绑定；纯 context/memory 可无 backend | 最小策略能捕获结果、更新记忆并进入下一次上下文，实际推理为零；需要推理却无能力时安装前拒绝 |
| R4 上下文 | context_scope 声明最终请求处理权；显式 policy compactor 优先 | 不覆盖配置对象；直接与包装都调用所选 compactor 一次，recent_limit=7；宿主不再先压一次 |
| R5 输入 | 公开文本窗口与默认 rich view 分离 | 普通 callable 可消费 context 与能力、调用实际 fixture 工具，再读取真实工具回执 |

用量观察和上下文作用域在失败后恢复原状态。身份失败不进入格式重试，但已经收到的响应照常记录。
完整请求计量与摘要仍由内置策略/后端的已有能力执行。第三方 SDK 需要连接受控后端或相应
观察能力；公开接口不能自动截获任意进程外调用，也不把受信 Python 插件变为隔离环境。

`DefaultDecisionHarness(model=..., model_provider=...)` 让身份由装配提供。
普通装配保存真实角色绑定，替换成没有 `.model` 的策略后仍可追踪来源。
动态提供者每次返回当前 ModelRef；不能返回 None 后退回旧身份，也不能把策略明确声明的
另一版本重新标记为当前版本。Feedback 使用同一解析入口，额外提供者不得与主决策分叉。
零推理不等于删除 provenance，纯记忆也不需要虚设 backend。

独立 policy compactor 和自定义身份提供者目前需要显式恢复适配。
内置 snapshot/durable 在不支持时明确拒绝；支持的宿主 compactor 配置、滚动状态与旧类型
继续走已有恢复链路。DSH 原生会话仍通过其独立适配与恢复扩展接入。

## 离线验收

新增 [test_decision_module_windows.py](../../tests/decision/test_decision_module_windows.py)，覆盖：

- 最小 harness、两种运行模式、权限撤销、guarded 截止、review 两种能力、非法返回。
- 包装前后有限重试/全失败用量、身份失败、压缩器优先级、作用域异常恢复及后端共享。
- 文本窗口、memory 与工具回执、引用复制、review 能力发现、旧类型序列化兼容。
- 零推理反馈、宿主身份更新、错误装配提前拒绝、未集成恢复能力拒绝保存。

新增 28 项回归全部通过。最终全量 1,555 项：1,543 通过、12 跳过，零失败/错误；
跳过的是未安装可选 SDK 或未授权真实服务的检查。源码及配置指纹在测试前后无变化。

修复前 [probe.json](README.md#archived-evidence) 与其验证指纹保留不变；
修复后写入 [probe-current.json](README.md#archived-evidence)。
最终测试及源码指纹见 [validation](README.md#archived-evidence)。

```sh
PYTHONPATH=src:tests .local/langmem-validation/bin/python -m unittest -q test_decision_module_windows
PYTHONPATH=src .local/langmem-validation/bin/python -m unittest discover -s tests -q
PYTHONPATH=src .local/langmem-validation/bin/python docs/review/repros/decision-module-boundary-probe.py
```

这些检查使用离线 CPU fixture，未调用真实模型、训练或外部 SDK 服务。
它们验证已声明的接线、权限、计数和恢复边界；长期 memory 的摘要语义、研究质量及
任意第三方实现的持久恢复需要单独验收，不能据此宣布全部即插即用或上下文问题彻底消失。
