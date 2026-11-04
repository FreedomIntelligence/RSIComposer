# Decision Harness / Policy 接口边界审查

日期：2026-11-04。范围：当前本地工作树，包括尚未提交的 Feedback 接口化调整。
本次只增加审查记录与离线复现材料，没有修复生产代码、修改 prompt、启动实验或 push。

本文件保留修复前的审查与反例。后续用户授权的实现见
[Decision 接口修复](decision-module-windows-20261104.md)；新探针结果另存 current 文件，
不覆盖本次历史证据。

结论：整套 harness 和内部 policy 都已有可替换入口，但还不能按“替换后宿主保证不变”验收。
复现得到四类行为问题；此外，默认 policy 输入尚未与默认 harness 状态结构解耦。

## 已确认的边界

- 整套 DecisionHarness 可以替换，不要求实现默认 memory、tools、policy 或 state。
- 默认 harness 内的 DecisionPolicy 也可以单独替换；最小协议只有 `decide(context) -> Decision`。
- 外部约定必要交互，内部状态、prompt、模型响应格式和算法由实现选择。
- Host 保留身份、权限、准入、计费、生命周期和恢复责任。
- context/memory 能力保持激活，但不要求固定实现或一定发生推理。
- review、后台推理、模型升级及持久恢复是按需提供的能力，不应全部变成基础必选方法。

## 已正常工作的部分

1. 只有 step/pending/receive_feedback 的替代 harness 可以完成一次离线任务，不需要 policy 或 memory 属性。
2. 只有 decide 方法的替代 policy，在未接反馈的默认 harness 中可以正常返回动作。
3. 独立 FeedbackAdapter 已支持自定义 harness；需要适配器本身不是缺陷。
4. 默认 prompt 已拆成九个可替换组件；它们属于默认 policy，实现替代 policy 不必复用这些文本。
5. 默认 snapshot 对未集成的自定义实现明确拒绝，而不是静默丢失状态。这是已声明的能力边界；
   不能由“可替换”推导出任意第三方状态已经可自动恢复。

## R1 / P1：合规替代 harness 收到未授权过滤的 Action Space

位置：[loop.py:14](../../src/rsisys/loops/decision_making/loop.py#L14)、
[access.py](../../src/rsisys/loops/decision_making/action_space/access.py)、
[runner.py:97](../../src/rsisys/runtime/runner.py#L97)。

普通 `DecisionMakingLoop.step` 原样传入 action_space；step_guarded 才包装授权视图。
Runner 默认 ResearcherRunPolicy 不走 step_guarded。默认 harness 自己再次包装，因而内置链路正常。
替代 harness 若按公开协议调用 action_space.validate_directive，得到的只是模式校验，
没有 ActionPermissions 检查。此缺口不需要插件主动绕开校验。

复现：同一最小 harness、同一 `ActionPermissions(directives=())`，提交 execute 指令：

| 运行模式 | 实际结果 |
| --- | --- |
| researcher，默认运行模式 | 完成一个被配置禁止的离线任务 |
| time | PermissionError，未提交 |

建议：由共同宿主边界始终提供授权能力视图；在正式转交 dispatch 时统一复核当前权限。
review 的操作入口也需保持相同规则，避免依赖某个 harness 自行重复实现。
这些保证限定于受控工具和 Lab 交付路径，不把进程内 Python 插件说成了安全隔离边界。

验收：同一允许/禁止/执行前撤销用例，在默认与最小替代 harness、researcher/time 模式下结果一致；
拒绝的任务没有到达 Lab，也没有产生执行副作用。

## R2 / P2：用量观察依赖具体 ModelDecisionPolicy 类型

位置：[engine.py:573](../../src/rsisys/loops/decision_making/harness/engine.py#L573)。

只有 isinstance(policy, ModelDecisionPolicy) 才进入 track_usage；其余实现仅在成功返回
Decision/ReviewDecision 后读取最终 result.usage。即使透明包装转发了 track_usage、model、backend，
也会失去每次响应的观察路径。

复现：每次 fixture 响应为 input=11、output=3，生成两次：

| 结果 | 未包装：harness 记录 | 透明包装：harness 记录 | backend 实际响应累计 |
| --- | --- | --- | --- |
| 第一次格式错误，第二次成功 | 22 / 6 | 11 / 3 | 22 / 6 |
| 两次均格式错误 | 22 / 6 | 0 / 0 | 22 / 6 |

这是 harness token 统计漏记的直接证据，不证明所有部署的 backend 调用准入都已被绕过。
后端若有独立计数仍可能正确；两者必须分别验收。

建议：把调用级观察与计费接入宿主提供的模型调用窗口，或显式的可选调用观察能力，
而非检查继承关系。纯函数 policy 不必伪装成模型；失败、摘要和恢复调用同样需要真实记录。

验收：直接 policy 与透明包装在成功、解析失败、有限重试下记录相同调用/用量，既不漏记也不双记。

## R3 / P2：零推理 Feedback 接入仍隐含要求 policy.backend / policy.model

位置：[learning_harness.py:386](../../src/rsisys/loops/decision_making/harness/learning_harness.py#L386)、
[learning_harness.py:41](../../src/rsisys/loops/decision_making/harness/learning_harness.py#L41)、
[service.py:60](../../src/rsisys/loops/decision_making/feedback/service.py#L60)。

ResearchAssembly.requires_model=false 可以正确表达纯 context/memory 实现。
但 decision 侧接线仍按具体 policy 类型推导 backend；默认 adapter 每次捕获也直接读取 policy.model。
而这些属性没有出现在 DecisionPolicy 最小协议中。

复现：默认 harness 内换成仅有 decide 的 PlainPolicy，装配纯 `query -> str` memory：

1. 不额外提供 feedback backend：装配拒绝，报 Session/custom policies require an explicit feedback backend。
2. 额外提供 backend：装配和 decision 均通过；首份结果 observe_output 时抛出 AttributeError，缺少 model。
3. 上述过程真实模型调用数均为零。

建议：把“当前 meta-agent 身份提供者”和“受控推理能力”在装配中显式接入。
模型身份仍应真实、可追踪，不能伪造或因无需推理就删除；但不应要求算法组件必须持有某个属性。
推理能力只在所选组件需要它时要求；外部 SDK/自定义会话可由 adapter 连接宿主模型能力。

验收：最小 policy 配宿主显式身份、纯 context/memory 时能完成装配、结果捕获和下一次上下文读取，
无需空壳 backend；需要推理的组合缺少模型能力时，在运行前明确失败。

## R4 / P2：压缩所有权随 policy 的包装方式改变

位置：[engine.py:579](../../src/rsisys/loops/decision_making/harness/engine.py#L579)、
[policy/__init__.py:293](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L293)。

直接使用 ModelDecisionPolicy 时，harness 在调用前覆盖 policy.context_compactor；
换成透明包装后，harness 先压缩 context，内部 policy 再使用自己的 compactor。
这使组合方式改变内部配置及压缩调用次数。

复现：为 policy、harness 各装一个符合 compact(context, *, recent_limit=32) 的记录器，
将 harness.memory_limit 设为 7：

| 组合 | policy compactor | harness compactor |
| --- | --- | --- |
| 直接 policy | 被覆盖，原对象未调用 | 调用一次，recent_limit=32 |
| 透明包装 | 保留，调用一次，recent_limit=32 | 先调用一次，recent_limit=7 |

记录器只证明调用与配置不一致；本轮没有真实摘要模型，也没有证明发生了语义丢失。
当两层均有真实压缩逻辑时，这种隐含组合会增加定位上下文问题的难度。

建议：在装配时明确上下文处理的所有者和交接点，不在调用中根据内置类身份替换插件属性。
默认 harness 提供事实与当前工作视图；原生会话 policy 若自行维护会话，需显式声明接管窗口。
宿主对完整模型请求的准入独立保留，不要求外部实现复制 BasicContextCompactor 算法。

验收：直接/包装两种组合使用相同压缩配置、次数和完整请求准入；身份错误与容量错误仍分别处理。

## R5：输入契约尚未从默认实现状态中分离

位置：[DecisionPolicy](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L26)、
[DecisionContext](../../src/rsisys/loops/decision_making/harness/state.py#L45)、
[默认请求序列化](../../src/rsisys/loops/decision_making/harness/policy/__init__.py#L73)。

当前 policy 协议直接依赖默认 harness.state.DecisionContext。输入混合目标、工具能力、
真实回执、内部 Memory/HistoryEntry、候选账本、research_agenda、decision_graph、research_state、
context_summary 和 memory_context。默认请求 builder 再将 dataclass 整体序列化。

这些信息在内置实现中有用；并不意味着所有替代实现都应该复刻这套组织方式。
本项是接口设计缺口，不是已复现的当前运行失败；也不能据此断言必须删除全部结构化事实。

建议 checkpoint：

- 通用输入只要求目标、当前可用 context、能力及必要引用窗口。确切字段在下一次讨论确认。
- context 的内部组织留给实现；候选、指标和生命周期等真实事实仍由宿主提供，不能由摘要覆盖。
- 默认 DecisionContext 留作默认实现的便利视图，由 adapter 接通；不直接升格为所有实现的必选状态结构。
- 普通决策返回可校验动作，模型响应可以是 JSON 或原生 tool calling，解析属于 policy 实现。
- review、异步观察、用量事件、身份/模型升级及恢复分别描述能力；不把整个功能集合塞入基础协议。

## 待确认的实施次序

1. 先确定通用输入窗口和必要动作输出；保留现有默认状态作为 adapter 后的实现细节。
2. 将权限与调用级用量保证放到共同宿主边界，消除内置类型判断带来的行为差异。
3. 明确 context 处理所有权及当前身份/推理能力的装配方式，接通纯 policy 与自带会话实现。
4. 分别核验 review、异步、关闭和恢复能力；没有提供的能力应装配时明确拒绝，不假装通用支持。
5. 边界确认并落实后再逐句审默认 prompt，不借接口重构改变研究策略或运行预算。

## 离线验证

- 现有测试：test_harness_composition、test_assembly、test_structured_decisions、
  test_decision_validation_retry、test_feedback_decoupling、test_research_assembly，63 项通过。
- 补查：test_action_permissions、test_learning_policy、test_m03_decision_policy、
  test_decision_interaction，46 项通过。
- 合计 109 项通过，零跳过、零失败；未重复运行全量回归。
- 新增[独立探针](repros/decision-module-boundary-probe.py)与[观察结果](repros/decision-module-boundary-probe.json)，
  包含正向最小实现与上述四类反例。反例记录当前问题，并非修复通过的验收报告。
- [验证及源码指纹](repros/decision-module-boundary-validation.json)标记此次工作树范围。

复现命令：

```sh
PYTHONPATH=src .local/langmem-validation/bin/python docs/review/repros/decision-module-boundary-probe.py
```

所有生成均为离线 fixture；没有真实模型推理、训练、第三方存储恢复或长时上下文测试。
本轮不据此声称研究质量、memory 摘要语义或任意第三方组件即插即用已通过验收。
