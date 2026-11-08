# 2026-11-04 全模块审查

本次按 [M01–M15 地图](../modules.md) 检查当前实现，修复可复现的实现缺陷，为后续逐模块调整留下入口。
这是一轮源码、故障注入和本地集成审查，不是对所有可能缺陷、自然语言语义或真实后端的无缺陷认证。

审查基于 `main` 的 `b0378aa4ceb1cc12ef40b2ca5aaf15f4a2b96d9c` 加本地工作树。
前序的组件化与 prompt 修改原本就未提交，本轮继续保留；最终文件指纹覆盖合并后的 `src/examples/tests` 源码、模板与静态文件。
没有启动新实验、真实训练、远程模型请求或修改历史冻结 run，也没有提交或推送本轮变更。

## 入口与模块覆盖

| 模块 | 本轮检查内容 | 详细审查记录 |
| --- | --- | --- |
| M01 共享契约/装配 | 身份与 Lab 回执，双循环职责，可替换边界，资源回调 | [装配/上下文/监控](context-integration.md) |
| M02 Host/Runner | 交付和确认顺序，lost response，取消、UNKNOWN、截止与清理 | [运行/持久恢复](runtime-recovery.md) |
| M03 decision policy | prompt 组件、schema/parser、拒绝前校验、模型身份与调用计量 | [本轮核对](context-integration.md)、[完整 M03/prompt 请求](../m03.md) |
| M04 工具/Sandbox | 动作授权、schema、路径约束、进程启动/超时/收尾 | [执行/后端](execution-backends.md) |
| M05 决策账本 | 依赖/引用/采纳归因，候选来源、评测索引，全量校验 | [装配/上下文/监控](context-integration.md) |
| M06 context/compact | 必要事实、archive 读回、常驻工作视图、旧摘要和恢复原子性 | [装配/上下文/监控](context-integration.md) |
| M07 feedback 生命周期 | capture、队列、native/legacy 输出、恢复、计费与发布重试 | [反馈](feedback.md) |
| M08 Attempt/claims | 来源验证、版本与 simulation 传承、候选生产者、解释与事实分离 | [反馈](feedback.md) |
| M09 反馈组件 | 必需 memory、可选提案、轨迹提取、组件配置和恢复一致性 | [反馈](feedback.md) |
| M10 Exploration/Lab | 指令分派、任务所有权、失败回执、训练与评测边界 | [执行/后端](execution-backends.md) |
| M11 训练/部署 | 指标合法性、initializer/parent、进程、权重与模型身份 | [执行/后端](execution-backends.md) |
| M12 评测/采纳 | 候选/协议绑定、缓存身份、扩展回执、失败前的应用顺序 | [执行/后端](execution-backends.md) |
| M13 存储/恢复 | 严格持久化、快照预检和失败收尾、恢复配置、检索一致性 | [运行/持久恢复](runtime-recovery.md) |
| M14 医疗/Qwen 装配 | 固定数据、host 模式、真实后端接线与示例差异、终态落盘 | [装配/上下文/监控](context-integration.md)、[执行/后端](execution-backends.md) |
| M15 监控/前端 | 公开投影、修订窗口、隐私字段、图/缓存/轮询/多语言 | [装配/上下文/监控](context-integration.md) |

“已审查”指上述边界已做本轮检查；用户对产品逻辑、prompt 内容和架构的验收复选框继续保留，未代为勾选。

## 主要修复

- **上下文与收尾**：旧摘要被错误算入不可压缩最小值；最终 monitor 错误阻止正式结果落盘；清理回调失败跳过其余释放；公开历史遗漏当前 Attempt 修订；decision learner 非法指标被缓存为成功。
- **反馈生命周期**：恢复结果绕过证据校验、宿主字段/计费被 processor 输出覆盖、未知恢复错误传播、截断解释被接受、simulation 来源丢失、组件收据与配置/输入不匹配、可选阶段恢复失败连带已完成 memory 等。
- **执行、训练和评测**：非法训练指标、系统同版本号缓存复用、扩展解释覆盖测量身份、prompt 读取失败发生在采纳之后、进程组在 leader 回收后才清理、argv/timeout 契约不一致、本地推理丢失 chat history/静默忽略原生约束。
- **持久恢复**：NaN 预检漏过后才发生副作用、自定义 hooks 被快照静默丢弃、失败恢复遗留目录/资源、SQLite 与内存的 Unicode 检索不一致。

完整复现、改动与每项测试名见对应分组记录。新增 **74 个测试方法**，包含额外参数化故障用例；同一缺陷跨模块出现时分别补齐，不把分组条数作为独立 bug 数。

## 明天逐模块核对

建议从 **M07 → M08 → M09 → M06** 开始，把反馈输入、事实与解释、组件产物及下一次 decision 实际收到的内容连起来看。
其后用 **M03 → M04 → M05 → M10–M12** 核对提案如何进入正式决策、执行、评测和采纳，再看 Host/恢复/领域/前端。

- 每模块的问题清单：[modules.md](../modules.md)。
- 逐条模型输入和 prompt 入口：[prompts.md](../prompts.md)。
- 九个通用 decision prompt、schema 和八份合成完整请求：[m03.md](../m03.md)。
- memory / 轨迹提取 / harness proposal 的当前工作流：[feedback-workflow.md](../feedback-workflow.md)。

仍保留已讨论的架构：meta-agent 做 decision；Lab 执行固定目标下的任务；反馈分身使用同一捕获权重和独立上下文；context/memory 能力必须激活而实现可替换；轨迹提取默认生效但可配置；harness 按需给提案，由正式 decision 选择后续构建。

## 不能由本轮通过推定的事项

- memory 自由文字是否保留了所有重要事实、旧记忆合并是否合理，仍需真实判例与语义验收。
- 此前 62,815-token 历史失败没有完整真实模型重放；有界视图不证明原始存储和遍历开销有界。
- 新组件 pipeline 并未自动替换所有领域示例的兼容 feedback_learning 路径；通用多 Lab 调度、harness 激活/回滚和轨迹转训练材料仍有后续工作。
- 本次没有验证真实模型部署、真实 SFT/RL、医疗开发分数或医学泛化；fixture、预检、Attempt 和提案不计为真实实验或系统改善。
- baseline 别名对应的实际权重、跨推理后端数值等价和 DSH 真实服务行为，仍不能从 CPU 回归推定。

## 最终验证材料

| 验证 | 最终结果 | 证据 |
| --- | --- | --- |
| Python 完整 optional 环境 | **1426 总计，1414 通过，12 跳过，0 失败/错误**；runner 84.854 秒、含 discovery 的实际耗时 85.831 秒 | [完整日志](full-python.log)、[结构化结果/逐项跳过原因](python-validation.json) |
| 前端 Node 测试 | **140 通过，0 失败/跳过** | 工具控制台输出；[汇总](validation.json)，未另保存独立 Node 日志 |
| 改动空白检查 | `git diff --check` 通过 | [汇总](validation.json) |
| 测试期间源码一致性 | 前后指纹无变化，`source_changes_during_run=[]` | [287 个文件指纹](source-manifest.json) |

测试于北京时间 **2026-11-04 00:51:33–00:52:59** 执行。
12 个跳过项是 6 个 DSH SDK 路径和 6 个真实推理 canary，没有静默补装依赖或调用外部服务。
完整 suite 包含已安装 LangMem/Trustcall 的本地固定 backend 测试；它们不是对真实模型记忆质量的验证。

可等价重跑：

```sh
PYTHONPATH=src:tests .local/langmem-validation/bin/python -m unittest discover -s tests -v
node --test tests/monitor/frontend/*.test.js
git diff --check
```

[validation.json](validation.json) 保存汇总、边界与日志 SHA；`source-manifest.json` 文件 SHA256 为
`33651ce0f2f8f8afdedc9455343986a8dde80e428fce221d3537ddc3ee03fa49`。
各定向回归与初始失败复现见四份分组记录。分组测试存在重叠，不能相加成独立通过项。

来源指纹包含 `src/examples/tests` 下 `.py/.js/.txt/.yml/.toml/.html/.css` 文件，不是整个项目、数据或真实模型权重的冻结包。
本轮没有测到的硬件/服务分支、历史失败的完整重放与语义边界仍按上文保留，不能由全量通过推定。
