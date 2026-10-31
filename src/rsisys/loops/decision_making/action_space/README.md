# Decision Action Space

独立于 [Decision Harness](../harness/README.md) 的动作定义与校验边界。覆盖本地 research 操作和发往 Exploration Harness 的系统指令；实际操作由 Harness 的 [tools](../harness/tools/README.md) 执行。

## 接口

[DecisionActionSpace](__init__.py) 提供：

- `describe() → tuple[OperationSpec, ...]`：描述本地操作名称与参数。
- `validate(call: OperationCall) → OperationCall`：校验并规范化操作，不执行副作用。
- `directive_modes() → tuple[str, ...]`：列出系统指令模式。
- `validate_directive(directive) → None`：校验指令模式；具体 payload 由 Exploration Harness 接收时校验。

默认模式为 execute / sft / rl / evaluate / configure_prompt。它们是 decision policy 可发出的系统指令，不是 ActorPolicy.act 的动作；外部系统也可直接向 Exploration 端点提交 Directive。PromptSpec(instructions) 的约定见[共享协议](../../contracts/README.md)。

## 当前实现

`ModularDecisionActionSpace(definitions, *, directive_modes=...)` 接收带 spec 和 `validate(arguments)` 的定义。名称必须唯一，校验结果必须是同名 OperationCall；输入与返回值作副本处理。JSON Schema 用于描述，实际校验由 definition 实现，不自动运行通用 schema 校验。

`LocalDecisionActionSpace()` 提供下列操作，配合 `LocalDecisionTools(artifacts)` 使用：

| 操作 | 参数校验 |
|---|---|
| prepare_data | 仅 text 字符串，允许为空 |
| search_literature | 仅 query 字符串，不可全为空白 |
| store_strategy | 仅 strategy 字符串，不可全为空白 |

`DemoDecisionActionSpace()` 只描述 prepare_data，配合 DemoDecisionTools 使用。构造 Action Space 不需要 sandbox 或产物存储；新增操作时同时提供纯校验定义与 Harness 执行绑定。

OperationCall 包含 name、arguments；OperationResult 描述工具执行后的 summary、artifacts、simulated、status 和可选 error。结果必须为 COMPLETED / FAILED / CANCELLED，FAILED 必须有 error，COMPLETED 不能有 error。异步结果回填、重试与记录由 Harness 管理。
