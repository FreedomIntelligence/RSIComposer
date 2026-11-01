# Decision Action Space

操作的 `OperationSpec.read_only` 默认 false。运行中 researcher review 只允许显式标记为只读的操作；这是接入者的行为声明，具体执行仍由 Harness tools 负责。[QueryActionDefinition](queries.py) 将 Exploration 广告的查询转换为只读操作，默认装配自动注入。两个默认 Action Space 可通过 `extra_definitions=()` 添加这些定义，自定义操作仍使用同一验证接口。

独立于 [Decision Harness](../harness/README.md) 的动作定义与校验边界。覆盖本地 research 操作和发往 Exploration Harness 的系统指令；实际操作由 Harness 的 [tools](../harness/tools/README.md) 执行。

## 接口

[DecisionActionSpace](__init__.py) 提供：

- `describe() → tuple[OperationSpec, ...]`：描述本地操作名称与参数。
- `validate(call: OperationCall) → OperationCall`：校验并规范化操作，不执行副作用。
- `directive_modes() → tuple[str, ...]`：列出系统指令模式。
- `validate_directive(directive) → None`：校验指令模式；具体 payload 由 Exploration Harness 接收时校验。

默认模式为 execute / sft / rl / evaluate / configure_prompt。它们是 decision policy 可发出的系统指令，不是 ActorPolicy.act 的动作；外部系统也可直接向 Exploration 端点提交 Directive。PromptSpec(instructions) 的约定见[共享协议](../../contracts/README.md)。

整项训练通过可选的 `training_directive()` 定义 `train` 模式，解析 recipe_ref、method 与可选 apply_result 为 TrainSpec。它不包含在 builtin_directives() 中；需显式加入 catalog / directive_modes，并在 Exploration Harness 注册 TrainingCommandHandler、注入 learner。后端可独立完成 rollout 与训练；默认 `rl` 仍执行本地采样循环。完整接入见[使用指南](../../../../../docs/GUIDE.md#接入整项训练任务)。

候选应用通过可选的 `apply_candidate_directive()` 定义 `apply_candidate` 模式，同样不加入默认模式。参数只允许 candidate_id，必须是非空字符串且存在于传给 parser 的 candidates 中，返回 `ApplyCandidateSpec(candidate_id)`，不需要 job ID。将该 builder 显式加入与 policy 共享的 catalog，并在探索侧装配同名 handler；例如可将 train / evaluate / apply_candidate / execute 组合成训练、评测、选用和执行链路。模型可发送 `{"kind":"explore","mode":"apply_candidate","strategy":"use evaluated candidate","arguments":{"candidate_id":"candidate-id"}}`，候选来源沿用跨轮 context 和最新反馈兼容逻辑。

## 当前实现

`ModularDecisionActionSpace(definitions, *, directive_modes=..., directives=...)` 接收带 spec 和 `validate(arguments)` 的定义。名称必须唯一，校验结果必须是同名 OperationCall；输入与返回值作副本处理。JSON Schema 用于描述，实际校验由 definition 实现，不自动运行通用 schema 校验。

`LocalDecisionActionSpace()` 提供下列操作，配合 `LocalDecisionTools(artifacts)` 使用：

| 操作 | 参数校验 |
|---|---|
| prepare_data | 仅 text 字符串，允许为空 |
| search_literature | 仅 query 字符串，不可全为空白 |
| store_strategy | 仅 strategy 字符串，不可全为空白 |

`DemoDecisionActionSpace()` 只描述 prepare_data，配合 DemoDecisionTools 使用。构造 Action Space 不需要 sandbox 或产物存储；新增操作时同时提供纯校验定义与 Harness 执行绑定。

OperationCall 包含 name、arguments；OperationResult 描述工具执行后的 summary、artifacts、simulated、status 和可选 error。结果必须为 COMPLETED / FAILED / CANCELLED，FAILED 必须有 error，COMPLETED 不能有 error。异步结果回填、重试与记录由 Harness 管理。

## 扩展系统指令

[directives.py](directives.py) 提供 `DirectiveSpec(mode, description, input_schema)`、`DirectiveDefinition(spec, parser)` 和 `DirectiveDefinitions(definitions=None)`。每个装配实例独立持有定义；无全局注册表。省略 definitions 使用内置模式，显式传入则替换整个目录；追加模式使用 `(*builtin_directives(), custom_definition)`。

parser 接收 `(arguments, candidates)`，返回 payload；它负责参数验证，JSON Schema 仅向模型描述参数。payload 应为可被 JSON 序列化的值或仅包含这些值的 dataclass，以适配默认 Harness 的 history 与事件记录。返回可序列化 payload 不代表具备快照支持。

ModelDecisionPolicy 传入跨轮 context.candidates，并兼容合并 last_feedback.candidates，同 ID 只传一次。内置 evaluate 按 candidate_id 查找此集合，允许重复评测较早的训练候选，不要求最近反馈再次携带候选。

```python
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveSpec, DirectiveDefinition, DirectiveDefinitions, builtin_directives,
)

# parse_custom(arguments, candidates) 由调用方提供，必须完成参数校验。
catalog = DirectiveDefinitions((*builtin_directives(), DirectiveDefinition(
    DirectiveSpec("custom", "Run a custom system instruction", {
        "type": "object", "properties": {}, "additionalProperties": False,
    }), parse_custom,
)))
space = LocalDecisionActionSpace(directives=catalog, directive_modes=("execute", "custom"))
# 构造 ModelDecisionPolicy(..., directives=catalog)，共享同一份定义。
# 构造 DefaultExplorationHarness(..., command_handlers={"custom": handler})。
```

`directive_modes` 是允许发出的子集，必须存在于 catalog；不提供时默认使用 catalog 全部模式。探索侧必须提供同名 [CommandHandler](../../exploration/harness/README.md#扩展系统指令)。新模式或覆盖内置模式均需显式快照集成。
