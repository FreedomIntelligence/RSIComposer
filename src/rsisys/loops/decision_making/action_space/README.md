# Decision Action Space

操作的 `OperationSpec.read_only` 默认 false。read_only 是能力属性，默认权限配置用它筛选运行中操作；外部配置可开放写操作或所有已接入操作，具体执行仍由 Harness tools 负责。[QueryActionDefinition](queries.py) 将 Exploration 广告的查询转换为只读操作，默认装配自动注入。两个默认 Action Space 可通过 `extra_definitions=()` 添加这些定义，自定义操作仍使用同一验证接口。

独立于 [Decision Harness](../harness/README.md) 的动作定义与校验边界。覆盖本地 research 操作和发往 Exploration Harness 的系统指令；实际操作由 Harness 的 [tools](../harness/tools/README.md) 执行。

## 接口

[DecisionActionSpace](__init__.py) 提供：

- `describe() → tuple[OperationSpec, ...]`：描述本地操作名称与参数。
- `validate(call: OperationCall) → OperationCall`：校验并规范化操作，不执行副作用。
- `directive_modes() → tuple[str, ...]`：列出系统指令模式。
- `validate_directive(directive) → None`：校验指令模式；具体 payload 由 Exploration Harness 接收时校验。

默认模式为 execute / sft / rl / evaluate / configure_prompt。它们是 decision policy 可发出的系统指令，不是 ActorPolicy.act 的动作；外部系统也可直接向 Exploration 端点提交 Directive。PromptSpec(instructions) 的约定见[共享协议](../../contracts/README.md)。

整项训练通过可选的 `training_directive()` 定义 `train` 模式，解析 recipe_ref、method 与可选 apply_result 为 TrainSpec。它不包含在 builtin_directives() 中；需显式加入 catalog / directive_modes，并在 Exploration Harness 注册 TrainingCommandHandler、注入 learner。后端可独立完成 rollout 与训练；默认 `rl` 仍执行本地采样循环。完整接入见[使用指南](../../../../../README.md#接入整项训练任务)。

训练数据由外层 `input_ids` 选择；`origin` 仅描述实验父版本、初始权重和恢复方式。生成 schema 与 `TrainingOrigin` 校验一致：`restart` 的 checkpoint 只能省略或为 null；`weights_only` 可选权重引用；`full_state` 必须给出 checkpoint。仅暴露后端声明的恢复模式，不支持 restart 时必须显式提供 origin 与 resume_mode。支持 restart 时可省略 origin，具体装配负责确定默认初始化语义；checkpoint 导出能力不决定已有 checkpoint 的消费能力。

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

## Lab 声明动作出口

`Application(..., lab_factory=...)` / `build_demo(..., lab_factory=...)`
将 [LabExecution](../../contracts/execution.py) 的工具目录接入默认 action space。
Lab 只提供 OperationSpec 和 execute(call)，无需构造下述内部 Capability / validator 对象。
默认适配器校验 JSON 封装，具体参数语义由 Lab 在执行前校验；权限仍在 host 分派前复核。
可暴露通用代码执行工具，SFT、RL、evaluate 不再是该路径的固定动作分类。
示例、非阻塞任务与限制见 [实现说明](../../../../../docs/review/open-lab-execution-20261104.md)。

## 显式能力装配

[capabilities.py](capabilities.py) 提供独立的 `Capability(definition, handler)` 与 `Capabilities(capabilities=())`。已有 definition 必须提供 `OperationSpec` 和 `validate(arguments) → OperationCall`；只有 spec 时使用 `Capability.from_spec(spec, validator, handler)`。handler 接收校验后的参数字典，返回 `OperationResult | JobRef`。重复名、空名、缺失或不可调用的 validator / handler、无绑定的 provider 输出都会在装配时失败。

`CapabilityProvider.describe()` 返回有限的 `Iterable[Capability]`；仅提供 `discover()` 的自定义 provider 也可通过显式选择方法适配。`Capabilities.from_provider(provider, method="describe", explicit=())` 调用所选方法一次并合并显式能力；使用 `method="discover"` 选择发现入口，失败直接向调用方传播。不会自动发现 provider、调用 connect 或实现 MCP / OpenAPI 网络客户端。外部描述需由用户适配器转换成含校验器和 handler 的 Capability。

```python
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.capabilities import Capability, Capabilities
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools

# provider、note_definition、save_note 由调用方提供。
capabilities = Capabilities.from_provider(
    provider, explicit=(Capability(note_definition, save_note),),
)
# 全部显式提供时：Capabilities((Capability(note_definition, save_note),))
action_space = ModularDecisionActionSpace(capabilities.definitions)
tools = BoundDecisionTools(capabilities.bindings)
# 将 action_space 和 tools 注入现有 loop / Harness。
```

集合可迭代，因此多个来源可以用 `Capabilities((*first, *second, explicit))` 再次装配并检查重名。`definitions` 是 tuple，`bindings` 每次返回新字典；底层 definition 与 handler 仍由调用方持有，装配后不应改变定义。每个集合独立，无全局插件中心；空集合不提供任何能力。通用 shell / Python 可由用户显式提供 handler，本模块不默认开放 host shell，也不负责权限决策。schema 只作描述，参数验证仍由 validator 执行，执行权限由 Harness 的权限机制决定。

## 扩展系统指令

[directives.py](directives.py) 提供 `DirectiveSpec(mode, description, input_schema)`、`DirectiveDefinition(spec, parser)` 和 `DirectiveDefinitions(definitions=None)`。每个装配实例独立持有定义；无全局注册表。省略 definitions 使用内置模式，显式传入则替换整个目录；追加模式使用 `(*builtin_directives(), custom_definition)`。

`DirectiveSpec.accepts_input_refs=False` 表示该指令不接收额外输入引用；默认 policy 在省略 input_ids 时不继承历史产物，并拒绝非空 input_ids。Action Space 也在发出指令前校验这一声明。默认 True 保留显式选取或继承输入的行为，具体内容仍由执行后端校验。

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

## 外部权限配置

[ActionPermissions](access.py) 与能力目录分开。`operations` 和 `directives` 分别限制普通操作与实验指令，None 表示全部已接入能力；空集合表示全部关闭。`review_operations` 可以是默认 `"read_only"`、`"all"` 或操作名称集合，并与 operations 取交集。`allow_cancel=False` 关闭模型的取消权限，continue 始终作为不发起新修改的选择保留。

```python
from rsisys.loops.decision_making.action_space.access import ActionPermissions

permissions = ActionPermissions(
    operations={"search_literature", "prepare_data", "query_exploration_metrics"},
    directives={"train", "evaluate"},
    review_operations={"search_literature", "prepare_data", "query_exploration_metrics"},
    allow_cancel=False,
)
# 完全开放所有已接入能力：ActionPermissions.unrestricted()
```

给 ModularDecisionActionSpace / LocalDecisionActionSpace / DemoDecisionActionSpace 传入 `permissions=...`，或给 Application / build_demo 传入 `action_permissions=...`。运行期间可以替换 `loop.action_space.permissions`，下一次展示与执行检查立即使用新配置；不追溯取消已经授权并提交的任务。

能力目录的 describe/validate 负责接口定义与参数校验；[authorized_space(space, phase)](access.py) 提供当前授权视图并在执行前检查归一化后的请求。默认 Harness 对普通决策和复查都使用该视图。发现时的过滤不能代替执行检查。替代 Harness 需复用该视图或实现等效检查，外部 Python 代码和宿主权限不由这个接口隔离。

可选 `check(AccessRequest) -> bool` 支持参数/资源级规则：发现阶段 action=None，执行阶段 action 为归一化后的 OperationCall、Directive 或 ReviewDecision。回调可以读取外部动态配置；未知参数在发现阶段可先允许展示，实际执行时再校验。回调必须返回 bool。静态配置可用 to_dict/from_dict 保存，内置 Snapshot 保留它；自定义回调需显式快照适配。

开放全部能力不安装新工具、不自动开放 shell，也不取消单个在途实验指令等生命周期约束。需要自由代码执行时，由使用者显式接入对应能力和执行后端。
