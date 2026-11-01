# 使用指南

以下命令在项目根目录执行。组件实现和限制见[组件导航](../README.md)。

## 启动

| 命令 | 用途 |
|---|---|
| `pixi run demo` | 内存中的准备与执行演示 |
| `pixi run loop-demo` | 两个 loop 的本地学习、Actor RL 和评测 |
| `pixi run coordination-demo` | 跨 loop 的 SFT → 评测 → RL 指令链 |
| `pixi run backend-demo` | 独立子进程中的真实 CPU 标量回归训练 |
| `pixi run local-demo` | Mock 模型、真实本地 echo、持久化与快照 |
| `pixi run test` | 行为测试 |

Pixi 任务通过 PYTHONPATH=src 加载源码，无需全局安装：

```bash
PYTHONPATH=src pixi run python -m rsisys --help
```

`__main__.py` 和安装后的 `rsisys` 命令都调用 `cli.main()`；`__init__.py` 不启动 CLI。

## 配置运行

- [demo.json](../configs/demo.json)：内存演示，由 config.py 装配。
- [local.json](../configs/local.json)：本地持久化，由 application.py 装配。
- [model-service.example.json](../configs/model-service.example.json)：HTTP 模型模板。

backends 定义调用后端，roles 分别绑定 decision_maker / actor；execution 选择 mock / local_process，learning 选择 none / mock。默认 Decision Harness 的 experience_limit 限制系统反馈条数，memory_limit 限制自身 history 与工作记忆条数；默认 Exploration Harness 仅使用 memory_limit。这些限制不计算 token。JSON 配置装配内置组合；Python 可通过下述工厂替换任一 loop。

```bash
PYTHONPATH=src pixi run python -m rsisys run \
  --config configs/local.json --data-dir runs/example \
  --literature path/to/paper.txt
```

配置 data_dir 相对配置文件解析，命令行 --data-dir 相对当前目录解析。已有目录不覆盖，同一目录不能由两个 Application 同时打开。--literature 导入普通文本供 [Decision tools](../src/rsisys/loops/decision_making/harness/tools/README.md) 使用。

本地目录包含 run.json、config.json、experience.sqlite3、decision_memory.json、exploration_memory.json、artifacts/、工作目录和 snapshots/。Experience SQLite 使用 schema 2，显式拒绝旧 schema，不自动迁移。

## 连接模型

复制模型服务模板，填写 endpoint、model_id、revision。endpoint 为 API base，后端请求其 /chat/completions；两个角色可选相同或不同后端。

需要鉴权时 credential_env 填环境变量名，启动前设置对应环境变量。不支持 JSON response_format 的服务可将 json_mode 设为 false，但内置模型 policy 仍要求回复合法 JSON。传输能力、版本验证和本地进程限制见 [Runtime](../src/rsisys/runtime/README.md)。

### vLLM 推理与部署

[适配器 README](../src/rsisys/adapters/README.md#serving) 提供 VLLMBackend、部署与 LoRA 启用的完整配置。
服务由操作者独立启动，endpoint 包含 `/v1`；默认只观察既有部署，动态 LoRA 需显式开启。
只有 READY 且模型身份匹配才允许绑定；服务别名不证明权重不可变，query / reattach 不重放加载。

## Researcher 查询实验

默认装配会将 Exploration 的可用查询注册到 Decision Action Space，名称为 `query_exploration_<name>`。researcher 可在普通决策或运行中 review 中调用；review 默认权限仅开放 read_only 操作，可通过外部权限配置开放其他操作，默认最多调用 4 次，再作 continue / cancel 判断。训练继续由后端执行，控制器的同步查询应由对应后端设置超时。

```json
{"kind":"invoke","operation":"query_exploration_metrics","arguments":{"filters":{"directive_id":"实际指令ID"}}}
```

模型在下一次调用的 `operation_results` 中读到结果，`data` 包含来源、读取时间、内容版本、正文和引用。默认可查 status、metrics、tasks、logs、candidates、model、artifacts、experience；提供本地 ArtifactStore 时还有 artifact_read，可按已知引用分页读取小型文本。logs 是已记录的本地摘要，不是未接入的远程训练日志。指标来自最近一次后端轮询，查询不会触发训练推进；后端提供时保留原始样本时间，未知时间保持未知。

外部 Harness 可实现 [QueryProvider](../src/rsisys/loops/contracts/queries.py) 的 `describe_queries()` 和 `query(request)` 来暴露其他实验信息；未实现时不广告这些能力。自定义装配可使用 [QueryTools](../src/rsisys/loops/decision_making/harness/tools/queries.py) 的 definitions 和 bindings 分别注入 Action Space 与 tools。协议只统一查询封装，不转换训练数据格式，也不要求把执行历史搬进 Experience。

Qwen toy 已绑定同样的查询入口。`tests/test_researcher_integration.py` 使用 CPU HTTP fixture 验证「后端训练进度 → researcher 查询 → 模型判断 → 实验继续完成」；该测试不加载真实 Qwen 权重。

## 配置能力与权限

能力可以显式定义，也可以通过自定义 provider 发现后注入；[Capabilities](../src/rsisys/loops/decision_making/action_space/capabilities.py) 将操作定义与执行绑定配对，装配时拒绝重复名称和缺失 handler。它不包含网络发现客户端，也不自动开放发现的能力。

```python
from rsisys.application import Application
from rsisys.loops.decision_making.action_space.access import ActionPermissions

with Application(config, action_permissions=ActionPermissions.unrestricted()) as app:
    app.run()
```

完全开放仅针对已接入能力，仍保留参数校验、实验容量和交付一致性等协同规则。默认使用 ActionPermissions()：普通决策允许全部已接入能力，运行中仅允许只读操作。也可指定 operations / directives / review_operations 名称集合，或 check 回调检查实际请求的参数和资源。运行时替换 `app.runner.decision_making.action_space.permissions` 即可更新默认 Harness 的权限；已提交任务不会被自动撤回。静态权限配置会随内置快照恢复；外部权限回调需自定义快照适配。

显式/provider 能力装配和参数级检查见 [Action Space](../src/rsisys/loops/decision_making/action_space/README.md)。整体替换 Harness 时，应在操作执行前复用 authorized_space 或提供等效权限检查。

## 使用 Python 接入组件

[Application](../src/rsisys/application.py) 和 [build_demo](../src/rsisys/config.py) 接受可选的 `exploration_factory(context)` 与 `decision_factory(context, exploration)`，分别返回 ExplorationLoop 与 DecisionMakingLoop。先构造 exploration，再构造 decision；未传入的一侧使用 [assembly.py](../src/rsisys/assembly.py) 的 `build_default_exploration` / `build_default_decision`，也可在自定义工厂内直接调用这两个函数。默认 decision 的 allowed_actions 从已构造的 exploration.action_space 读取；directive_modes 根据实际未被覆盖的内置 command handlers，以及 learner/evaluator、模型和 rollout 能力推导，不从 with_learning 猜测。

`AssemblyContext` 提供共享 experience、artifacts、backends、sandbox、roles，以及 data_dir（demo 为 None）、execution、with_learning、experience_limit 和 memory_limit。Application 管理共享资源的创建与关闭，自定义工厂无需重复创建存储或构造后修改属性。例如只替换 exploration：

```python
from rsisys.application import Application
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions, builtin_directives
from rsisys.loops.exploration.loop import ExplorationLoop

# config、MyHarness 和 MyActionSpace 由调用方提供；本例 MyHarness 支持 execute。
def exploration_factory(context):
    harness = MyHarness(context.experience, context.artifacts)
    context.add_cleanup(harness.close)  # 若实现持有自有资源，在取得资源后立即注册。
    context.exploration_directives = DirectiveDefinitions(
        definition for definition in builtin_directives() if definition.spec.mode == "execute"
    )
    return ExplorationLoop(harness, MyActionSpace())

with Application(config, exploration_factory=exploration_factory) as app:
    app.run()
```

未知自定义 Harness 必须显式设置 `context.exploration_directives`，或提供 `decision_factory`。新增或覆盖 handler 的模式不会自动向默认 decision 开放，也需 declaration（`DirectiveDefinitions`）或由自定义 decision_factory 配置。显式 catalog 完整指定可用模式及参数解析器，会同时注入默认 policy 和 Action Space；它不是追加列表。对默认 Harness，装配还会检查声明的 handler 是否存在、内置模式所需能力是否齐备。

工厂自有资源用 `context.add_cleanup(callback)` 登记无参清理函数。回调逆序执行且只执行一次；某个回调失败不跳过其余清理。正常关闭和任一工厂装配失败均会清理；Application 在回调完成后关闭共享存储与执行后端。替代 Harness 不必提供 policy 或 memory。

`build_demo` 保持 `(runner, store)` 返回形式，同样接受两个工厂。返回的 runner 支持 `close()` 和上下文管理；自定义工厂使用资源时须显式关闭：

```python
from rsisys.config import build_demo

runner, store = build_demo(demo_config, exploration_factory=exploration_factory)
with runner:
    runner.run(demo_config.objective, rounds=demo_config.rounds)
```

工厂仅通过 Python 参数注入，不写入 JSON 配置，也不会自动获得快照恢复支持；自定义装配仍需显式集成 Snapshot。

先组合 Harness 与独立 Action Space，再交给 loop。默认实现内部的 policy、tools、memory、learner/evaluator 通过 Harness 构造函数注入：

```python
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.action_space import LocalDecisionActionSpace
from rsisys.loops.decision_making.harness.tools.local import LocalDecisionTools
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.action_space import EchoActionSpace

# decision_policy、actor_policy、environment、experience 和 artifacts 由调用方提供。
decision = DecisionMakingLoop(
    DefaultDecisionHarness(decision_policy, LocalDecisionTools(artifacts), experience),
    LocalDecisionActionSpace(),
)
exploration = ExplorationLoop(
    DefaultExplorationHarness(actor_policy, environment, experience),
    EchoActionSpace(),
)
```

内部控制通过 `loop.harness`，动作定义通过 `loop.action_space` 访问。例如 `decision.harness.resume(job, result)`、`exploration.harness.submit_learning(...)`；本地执行使用 `exploration.harness.run(directive, exploration.action_space)`，本地 RL 推进使用 `exploration.harness.step_rl(exploration.action_space)`。loop 不提供旧 policy / memory / space 属性或本地控制方法别名。

默认实现将执行放在现有组件中：Decision 的 tools 统一普通操作与运行中 review 的执行/记录；Exploration 的 environment、learner、evaluation 和 rl.py 分别负责 episode、学习、评测和本地 RL 推进。对外仍使用上述 Harness 入口，以保留占用检查；内部执行组件不负责跨 loop 调度。

也可直接实现 Harness 协议并整体替换，无需提供 memory 或 tools 子接口。示例：

- [loop_internals.py](../examples/loop_internals.py)：训练请求、查询、版本应用和独立评测。
- [coordination.py](../examples/coordination.py)：自定义 policy 消费 last_feedback 并发出下一条指令。
- [backend_training.py](../examples/backend_training.py)：通过可选 train 指令委托独立 CPU 训练进程并回传结果。

两个默认 Harness 内部分别使用 DecisionMemory / ExplorationMemory，可在默认实现中替换。DecisionContext 同时包含全部来源的 experience、自身 history 和 memories；ActorContext 读取当前任务 history 与自身 memories，不默认注入系统经验。scope 仅标记反馈来源，优化对象由 Decision policy 选择。

默认 Harness 的本地 RL 请求使用 trajectory_refs 引用 Actor 本地 history；Feedback.evidence_refs 引用系统记录。两侧训练请求均为后端中立：method / recipe_ref 是非空字符串，base 为 ModelRef，引用要求正确类型的 tuple，但允许为空。DecisionTrainingRequest 保留 `(base, experience_refs, recipe_ref)` 位置参数，追加 `data_refs=()` 与 `method="sft"`；可使用系统反馈 experience_refs、外部数据 data_refs，或由后端根据 recipe 获取数据。

调用 runner.step(objective) 推进一步，或 runner.run(objective, rounds=1) 推进到本次反馈交付额度完成、等待或结束；waiting 后由调用方安排下一次推进。Runner 通过 decision_making.receive_feedback(directive, feedback) 交付反馈。取消用 runner.cancel()，之后继续推进直到终态。协同语义见[系统架构](ARCHITECTURE.md)。

## 选择运行策略

Python 装配的 `Runner`、`build_demo` 和 `Application` 均接受 `run_policy=` 与 `cleanup_timeout_seconds=30`。下例复用前面构造的 decision / exploration；省略参数即使用 `ResearcherRunPolicy()`，研究者 finish 后停止。时间模式允许研究者提前 finish 后再开启一轮，持续到指定时限：

```python
from rsisys.runtime.runner import Runner
from rsisys.runtime.run_policy import TimeRunPolicy

runner = Runner(
    decision, exploration,
    run_policy=TimeRunPolicy(seconds=600, restart_delay_seconds=1),
    cleanup_timeout_seconds=30,
)
result = runner.run("改进 Actor 并评测候选", rounds=1)
```

`rounds=1` 限制本次调用交付的指令数量，不表示研究轮次数或运行策略已完成；`completed` 也不证明目标达成。保留同一 Runner，在 `waiting` 后由事件循环或调度器安排后续 `step()` / `run()`；返回 `finished` 后检查 `shutdown_complete` 与 `run_stop` 中的未确认任务，清理到限不等于全部任务已终结。Runner 不在轮次间 sleep；新轮次通过 `restart_research` 显式开启，保留 memory、经验和候选。自定义 Decision 端点要支持此重启接口，才能接入跨轮次继续研究。

目标模式使用 `GoalRunPolicy(goal, restart_delay_seconds=1, max_seconds=None)`，其中 goal 是可替换的 `goal(context) -> bool` callable，读取 RunContext 中已交付的 outcomes。可用 `MetricGoal` 表达绑定评测协议的指标阈值目标，默认不接受 Mock 评测；复杂目标可自行验证证据。`max_seconds=None` 表示没有目标模式的运行时限；需要时间兜底时显式传入秒数。例如复用已有配置、按验证集 accuracy 目标装配：

```python
from rsisys.config import build_demo
from rsisys.runtime.run_policy import GoalRunPolicy, MetricGoal

# protocol_ref 须对应 evaluator 实际使用的评测协议。
goal = MetricGoal("accuracy", 0.8, protocol_ref="validation-v1")
runner, store = build_demo(
    demo_config,
    run_policy=GoalRunPolicy(goal, max_seconds=600),
    cleanup_timeout_seconds=30,
)
with runner:
    result = runner.run(demo_config.objective, rounds=1)
    # 持续运行时，在此资源作用域内由调用方调度后续 step / run。
```

上例阈值与协议由调用方选择；默认 Mock evaluator 不能满足该目标。损失等越低越好的指标使用 `higher_is_better=False`；`allow_simulated=True` 仅显式允许模拟结果参与判断。`Application(config, run_policy=..., cleanup_timeout_seconds=30)` 使用相同注入方式。

停止后的 drain / cancel 由 policy decision 指定；超时默认 cancel，清理另受 `cleanup_timeout_seconds`（默认 30 秒）限制。停止禁止新实验，但继续确认与交付已有反馈；未确认任务不会被记为成功。收尾异常原样抛给调用方，保留状态供后续 `step()` 重试；时限检查依赖调用方继续推进，不会自动硬杀后端。完整接口、结果与同步调用限制见 [Runtime](../src/rsisys/runtime/README.md#运行策略与研究轮次)。

## 系统指令与 Actor 动作

Decision Action Space 覆盖 research 操作和 execute / sft / rl / evaluate / configure_prompt 系统指令。Directive 由 decision policy 生成，或由外部系统通过 `exploration.accept(directive)` 提交，再调用 advance / inspect / cancel / acknowledge 管理生命周期。外部直接提交时，由调用方负责反馈消费与 acknowledge。

Exploration Action Space 仅包含 Actor 环境动作。execute episode 和 RL rollout 使用 ActorPolicy.act；SFT、prompt 配置和评测绕过它。真实推理评测需要替换 evaluator；当前默认训练与评测仍是 Mock。`configure_prompt` 使用 `PromptSpec(instructions)` 替换 Actor 指令文本，空字符串清空配置，见[共享协议](../src/rsisys/loops/contracts/README.md)。

## 接入整项训练任务

默认 `rl` 继续由 Harness 执行本地 rollout → learner 训练 → 应用版本 → 下一轮 rollout。若要由外部后端自主完成整个任务的采样与训练，显式启用 `train`：决策侧加入 `training_directive()`，探索侧注册 `TrainingCommandHandler()` 并注入实现作业协议的 learner。

```python
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinitions, builtin_directives, training_directive,
)
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler

catalog = DirectiveDefinitions((*builtin_directives(), training_directive()))
# 手动装配时，将 catalog 同时传给 Decision policy 与 Action Space，允许 train。
# 工厂装配时，设置 context.exploration_directives = catalog；
# catalog 中其他模式也必须具备对应的 handler 和能力，可按实际能力筛选。
exploration_harness = DefaultExplorationHarness(
    actor_policy, environment, experience,
    learner=external_learner,
    command_handlers={"train": TrainingCommandHandler()},
)
```

上述 actor_policy、environment、experience 与 external_learner 由调用方提供。`train` payload 为 `TrainSpec(recipe_ref, method, base=None, apply_result=False, origin=None)`；handler 将 input_refs 作为 data_refs，以单个 ActorTrainingRequest 提交，不预先生成 trajectory_refs，也不运行本地 RL 循环。后端决定 method、数据与 rollout 的语义，Harness 复用提交重试、轮询、取消、证据记录和可选版本应用。直接向 Exploration 端点提交时仍需注册 handler，调用方负责反馈消费与 acknowledge。

默认 MockActorLearner 只支持 sft / rl，并在提交时分别要求非空 data_refs / trajectory_refs，不能替代自主采样的后端。具体框架通过 Harness 外的适配器接入，原生配置与数据格式仍由调用方明确提供。

### 具体训练接入与组合版本

[适配器 README](../src/rsisys/adapters/README.md#training) 提供 ProcessTrainingBackend 的完整 recipe、
原生命令、数据格式和结果读取接口。默认 worker 支持 Transformers / PEFT SFT，外部依赖自行安装；
LLaMA-Factory 只是原生命令扩展，未验证 verl 适配。原生作业重连保留同一 JobRef，不自动恢复控制器。

[TrainingOrigin](../src/rsisys/loops/contracts/README.md#可选训练来源与后端能力) 区分实验父系统与初始权重。
TrainSpec 可传 origin；显式 base 必须等于 initial_model，成功候选须保留 parent_system。
restart / weights_only / full_state 必须按后端声明执行；默认 worker 不支持 full_state，不隐式转换数据。

[SystemRegistry](../src/rsisys/runtime/README.md#系统组合与显式应用) 管理组件清单。下例仅演示引用级 prompt
组合；initial_model、prompt_ref 和 next_prompt_ref 由调用方提供：

```python
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.runtime.systems import ReferenceSystemActivator, SystemRegistry

initial = SystemVersion("system-v1", {"prompt": prompt_ref}, initial_model)
systems = SystemRegistry(initial, ReferenceSystemActivator(initial_model))
proposal = Candidate("prompt-v2", initial_model.revision,
                     {"prompt": next_prompt_ref}, ("actor",), parent_system=initial.version)
systems.register(proposal)  # current 仍为 system-v1。
systems.activate(proposal.candidate_id, idempotency_key="apply-prompt-v2")
```

register 不采纳候选；activate 检查父版本等于 current，但不做质量评测。ReferenceSystemActivator
不部署或修改角色；真实组合需显式 SystemActivator，registry 与外部回执的恢复也需单独集成。
仅 SystemActivationRejected 确认无副作用或已回滚时释放 pending；未知异常继续冻结原意图。

运行 `pixi run backend-demo` 查看 [backend_training.py](../examples/backend_training.py)。示例使用独立子进程在 CPU 上优化标量回归模型，返回真实权重产物与 MSE；它展示训练作业接入，不是 LLM 训练、RL rollout 或 verl 的现成适配。请求字段及结果要求见 [Actor learner](../src/rsisys/loops/exploration/harness/learner/README.md) 和 [Decision learner](../src/rsisys/loops/decision_making/harness/learner/README.md)。

## 模型启用与外部引用

真实后端返回训练结果后，若要 apply_learning 或 apply_result，需要在对应 Harness 构造时传入 `model_activator`。该适配器实现 `activate(result, *, idempotency_key)`：解析候选权重引用，完成部署或确认既有部署，并返回与 result.model 相同的 ModelRef。等待状态抛出 ModelActivationPending，由调用方稍后继续推进；不在框架内部持续轮询。角色绑定仅在确认后改变。

`backend-demo` 还展示标量模型的加载与推理：独立进程输出权重，ScalarModelService 加载并确认，Harness 更新绑定，最后通过新版本进行预测。具体推理服务通过独立适配器接入；prepare / query / release / reattach 的共享协议见 [backend.py](../src/rsisys/loops/contracts/backend.py)。部署 JobRef 成功不等于 READY，适配器必须核对精确模型身份后才允许绑定。

外部数据可以直接使用 `ArtifactRef("dataset-id", "dataset", "s3://bucket/path")` 等引用，无需先写入框架存储；URI 和数据格式由使用者的适配实现解释。未知的 version/digest 保留 None，不伪造摘要。

## 恢复运行

CLI 在空闲边界生成快照，并在 JSON 输出中给出路径：

```bash
PYTHONPATH=src pixi run python -m rsisys inspect --snapshot <快照目录>
PYTHONPATH=src pixi run python -m rsisys resume \
  --snapshot <快照目录> --data-dir runs/restored --rounds 1
```

恢复目标必须是新目录，原目录不被修改。快照保存内置 Harness 的控制状态、两份 memory 和共享 Experience；内置快照仅兼容默认 ResearcherRunPolicy 和默认清理预算；time / goal、自定义运行策略或清理预算会被明确拒绝，需显式集成保存恢复。整体替换 Harness 或替换持久化组件也需单独集成。旧快照受源码版本精确匹配约束，不提供自动迁移。保存内容、版本一致性要求及自定义组件边界统一见 [Snapshot](../src/rsisys/snapshot/README.md)。

CLI 的 `resume --snapshot` 仍只处理空闲快照。持续研究可使用 Python 的运行恢复入口：

```python
from pathlib import Path
from rsisys.application import Application, ApplicationConfig
from rsisys.runtime.run_policy import TimeRunPolicy

config = ApplicationConfig.load(Path("configs/local.json"))
with Application(config, durable=True, run_policy=TimeRunPolicy(3600)) as app:
    app.runner.step(config.objective)

# 重新打开原工作目录；不刷新已经消耗的时间预算。
with Application.resume(config.data_dir) as app:
    app.runner.step(app.config.objective)
```

每个 Runner `step/cancel/update` 前记录意图，结束后原子保存回执、控制状态与运行策略。
离线时间也计入已开始的时间预算。内置恢复覆盖串行默认组合、Mock 训练/评测作业、反馈交付，
不重建活跃 shell、GPU 进程或并行实验。外部组件用 `RecoveryContract.capture/restore` 显式
保存并重新关联真实 job；恢复不能再次提交作业。发生未确认操作时，运行被标记为
`RecoveryRequired`，需根据回执调用 `app.recovery.reconcile(repair, evidence=...)`；
它没有盲目重试或清除未知状态的快捷入口。完整边界见 [Runtime](../src/rsisys/runtime/README.md)。

## 自主研究与并行实验

[Qwen toycase](../examples/qwen_toy/README.md) 使用模型自主选择的步骤，外部配置能力、权限和预算。提前结束、没有训练或未采纳候选都按实际结果记录，不自动补齐工作流。

[parallel_experiments.py](../examples/parallel_experiments.py) 演示两个隔离实验、乱序完成、运行中更新和候选比较。它使用明确标记的 Mock 后端与脚本 policy，仅验证接口；不代表实际 GPU 并行训练。装配使用 `ConcurrentExplorationHarness(factory, max_in_flight=2)`、`DefaultDecisionHarness(max_outstanding=2)` 与 `Runner(parallel=True)`；factory 决定后端资源放置。

`ExperimentControls(exploration_loop, decision_loop)` 的 definitions / bindings 接入普通 Action Space / tools，可供模型调用 inspect、cancel、update、compare，沿用外部权限。动态更新只对实现 validate_update / update 的后端开放；它不是任意训练参数热更新。比较与候选应用仍检查来源、协议和版本，不隐式改动其他实验的模型。

反馈提取可通过 `DefaultDecisionHarness(..., feedback_selector=...)` 替换；接口和默认实现见 [Experience](../src/rsisys/experience/README.md)。工作区基础读取见 [tools](../src/rsisys/loops/decision_making/harness/tools/README.md)，宿主资源隔离由沙箱/部署层实现。

## 独立医学研究示例

[medical README](../examples/medical/README.md) 提供数据格式、prepare / check / run 命令、Python 装配、
固定评测及证据边界。领域代码位于 examples/medical，通过两个默认 Harness 与 Runner 独立装配。
prepare / check 只读；CLI run 支持 prompt 研究，原生训练通过 Python 注入 learner 与显式输入绑定。
缓存数据准备 CLI 与单 GPU 自主 LoRA SFT 入口、预算和恢复限制见 [medical README](../examples/medical/README.md#single-gpu-autonomous-experiment)。

医疗 prompt / evaluate / apply_candidate 声明 `accepts_input_refs=False`，省略 input_ids 即为空，
显式非空拒绝并可进入有界输出纠错；固定评测后端仍拒绝非空 data_refs。train 保留输入引用。
原生 worker 的 tokenized JSONL 与医学题目格式不同，转换和消费证据由接入实现提供。
[native.py](../examples/medical/native.py) 的 MedicalTrainingBackend / MedicalServing 提供显式
tokenize / 进程训练包装与候选路由准备；完整接线见上述 README。
评测可注入 prepare_inference / evaluator_factory，默认 poll 每次至多推进一个样本；明确应用拒绝返回反馈。

已验证 native-final 的 3 步训练、6 行／48 input tokens 消费及同 job 重连一致。
真实医疗运行仅为 prompt→评测→持平拒绝 apply→finish，current 未变；使用 4 条 train 和
4 条原 source TRAIN 子集作 dev。尚无真实 medical SFT / LoRA 部署或训练→评测→apply 全链路证明，
这些接线验证不代表医学增益。

## 默认长程研究能力

默认 Harness 的 `ContextCompactor.compact(context, recent_limit=...)` 可替换；
`BasicContextCompactor` 以字符预算保留近期完整记录，对超出的历史生成带来源与摘要哈希的摘录导航，
完整历史仍可查询。它不是语义推理模型，也不是整个模型输入的 token 上限；目标、能力 schema、
候选和运行控制字段不被静默裁剪。`IdentityContextCompactor` 可用于已有自己上下文管理的策略。

`ResearchAgenda` 保存模型维护的 agenda、todo、stage_summary；默认装配注册
`research_read`、`research_patch`、`research_archive`。模型按稳定 ID 增量修改，带 expected_revision
避免覆盖更新，分页读取完整档案；短上下文只显示导航。没有要求先写计划再训练，也不把模型笔记
当作已验证结论。这些能力遵循相同的外部权限设置，组件说明见
[Decision Harness](../src/rsisys/loops/decision_making/harness/README.md)。

## 使用 DSH Harness

可选依赖固定为测试过的 SDK / runtime 版本，默认安装不引入 Node 或 DSH：

```bash
python -m pip install -e '.[dsh]'
PYTHONPATH=src python examples/dsh_smoke.py
```

smoke 启动真实 DSH runtime 和本地确定性 HTTP 响应服务，验证会话恢复及 RSISys 指令链路；
不使用真实大模型，不训练真实 Actor。

```python
from pathlib import Path
from rsisys.application import Application, ApplicationConfig
from rsisys.loops.decision_making.harness.dsh import DSHConfig, dsh_decision_factory

config = ApplicationConfig.load(Path("configs/local.json"))
dsh = DSHConfig(
    home=Path("/absolute/path/to/dsh-home"),
    workspace=Path("/absolute/path/to/research-workspace"),
    model="deepseek-v4-flash", session_id="research-001",
    api_key_env="DEEPSEEK_API_KEY",  # 只存环境变量名
)
with Application(config, decision_factory=dsh_decision_factory(dsh), durable=True) as app:
    app.runner.step(config.objective)  # waiting 时由调用者稍后继续 step
```

DSH 维护持久会话、模型调用及原生压缩；适配器复用默认回执协调，通过异步 worker 提出结构化
Decision。真正的操作仍在 Runner 线程执行，每次重新检查 Action Space 权限和 RunPolicy。
DSH 自身 turn 完成只产生一个提案；`finish` 是否结束整个 RSI 由外部 RunPolicy 决定。
原生 shell / 文件操作在附带的 sdk-minimal patch 中关闭，所有研究能力经 RSISys 提交。
这不是操作系统隔离，也尚未提供任意 DSH native tool 与 RSISys 权限之间的自动映射。

`base_url` 由该 DSH provider 解释；验证过的 rc2 `deepseek-official` 请求使用 `/v1/messages`，
不能假定任意只实现 `/v1/chat/completions` 的服务可直接使用。模型身份记录为 provider-managed，
不宣称验证了远端权重版本；未取得的 token 用量保持未知。SDK 官方接口见
[DeepSeek Harness Python SDK](https://github.com/deepseek-ai/deepseek-harness/blob/master/python/sdk/README.md)。

恢复 DSH 装配时需向 `Application.resume(..., decision_factory=同一工厂)` 提供原配置。
已缓存的完整提案可恢复；未完成的推理会保留为待核对状态，不自动重复请求。
`harness.resume_interrupted_inference(evidence=...)` 显式恢复原生会话、记录核对说明并丢弃未确认建议，
不发送新模型输入；若 Runner journal 同时中断，应在 `app.recovery.reconcile` 回调中调用。
`DSHConfig(resume=True)` 单独只恢复 DSH 原生会话，不等于恢复 RSISys 实验或外部进程。
