# 使用指南

以下命令在项目根目录执行。组件实现和限制见[组件导航](../README.md)。

## 启动

| 命令 | 用途 |
|---|---|
| `pixi run demo` | 内存中的准备与执行演示 |
| `pixi run loop-demo` | 两个 loop 的本地学习、Actor RL 和评测 |
| `pixi run coordination-demo` | 跨 loop 的 SFT → 评测 → RL 指令链 |
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

backends 定义调用后端，roles 分别绑定 decision_maker / actor；execution 选择 mock / local_process，learning 选择 none / mock。默认 Decision Harness 的 experience_limit 限制系统反馈条数，memory_limit 限制自身 history 与工作记忆条数；默认 Exploration Harness 仅使用 memory_limit。这些限制不计算 token。配置入口目前只装配内置组合，Harness 构造函数参数见各 Harness 文档。

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

## 使用 Python 接入组件

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

也可直接实现 Harness 协议并整体替换，无需提供 memory 或 tools 子接口。示例：

- [loop_internals.py](../examples/loop_internals.py)：训练请求、查询、版本应用和独立评测。
- [coordination.py](../examples/coordination.py)：自定义 policy 消费 last_feedback 并发出下一条指令。

两个默认 Harness 内部分别使用 DecisionMemory / ExplorationMemory，可在默认实现中替换。DecisionContext 同时包含全部来源的 experience、自身 history 和 memories；ActorContext 读取当前任务 history 与自身 memories，不默认注入系统经验。scope 仅标记反馈来源，优化对象由 Decision policy 选择。

默认 Harness 的 RL 请求使用 trajectory_refs 引用 Actor 本地 history；Feedback.evidence_refs 引用系统记录。决策者自身训练仍使用系统反馈的 experience_refs。

调用 runner.step(objective) 推进一步，或 runner.run(objective, rounds=1) 推进到轮次完成、等待或结束；waiting 后由调用方安排下一次推进。Runner 通过 decision_making.receive_feedback(directive, feedback) 交付反馈。取消用 runner.cancel()，之后继续推进直到终态。协同语义见[系统架构](ARCHITECTURE.md)。

## 系统指令与 Actor 动作

Decision Action Space 覆盖 research 操作和 execute / sft / rl / evaluate / configure_prompt 系统指令。Directive 由 decision policy 生成，或由外部系统通过 `exploration.accept(directive)` 提交，再调用 advance / inspect / cancel / acknowledge 管理生命周期。外部直接提交时，由调用方负责反馈消费与 acknowledge。

Exploration Action Space 仅包含 Actor 环境动作。execute episode 和 RL rollout 使用 ActorPolicy.act；SFT、prompt 配置和评测绕过它。真实推理评测需要替换 evaluator；当前默认训练与评测仍是 Mock。`configure_prompt` 使用 `PromptSpec(instructions)` 替换 Actor 指令文本，空字符串清空配置，见[共享协议](../src/rsisys/loops/contracts/README.md)。

## 恢复运行

CLI 在空闲边界生成快照，并在 JSON 输出中给出路径：

```bash
PYTHONPATH=src pixi run python -m rsisys inspect --snapshot <快照目录>
PYTHONPATH=src pixi run python -m rsisys resume \
  --snapshot <快照目录> --data-dir runs/restored --rounds 1
```

恢复目标必须是新目录，原目录不被修改。快照保存内置 Harness 的控制状态、两份 memory 和共享 Experience；整体替换 Harness 或更换其持久化组件后，需显式集成保存恢复。旧快照受源码版本精确匹配约束，不提供自动迁移。保存内容、版本一致性要求及自定义组件边界统一见 [Snapshot](../src/rsisys/snapshot/README.md)。

CLI 不是常驻作业服务：waiting 时不会保存快照，resume 不能找回在途任务。需要持续推进异步任务时使用进程内 Runner。
