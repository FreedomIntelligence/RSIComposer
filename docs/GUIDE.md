# 使用指南

以下命令在项目根目录执行；lion 上路径为 `/home/jovyan/projects/RSISys`。组件实现和限制见[组件导航](../README.md)。

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

backends 定义调用后端，roles 分别绑定 decision_maker / actor；execution 选择 mock / local_process，learning 选择 none / mock。Decision loop 的 experience_limit 限制系统反馈条数，memory_limit 限制自身 history 与工作记忆条数；Actor loop 仅使用 memory_limit。这些限制不计算 token。配置入口目前只装配内置组合，Python 构造函数参数见各 loop 文档。

```bash
PYTHONPATH=src pixi run python -m rsisys run \
  --config configs/local.json --data-dir runs/example \
  --literature path/to/paper.txt
```

配置 data_dir 相对配置文件解析，命令行 --data-dir 相对当前目录解析。已有目录不覆盖，同一目录不能由两个 Application 同时打开。--literature 导入普通文本供 [Decision space](../src/rsisys/loops/decision_making/decision_space/README.md) 使用。

本地目录包含 run.json、config.json、experience.sqlite3、decision_memory.json、exploration_memory.json、artifacts/、工作目录和 snapshots/。Experience SQLite 使用 schema 2，显式拒绝旧 schema，不自动迁移。

## 连接模型

复制模型服务模板，填写 endpoint、model_id、revision。endpoint 为 API base，后端请求其 /chat/completions；两个角色可选相同或不同后端。

需要鉴权时 credential_env 填环境变量名，启动前设置对应环境变量。不支持 JSON response_format 的服务可将 json_mode 设为 false，但内置模型 policy 仍要求回复合法 JSON。传输能力、版本验证和本地进程限制见 [Runtime](../src/rsisys/runtime/README.md)。

## 使用 Python 接入组件

先实现组件目录 README 中的协议，再通过 loop 构造函数注入。可运行示例：

- [loop_internals.py](../examples/loop_internals.py)：训练请求、查询、版本应用和独立评测。
- [coordination.py](../examples/coordination.py)：自定义 policy 消费 last_feedback 并发出下一条指令。

两个 loop 分别注入可替换的 memory，默认实现为 DecisionMemory / ExplorationMemory。DecisionContext 同时包含全部来源的 experience、自身 history 和 memories；ActorContext 读取当前任务 history 与自身 memories，不默认注入系统经验。scope 仅标记反馈来源，优化对象由 Decision policy 选择。

RL 请求使用 trajectory_refs 引用 Actor 本地 history；Feedback.evidence_refs 引用系统记录。决策者自身训练仍使用系统反馈的 experience_refs。

调用 runner.step(objective) 推进一步，或 runner.run(objective, rounds=1) 推进到轮次完成、等待或结束；waiting 后由调用方安排下一次推进。Runner 通过 decision_making.receive_feedback(directive, feedback) 交付反馈。取消用 runner.cancel()，之后继续推进直到终态。协同语义见[系统架构](ARCHITECTURE.md)。

## 恢复运行

CLI 在空闲边界生成快照，并在 JSON 输出中给出路径：

```bash
PYTHONPATH=src pixi run python -m rsisys inspect --snapshot <快照目录>
PYTHONPATH=src pixi run python -m rsisys resume \
  --snapshot <快照目录> --data-dir runs/restored --rounds 1
```

恢复目标必须是新目录，原目录不被修改。快照分别保存两个 loop 的 memory 状态和共享 Experience，当前只支持内置实现；自定义 memory 需显式集成保存恢复。旧快照受源码版本精确匹配约束，不提供自动迁移。保存内容、版本一致性要求及自定义组件边界统一见 [Snapshot](../src/rsisys/snapshot/README.md)。

CLI 不是常驻作业服务：waiting 时不会保存快照，resume 不能找回在途任务。需要持续推进异步任务时使用进程内 Runner。
