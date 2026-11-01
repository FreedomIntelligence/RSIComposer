# RSISys

可客制化的双 loop RSI 框架。Decision-making 选择准备操作与系统指令，Exploration 接收指令并执行、学习或评测；两者共享 Experience 系统证据库。

每个 loop 由可整体替换的 **Harness** 和独立的 **Action Space** 组成，`loop.py` 只保留薄端点与有界推进。默认 Harness 封装策略、上下文、记忆、工具及对应的学习/评测流程；替代实现可以不含 memory。目录与消息边界见[架构](docs/ARCHITECTURE.md)。

**Policy 位于默认 Harness 内部**：`DecisionPolicy.decide` 选择操作、系统指令、等待或结束；Exploration Policy 即 `ActorPolicy.act`，只选择环境动作。Harness 组织执行，Action Space 定义可用操作/动作及约束。

Python 装配支持分别注入两个 loop 的工厂；系统指令通过决策侧定义与探索侧 handler 映射扩展，无需修改核心分派。接入方式见[使用指南](docs/GUIDE.md#使用-python-接入组件)。

长任务可通过 `review_after_seconds` 开启决策复查，选择继续、取消或调整下次复查时间；后端轮询不必调用模型。默认端点支持 Runner 重建后的未确认反馈交付恢复，默认串行，可选隔离的并行实验；待交付状态仍在进程内，见 [Runtime](src/rsisys/runtime/README.md)。

## 快速开始

在项目目录使用 Pixi（当前锁定 Linux / Python 3.12）：

```bash
pixi run demo
pixi run coordination-demo
pixi run backend-demo
pixi run test
```

默认模型、训练与评测使用 Mock。版本引用变化不代表真实梯度训练或能力提升，Mock 评测不执行真实推理打分。HTTP 模型、本地进程、持久化和空闲边界恢复的范围见组件说明。

训练请求保持后端中立，允许自定义 method 和外部数据；Decision 请求也支持 data_refs 与 method。默认 `rl` 保留本地采样、训练、应用版本的简单循环；显式启用 `training_directive()` + `TrainingCommandHandler` 并注入 learner 后，`train` 可把整个任务的 rollout 与训练交给后端。接口开放不等于已有具体框架适配，接入见[使用指南](docs/GUIDE.md#接入整项训练任务)。

`pixi run backend-demo` 运行 [backend_training.py](examples/backend_training.py)：在独立子进程中进行真实 CPU 标量回归并回传权重产物和 MSE。它不是 LLM 训练或 verl 现成适配，也不演示 RL rollout。

[Qwen3.5 toycase](examples/qwen_toy/README.md) 提供模型自主选择步骤的真实决策、LoRA 训练、评测、候选应用和 Actor 执行装配；外部配置能力、权限及预算，未预设流程或采纳阈值。服务默认使用 CPU，小规模验证使用 0.8B，也可接入 9B。[监视前端](web/README.md) 使用标准库 HTTP 服务读取 `runs/*/monitor.json`，展示双 loop、作业、候选指标和事件时间线。

[并行实验示例](examples/parallel_experiments.py) 展示隔离实验、动态控制和候选比较（Mock 验证）；反馈提取与工作区查询都可替换。

## 系统文档

- [架构](docs/ARCHITECTURE.md)：Harness / Action Space 边界、双 loop 协同与图示。
- [使用指南](docs/GUIDE.md)：启动、配置、组件接入与恢复。

## 组件文档

| 组件 | 内容 |
|---|---|
| [Decision-making](src/rsisys/loops/decision_making/README.md) | Decision Harness 与 Action Space |
| [Exploration](src/rsisys/loops/exploration/README.md) | Exploration Harness 与 Actor Action Space |
| [共享协议](src/rsisys/loops/contracts/README.md) | 系统指令、反馈、任务、候选与模型引用 |
| [Experience](src/rsisys/experience/README.md) | 系统反馈、来源标签、检索和存储 |
| [Snapshot](src/rsisys/snapshot/README.md) | 内置实现的保存恢复及自定义边界 |
| [Runtime](src/rsisys/runtime/README.md) | Runner、模型调用、执行后端、作业和产物 |

## 文档维护

架构图标出 Harness 内的 Decision Policy 与 Exploration Policy（ActorPolicy），以及协调层和具体执行组件的边界；组件细节见各自 README。每个职责组件保留一个 README，backends/ 的具体实现由所属组件统一说明，单文件模块由所在目录 README 说明。字段与签名以链接源码为准，运行示例放在 examples/，不另建全局说明或旧版本副本。
