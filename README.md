# RSISys

可客制化的双 loop RSI 框架。Decision-making 选择准备操作与系统指令，Exploration 接收指令并执行、学习或评测；两者共享 Experience 系统证据库。

每个 loop 由可整体替换的 **Harness** 和独立的 **Action Space** 组成，`loop.py` 只保留薄端点与有界推进。默认 Harness 封装策略、上下文、记忆、工具及对应的学习/评测流程；替代实现可以不含 memory。目录与消息边界见[架构](docs/ARCHITECTURE.md)。

**Decision／Actor Policy 位于默认 Harness 内部**：`DecisionPolicy.decide` 选择操作、系统指令、等待或结束；Exploration Policy 即 `ActorPolicy.act`，只选择环境动作。Harness 组织执行，Action Space 定义可用操作/动作及约束。

外部运行控制由 runtime 的可替换 **RunPolicy** 负责：默认 `researcher` 在研究者 finish 后停止；`time` 持续到时限；`goal` 在目标未达时继续研究。后两种模式可显式开启新研究轮次，保留 memory、经验与候选。停止后禁止新实验，按策略 drain 或 cancel 并保留反馈交付；接入见[使用指南](docs/GUIDE.md#选择运行策略)，清理边界见 [Runtime](src/rsisys/runtime/README.md)。

Python 装配支持分别注入两个 loop 的工厂；系统指令通过决策侧定义与探索侧 handler 映射扩展，无需修改核心分派。接入方式见[使用指南](docs/GUIDE.md#使用-python-接入组件)。

长任务可通过 `review_after_seconds` 开启决策复查，选择继续、取消或调整下次复查时间；后端轮询不必调用模型。默认端点支持 Runner 重建后的未确认反馈交付恢复，默认串行，可选隔离的并行实验；使用 `Application(..., durable=True)` 可把串行运行的回执、在途 Mock 作业和 RunPolicy 状态持久化，并通过 `Application.resume()` 重启；中断时结果不明的操作须先核对，见 [Runtime](src/rsisys/runtime/README.md)。

默认 Decision Harness 使用可替换的 `BasicContextCompactor(trigger_tokens=80000, target_tokens=24000, summary_tokens=4096)`，在完整 `GenerationRequest` 构造后的 request 阶段计量输入 token，达到 80k 才压缩，24k 是压缩后总请求输入目标。VLLM 通过 `/tokenize` 精确计数；无 tokenizer 的 backend 使用明确标记的 UTF-8 保守上界。原始历史与工具回执持久保存在 raw archive，compact 不改写或删除归档原文，滚动摘要保留来源；最新工具结果及其最近 history 配对完整保留。若固定规则与最新完整反馈本身超过目标，返回明确预算错误，需分页或调整预算。模型仍自主维护计划 / todos / 阶段总结，并可分页查询原始记录；摘要只是导航，不替代原始证据或规定研究步骤。

可选 [DSH Harness](src/rsisys/loops/decision_making/harness/dsh.py) 通过 `dsh_decision_factory(config)` 替换默认 Harness，可用于上下文实现的对照。持久 session 与会话压缩由原生 DSH 唯一负责，不先经过默认 context compactor；操作仍通过同一 Action Space，RSISys 保留权限、预算与双 loop 协调。其他 Harness 继续使用公开协议与 Python 工厂接入，见[使用指南](docs/GUIDE.md#使用-dsh-harness)。

## 快速开始

在项目目录使用 Pixi（当前锁定 Linux / Python 3.12）：

```bash
pixi run demo
pixi run coordination-demo
pixi run backend-demo
pixi run test
```

默认模型、训练与评测使用 Mock。版本引用变化不代表真实梯度训练或能力提升，Mock 评测不执行真实推理打分。HTTP 模型、本地进程、持久化和空闲边界恢复的范围见组件说明。

训练请求保持后端中立，允许自定义 method 和外部数据；Decision 请求也支持 data_refs 与 method。默认 `rl` 保留本地采样循环；显式注册 `training_directive()` + `TrainingCommandHandler` 后，`train` 可把整个任务交给后端。[共享协议](src/rsisys/loops/contracts/README.md) 单独定义可选的训练来源、能力声明与部署生命周期；具体进程训练和推理服务接入位于 Harness 外的 [adapters/](src/rsisys/adapters/)，接入见[使用指南](docs/GUIDE.md#接入整项训练任务)。

[SystemVersion / Candidate](src/rsisys/loops/contracts/candidate.py) 描述组件组合及其变更，[SystemRegistry](src/rsisys/runtime/systems.py) 管理父版本、候选注册与当前版本的显式应用。实验父系统 ID、初始训练权重、服务部署和角色绑定分别表达；`TrainingOrigin` 明确 restart / weights_only / full_state，不能根据父版本暗中决定训练起点。

可选 ProcessTrainingBackend 接入原生训练进程，默认 worker 支持 Transformers / PEFT SFT；VLLMBackend / VLLMDeploymentBackend / VLLMLoRAActivator 接入独立运行的推理服务。已单独验证 CPU tiny random GPT-2 的 3 步 LoRA 训练链路，不代表医学增益；LLaMA-Factory 仅提供原生命令扩展，未声称实际安装或训练，verl 适配尚未验证。

`pixi run backend-demo` 运行 [backend_training.py](examples/backend_training.py)：在独立子进程中进行真实 CPU 标量回归并回传权重产物和 MSE。它不是 LLM 训练或 verl 现成适配，也不演示 RL rollout。

[Qwen3.5 toycase](examples/qwen_toy/README.md) 提供模型自主选择步骤的真实决策、LoRA 训练、评测、候选应用和 Actor 执行装配；外部配置能力、权限及预算，未预设流程或采纳阈值。服务默认使用 CPU，小规模验证使用 0.8B，也可接入 9B。[监视前端](web/README.md) 使用标准库 HTTP 服务读取 `runs/*/monitor.json`，展示双 loop、作业、候选指标和事件时间线。

[并行实验示例](examples/parallel_experiments.py) 展示隔离实验、动态控制和候选比较（Mock 验证）；[决策图示例](examples/decision_graph.py) 验证两项任务并行、多个结果确认采纳后汇合、后续执行消费组合改动，并输出监控图谱。运行 `PYTHONPATH=src:. python examples/decision_graph.py --run-dir runs/decision-graph-demo`（目录须未使用）。示例使用脚本策略与 CPU 产物任务，不代表自主模型研究或权重融合。反馈提取与工作区查询都可替换。

[medical/](examples/medical/README.md) 是独立装配示例：冻结本地医学选择题数据，注入 researcher 模型、候选推理、learner 与系统启用器，再通过可选 default / DSH 决策 Harness、Exploration Harness 和 Runner 研究 prompt / 训练候选。native.py 显式接通 tokenize、原生训练与候选服务准备；数据协议、固定评测与验收规则留在示例中。准备与检查不会下载模型、启动训练或服务，接线与边界见[使用指南](docs/GUIDE.md#独立医学研究示例)。

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
| [适配器](src/rsisys/adapters/README.md) | 原生进程训练、vLLM 推理与部署 |
| [医学装配](examples/medical/README.md) | 领域数据、评测与 native 包装接线 |

## 文档维护

架构图以实线灰框标记协议与协调骨架，以彩色虚线框标记可替换实现；adapters/ 和 examples/medical 位于 Harness 外。组件细节见各自 README，具体适配器的使用见现有指南；字段与签名以链接源码为准，运行示例放在 examples/，不另建全局说明或旧版本副本。
