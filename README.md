# RSISys

可客制化的双 loop RSI 框架。Decision-making 选择优化目标与改进策略，Exploration 执行、学习并返回反馈；policy、space、learner、各 loop 的 memory 等实现通过接口替换。Experience 是共享系统反馈库，不保存原始 test-time history。

每个 loop 目录的组件共同组成对应角色的 harness；loop 负责运行循环和策略流程，memory 是 harness 的子模块。具体映射见[架构](docs/ARCHITECTURE.md)。

## 快速开始

在项目目录使用 Pixi（当前锁定 Linux / Python 3.12）：

```bash
pixi run demo
pixi run coordination-demo
pixi run test
```

演示使用模拟模型、训练与评测后端；版本引用变化不代表真实权重训练或能力提升。已有 HTTP 模型接口、本地进程执行、持久化和空闲边界恢复，具体范围见各组件说明。

## 系统文档

- [架构](docs/ARCHITECTURE.md)：模块关系、双 loop 协同与图示。
- [使用指南](docs/GUIDE.md)：启动、配置、驱动和恢复。

## 组件文档

| 组件 | 内容 |
|---|---|
| [Decision-making](src/rsisys/loops/decision_making/README.md) | 决策流程，以及 policy / memory / decision_space / learner 的入口 |
| [Exploration](src/rsisys/loops/exploration/README.md) | 执行与学习流程，以及 policy / memory / action_space / environment / learner / evaluation 的入口 |
| [共享协议](src/rsisys/loops/contracts/README.md) | 指令、反馈、任务、候选与模型引用 |
| [Experience](src/rsisys/experience/README.md) | 系统反馈、来源标签、检索和存储 |
| [Snapshot](src/rsisys/snapshot/README.md) | 保存恢复范围及客制化边界 |
| [Runtime](src/rsisys/runtime/README.md) | Runner、模型调用、执行后端、作业和产物 |

## 文档维护

组件的职责、接口、具体实现与限制只维护在对应目录的 README；loop README 负责本地流程和子组件导航，全局文档负责跨组件关系与操作方法。组件变更同步修改所属 README，不另建阶段报告或旧版本副本。

以职责组件为单位建 README；backends/ 下的具体实现由所属 learner README 统一说明，单文件组件由所在模块 README 分节维护。字段与签名以链接的源码为准，运行示例放在 examples/。
