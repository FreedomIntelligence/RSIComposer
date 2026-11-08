# 开发自己的决策器与 Lab

此目录可以复制到库外，仅依赖安装后的 RSISys。plugin.py 提供两个核心扩展示例：
TargetController 接收 DecisionInput 并返回 Decision；CounterLab 接收 LabRequest 并返回 LabFeedback。
这些名字使用框架现有序列化和运行契约，没有第二套执行器。

```sh
rsisys run --config single-loop.json --plugin plugin.py
rsisys run --config custom-controller.json --plugin plugin.py
```

两个配置复用同一个 Lab，只替换决策器。示例不调用模型、不训练权重。
meta-agent.json 展示使用默认 harness 的模型驱动决策器，运行前填写模型服务配置。
已适配的 DSH 可以通过 meta_agent 的 harness=dsh 和 dsh 原生配置选择，见注册指南。

内部组件不是必填槽位：固定算法无需 LLM harness 或 memory；复杂 Lab 可以自行组装
校验器、推理引擎和训练引擎。常用的数学组件见 [rsisys.labs](../../src/rsisys/labs/README.md) 与 [ReST^EM 实验](../../experiments/rest_em/README.md)。
自行获得资源的工厂需注册生命周期清理；跨进程恢复、并发调度等需遵循底层能力契约。
