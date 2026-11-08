# 示例导航

框架的两个扩展边界是 RSI 决策器和 Lab；harness、memory、校验器及训推引擎可在模块内部组装。
本目录只保留入门和接口演示，模型训练及数据集实验统一放在 [experiments](../experiments/README.md)。

| 目标 | 入口 |
| --- | --- |
| 无 GPU、无模型服务跑通 ReST^EM | [Quickstart](quickstart/README.md) |
| 开发并注册自己的决策器 / Lab | [注册示例](registered/README.md) |
| 了解调度、反馈、并行等内部机制 | [组件示例](components/README.md) |
| 固定模型、只进化 Python harness | [RRSI 实验](../experiments/rrsi/README.md) |
| 运行 Qwen2.5-7B 的 GSM8K / MATH 实验 | [ReST^EM 实验](../experiments/rest_em/README.md) |

算法实现位于库内，通过配置选择和注册扩展；examples 不复制算法实现。
改变调度、反馈交付或恢复语义时，需要扩展底层库并补充相应测试。
运行产物写入 runs，不与示例源码混放。
