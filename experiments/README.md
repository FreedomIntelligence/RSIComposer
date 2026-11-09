# 实验配方

这里维护需要模型、数据集或专门运行环境的完整实验；入门及接口教程见 [examples](../examples/README.md)。
所有命令默认从仓库根目录执行，配置保留完整参数，运行产物写入 runs 或显式指定目录。
先查看[方法支持与验证范围](../docs/methods.md)，区分参考实现、原生适配与完整论文复现。
实验完成、真实训练与效果提升分别验收；各目录的结果与失败记录是对应配置的证据。

| 实验 | 内容 |
| --- | --- |
| [RRSI](rrsi/README.md) | 固定模型、纯 Python harness 进化；MATH；无 RL |
| [SEAL](seal/README.md) | self-edit、内层 SFT、top-1、外层 SFT；SQuAD；8 卡配方 |
| [TTRL](ttrl/README.md) | 无标签多数奖励、真实 GRPO 更新；MATH；目标适配及独立迁移指标 |
| [FT-Agent](ft_agent/README.md) | 上游研究假设、代码生成、反馈与历史选择；本地 GSM8K / 原生 LLaMA-Factory 参考适配 |
| [AIDE](aide/README.md) | 原生代码搜索决策器；通过隔离 Lab 搜索 LLM 数学推理程序 |
| [ReST^EM](rest_em/README.md) | Qwen2.5-7B Base；GSM8K / MATH；单卡与 8 卡配置 |
| [Qwen toy](qwen_toy/README.md) | 模型驱动的研究循环、LoRA、评测及监控 |
| [医疗实验](medical/README.md) | 领域数据、固定评测及自主研究装配 |

算法和通用组件属于 src/rsisys；实验目录负责数据准备、配置及场景装配。
实验相关回归测试位于 tests/experiments，通用框架测试按模块分类。
