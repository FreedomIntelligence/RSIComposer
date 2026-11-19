<h1 align="center">RSISys</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-009688.svg" alt="许可证：Apache 2.0"></a>
  <a href="https://github.com/satsuki2486441738/RSISys/stargazers"><img src="https://img.shields.io/github/stars/satsuki2486441738/RSISys?logo=github" alt="GitHub 星标数"></a>
</p>

<p align="center"><strong>通用的递归自改进框架：研究、探索，并改进整个系统</strong></p>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

<p align="center"><a href="#框架概览">框架概览</a> · <a href="#最新动态">最新动态</a> · <a href="#亮点">亮点</a> · <a href="#快速开始">快速开始</a> · <a href="docs/README.md">文档</a> · <a href="#已适配的方法">支持的方法</a> · <a href="#引用">引用</a></p>

RSISys 把递归自改进（RSI）实验组织成一个闭环：Controller 决定下一步做什么，Lab 运行实验并返回回执，Controller 根据证据决定下一步或采纳哪个候选。提示词搜索、harness 搜索、监督微调和强化学习都使用这个闭环，因此可以替换 Controller、复用 Lab，或在不改核心代码的前提下新增方法。

## 框架概览

![RSISys 架构分为两部分。（a）Controller 向 Domain Labs 请求操作，Labs 将轨迹、指标和产物作为证据返回。（b）Self-Reflection Lab 构建 Controller 更新，可用嵌套试验验证；当前 Controller 决定是否采纳更新。](assets/rsisys-overview.png)

**Controller** 决定下一步试什么、采纳什么。Policy 负责决策，Harness 准备输入和工具，可选的 Memory 提供上下文。**Lab** 使用配置好的输入执行评测、采样或训练等操作。宿主负责转发调用、执行预算和数据划分规则，并保存回执以支持恢复。

## 最新动态

- **[2026 年 11 月]** RSISys 开源，内置 16 个已注册的方法，支持可恢复运行。

## 亮点

**一个闭环覆盖所有 RSI 方法。** RSISys 把递归自改进描述为一个闭环：Controller 决定下一步做什么，Lab 在目标环境中运行实验，返回的证据驱动下一次决策。改进对象不固定，可以是模型权重、提示词、程序、agent harness，也可以是 Controller 自身。自训练、提示词与程序搜索、harness 演化和自主研究 agent 都能放进这个闭环，不需要第二种结构；各方法的对应关系见[方法表](#已适配的方法)。

**研究与探索是角色，不是额外组件。** Controller 承担研究角色，每个 Lab 承担探索角色。把两个角色交给不同的 agent，例如由强模型决定如何训练一个小模型，系统就是解耦的；交给同一个 agent，就回到常见的单 agent 自改进闭环。RSISys 用同一套接口表达这两种情况，切换时不需要重写方法。

**Controller 自身也能被改进。** 大多数系统里，自改进算法本身是固定的。在 RSISys 中 Controller 是可替换的组件：它可以在嵌套实验中试跑自己的新版本，并在 checkpoint 处切换过去，研究闭环本身也在改进范围之内。

## 快速开始

在 Linux 或 macOS 上用 Python 3.12 或 3.13 安装。核心没有第三方依赖；各方法需要的[可选依赖](docs/components.md#methods)列在组件页里。

```sh
git clone https://github.com/satsuki2486441738/RSISys.git
cd RSISys
python -m pip install -e .
```

**不用 GPU 先跑一次。** 下面的命令在示例数据上用模拟模型跑两轮 ReST^EM，用来检查安装是否成功，并了解一次运行会产出什么。

```sh
rsisys run --config src/rsisys/algorithms/rest_em/configs/cpu/rest_em.json --data-dir runs/quickstart
```

**用 GPU 模型运行。** 先[启动一个 Chat Completions 服务](src/rsisys/labs/_template/README.md#connect-a-gpu-model)，再运行方法模板，它会搜索提示词和 few-shot 设置：

```sh
rsisys run --plugin src/rsisys/labs/_template --plugin src/rsisys/algorithms/_template \
  --config src/rsisys/algorithms/_template/configs/run.json --data-dir runs/gpu-quickstart
```

[Quickstart](docs/quickstart.md) 介绍了如何读取结果、用 RSIMonitor 查看运行，以及如何恢复中断的运行。

## 文档

| 我想… | 阅读 |
| --- | --- |
| 了解基本概念：Controller、Lab、operation、receipt、split | [Concepts](docs/concepts.md) |
| 查看内置的方法、Controller、Lab 和组件 | [Components](docs/components.md) |
| 把 Controller 换到别的 Lab 上，或加载自己写的 Controller | [Compose a system](docs/compose.md) |
| 编写新方法 | [方法模板](src/rsisys/algorithms/_template/README.md) |
| 编写 Lab 或接入自己的评测集 | [Lab 模板](src/rsisys/labs/_template/README.md) |
| 查询配置字段或命令行选项 | [Run configuration](docs/config.md) |
| 查看和对比运行 | [RSIMonitor](src/rsimonitor/README.md) |

文档目前只有英文版。

## 已适配的方法

| 类别 | 方法 | Controller（研究） | Lab（探索） | 改进对象 |
| :-- | :-- | :-- | :-- | :-- |
| 自训练 | [ReST^EM](https://arxiv.org/abs/2312.06585) · [STaR](https://arxiv.org/abs/2203.14465) · [TTRL](https://arxiv.org/abs/2504.16084) · TTRL-verl | 调度生成、筛选、训练各轮 | 模型采样作答，经评分后微调 | 模型权重 |
| 自我编辑 | [SEAL](https://arxiv.org/abs/2506.10943) | 调度自我编辑各轮 | 模型为自己生成训练数据并据此更新 | 模型权重（统一） |
| 提示词搜索 | [OPRO](https://arxiv.org/abs/2309.03409) · [GEPA](https://arxiv.org/abs/2507.19457) · GEPA-Merge · [PromptPG](https://arxiv.org/abs/2209.14610) | 在候选提示词中搜索 | 用每个提示词评测任务模型 | 提示词 |
| 程序搜索 | [FunSearch](https://github.com/google-deepmind/funsearch) | 演化程序种群 | 执行程序并打分 | 程序 |
| Harness 搜索 | [RRSI](https://arxiv.org/abs/2609.24972) · Meta-Agent · [Coding Meta-Agent](src/rsisys/algorithms/coding_meta_agent/README.md) | 模型提出 harness 修改 | agent 用候选 harness 解题 | Agent harness |
| 反思 | [Reflexion](https://arxiv.org/abs/2303.11366) | 决定何时重试 | 同一个 agent 写代码、跑测试并反思失败 | 自身的反思（统一） |
| 研究型 agent | [AIDE](https://github.com/WecoAI/aideml) · [FT-Agent](https://github.com/microsoft/RD-Agent) | 上游 agent 起草、调试并选择方案 | 运行候选程序和微调任务 | 解决方案与微调模型 |
| 自我反思 | [Self-Reflection Lab](src/rsisys/labs/scenes/self_reflection/lab.py)（是一个 Lab，需搭配 Controller 使用） | 当前 Controller | 候选 Controller 在嵌套实验中运行 | Controller 自身 |

运行 `rsisys catalog` 查看已注册的方法。各方法的预设配置、硬件和依赖见 [Components](docs/components.md#methods)。目前只有 ReST^EM、GEPA 和 Coding Meta-Agent 有使用指南，其他方法只有预设配置。预设配置只是起点，不代表已经复现论文结果。

## 参与贡献

请从[方法模板](src/rsisys/algorithms/_template/)或 [Lab 模板](src/rsisys/labs/_template/)开始，插件的写法见 [Compose a system](docs/compose.md)。新增方法或 Lab 时，请提供可运行的配置，说明它对照哪些结果验证过，并标注改编来源。框架接口统一从 `rsisys.api` 导入。

## 引用

如果 RSISys 对你的研究有帮助，请引用：

```bibtex
@misc{rsisys2026,
  title        = {RSISys: A Modular Runtime for Recursive Self-Improvement Research},
  author       = {{RSISys contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/satsuki2486441738/RSISys}}
}
```

## 许可证

RSISys 以 [Apache 2.0](LICENSE) 许可证发布。改编的上游代码在实现旁保留署名。
