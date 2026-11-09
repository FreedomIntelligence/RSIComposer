# 方法支持与验证范围

所有方法都通过 **controller → Lab → feedback** 进入公共运行时。
注册名称存在、接口回归通过、真实训练完成和论文收益复现是不同结论。
本页汇总来源与范围；数据、预算、上游 revision 和失败记录以各配方及结果为准。

## 通用模板

| 决策器 | 用途 | 依赖与范围 |
| --- | --- | --- |
| `single_loop` | 按固定次数调用操作 | CPU 注册示例；不是某篇论文算法的复现。 |
| `meta_agent` | 用 LLM 自主选择已授权操作 | default / DSH harness、可选 memory；需要已配置模型端点。它是可组装模板，不是研究能力基准。 |

从[注册示例](../examples/registered/README.md)开始开发自己的 controller 与 Lab。
Lab 的 schema 和语义仍须匹配决策器，注册目录不保证任意组合都可运行。

## 已注册研究方法

| 决策器 / Lab | 实现来源与改动 | 已验证的范围 | 主要限制与证据 |
| --- | --- | --- | --- |
| `rest_em` / `math` | 独立轻量实现 Generate/Filter/Improve，每轮从初始模型训练 | 模拟 CPU 入门、真实 LoRA 训练及 Qwen2.5-7B GSM8K/MATH 配方 | LoRA、模型与预算均不同于原 PaLM 实验；[配方与运行记录](../experiments/rest_em/README.md)。 |
| `rrsi` / `harness_math` | 算法移植；保留正则化、critic、筛选规则，改用完整 Python 文件提案 | 固定 Qwen 的三轮 MATH 搜索与独立留出；没有权重训练 | 领域、搜索预算、编辑工具与上游不同；[配方](../experiments/rrsi/README.md)、[结果及失败](../experiments/rrsi/RESULTS.md)。 |
| `seal` / `seal`，reference profile | self-edit、内层适配、选择与外层 SFT；独立 LoRA 参考训练器 | Qwen2.5-7B、两轮、8 卡真实训练 | 使用 SQuAD EM/F1 与小预算，未测得收益；[配方](../experiments/seal/README.md)、[结果](../experiments/seal/RESULTS.md)。 |
| `seal` / `seal`，paper profile | 适配论文数据、按行序列、原生 Trainer/SFTTrainer 与语义评分路径 | 原生训练/权重交接的小规模验证；提示、序列与选择对照 | 实跑使用显式 lexical smoke scorer，未验收论文 GPT-4.1 语义服务或完整收益；[配置](../experiments/seal/paper.json)、[验证记录](review/weight-paper-adaptation-20261109.md)。 |
| `ttrl` / `ttrl` | 独立 LoRA GRPO 参考实现、多数伪标签 | 两轮、8 卡真实梯度更新，目标题与独立迁移题分开计量 | 预算、loss、调度及判分偏离上游，未测得收益；[配方](../experiments/ttrl/README.md)、[结果](../experiments/ttrl/RESULTS.md)。 |
| `ttrl_verl` / `ttrl_verl` | 固定官方 verl/全参数后端，一次 fit 保持完整日程状态 | 32 道目标题、64 票/32 响应、1 epoch；评估每题 16 次的原生 GPU 验证 | 是 transductive 目标题指标，未完成 500 题 × 10 epochs；[配置](../experiments/ttrl/paper.json)、[验证记录](review/weight-paper-adaptation-20261109.md)。 |
| `ft_agent` / `llm_finetune` | 部分原生参考适配：加载固定上游假设、代码生成、接受与历史选择策略；执行交给 Lab | GSM8K 三个原生 LLaMA-Factory SFT 实验，数值合并/重载与最终留出评测 | 未接全 CoSTEER 知识/调试链；替换 scenario、执行检查和评测；最终留出下降；[范围](../experiments/ft_agent/README.md)、[结果及失败](../experiments/ft_agent/RESULTS.md)。 |
| `aide` / `aide_program` | 原生 Agent/search/review/Journal；执行回调适配为 Lab，任务改为 LLM 数学推理程序 | 三节点搜索、原生选择、独立留出；模型权重固定 | 不是原表格基准；留出提升伴随更多模型调用与 token，不是等成本收益；[范围](../experiments/aide/README.md)、[结果](../experiments/aide/RESULTS.md)。 |

FT-Agent 和 AIDE 都是研究型 meta-agent 决策器。FT-Agent 优化目标模型的训练方案；
AIDE 在这里搜索固定模型的推理程序。两者都没有训练 meta-agent 自身的权重，
也没有提供原生搜索进程的断点恢复。

## 如何判断一次运行

先检查 host 的停止原因、方法报告和各次回执；失败实验仍是搜索历史的一部分。
真实训练应有 optimizer steps、数据及已发布权重变化等证据；harness 方法则检查
实际执行的程序、采纳路径和模型调用。比较效果时固定数据、判分、采样、预算和选择协议，
同时报告成本。目标题参与无标签适配的指标须标记 transductive。

当前没有完成上述方法的全规模论文数值复现。小规模适配跑通可以验证框架执行边界，
不能据此把论文收益不复现归咎于论文；已发现的本地偏差与对照见
[SEAL/TTRL 归因审查](review/weight-reproduction-diagnosis-20261109.md)。

CPU、GPU、可选服务、恢复和资源调度的公共边界见[兼容与支持](compatibility.md)。
