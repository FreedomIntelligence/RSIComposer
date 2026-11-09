# 2026-11-09：SEAL / TTRL 论文协议适配

本轮修复了[归因审查](weight-reproduction-diagnosis-20261109.md)发现的实现偏差，
并在 B200/B20Z × 8 上通过注册入口验证了两个原生训练闭环。
这证明训练后端与框架能衔接，不证明论文的完整性能收益已经复现。
原小规模实验不足以把收益缺失归因到论文；当时存在已经确认的本地协议差异。

## SEAL

入口为 [paper.json](../../experiments/seal/paper.json)，仍使用 `seal` 决策器和 Lab，
通过 `profile: paper` 选择 Trainer/SFTTrainer 后端。

- 两轮分别使用发布数据中的 50 篇互不重叠文档；独立评估为 200 篇、974 题。
  只复用文档身份，丢弃上游已经生成的 self-edit，由当前模型重新生成。
- 每篇 5 候选，每候选 3 次独立内层训练，以三次语义正确率均值选择稳定 top-1。
  内层种子显式记录、可重放，替代原服务器的 wall-clock 种子。
- 显式开启按行构造，覆盖 self-QA 和前导文字分支。内层使用原版 HF Trainer，
  保留上游固定长度 padding、未屏蔽的 EOS padding 标签、线性日程和默认 weight decay。
- 内层 adapter 由 vLLM 实际加载后答题，采用与上游相同的简答提示、greedy、64 token 和 EOS 终止。
- 外层使用原版 TRL SFTTrainer。论文配置为 rank 64、2 epochs、lr 3e-4，
  2 GPU × microbatch 1 × accumulation 5，有效 batch 10；8 GPU 用于独立内层任务。
- 语义评分使用上游精确提示及 GPT-4.1。传入完整问题/标题；空回答计错，
  网络失败或含糊返回终止阶段。无词面评分回退，也不把服务失败当负奖励。
  主指标按文档等权；逐题加权准确率单独记录。

实际后端验证用 4 篇训练、2 篇留出、每篇 1 题、1 候选、3 内层种子，
缩短至内层 1 epoch / 256 tokens、外层 1 epoch / accumulation 1。
12 次训练侧内层适配使用全部 8 卡；初末评测另有 4 次内层适配。
2 卡外层 SFT 完成 2 个优化器步，跨卡 adapter 哈希一致，
合并至发布 dtype 后检测到 3,382,454,812 个元素变化，最终评测实际加载新模型。
6 条操作回执均 completed，决策器报告 completed / training_performed=true。

**该运行显式注册 `squad-smoke-only` 词面评分器，仅验证后端。**
B200 启动环境尚未提供 `OPENAI_API_KEY`，没有执行真实 GPT-4.1 语义评分，
没有执行完整 50 + 50 篇外层训练及 200 篇语义评测。
该运行的 accuracy 字段不能解释为论文语义准确率。

## TTRL

入口为 [paper.json](../../experiments/ttrl/paper.json)，注册名为 `ttrl_verl`。
决策器执行 `evaluate → fit → evaluate`；一次 fit 调用固定 revision 的官方
verl/Ray/FSDP 完整日程，保持优化器、reference、on-policy 刷新和 cosine 日程，
避免拆成每轮重新初始化的 LoRA 训练器。

- 全量配置使用 MATH-500 全部 500 道目标题，10 epochs、每批 32 题，
  每题 64 票生成伪标签，取前 32 条训练响应；全参数 GRPO、token-mean loss。
- 直接使用固定上游完整数学奖励包及原生多数选择，不再近似其数学归一化。
- 训练入口只接收 `id/prompt`；原生 parquet 的 ground_truth 为 `"None"` 占位。
  真实答案只在 Lab 的初末评测路径中使用；原生 original-GT 诊断没有真实意义，
  原生内部验证关闭，不能把相关训练日志当标签准确率。
- 初末每题采样 16 次，T=0.6、top-p=0.95，计算平均正确率，不取 best-of-16。
  目标题为 transductive 口径，不能称为独立留出泛化。
- 发布配置日程的最后一个原生 checkpoint，经官方 merger 合并后再检查实际数值，
  不按标签分数选 checkpoint。保护 Lab 管理的数据、模型、奖励和发布路径，
  拒绝用父级 Hydra map 替换绕过配置保护。

第一轮原生验证为固定 8 题、8 票/4 训练响应、1 epoch、256 token 上限，
8 卡完成一个原生训练步、导出 checkpoint，并完成新权重上的最终评测。
3 条操作回执均 completed，决策器报告 completed / training_performed=true。
独立使用统一 float64 数值表示复核全部 339 个参数张量，293 个发生数值变化；
单纯 dtype 转换不计作训练更新。后续实现已将此检查纳入发布路径。

补充的原协议采样预算验证使用固定上游前 32 题、1 epoch，保持 64 票/32 训练响应、
3072 token 和 16 次评测采样，使用 source-v5。1 个原生 rollout/train step 包含
32 次 PPO mini-batch 更新；原生显存比例 0.65、eager、dataloader workers 0。
8 卡完成训练、官方合并、数值检查及最终评测，3 条回执 completed，控制器 completed。
318 个参数张量发生数值变化，最后的评测实际加载该 checkpoint。

| 固定 32 道 transductive 目标题，16 次采样平均 | 训练前 | 训练后 |
| --- | --- | --- |
| 正确响应 / 全部响应 | 324/512 | 346/512 |
| 平均正确率 | 63.28% | 67.58% |

这个单种子、小规模验证出现了正向变化，但没有执行完整 500 题 × 10 epochs，
不能把 4.30 个百分点的变化当成论文主表收益复现。
题目在任何新评测前按上游顺序固定，未根据标签筛题或挑 checkpoint。

## 通用框架修复

首次原生 TTRL 验证遇到多卡通信初始化停滞。原 NCCL 2.26.2 替换为 2.28.9，
同时关闭 CUMEM host / CUMEM / NVLS 后，独立 8 卡 all-reduce 及原生训练通过。
这是整套环境组合的验证，没有隔离每个开关的单独作用。
配置和安装方式记录在[隔离环境说明](../../experiments/paper-environment.md)。

长错误日志同时暴露了反馈交付缺陷：文本证据投影移除了 operation 及原始 result.data，
决策器因此把终态失败误判为没有匹配回执。
默认 harness 现在通过可选的 `operation_feedback_scope` 传递最新准确的结构化操作/回执；
文本仍可压缩，原始证据仍保存在 memory/archive。
该路径按回执内容摘要匹配源记录，交付独立副本，退出调用后清理作用域。
回归测试实际经过大日志证据卡片投影，确认失败可交付且决策器无法修改宿主状态。

训练期间没有由维护者指定 candidate、伪标签或 checkpoint。
控制器负责配置日程，Lab/原生算法负责实际执行及确定性选择，host 只处理执行与反馈。
TTRL 每次 fit 持有独立 Ray 集群，并负责关闭，未附着或停止其他任务的集群。

## 验证与证据

- 离线全量 2070 项测试通过，75 项按可选依赖/服务条件跳过。
- 随后新增的数值发布测试在 B200 的隔离环境中执行，8 项适配测试全部通过、无跳过。
- 147 项 Node 测试、致命静态检查、文档链接及 whitespace 检查通过。
- SEAL 对照覆盖 480 种序列构造、800 种稳定 top-1，及 self-edit、答题、语义判分提示。
- 核心/monitor wheel、sdist 构建与 metadata 检查通过；干净环境安装、公共入口与资源检查通过。

[机器可读记录](../../experiments/paper-adaptation-validation-20261109.json)保存运行身份、
源码快照摘要及终态产物摘要。原始请求、日志、回执和权重位于 B200 忽略目录
`.local/paper-adapters-20261109/`，未放入 Git。
原有 dirty checkout 保留，实跑使用独立源码快照。
TTRL 首轮验证使用 source-v3，协议预算验证使用 source-v5；SEAL 使用 source-v4。
最终源码快照 source-final 的 8 项适配测试也在 B200 环境通过。两个原生 Ray 会话结束后未残留其持有的进程。

完整收益验证仍需执行两份全量配置，并按相同评分器、采样、数据及计算预算比较。
SEAL 的语义服务凭据尚缺；在这些条件补齐前，不能宣布论文收益复现或论文结论有误。
