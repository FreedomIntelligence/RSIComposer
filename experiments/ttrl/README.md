# TTRL：无标签多数奖励 → GRPO → 下一轮模型

新增 [paper.json](paper.json) 使用内置 `ttrl_verl` 决策器和 Lab，直接运行固定版本的官方
verl/Ray/FSDP 全参数训练。旧 [8gpu.json](8gpu.json) 继续使用独立 LoRA 小规模参考实现。

```bash
git clone https://github.com/PRIME-RL/TTRL.git .local/upstream/ttrl
git -C .local/upstream/ttrl checkout 5806e119789132fcb43552beaf2532d7d0447213
python experiments/ttrl/prepare.py --upstream .local/upstream/ttrl --paper --output runs/ttrl-paper-data
rsisys run --config experiments/ttrl/paper.json
```

`--paper` 使用全部 MATH-500 题作为无标签适配目标，迁移集为空；这符合论文的 transductive 口径。
训练入口只接收 `id/prompt`；原生 parquet 的 ground-truth 字段填占位符，真实标签仅供 Lab 的独立评测。
因此上游 original-GT 诊断不再有真实含义，原生内部验证关闭；框架仍记录训练前后完整评测。

决策器执行 `evaluate → fit → evaluate`。一次 `fit` 内由原生引擎维持全部 on-policy 刷新、
优化器、cosine 日程和固定 reference，避免每一轮重新初始化这些状态。
Lab 的 `total_epochs=10`、每批 32 题，每题 64 票生成伪标签、取前 32 条训练，
token-mean objective，输出上限 3072。决策器的 `samples_per_prompt=64` 表示投票数，
Lab 的同名字段 `32` 表示实际训练响应数；配置显式保留两个值。
评估对每题采样 16 次，T=0.6、top-p=0.95，报告平均正确率，不是 best-of-16。
数学评分直接加载固定上游完整 reward 包，不再用另一个 verifier 近似。
原生训练保留上游 dataloader seed 1 / rollout seed 0，写入 effective YAML；
Lab 的 `seed` 用于框架侧评测采样，原生种子可通过 `native_overrides` 显式调整。
当前 JSON 保留已验证的 B200 执行设置：eager、原生 vLLM 显存比例 0.65、
dataloader workers 0；这些与上游脚本的图执行/显存配置不同，训练预算和数学目标不变。

需要 [隔离依赖](../paper-requirements.txt)、与 torch/CUDA/Python ABI 匹配的 flash-attn，
以及干净、固定 revision 的 `upstream_root`。模型和解释器路径需按机器调整。
安装步骤及 B200 通信兼容配置见 [环境说明](../paper-environment.md)。
`native_overrides` 可调整原生 FSDP/显存等参数；数据、奖励器、总日程、模型和发布路径由 Lab 管理。
每次运行保存完整 resolved Hydra 配置、命令、日志和原生 checkpoint，
只合并配置日程的最后一步，校验发布后的实际 tensor 变化，不按评估分数挑 checkpoint。

新增适配不代表已获得论文收益。下文及 [RESULTS.md](RESULTS.md) 描述的是旧 LoRA 配方。

`ttrl` 决策器与 `ttrl` Lab 均为内置注册项，使用统一的 `rsisys run --config` 入口。
实现依据 *TTRL: Test-Time Reinforcement Learning*（arXiv:2504.16084），上游
`PRIME-RL/TTRL` 固定到 `5806e119789132fcb43552beaf2532d7d0447213`。
这是独立 LoRA 参考后端上的方法移植，不是上游 verl 全量训练及论文数值复现。
两轮实跑未得到提升，见 [RESULTS.md](RESULTS.md)。

## 运行

准备与 SEAL / ReST^EM 相同的固定 Qwen2.5-7B Base 权重，然后执行：

```sh
python experiments/ttrl/prepare.py --output runs/ttrl-data
PYTHONPATH=src python3.12 -m rsisys run --config experiments/ttrl/8gpu.json
```

从仓库根目录启动，按实际环境修改 JSON 中的模型及数据路径。
框架本身不依赖 torch；推理解释器需要 vLLM，训练解释器需要 PyTorch / Transformers /
PEFT；Lab 主进程需要 `pip install -e '.[math]'`。
分别设置 `inference_python`、`training_python` 与可选的 `inference_cuda_home`、
`training_cuda_home`；未指定解释器时使用当前 Python。

准备脚本使用上游固定版本的 MATH-500，seed=7309，默认取 32 道目标题和 128 道
独立迁移题。`unlabeled.json` 只含 `id/prompt`，不含答案；`evaluation.json` 单独保存
目标题和迁移题的标签。可用 `--target` / `--transfer` 设置新的规模。
数据及运行输出目录不得复用，源文件和拆分哈希写入 manifest。

## 方法和隔离

决策器依次请求基线评测，然后每轮 `rollout → reward → train`，最后评测一次。
不会根据标签分数挑 checkpoint，也不使用维护者指定的正确解。

- rollout：当前模型对每道无标签目标题生成 16 个解答，保存实际采样 token IDs。
- reward：仅从这些回答中提取 boxed 答案、做稳定多数选择；无合法答案时奖励全零。
  回答与伪答案等价得到 1，否则为 0。按题用样本标准差归一化 GRPO 优势。
- train：使用这些正、负优势做真实 token 概率梯度，PPO ratio clip=[0.8,1.2]，
  initial-reference KL 系数 0.001；不是把多数解答当 SFT 标签训练。
  旧策略概率不反传，提示词及 padding 不计入损失，采样 token IDs 不重新编码。
- 模型：rank-32 LoRA、lr=5e-7，每轮一个 on-policy batch、一次 epoch；8 卡 DDP，
  每卡 microbatch=1。32×16 条响应对应每轮 64 个优化器更新，共两轮。
  解题模型、采样模型和下一轮起点是同一个当前 checkpoint。

推理使用 8 个单卡 vLLM 副本；训练前释放推理进程。每轮训练后合并并发布 checkpoint，
固定初始模型仅作为 KL reference。训练回执保存梯度、优势数量、KL、参数哈希及跨卡一致性。
新后端版本另检查发布 dtype 的舍入不会把所有更新抹掉。

**目标题指标是 transductive。** 这些题的题面参与了无标签适配，即使没有用答案，也不能
称为独立留出泛化结果。迁移题完全不参与 rollout / reward / train，单独报告 accuracy。
Lab 校验两组不重叠、适配集合精确匹配目标题，并拒绝学习输入中出现 answer 等附加列。
原始上游 MATH 配方对相同 500 题进行训练与测试；本配方特意增加独立迁移观察。

主要差异：LoRA 参考实现替代上游 verl 全参数后端；32 道目标题、每题 16 票且全部
用于更新，替代上游每题 64 票取前 32 条；两轮替代完整训练日程；生成上限为 2048。
boxed Math-Verify 替代上游数学归一化/判分器，优势标准化已单独对照 600 种输入。
这些验证不意味着整个 verl 或完整 TTRL 奖励器数值等价。

[后续归因审查](../../docs/review/weight-reproduction-diagnosis-20261109.md)补充了重要差异：
论文 v3 主表按每题 16 次非零温度采样估计 pass@1，本配方只做单次 greedy；
本地常数学习率和 sequence-mean loss 也没有对齐固定上游的 cosine / token-mean。
官方判分器重算最终 160 题后分数不变，但相同训练回答上的投票/奖励仍不完全一致。
上述小规模结果不能用于判断论文收益是否成立。

## 框架和组件

`TTRLLab(context, config, engine=..., verifier=...)` 可由库外注册工厂装配。
engine 声明 `name`、`simulated`、`generate/train`；verifier 复用数学组件的
`parse/is_correct`。换为真正的 verl / FSDP 后端应替换 Lab 内的 engine，决策器不变。
旧参考实现保持独立；新增 `ttrl_verl` Lab 使用 `generate/fit` 引擎接口，
其决策器以单次 fit 表达完整原生日程，不能沿用逐轮重新初始化的训练语义。

同步终态协议目前不支持断点续跑。`controller_report` 提供算法状态及完成轮数；
原始回执、采样、奖励、模型来源和 worker 日志都在运行目录，临时证据不进 Git。

```sh
PYTHONPATH=src python3.12 experiments/ttrl/check_upstream.py /path/to/TTRL --output conformance.json
PYTHONPATH=src python3.12 -m unittest tests.controllers.test_weight_evolution tests.labs.test_ttrl tests.integration.test_weight_framework
# 有 PyTorch / PEFT 时可在 CPU 上验证实际张量梯度与合并舍入。
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest tests.integration.test_weight_objectives
```

原始 8 卡实验记录见 [机器可读结果](results-20261109.json)，本地完整证据位于
忽略的 `.local/ttrl-rl-20261109-v1/`；原始 checkpoint 保留在 b200 的相同相对目录。
