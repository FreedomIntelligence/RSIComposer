# ReST^EM 实验：Qwen2.5-7B Base / GSM8K / MATH

内置 `rest_em` 决策器与 `math` Lab 由配置组装，无需示例插件。
无 GPU 的流程入口见 [quickstart](../../examples/quickstart/README.md)。
本目录维护数据准备脚本与 configs 下的完整实验配置，命令均从仓库根目录运行。

## 算法契约

实现依据 Singh et al., *Beyond Human Data: Scaling Self-Training for Problem-Solving with
Language Models*，arXiv:2312.06585，Section 3 的 Generate/Improve 定义：
`https://arxiv.org/html/2312.06585v2`。

1. 在固定留出集上评测初始模型（工程观测步骤）。
2. 用当前模型在训练题上生成多个解答。
3. 用已知答案提供二值反馈，保留正确解答，按题限制数量。
4. **从初始模型重新训练**，只使用本轮保留的模型生成解答，不累积过去轮次数据。
5. 评测新模型；将它用于下一轮生成。

决策器不引入 meta-agent，不根据评测涨跌自动选最优模型，也不向模型提供答案进行 rationalization。
LoRA 是此 Lab 的轻量实现选择，并非原论文 PaLM 实验的完整复现。
每轮成功依赖真实 Lab 回执；失败/取消或空接受集会停止，回执保留原因。
`finished` 表示决策器停止，不自动证明所有轮次成功，需检查回执和最终停止原因。

## 使用本地模型

在已有 PyTorch / Transformers / PEFT 环境中安装 RSISys，准备固定版本的本地模型、
训练 JSONL 和独立留出 JSONL。以 `configs/gsm8k.json` 或 `configs/math.json` 为起点修改
模型与数据路径；无需修改库或加载插件。框架主体不安装 ML 依赖，训练环境由使用方准备。
默认 `model_kind=causal_lm` 使用 AutoModelForCausalLM；模型目录应是固定的本地快照。

可选张量测试在具备训练依赖的环境运行，不下载模型，并强制隐藏 GPU：

```sh
CUDA_VISIBLE_DEVICES= RSISYS_REST_EM_CPU_INTEGRATION=1 PYTHONPATH=src \
  python3.12 -m unittest tests.integration.test_rest_em_cpu -v
```

该测试已在 b200 的 torch 2.13.0+cu130、transformers 5.19.0、peft 0.21.2 环境通过，
该 CPU 回归使用极小 Llama 架构；两个 Qwen 模型的兼容性另由 GPU 实际运行验证。

## Qwen2.5-7B Base / GSM8K pilot

从仓库根目录执行。准备脚本会显式下载固定版本的模型和数据，需要网络、磁盘空间以及
`pyarrow`；运行阶段使用本地文件，无需模型服务。训练环境需另装支持该架构的
PyTorch、Transformers 和 PEFT（上面的版本组合已实际运行）。

```sh
python experiments/rest_em/prepare_gsm8k.py \
  --model-dir models/Qwen2.5-7B-d1497293 \
  --data-dir runs/gsm8k-data

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  rsisys run --config experiments/rest_em/configs/gsm8k.json
```

模型固定为 `Qwen/Qwen2.5-7B`（Base）的 `d149729398750b98c0af14eb82c78cfe92750796`，
数据固定为 `openai/gsm8k` 的 `740312add88f781978c0658806c59bc2815b9866`。
下载验证 LFS SHA256，manifest 另记录所有下载文件和输出 JSONL 的哈希。
seed=73 从官方训练集拆出 64 道训练题和 128 道开发题；另导出完整官方测试集
`test.jsonl`，不用于训练或迭代评估。另从官方训练集顺序选取 8 道题作为固定的
few-shot 示范，排除 train/dev/test 的全部目标题目，保存于 `few-shot.jsonl` 并记录哈希。
数据目录必须是新目录，运行目录也不能复用。

Base 使用 `prompt_format=completion`，不套用 chat template。固定示范与目标统一为
`Question: ...\nAnswer: ...#### number`；生成在下一题 `\nQuestion:` 处停止，移除该分隔符后
再验证本题答案。训练和评估使用同一提示构造，示范及目标问题均被训练 loss 屏蔽，
只有本轮正确的模型生成 completion 进入 SFT 目标。few-shot 示范的答案是显式提示上下文，
不是目标问题的答案；参考解题过程不会替代模型生成 completion。

配置执行初始开发集评估和两轮 Generate → Filter → Train → Evaluate。
每题采样 8 个解答，最多保留 4 个；每轮从初始模型训练新的 rank-8 LoRA，
仅作用于 `q_proj` / `v_proj`，1 epoch、学习率 2e-5、microbatch=2、梯度累积=4。
生成批量 32 用于减少单卡等待，可根据自己的显存降低。
这是算法流程的轻量复现：模型、GSM8K、LoRA 和数据规模均与原论文实验不同，
不代表原论文 MATH/APPS 指标复现。少量开发题上的变化不能据此推断通用数学能力提升。

如需准备其他模型，显式指定配对的 `--model-repo` 和 `--model-revision`；固定 revision
是校验与留档的一部分，不自动追踪上游最新版。Qwen3.5-9B 的旧预跑使用 chat 格式，
其分数不与当前 Base 的 few-shot 分数作严格的模型能力比较。

## MATH / MATH-500

MATH 作为主要数学实验，GSM8K 用作轻量流程检查；两者独立训练和评测，不混用指标。
复用同一个 `rest_em` 决策器和 `math` Lab，通过配置替换答案格式、校验器和数据。
先按上面的步骤准备固定版本的 Qwen2.5-7B Base，然后从仓库根目录执行：

```sh
pip install -e '.[math]'
python experiments/rest_em/prepare_math.py --data-dir runs/math-data
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  rsisys run --config experiments/rest_em/configs/math.json
```

`math` extra 只安装符号校验组件，不安装 GPU 训推环境。当前固定 Math-Verify 0.9.0、
latex2sympy2_extended 1.11.0 和 ANTLR 4.13.2；PyTorch / Transformers / PEFT 仍需自行配置。
原始 MATH 镜像为 `EleutherAI/hendrycks_math`，revision
`21a5633873b6a120296cce3e2df9d5550074f4a3`；最终评测使用 `HuggingFaceH4/MATH-500`，
revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`。

默认从 7500 道原始训练题中选 128 道训练题、128 道开发题和 4 道固定 few-shot 示范，
seed=73，排除 MATH-500 目标题目，并对四个分区按题目文本去重。
完整 500 道测试题保存于 `test.jsonl`，不参与迭代评估或训练。准备时校验所有选中题目的
参考答案均可解析；遇到不支持的参考答案直接失败，不静默删除测试题。

配置使用 completion 提示、LaTeX `\\boxed{...}` 最终答案、`verifier=math_verify`，
每题采样 8 次、最多保留 4 次、迭代 2 轮。生成预算 2048 tokens、总长度上限 8192，
生成 batch=32，训练 microbatch=2、梯度累积=4，其余训练设置与 GSM8K pilot 一致。
两个 pilot 都是小规模 LoRA 适配，不能当作原论文数值结果复现。

符号校验只提取最后一个完整 `\\boxed{}` / `\\fbox{}` 的内容，支持嵌套 LaTeX，
然后使用 Math-Verify 的严格等价判断，解析和验证各限时 2 秒。没有合法最终答案时判错，
不从推导过程中的数字猜测答案。评测协议明确为该固定版本的 boxed Math-Verify，
不声称与所有论文或 leaderboard 的判分规则完全相同。

数据每行：

```json
{"id":"train-001","prompt":"题目文本","answer":"参考解题过程\n#### 42"}
```

训练与留出集不得重叠（Lab 检查相同 id 和完整题目文本）。训练只读取筛选后的
`prompt + completion`，参考答案不会作为训练目标注入。数据清洗、去重和题目拆分由接入方负责。
严格数值校验器支持 `#### 42`、数值 `\\boxed{42}`、十进制或普通分数；不尝试执行生成文本。
它适合先接 GSM8K，**不覆盖完整 MATH 的表达式、集合、单位和符号等价**。
MATH 请使用上述符号校验配置；不应把 strict-numeric 的准确率当成 MATH 指标。

生成采用 temperature=0.7、top_k=40；评测采用 greedy 单次生成。训练使用每轮新建的
rank-8 LoRA、新优化器、仅 completion 计损失。
通过 `epochs`、`learning_rate`、`max_new_tokens`、`max_length`、`seed` 配置；
`generation_batch_size`、`train_batch_size` 和 `gradient_accumulation_steps` 默认为 1。
生成左侧补齐，训练右侧补齐并屏蔽 padding 的损失；最后不足一组的累积也执行更新。
超长样本直接失败，不截断掉最终答案。需要更大规模时替换 Lab 后端，不改决策器。

## 替换为自己的 Lab

通过 `register_lab` 暴露四个操作，名称由 controller 的 `lab` 指定别名前缀：

| 操作 | 请求字段（均另带 `round`） | 成功回执 `data` |
| --- | --- | --- |
| `generate` | `model`, `dataset`, `samples_per_prompt` | `request`, `dataset`, `count` |
| `filter` | `samples`, `max_accepted_per_prompt` | `request`, `dataset`, `count` |
| `train` | `initial_model`, `dataset` | `request`, `model` |
| `evaluate` | `model`, `dataset` | `request`, `quality_metrics` |

四种操作是该数学 Lab 的能力契约；ReST^EM 决策器负责调度顺序。
`request` 必须完整回显收到的参数，用于拒绝错轮/错模型反馈。数据和模型是不可变的引用；
决策器不传递大段样本。Lab 负责实际采样、校验、训练、评测和证据真实性。
成功返回终态 `OperationResult`；后台作业提交成功不等于训练成功，本版同步协议不能直接接
`JobRef`。需要异步调度可实现自己的适配 Lab 或另注册决策器。

运行产物位于 `<data_dir>/math-lab/round-N/`：生成样本、筛选样本、评测明细，及模型 manifest。
训练模式还包含 adapter 权重和训练步数/loss。数据回执保留 SHA256，拒绝复用或修改前一阶段
产物。输入模型及其依赖版本需要由使用方固定；manifest 不是所有模型文件的全量哈希清单。
本示例在进程内保留算法状态，不支持断点续跑；操作失败后使用新 run。
`iterations=N` 正常需要 `4*N+2` 次决策（含基线和 finish），配置预算至少覆盖这个数量。

## 8 卡离线执行

`math-8gpu.json` 和 `gsm8k-8gpu.json` 复用同一 ReST^EM 决策器和 MathLab，设置 `engine=vllm`。
`gpu_ids=[0,1,2,3,4,5,6,7]` 启动 8 个独立推理副本，每个进程独占一张卡，
请求均匀分片、结果按原题目与采样序号还原。推理子进程只接收 prompt，不接收参考答案。
连续批处理、前缀缓存和分块 prefill 由 vLLM 执行；不用部署 API 服务。

可以分别设置 `inference_python` 和 `training_python` 为两个环境的 Python 绝对路径；
默认使用当前解释器。推理环境需要 vLLM；训练环境需要 PyTorch、Transformers、PEFT。
在 B20Z 上验证的版本为 vLLM 0.32.0、PyTorch 2.13.0、Transformers 5.19.0、PEFT 0.21.2。
若 FlashInfer 需要 JIT 编译，需提供 CUDA toolkit 与 ninja，首次启动包含内核编译开销。
这些 GPU 依赖不由框架默认安装。

`gpu_memory_utilization=.85` 是推理权重与 KV cache 的显存预算，不是实际活跃 token 的占用率。
`max_num_seqs=128` 为每卡同时执行的请求上限，`max_num_batched_tokens=16384` 为调度 token 预算。
应根据实际题量和输出长度测吞吐；增加缓存预留不会自动加速少量请求。
`generation_batch_size` 仅作用于 Transformers 后端。GSM8K 的 8 卡配置保持 64 道训练题、
128 道开发题、固定 8-shot 提示、1024-token 输出预算和 4096 总长度；正式测试用独立
单轮 `math.evaluate` 对基线和固定第二轮模型各评测完整 1319 题，不参与模型选择。

训练使用 torchrun / DDP，在全部指定 GPU 上从最初 base 重新建立 LoRA 和优化器。
`train_batch_size` 是每卡 microbatch；全局有效批量为它乘卡数再乘
`gradient_accumulation_steps`。8 卡配置用 `1×8×1=8`，每轮只训练本轮接受的数据。
损失按全局 completion token 数归一化；末组不足 8 条时不复制样本，不增加监督权重。
这与单卡原实现的 microbatch 均值累积有数值差异，应作为独立训练配置记录结果。
`train_gradient_checkpointing=false` 用更多激活显存换取速度；显存不足时可启用。
`worker_timeout` 为单次推理/训练操作的秒数上限，默认 3600。

```sh
# 模型、数据准备与单卡相同；GPU 环境已安装。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  rsisys run --config experiments/rest_em/configs/math-8gpu.json
```

每个 worker 输出实际请求数、生成 token 数和耗时，训练保存全局批量、卡数与损失归一化方式。
启动阶段与真实执行分开计时。任何 worker 失败或超时都会终止本次操作的其余进程；
框架主进程收到中断时清理它启动的 worker。不能在同一 run 目录内重试，需使用新目录。
