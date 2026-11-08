# RRSI：固定权重的 Python harness 进化

复现对象是 Google Research 的 *Regularized Recursive Self-Improvement of Agent Harnesses*
（arXiv:2602.10556），官方实现 `google-research/rrsi`，对照版本
`be50316e1db05914068a973f322770ef08ed7ba1`。
本配方是 **MATH 场景的小规模算法移植**，不是原论文指标复现。模型不训练、不更新权重；
同一个固定模型承担解题、分析、提出改动及审查四个角色。
已完成的小规模实测、失败运行及限制见 [结果记录](RESULTS.md)。

## 模块边界

- 内置 `rrsi` 决策器：搜索历史、失败/成功轨迹分析、完整 Python 源码提案、独立 critic、
  有界修复、候选筛选与采纳、下一轮父版本选择。
- 内置 `harness_math` Lab：隔离执行 `step(problem, history)`、调用固定推理端点、
  运行独立校验器、统计 token 成本和成功率，返回不可变证据路径。
- `seed_harness.py`：最初的一次调用解题策略。可进化复核、重试、上下文组织和输出处理，
  不局限于改 prompt 文本。

正式接口在库中，见 [接口边界](../../docs/api-contracts.md)；本目录只有实验源码和配方。
新增领域可以替换 Lab 和 researcher 组件，无需修改 RRSI 的选择规则。

候选的分析、提案、critic、修复、采纳及父版本切换都由决策器执行，Lab 只交付测量结果。
运行维护者负责服务、传输、隔离和接口适配；候选的解题策略或输出处理错误应作为反馈交给决策器。
维护者的诊断不追加到运行中的研究提示，不手工修改候选、指定赢家或依据分数调整采纳规则。
框架或适配器修复使用新的源码快照和运行目录；输出协议及角色预算的变化也属于新的实验条件。

## 算法保留与差异

保留退火 L0 编辑预算、组件标签、失败与成功证据、无收益组件的剪枝提示、停滞后的探索槽、
critic 与有界修复、单次 k-trial 基线 bootstrap 噪声校准、最佳历史质量下限，以及 Algorithm 2
的相对 token 成本与结构新颖性规则。原实现中的退火、校准和选择规则已做数值对照（1152 组选择案例及退火/校准样例）。
运行中的错误 trial 保留完整分母；错误超过 15% 的候选不允许被采纳。

差异：改用 Qwen3.5-9B 和数学任务；三轮、每轮两个候选；提案通过完整文件 JSON 交付，
替代原实现的交互式 git 编辑工具；每题最多三次模型调用。
提案和审查默认使用 OpenAI-compatible JSON Schema 约束输出，避免代码字符串中的反斜杠破坏交付格式；
不支持该能力的端点可显式设置 researcher.structured_outputs=false，保留严格解析与有界修复。
目前不提供联网工具、跨题持久记忆和子代理，组件标签不等于已实现对应能力。
这不是完整原论文多领域/长时程搜索，也不声称完整复刻原版历史归因/digester 工具链。
若需要验证这些机制的收益，应扩大搜索轮数、补充领域工具并进行消融。

## 运行

需要 Linux `bubblewrap`、Python 3.12+、已固定版本的本地模型，以及可独立使用的 vLLM 环境。
框架与评分端安装 `pip install -e '.[math]'`；不需要训练依赖或 RL 引擎。
示例已在 vLLM 0.32.0 验证。先从仓库根目录启动八个本地推理副本：

```sh
PYTHONPATH=src python3.12 experiments/rrsi/serve.py \
  --python /absolute/path/to/vllm-env/bin/python \
  --model /absolute/path/to/Qwen3.5-9B \
  --log-dir runs/rrsi-serve
```

等待 `http://127.0.0.1:18400/v1/models` 至 `18407` 全部就绪。
若只使用一个已有端点，将配置中两处 endpoints 改成同一个 URL 即可。
运行结束后在启动服务的终端按 Ctrl+C，关闭本配方拥有的副本。

准备两个不重叠的 JSONL：`runs/rrsi-data/search.jsonl` 和 `heldout.jsonl`，
每行包含 `id`、`prompt`、`answer`；MATH 使用参考最终 boxed 答案。
题目 ID 和题目文本在分区间必须不重复；准备方式可复用 [ReST^EM 数学数据脚本](../rest_em/prepare_math.py)。
已有该脚本准备的 MATH 数据时，冻结与 pilot 相同的拆分方式：

```sh
python3.12 experiments/rrsi/prepare.py --source-dir runs/math-data --output-dir runs/rrsi-data
```

默认使用 train.jsonl 前 64 题；以 seed=7308 打乱独立 test.jsonl 后选 128 题。
脚本检查重复及跨分区重叠，记录输入/输出哈希，并拒绝覆盖已有目录。
固定数据、模型和源版本，并记录文件哈希，然后：

```sh
PYTHONPATH=src python3.12 -m rsisys run --config experiments/rrsi/math.json
```

命令从仓库根目录运行；配置里的数据和初始 harness 路径相对工作目录。
运行目录必须是新目录。重复实验可通过 `--data-dir` 指定新目录。

## 八卡评测吞吐

八卡配方使用 8 个单卡 vLLM 副本，默认开启编译和 CUDA Graph，
每卡 `max_num_seqs=128`、`max_num_batched_tokens=32768`。
Lab 的 `workers=512` 限制在途解题任务，`sandbox_workers=32` 独立限制沙箱进程，
`grading_workers=16` 并行执行 Math-Verify。生成完成的题立即交付评分，最终回执保留原题目顺序。
沙箱准入等待不消耗候选的执行超时；每次实际隔离执行的资源限制保持一致。
自定义校验器继续在主进程主线程运行。

同一份固定初始 harness、128 道留出题各 4 次采样的 512 次调用实测：
原配置耗时 292 秒；优化后连续两次为 17.0 / 15.6 秒，均无执行错误，
输出吞吐为 3.19万 / 3.51万 token/s。详细对照与失败压测见 [吞吐记录](THROUGHPUT.md)。
这是执行侧压测，不调用决策器，不改变搜索样本量或采纳规则。

可用实际运行生成的绝对路径配置，独立压测已有服务：

```sh
PYTHONPATH=src python3.12 experiments/rrsi/benchmark.py \
  --config /absolute/path/to/experiment.json --output runs/rrsi-benchmark \
  --workers 512 --grading-workers 16 --trials 4
```

`benchmark.py` 记录总耗时、实际输出 token、逐卡 GPU 利用率采样和完整评测回执。
有效吞吐比较要求全部 trial 实际执行且 `harness_errors=0`；应同时检查输出 token 数和准确率，
避免把失败或缩短输出当作提速。服务启动时间不包含在稳态压测耗时中。
小搜索集或单条 meta 推理的可并行工作量较少，以上吞吐对应批量 Lab 评测。
需要诊断时可给 `serve.py` 添加 `--enforce-eager` 恢复 eager 模式。

## 隔离与证据

harness 在新的 Linux namespace 中运行，只挂载 Python 标准运行环境、当前候选和私有 runner。
它只通过 stdin 获得当前问题和本 trial 历史，不得到数据文件、参考答案、评分器或网络。
CPU、内存、文件尺寸和运行时间有上限；源码执行行覆盖记录到轨迹。
评分器由宿主执行。搜索阶段 researcher 只读取 search 轨迹，不读取 heldout 数据或指标；
全部搜索完成后，才对初始和最终 harness 分别执行 heldout 评测。

`rrsi-research/` 保存模型调用、候选源码、critic、修复与逐轮 selection；
`harness-math/` 保存每次实际执行的逐题轨迹、哈希、质量、token 成本和使用的端点。
`simulated=false`、`training_performed=false`；准确率由真实模型回答计算。
`finished` 本身不等于提升或复现成功，应检查完整回执、实际改动及独立留出结果。

目前仅支持同步执行；搜索状态保存在进程内，未提供断点续跑。
