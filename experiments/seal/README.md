# SEAL：self-edit → 内层 SFT → 选择 → 外层 SFT

`seal` 决策器和 `seal` Lab 均为内置注册项，通过 `rsisys run --config` 运行。
它们不启动上游 shell pipeline；调度、回执、训练和模型交接都经过 RSISys。
实现依据 *Self-Adapting Language Models*（arXiv:2506.10943）知识自适应分支，
上游 `Continual-Intelligence/SEAL` 固定到 `6d9c9f9ee392c6cc618e771f399d436d190f6ca4`。
提示和训练序列构造的许可见 [NOTICE](NOTICE)。

这是方法移植配方，**不是论文数值复现**。已完成的两轮 8 卡实跑没有测得收益，见
[RESULTS.md](RESULTS.md) 和 [机器可读结果](results-20261109.json)。

## 运行

先准备固定版本的 Qwen2.5-7B Base；本轮使用的 revision 为
`d149729398750b98c0af14eb82c78cfe92750796`，与 ReST^EM 配方相同。
所有角色都使用当前轮同一个 checkpoint；没有外部提案模型或付费判分服务。

```sh
python experiments/seal/prepare.py --output runs/seal-data
PYTHONPATH=src python3.12 -m rsisys run --config experiments/seal/8gpu.json
```

命令从仓库根目录运行。修改 JSON 中的模型及数据路径；`data_dir` 相对配置文件，
Lab 输入路径默认相对启动目录。准备脚本默认固定 seed=7309、8 篇训练文档和 8 篇独立
验证文档；原始数据来自固定上游版本，原文和输出 SHA256 记录在 manifest 中。
`--documents` 可指定新的数据规模，运行目录与数据输出目录必须是新目录。

框架解释器不需要 ML 依赖。推理需要 vLLM，训练需要 PyTorch / Transformers / PEFT。
JSON 可分别设置 `inference_python` / `training_python`，以及各自的
`inference_cuda_home` / `training_cuda_home`。省略解释器时使用当前 Python。
已验证环境为 vLLM 0.32.0、PyTorch 2.13.0+cu130、Transformers 5.19.0、PEFT 0.21.2。

## 决策和执行边界

1. 初始独立文档上生成一次 self-edit、实际内层 SFT，并记录适配前后问答表现。
2. 当前模型仅看训练文档，按配置生成多个 self-edit，不看对应 QA 的参考答案。
3. 对每个 self-edit，从当前 checkpoint 新建临时 LoRA；训练内容为生成的 implications
   和原文，以标题作为前缀。问答时不把原文放进上下文。
4. Lab 提供内层 SFT 后的分数，按文档稳定 top-1；同分保留先生成的候选。
   选择规则由算法实现，维护者不指定赢家。
5. 当前模型在选中 self-edit 上做 completion-only 外层 SFT，发布新的 checkpoint，
   供下一轮生成使用。这里从当前模型继续，与 ReST^EM 数学配方每轮回到初始模型不同。
6. 固定轮数后，在同一批独立文档上做最终评估，不使用其分数选择 checkpoint。

配置为两轮、每文档两候选、内层 rank-32 / 10 epochs / lr=1e-3，外层 rank-32 /
2 epochs / lr=3e-4，采样上限 1024 tokens。8 个独立推理或内层适配进程各用一张卡；
外层 SFT 使用 8 卡 DDP。训练后合并 LoRA，供下一轮从实际更新的权重开始。

相对上游的主要差异：数据和采样规模更小、每候选内层测量一次、输出预算缩短、
外层 rank 从 64 改为 32、使用常学习率的 PyTorch 参考训练器；本地 SQuAD EM/F1
替代 GPT-4.1 判分，选择奖励为每篇文档的平均 F1，报告总体指标为问题平均。
这会改变奖励口径，不能与论文的 GPT 判分准确率直接比较。

## 组件替换与证据

`SEALLab(context, config, engine=..., scorer=..., scorer_name=...)` 可在库外 Lab 工厂中装配。
engine 需声明 `name`、`simulated` 和 `generate/adapt/train`；scorer 返回 `f1` 与
`exact_match`，并提供显式版本名。默认组件位于 `src/rsisys/labs/weight_engine.py`，
无需把新训练库安装进框架环境。新调度语义仍需实现或扩展决策器。

每阶段保存不可变数据、完整小请求、SHA256、终态回执、worker 输入输出和日志。
内层保存步数及参数哈希；外层保存权重、优化器步数、跨卡一致性和权重来源。
后续版本还检查合并到发布 dtype 后仍有参数变化。
算法当前为同步终态协议；worker 超时由 Lab 处理，不支持中断后在原目录自动恢复。
`controller_report` 区分算法完成、失败、空数据停止和真实训练轮数；`finished`
只代表 host 生命周期结束。

纯函数对照和离线回归：

```sh
PYTHONPATH=src python3.12 experiments/seal/check_upstream.py /path/to/SEAL --output conformance.json
PYTHONPATH=src python3.12 -m unittest tests.controllers.test_weight_evolution tests.labs.test_seal tests.integration.test_weight_framework
```

对照覆盖精确 self-edit 提示、80 种训练序列和 800 种稳定 top-1 选择；不宣称整个训练
后端或自定义 F1 判分与上游数值等价。原始实跑证据位于忽略的 `.local/seal-sft-20261109-v1/`。
