# Qwen 自主研究 toycase

Decision-maker 自己选择研究步骤；没有 train → evaluate → apply → execute 阶段表，也没有固定的候选接受阈值。模型可以先执行当前 Actor、查询日志与源码、训练、重复评测、应用候选、修改 prompt，或直接结束。框架只检查已安装能力、外部权限、预算和请求/模型版本一致性；无脚本兜底决策。

这个示例使用真实 Qwen 服务、Transformers / PEFT LoRA SFT、独立样本 NLL 评测和版本化 Actor 推理。训练仍只支持本例的 `toy-lora` / `sft`，评测协议为 `heldout-nll`，不运行真实 RL。数据格式属于本例，核心不依赖 PyTorch，也不转换跨框架数据。

## 能力、权限与预算

- 系统指令：train、evaluate、apply_candidate、execute、configure_prompt。
- 研究操作：已有实验查询，以及 workspace_list / workspace_read / workspace_search。默认只挂载本次 run、框架 src 和本例源码三个命名工作区；远端 worker 的未共享文件仍需后端显式暴露。
- 默认开放所有已安装操作；`--permissions FILE` 使用 ActionPermissions JSON 覆盖权限。文件工具的路径约束不替代 OS / 容器沙箱。
- `--max-trainings` 默认 2，是训练提交上限；兼容旧参数名 `--rounds`，不再表示必须执行的轮数。当前 worker 最多允许两次训练尝试。
- `--max-directives` 默认 16；`--max-model-calls` 默认 32，包含研究决策、复查及解析失败重试；单次输出上限 200 tokens。`--max-context-chars` 默认 20000，按字符限制输入视图：先移除最旧记录，必要时省略大段 details/data 并明确标注，可查询完整来源；这不是 tokenizer 精确 token 计数。原始记录不裁剪。另有整体时间、HTTP 超时及推进 tick 预算。
- `--review-after-seconds` 仅提供提示，首次复查时间由模型写入 Directive，后续时间由 ReviewDecision 决定。`--require-training-review` 是可选的运行结果检查，不替模型插入复查。

训练次数上限会隐藏进一步 train 能力；实验指令预算用尽后仍可在剩余模型调用预算内查询或结束。模型直接 finish 会如实记录零训练，不会被宣称为训练链路成功。

## 启动

先创建新的 worker 数据目录，已有记录不覆盖：

```sh
pixi run python examples/qwen_toy/prepare.py --run-dir runs/qwen-autonomous
```

两个终端分别启动 baseline 与 worker。服务读取已有本地权重，默认 CPU；在本集群复用已有 Pixi 环境：

```sh
pixi run --as-is -m /home/jovyan/projects/vlm/SFT/Baseline/pixi.toml -e train \
  python examples/qwen_toy/serve.py --model /home/jovyan/models/Qwen3.5-0.8B \
  --served-name Qwen3.5-0.8B --device cpu --threads 4 --port 18095

pixi run --as-is -m /home/jovyan/projects/vlm/SFT/Baseline/pixi.toml -e train \
  python examples/qwen_toy/worker.py --model /home/jovyan/models/Qwen3.5-0.8B \
  --served-name Qwen3.5-0.8B --device cpu --threads 4 --steps 24 --rounds 2 \
  --run-dir runs/qwen-autonomous --port 8766
```

就绪后启动 controller；9B 服务使用对应 model-name：

```sh
PYTHONPATH=src pixi run python examples/qwen_toy/run.py \
  --baseline-url http://127.0.0.1:18095 --worker-url http://127.0.0.1:8766 \
  --model-name Qwen3.5-0.8B --run-dir runs/qwen-autonomous \
  --max-trainings 2 --max-directives 16 --max-model-calls 32 --timeout 1800
```

CUDA 部署使用 [serve.sh](serve.sh) 和 [train_worker.sh](train_worker.sh)，通过本集群 `~/tools/cluster/kjobctl create slurm` 提交单节点、单任务、单 GPU 的 zsh 脚本，分别设置 `QWEN_DEVICE=cuda`、`QWEN_MODEL_PATH`、`QWEN_SERVED_NAME`、端口和 worker 的 `QWEN_ROUNDS=2`。baseline 与 worker 各需一张分配卡；脚本只使用调度器分配的 GPU UUID。设置作业 activeDeadlineSeconds 与 backoffLimit，完成实验后清理本次资源，不直接使用未分配 GPU。CPU 服务默认只监听 localhost，CUDA 临时服务只用于受控集群网络。

## 实际模型更新与执行边界

worker 为每个候选保存独立 LoRA adapter；从父候选训练时复制父权重继续更新，不原地破坏旧版本。每次训练重新创建优化器。调用方负责相同底座配置；baseline revision 是操作方声明，候选在绑定前通过服务 revision 与真实推理 readiness 检查。

`train.apply_result` 由模型选择。模型也可先产生多个候选再评测或应用；应用要求候选 base 与当前 Actor 匹配。NLL 指标是证据，不自动触发采纳。未应用候选保持 proposed，不能根据负 delta 或缺少 apply 就标记为模型明确 rejected。

此 toy worker 的模型操作由一个锁串行保护，不代表两个 GPU 训练任务已经并行；框架的并行实验需要为实验工厂配置独立执行资源或支持并发的外部后端。默认 Decision-maker 仍使用 baseline，Actor 可切换版本；本例不自动训练 Decision-maker。

## 结果与监控

- result.json：模型选择的 `experiments` 序列、状态、预算、模型调用次数、训练提交次数、候选与 Actor 版本、取消确认等。
- round_results / round-N.json：每次 Actor execute 的观察记录。保留旧字段名供前端读取，**不是预设实验轮次或训练次数**。
- monitor.json：研究操作、实验指令、复查、任务进度、候选与版本，供[监控前端](../../web/README.md)读取。
- experience.sqlite3：系统反馈；两份 memory JSON：各自任务历史；artifacts：轻量引用对应的产物。
- worker 的 adapter、训练/评测结果和日志由 worker 数据目录保留。

`completed` 只表示模型选择结束；是否训练、评测、应用以及是否提升，必须查看具体证据。时间或调用预算耗尽记录为未完成，不合成 finish。HTTP 测试使用替身，只验证控制链路。

取消在训练优化步边界处理；提交/轮询幂等是进程内保证，服务重启不恢复作业。取消后不自动回滚已应用版本。当前 worker 在训练失败或取消后不再接收新的训练尝试，可继续读取既有证据；重新训练使用新 worker。自定义装配未接入 Snapshot，模型服务生命周期由部署环境管理。

## 真实自主验证记录

2026-11-01 使用 Qwen3.5-9B 与 CUDA SFT/评测验证。模型自行选择 train → evaluate → finish，未应用候选，也未调用 Actor execute；父版本 heldout NLL 约 0.044638，候选约 0.000014759。一次缺失 method/recipe_ref 的响应在获得具体校验反馈后由模型自行修正。结束表示研究者停止，不表示部署后的 Actor 已更新或通用能力提升。此前失败尝试保留，分别暴露了参数反馈不明确与上下文超限问题；未用脚本补齐模型步骤。
