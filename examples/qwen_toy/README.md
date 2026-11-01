# Qwen3.5 一到两轮真实 SFT 链路

这个例子把真实 Qwen 决策、Transformers / PEFT LoRA SFT、独立样本评测、候选验收与应用、更新后 Actor 执行和 Experience 反馈串起来。controller 和 worker 的 `--rounds` 默认都是 1，显式设为 2 可运行连续两轮；只支持 SFT，不运行真实 RL。适配器只属于本例，RSISys 核心不依赖 PyTorch，也不引入统一训练数据格式。

## 组成

| 文件 | 职责 |
|---|---|
| `prepare.py` | 生成 64 条训练数据和 12 条不重叠的评测数据 |
| `serve.py` / `serve.sh` | 临时基线模型服务，供 Decision Policy 使用 |
| `worker.py` / `train_worker.sh` | 单卡训练、评测及训练后候选推理服务 |
| `adapters.py` | 本例 HTTP 作业、模型路由、readiness 接口 |
| `run.py` | 装配双 loop，运行有界闭环 |
| `monitor.py` | 公开决策/执行摘要，以及已完成运行的历史导出 |

模型与数据格式参考 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B) 和 [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)。训练采用 PEFT LoRA：r=8、alpha=16、q_proj/v_proj、每轮默认 24 steps、batch=2、梯度累积=2。两轮时总计最多 48 个优化步。

## CPU 启动（默认）

两个服务默认使用 CPU。小规模验证可使用 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，也可传入已有的 9B 权重目录。先准备独立 run-dir；在本集群复用已有的 Transformers / PEFT Pixi 环境，分别在两个终端启动服务：

```sh
pixi run python examples/qwen_toy/prepare.py --run-dir runs/qwen-cpu

pixi run --as-is -m /home/jovyan/projects/vlm/SFT/Baseline/pixi.toml -e train \
  python examples/qwen_toy/serve.py --model /home/jovyan/models/Qwen3.5-0.8B \
  --served-name Qwen3.5-0.8B --device cpu --threads 4 --port 18095

pixi run --as-is -m /home/jovyan/projects/vlm/SFT/Baseline/pixi.toml -e train \
  python examples/qwen_toy/worker.py --model /home/jovyan/models/Qwen3.5-0.8B \
  --served-name Qwen3.5-0.8B --device cpu --threads 4 --steps 24 --rounds 2 \
  --run-dir runs/qwen-cpu --port 8766
```

就绪后在第三个终端运行 controller：

```sh
PYTHONPATH=src pixi run python examples/qwen_toy/run.py \
  --baseline-url http://127.0.0.1:18095 --worker-url http://127.0.0.1:8766 \
  --model-name Qwen3.5-0.8B --run-dir runs/qwen-cpu --timeout 1800 \
  --rounds 2 --review-after-seconds 2 --require-training-review
```

CPU 模式使用 float32、隐藏 CUDA 设备并限制线程数，默认只监听 `127.0.0.1`；无需 GPU 作业。只需单轮时将两个 `--rounds 2` 均改为 `--rounds 1`。服务只读取已有权重，不自动下载。更换模型时让两个服务和 controller 的 served-name/model-name 一致。

## 可选的本集群 GPU 启动

脚本使用 `/home/jovyan/models/Qwen3.5-9B` 和已有 Pixi 环境 `/home/jovyan/projects/vlm/SFT/Baseline/pixi.toml` 的 `train` 环境：Python 3.12、torch 2.10.0 cu128、transformers 5.2.0、peft 0.18.1。可通过 `QWEN_MODEL_PATH`、`QWEN_SERVED_NAME`、`QWEN_WORKER_PORT`、`QWEN_THREADS` 配置训练入口；`QWEN_ROUNDS` 默认 1，`QWEN_STEPS` 是每轮步数。迁移环境时使用自己的 Pixi 锁文件。

```sh
pixi run python examples/qwen_toy/prepare.py --run-dir runs/qwen-toy
```

仅在明确选择 CUDA 时，通过 `~/tools/cluster/kjobctl create slurm` 提交单节点、单任务、单 GPU 作业。可使用下述薄脚本，训练侧将末行改为 `exec /bin/zsh /home/jovyan/projects/RSISys/examples/qwen_toy/train_worker.sh /home/jovyan/projects/RSISys/runs/qwen-toy`：

```sh
#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
export QWEN_DEVICE=cuda
export QWEN_MODEL_PATH=/home/jovyan/models/Qwen3.5-9B
export QWEN_SERVED_NAME=Qwen3.5-9B
export QWEN_ROUNDS=2
exec /bin/zsh /home/jovyan/projects/RSISys/examples/qwen_toy/serve.sh
```

将薄脚本存为共享 run-dir 下的 `baseline-job.sh` / `train-job.sh`，先检查现有端点与作业，复用满足版本和轮数要求的服务，不重复申请 GPU。例如训练侧提交：

```sh
~/tools/cluster/kjobctl create slurm -- \
  --output '/home/jovyan/projects/RSISys/runs/qwen-toy/train-%j-%N.log' \
  --error '/home/jovyan/projects/RSISys/runs/qwen-toy/train-%j-%N.err' \
  /home/jovyan/projects/RSISys/runs/qwen-toy/train-job.sh
kubectl -n "$KUBE_USER_NAMESPACE" patch job JOB_NAME --type merge \
  -p '{"spec":{"activeDeadlineSeconds":7200,"backoffLimit":0}}'
```

CPU/控制器不申请 GPU；CUDA baseline 与 CUDA worker 若同时使用，各占一张分配卡。队列名可能含 H200，但本集群曾实际报告 H20Z，以作业设备日志为准。日志参数放在 kjob 的 `--` 后；创建后为本次 job 设置 `activeDeadlineSeconds=7200`、`backoffLimit=0`。本集群会暴露全部设备节点，脚本在 Python 导入 torch 前将调度器分配的 GPU UUID 映射到对应物理序号，只使用分配的那张卡。不要用未分配的设备绕过调度。

两个服务的 `/health` 就绪后，在能访问计算节点的环境启动 controller：

```sh
PYTHONPATH=src pixi run python examples/qwen_toy/run.py \
  --baseline-url http://BASELINE_NODE:18095 \
  --worker-url http://TRAIN_NODE:8766 \
  --run-dir runs/qwen-toy --timeout 3600 \
  --rounds 2 --review-after-seconds 2 --require-training-review
```

端点是临时集群内部服务，不提供公网身份验证。基线和 worker 必须加载相同权重；baseline revision 是操作方声明，结果中明确标为未由服务证明。候选服务返回 revision，并在角色切换前执行真实推理检查。

## 两轮父子版本与运行中 review

每轮按 `train → evaluate → apply_candidate / reject → execute` 运行，最后统一 `finish`。第一次训练从声明的 baseline 开始；应用成功后，下一轮训练请求的 `base` 必须是刚绑定的 Actor。若上一轮被拒绝，则保留当前 Actor，下一轮从该保留版本训练。Decision Policy 始终使用 baseline，不随 Actor 更新。

worker 在同一底座上保留每轮独立 LoRA adapter。第二轮通过 PEFT `load_adapter(..., is_trainable=True)` 从父 adapter 文件复制权重后继续 SFT，不会把父 adapter 原地训练或重新随机初始化；若父版本仍为 baseline，则创建新的初始 adapter。评测 `base_nll` 指**本轮父版本**，不是固定的最初 baseline。旧版本保留，可用 `model_revision` 进行精确推理路由。PEFT 多 adapter 行为参考[官方接口文档](https://huggingface.co/docs/peft/en/package_reference/peft_model)。

训练 Directive 的 `review_after_seconds` 由 controller 的 `--review-after-seconds` 指定（默认 2 秒）。`GuardedDecisionPolicy.review()` 委托真正的 `ModelDecisionPolicy.review()`，模型读取运行中 execution/progress 后返回 `continue` 或 `cancel`；无脚本 continue 兜底。每次最多尝试三次结构化解析，模型取消会被核心执行。`next_review_seconds=null` 表示本次复查后不再自动复查。

监控记录 `training_review_requested/completed/rejected`，完成事件包含 `source=llm`、round、directive/job ID、decision/actor revision 和 action；核心另将 review 写入 Experience。`--require-training-review` 要求每轮在训练期间至少一次真实模型 review，否则该次验证报告失败。很短的训练可能在 2 秒前完成；需要更短间隔时显式调整该参数，不插入假 review。测试中的 HTTP/模型替身仅验证协议，不能作为真实 SFT 或 LLM review 的证据。

baseline 和 worker 必须先启动并 ready；controller 会检查 worker 的 `max_rounds`，不把旧单轮 worker 当作两轮 worker。协作修改核心时，应等待核心与本例文件同步完毕再跑真实验证。

## 结果与边界

- `result.json`：最终结果、`rounds_requested/completed`、每轮 `round_results`、`training_reviews`、实际版本及错误状态。
- `round-1.json` / `round-2.json`：每轮父版本、候选、指标、应用结果、执行答案和 review 次数。
- `monitor.json`：供[只读前端](../../web/README.md)读取的原子更新摘要。
- `experience.sqlite3`、两个 memory 文件和 `artifacts/`：系统反馈及本地任务记录。
- `adapter/`、`train-result.json`、`evaluate-result.json`：第一轮 worker 产物，保留原单轮路径。
- `round-2/adapter/candidate-2/`、`round-2/train-result.json`、`round-2/evaluate-result.json`：第二轮独立产物，不覆盖第一轮。

评测比较同一组 heldout 样本上本轮父版本与候选的 assistant-token NLL。`delta_nll = base_nll - candidate_nll`，有限且大于零才允许应用；否则保留当前 Actor。Decision Policy 的指令由 Qwen 生成，工作流程和验收条件由显式 guard 约束，不提供脚本兜底决策。应用后 Actor 再回答一条未参与训练与评测的奇偶任务。

这个小任务验证链路可运行，不能作为通用能力提升的证据；它不是 RL / verl 适配，也不是吞吐优化服务。训练数据使用本例私有的 prompt/answer JSONL，跨框架对齐由使用者自己的 adapter 负责。

每个 worker 允许 `--rounds` 指定的一到两次训练尝试，默认一次；第三次会被拒绝。每轮有独立幂等 key 与 `training-attempt.json`，同 key 重试返回同一 job。训练取消在优化步边界检查；提交与轮询支持进程内幂等，服务重启不恢复 job。任一训练失败或取消后禁止继续训练；已成功的父 revision 仍可推理。重新实验需新 run-dir 和新 worker，不覆盖已有产物。每次新 Trainer 重置优化器状态，但 LoRA 权重沿父版本延续。自定义装配未接入系统 Snapshot；模型服务的生命周期由集群管理，完成临时实验后可删除本次创建的 job。

监控包含模型显式给出的 strategy / reason、决策时两侧版本、指令与任务关联、运行中复查、当前阶段和各轮结果。训练进度来自 Trainer callback 的实际步数与日志；没有数值时不估算百分比。可用 `PYTHONPATH=src python examples/qwen_toy/monitor.py --run-dir runs/COMPLETED_RUN` 为已有结束运行补导出真实决策历史；不会重新运行模型。具体字段与可视化见[监控说明](../../web/README.md)。
