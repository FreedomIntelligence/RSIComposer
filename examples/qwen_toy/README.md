# Qwen3.5 单轮真实链路

这个例子把真实 Qwen 决策、Transformers / PEFT LoRA SFT、独立样本评测、候选验收与应用、更新后 Actor 执行和 Experience 反馈串起来。适配器只属于本例，RSISys 核心不依赖 PyTorch，也不引入统一训练数据格式。

## 组成

| 文件 | 职责 |
|---|---|
| `prepare.py` | 生成 64 条训练数据和 12 条不重叠的评测数据 |
| `serve.py` / `serve.sh` | 临时基线模型服务，供 Decision Policy 使用 |
| `worker.py` / `train_worker.sh` | 单卡训练、评测及训练后候选推理服务 |
| `adapters.py` | 本例 HTTP 作业、模型路由、readiness 接口 |
| `run.py` | 装配双 loop，运行有界闭环并写监控摘要 |

模型与数据格式参考 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B) 和 [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)。训练采用 PEFT LoRA：r=8、alpha=16、q_proj/v_proj、默认 24 steps、batch=2、梯度累积=2。

## CPU 启动（默认）

两个服务默认使用 CPU。小规模验证可使用 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，也可传入已有的 9B 权重目录。先准备独立 run-dir；在本集群复用已有的 Transformers / PEFT Pixi 环境，分别在两个终端启动服务：

```sh
pixi run python examples/qwen_toy/prepare.py --run-dir runs/qwen-cpu

pixi run --as-is -m /home/jovyan/projects/vlm/SFT/Baseline/pixi.toml -e train \
  python examples/qwen_toy/serve.py --model /home/jovyan/models/Qwen3.5-0.8B \
  --served-name Qwen3.5-0.8B --device cpu --threads 4 --port 18095

pixi run --as-is -m /home/jovyan/projects/vlm/SFT/Baseline/pixi.toml -e train \
  python examples/qwen_toy/worker.py --model /home/jovyan/models/Qwen3.5-0.8B \
  --served-name Qwen3.5-0.8B --device cpu --threads 4 --steps 24 \
  --run-dir runs/qwen-cpu --port 8766
```

就绪后在第三个终端运行 controller：

```sh
PYTHONPATH=src pixi run python examples/qwen_toy/run.py \
  --baseline-url http://127.0.0.1:18095 --worker-url http://127.0.0.1:8766 \
  --model-name Qwen3.5-0.8B --run-dir runs/qwen-cpu --timeout 1800
```

CPU 模式使用 float32、隐藏 CUDA 设备并限制线程数；无需 GPU 作业。服务只读取已有权重，不自动下载。更换模型时让两个服务和 controller 的 served-name/model-name 一致。

## 可选的本集群 GPU 启动

脚本使用 `/home/jovyan/models/Qwen3.5-9B` 和已有 Pixi 环境 `/home/jovyan/projects/vlm/SFT/Baseline/pixi.toml` 的 `train` 环境：Python 3.12、torch 2.10.0 cu128、transformers 5.2.0、peft 0.18.1。迁移时修改两个启动脚本中的路径，使用自己的 Pixi 锁文件。

```sh
pixi run python examples/qwen_toy/prepare.py --run-dir runs/qwen-toy
```

仅在明确选择 CUDA 时，通过 `~/tools/cluster/kjobctl create slurm` 提交单节点、单任务、单 GPU 作业。可使用下述薄脚本，训练侧将末行改为 `train_worker.sh`：

```sh
#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
export QWEN_DEVICE=cuda
export QWEN_MODEL_PATH=/home/jovyan/models/Qwen3.5-9B
exec /bin/zsh /home/jovyan/projects/RSISys/examples/qwen_toy/serve.sh
```

日志参数放在 kjob 的 `--` 后；创建后为本次 job 设置 `activeDeadlineSeconds=7200`、`backoffLimit=0`。本集群会暴露全部设备节点，脚本在 Python 导入 torch 前将调度器分配的 GPU UUID 映射到对应物理序号，只使用分配的那张卡。不要用未分配的设备绕过调度。

两个服务的 `/health` 就绪后，在能访问计算节点的环境启动 controller：

```sh
PYTHONPATH=src pixi run python examples/qwen_toy/run.py \
  --baseline-url http://BASELINE_NODE:18095 \
  --worker-url http://TRAIN_NODE:8766 \
  --run-dir runs/qwen-toy --timeout 3600
```

端点是临时集群内部服务，不提供公网身份验证。基线和 worker 必须加载相同权重；baseline revision 是操作方声明，结果中明确标为未由服务证明。候选服务返回 revision，并在角色切换前执行真实推理检查。

## 结果与边界

- `result.json`：结果、实际版本、验收指标、采用与否和错误状态。
- `monitor.json`：供[只读前端](../../web/README.md)读取的原子更新摘要。
- `experience.sqlite3`、两个 memory 文件和 `artifacts/`：系统反馈及本地任务记录。
- `adapter/`、`train-result.json`、`evaluate-result.json`：worker 的真实产物。

评测比较同一组独立样本上基线与候选的 assistant-token NLL。`delta_nll = base_nll - candidate_nll`，有限且大于零才允许应用；否则保留基线。Decision Policy 的指令由 Qwen 生成，工作流程和验收条件由显式 guard 约束，不提供脚本兜底决策。应用后 Actor 再回答一条未参与训练与评测的奇偶任务。

这个小任务验证链路可运行，不能作为通用能力提升的证据；它不是 RL / verl 适配，也不是吞吐优化服务。训练数据使用本例私有的 prompt/answer JSONL，跨框架对齐由使用者自己的 adapter 负责。

每个 worker 只允许一次训练尝试。训练取消在优化步边界检查；提交与轮询支持进程内幂等，服务重启不恢复 job。失败后用新 run-dir 和新 worker 重试，不覆盖已有产物。自定义装配未接入系统 Snapshot；模型服务的生命周期由集群管理，完成临时实验后可删除本次创建的 job。
