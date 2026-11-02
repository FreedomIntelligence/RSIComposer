# Qwen 自主研究 toycase

Decision-maker 自己选择研究步骤；没有 train → evaluate → apply → execute 阶段表，也没有固定的候选接受阈值。模型可以先执行当前 Actor、查询日志与源码、训练、重复评测、应用候选、修改 prompt，或直接结束。框架只检查已安装能力、外部权限、预算和请求/模型版本一致性；无脚本兜底决策。

这个示例使用真实 Qwen 服务、Transformers / PEFT LoRA SFT、独立样本 NLL 评测和版本化 Actor 推理。训练仍只支持本例的 `toy-lora` / `sft`，评测协议为 `heldout-nll`，不运行真实 RL。数据格式属于本例，核心不依赖 PyTorch，也不转换跨框架数据。

## 能力、权限与预算

- 系统指令：train、evaluate、apply_candidate、execute、configure_prompt。
- 研究操作：已有实验查询、workspace_list / workspace_read / workspace_search，以及默认研究笔记工具 research_read / research_patch / research_archive。笔记工具与 DefaultDecisionHarness 共享 agenda 和 DecisionMemory；archive 可分页读取完整历史、工具回执、研究笔记修订和 experience。模型自行决定是否维护笔记，笔记不代表已验证结论。默认只挂载本次 run、框架 src 和本例源码三个命名工作区；远端 worker 的未共享文件仍需后端显式暴露。
- 默认开放所有已安装操作；`--permissions FILE` 使用 ActionPermissions JSON 覆盖权限。文件工具的路径约束不替代 OS / 容器沙箱。
- `--max-trainings` 默认 2，是训练提交上限；兼容旧参数名 `--rounds`，不再表示必须执行的轮数。当前 worker 最多允许两次训练尝试。
- `--max-directives` 默认 16；`--max-model-calls` 默认 32，包含研究决策、复查及解析失败重试；`--max-output-tokens` 必须为正整数，默认保留 200 tokens，可显式设为 512 或 1024，为 research_patch 等较长 JSON 响应留出空间。`--max-context-chars` 默认 20000，按字符限制输入视图：先移除最旧记录，必要时省略大段 details/data 并明确标注，可查询完整来源；这不是 tokenizer 精确 token 计数。原始记录不裁剪。另有整体时间、HTTP 超时及推进 tick 预算。
- `--review-after-seconds` 仅提供提示，首次复查时间由模型写入 Directive，后续时间由 ReviewDecision 决定。`--require-training-review` 是可选的运行结果检查，不替模型插入复查。

训练次数上限会隐藏进一步 train 能力；实验指令预算用尽后仍可在剩余模型调用预算内查询或结束。模型直接 finish 会如实记录零训练，不会被宣称为训练链路成功。跨研究周期重启时，训练提交、实验指令、模型调用及验证重试的预算累计保留，不重置；上下文、权限与 worker 上限继续生效。

## 运行结束策略

`--run-mode` 将运行结束条件交给核心 RunPolicy；研究步骤和候选应用仍由模型决定。

| 模式 | 参数与结束条件 | 模型选择 finish 后 |
| --- | --- | --- |
| `researcher`（默认） | ResearcherRunPolicy；研究者结束并完成清理后，记录 `completed` / `researcher_finished` | 结束本次运行，兼容原行为 |
| `time` | TimeRunPolicy；必须指定正数 `--run-seconds`，到期取消未完成工作 | 剩余时间内等待后重启研究 |
| `goal` | GoalRunPolicy + MetricGoal；必须指定 `--goal-metric`、有限数值 `--goal-threshold`；可用 `--run-seconds` 设置最长运行时间 | 目标未达成且未到期时等待后重启研究 |

`--goal-direction higher`（默认）表示指标 ≥ 阈值，`lower` 表示指标 ≤ 阈值。目标仅采用既有 `heldout-nll` 协议下已完成、非模拟评测的指标；例如 `delta_nll` 越高越好，`candidate_nll` 越低越好。指标达到目标不会自动应用候选，也不保证已经执行 Actor。目标达成使用 drain 收尾；目标模式的最长时间到期则使用 cancel。

`--restart-delay-seconds` 默认 1，必须是有限正数；time/goal 模式中 finish 只结束当前研究周期。控制器在 Runner 返回 waiting 时仍按 `--poll-interval` sleep，因此实际重启可能晚于指定间隔。

`--run-seconds` 是从 Runner 首次推进开始计算的运行策略时长，不是 HTTP 超时，也不会重置旧资源预算。`--timeout`（默认 3600 秒）继续作为整个控制器和服务调用的硬时间边界，包括健康检查、研究、重启等待及策略清理；`--request-timeout` 还限制单次 HTTP 调用。RunPolicy 不能抢占正在进行的同步服务调用，硬边界用于防止死等。策略清理窗口为 30 秒，受剩余 `--timeout` 限制；硬边界或异常退出后另有最多 5 秒的应急取消窗口。为正常策略结束留出余量，应让 `--timeout` 大于 `--run-seconds` 加服务启动检查与清理所需时间。`--max-steps` 等原有硬上限始终有效。

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

时间模式可在上述命令追加 `--run-mode time --run-seconds 600 --restart-delay-seconds 2`。
目标模式可追加 `--run-mode goal --goal-metric candidate_nll --goal-threshold 0.01 --goal-direction lower --run-seconds 1200`。
每次运行使用新的 controller 记录目录；持续研究仍可能先触及模型调用、训练或指令预算。

CUDA 部署使用 [serve.sh](serve.sh) 和 [train_worker.sh](train_worker.sh)，通过本集群 `~/tools/cluster/kjobctl create slurm` 提交单节点、单任务、单 GPU 的 zsh 脚本，分别设置 `QWEN_DEVICE=cuda`、`QWEN_MODEL_PATH`、`QWEN_SERVED_NAME`、端口和 worker 的 `QWEN_ROUNDS=2`。baseline 与 worker 各需一张分配卡；脚本只使用调度器分配的 GPU UUID。设置作业 activeDeadlineSeconds 与 backoffLimit，完成实验后清理本次资源，不直接使用未分配 GPU。CPU 服务默认只监听 localhost，CUDA 临时服务只用于受控集群网络。

## 实际模型更新与执行边界

worker 为每个候选保存独立 LoRA adapter；从父候选训练时复制父权重继续更新，不原地破坏旧版本。每次训练重新创建优化器。调用方负责相同底座配置；baseline revision 是操作方声明，候选在绑定前通过服务 revision 与真实推理 readiness 检查。

`train.apply_result` 由模型选择。模型也可先产生多个候选再评测或应用；应用要求候选 base 与当前 Actor 匹配。NLL 指标是证据，不自动触发采纳。未应用候选保持 proposed，不能根据负 delta 或缺少 apply 就标记为模型明确 rejected。

此 toy worker 的模型操作由一个锁串行保护，不代表两个 GPU 训练任务已经并行；框架的并行实验需要为实验工厂配置独立执行资源或支持并发的外部后端。默认 Decision-maker 仍使用 baseline，Actor 可切换版本；本例不自动训练 Decision-maker。

## 结果与监控

- result.json：模型选择的 `experiments` 序列、状态、预算、模型调用次数、训练提交次数、候选与 Actor 版本、取消确认，以及 `run_mode`、`run_stop`、`stop_reason`、`goal_reached`。
- round_results / round-N.json：每次 Actor execute 的观察记录。保留旧字段名供前端读取，**不是预设实验轮次或训练次数**。
- monitor.json：研究操作、实验指令、复查、任务进度、候选与版本，以及同样的停止原因和清理状态，供[监控前端](../../src/rsisys/monitor/README.md)读取。触发目标停止的最后一次反馈仍记录为完整实验，并更新候选指标。 作业沿用其所属决策的 round 和 directive_id；迟到的旧作业进度不会被归到当前新指令，也不会覆盖当前作业进度。
- experience.sqlite3：系统反馈；两份 memory JSON：各自任务历史；decision_research.json：模型首次 patch 后保存的完整研究笔记和修订日志；artifacts：轻量引用对应的产物。
- worker 的 adapter、训练/评测结果和日志由 worker 数据目录保留。

`decisions[]` 的公开解释字段为 `reason`、`changes`、`feedback_summary` 和 `error`。
`reason` 只来自 `Decision.reason`（复查使用 `ReviewDecision.reason`）；历史没有理由就留空，不从模型请求或推理内容补写。
`changes` 是**提交内容**，最多 20 项，每项 `field`、`after`、可选 `before` 均为字符串，每个文本最多 4000 字符。
默认不生成 `before`：前一次 prompt 提案即使 `completed` 也不代表 Actor 已应用，不能作为下一次的前值；只保留来源明确记录、且对应同一提交字段和值的已有 `before`。
`configure_prompt` 公开 Actor 的 `instructions`；训练公开显式的 `recipe_ref` / `method` / `apply_result`；评测公开协议、候选 ID、底座版本和候选组件名（不读取组件 artifact）；应用候选公开候选 ID。
`define_recipe` 仅公开有限数值超参数 `steps` / `lr` / `seed` / `weight_decay` / `warmup_ratio`。其他操作不展开任意参数或结果 `data`，也不公开训练样本、模型 request prompt 或 `reasoning_content`。
`feedback_summary` / `error` 保留实际 Feedback 或 OperationResult 的原文摘要和错误（最多 4000 字符），无回执就留空，状态 `completed` 不替代摘要。
例如 `Prompt proposed; evaluate before application` 表示产生了待评测提案，并未承诺 Actor 已更新；`evaluate` 的提交字段说明验证哪个候选的哪些组件，回执与数值指标说明观察结果。

部署后可显式补全已结束运行的决策记录（支持 medical 的无 `phase` 结果，也支持 `stopped`）：

```sh
PYTHONPATH=src:. python -m examples.qwen_toy.monitor --run-dir runs/<finished-run>
```

补全只写 `monitor.json` 的 `decisions`，保留已有决策 ID、timestamp、source 和全部其他监控字段（包括 `updated_at`、运行状态、workflow、current、模型绑定）；不修改 memory、steps、研究结果或训练文件。
工具使用 `decision_memory.json` 的结构化 payload/feedback；`kind=operation` 回执只与紧邻且操作及参数完全一致的决策配对。
可选 `steps.jsonl` 的指令反馈按 directive ID 补全，工具回执仅在完整决策顺序与历史一致时逐条匹配；截断、重复或缺失导致关联不明时留空，不按重复操作名猜测。
无法可靠对应的历史记录不会覆盖已有行；重复补全同一证据保持幂等。补全不会自动运行，也不改变运行结论。

`completed` 表示研究者选择结束，或配置目标已达成并完成清理；只有 `stop_reason=goal_reached` 才令 `goal_reached=true`。RunPolicy 时间预算到期且清理已确认时记录 `stopped` / `budget_exhausted`，CLI 返回 0；**stopped 不表示目标达成**。目标模式的 `--run-seconds` 到期也使用此语义。默认 researcher 模式保持 `completed` / `researcher_finished`，不宣称目标达成。是否训练、评测、应用以及是否提升，必须查看具体证据。

`--timeout`、模型调用或 tick 等硬预算耗尽，以及策略清理未确认时，记录 `timed_out`，CLI 返回 1；其他异常记录 `failed`、返回 1。这些情况不合成模型 finish。`run_stop` 保存核心返回的 `reason`、`shutdown`、`shutdown_complete`、`cleanup_timed_out`、`unresolved_directives`、`unresolved_research_job` 等字段；硬边界可能先于策略停止，此时 `run_stop` / `stop_reason` 为 null，以 `status` 和 `error_code` 表示异常。

清理未确认时，`remote_may_run=true`，兼容字段 `remote_job_may_still_run` 同步为 true；取消请求被接收不等于远端任务已终止。若硬边界中断了策略清理，保留最后一次未确认的 `run_stop`，不声称 Runner 已完成清理。HTTP 测试使用替身，只验证控制链路，不代表真实 GPU 训练结果。

取消在训练优化步边界处理；提交/轮询幂等是进程内保证，服务重启不恢复作业。取消后不自动回滚已应用版本。当前 worker 在训练失败或取消后不再接收新的训练尝试，可继续读取既有证据；重新训练使用新 worker。自定义装配未接入 Snapshot，模型服务生命周期由部署环境管理。

## 真实自主验证记录

2026-11-01 使用 Qwen3.5-9B 与 CUDA SFT/评测验证。模型自行选择 train → evaluate → finish，未应用候选，也未调用 Actor execute；父版本 heldout NLL 约 0.044638，候选约 0.000014759。一次缺失 method/recipe_ref 的响应在获得具体校验反馈后由模型自行修正。结束表示研究者停止，不表示部署后的 Actor 已更新或通用能力提升。此前失败尝试保留，分别暴露了参数反馈不明确与上下文超限问题；未用脚本补齐模型步骤。

## 可选 SwanLab 导出

本地监控不依赖 SwanLab。需要实验跟踪时，在使用者的环境安装可选 `swanlab`，从项目根目录运行：

```sh
PYTHONPATH=src:. python -m examples.qwen_toy.export_swanlab runs/<run-id>/monitor.json --project RSISys
```

默认 `offline`，只在本地写入 `swanlog`；显式 `--mode online` 才向已配置的 SwanLab 服务上传。每个有指标的 job 导出为一个实验；重复执行会创建新实验，不做自动恢复或去重。只导出公开指标投影，不发送模型输入和训练样本。导出已有快照，不是实时 SwanLab 同步；快照缺少历史时仅导出最新值，不补造曲线。SDK 的 step 是记录序号，实际训练步保存为 `optimizer_step`。

接口参考：[SwanLab init](https://docs.swanlab.cn/api/py-init.html)。

训练过程由 Trainer 实际日志采样（当前每 2 步），记录 loss、学习率、梯度范数、epoch 等有限数值；不为没有上报的步骤插值。worker 在 `<job_id>.metrics.jsonl` 追加完整采样，HTTP 和 `monitor.json` 的 `jobs[].metric_history` 保留最近 1000 点。历史从任务接口传到监控，不依赖前端刷新频率。旧运行缺少历史时只显示最终指标。
## Run ancestry

The controller accepts `--run-dir runs/run6 --parent-run-id run5`. This is an
explicit ancestry declaration, not a resume operation: model revision selection
is unchanged and no prior memory, weights or optimizer state is restored.
Omit the parent flag for an unknown/root source. The controller writes `run_id`
and `parent_run_id` to `run.json`, `monitor.json` and `result.json`.

`Monitor(..., source=None)` reads the current `run.json`; a supplied
`RunSource('run6', 'run5')` must agree with that metadata. Monitored run IDs must
equal their directory names, including the parent directory (`runs/run5`), since
the web viewer identifies runs by directory name. Mismatches are rejected before
monitor writes. Core Applications with other ID/directory combinations require
an aligned integration. Offline `enrich_completed_run` validates recorded ancestry
and preserves the snapshot's ancestry fields and observation time; it never fills
missing parent fields or infers parents from history or sibling directories.

监控在策略提出决策时不发布未确认的研究尝试归属；每次 `runner.step()` 后，以 `monitor.sync_research_attempts(harness.state.decision_records)` 读取已接受关联，再调用 `monitor.advance()` 写入快照。历史导出读取 Harness 保存的明确 `research_attempt_id` / `research_attempt_title`，不推断旧实验分组。
