# Actor learner

提供后端中立（backend-neutral）的训练请求与作业接口。[默认 Exploration Harness](../README.md) 保留简单的本地 SFT/RL 流程，也可显式启用整项训练任务委托。

[LearningExecution](__init__.py) 封装默认执行流程：使用 JobBook 提交、查询、取消作业，校验并记录终态、保留候选及应用版本。它与训练算法后端分离；本地控制、SFT/train 指令和 LocalRLExecution 共用这条路径。外部调用仍通过 Harness 入口遵守占用检查。

## 接口与输入

[ActorLearner](__init__.py) 提供 `submit(request, *, idempotency_key)`、`poll(job)`、`cancel(job)`、`result(job)`；后端 cancel 返回 None，默认 Harness 的 cancel_learning 返回 JobOutcome。ActorTrainingRequest 保留原位置参数 base、data_refs、recipe_ref、method 和 trajectory_refs，末尾追加可选 `origin=None`。

请求只校验结构：base 为 ModelRef，method / recipe_ref 为非空字符串，data_refs 为 ArtifactRef 元组，trajectory_refs 为字符串元组。method 默认 sft，但允许后端自定义名称；两类引用均可为空，数据来源、算法与采样要求由后端解释。例如外部 RL 后端可自行 rollout，无需预先提供本地轨迹。

`TrainingOrigin(parent_system, initial_model, resume_mode=ResumeMode.RESTART, checkpoint=None)`
将实验父系统 ID 与初始权重分开。传入时 initial_model 必须等于请求 base，成功结果的
candidate.parent_system 必须等于 origin.parent_system。restart 禁止运行 checkpoint；
weights_only 只恢复权重并重新初始化优化器、调度器、步数与 RNG；full_state 要求完整状态
checkpoint，后端不支持时明确拒绝。SystemVersion / SystemRegistry 管理组件组合与 current，
不会根据父系统隐式决定训练起点；细节见[共享协议](../../../contracts/README.md)。

后端可选实现 `BackendCapabilityProvider.capabilities()`；LearningExecution 在提交前校验其
声明的 method 与 resume_mode。update_fields、checkpoint 导出和 reattach 是独立能力，
未声明支持的操作明确失败。已有 learner 无需增加该端口；具体 recipe / 数据 / checkpoint
内容和更新值仍由后端验证。

提供 trajectory_refs 时，默认 Harness 仍校验它们引用 Actor 本地 memory/history 中的轨迹。这些引用不指向 Experience，也不能用 Feedback.evidence_refs 替代。训练终态仍以 training_run 写入共享 Experience。成功的 TrainingResult.base 必须匹配请求，candidate.affected_roles 必须包含 actor。幂等、终态与版本约定见[共享协议](../../../contracts/README.md)。

## 实现与替换

[MockActorLearner](backends/mock.py) 仅支持 sft / rl，并在 submit 时要求 sft 有 data_refs、rl 有 trajectory_refs；这些是该 mock 的局部能力限制，不是请求协议的限制。它读取数据产物并生成模拟版本清单，metrics 为空，不优化真实权重或计算能力提升。真实后端通过 DefaultExplorationHarness 的 learner 参数注入；外部 RL/SFT 框架需要实现相应适配，并非自动兼容。

可复用的具体接入放在 Harness 外的 [adapters/training.py](../../../../adapters/training.py)：
ProcessTrainingBackend 组合持久作业目录、原生命令与结果解析，默认 worker 可执行
Transformers / PEFT SFT；其他 CLI 通过显式命令和 result_reader 接入。
训练配置、tokenized 数据、设备和可选依赖均由调用方提供，见
[使用指南](../../../../../../README.md#接入整项训练任务)。领域任务装配和验收规则留在
[examples/medical/](../../../../../../examples/medical/)，不写入 learner。

成功结果默认形成候选。显式 `apply_learning`、`SFTSpec(apply_result=True)`、`TrainSpec(apply_result=True)` 或默认 RL 内部更新改变 Actor 绑定；应用要求 policy 支持 ModelUpdatable、当前模型与训练 base 一致，且无环境交互或其他活跃训练。同一 job 重复应用不再次更新；这些操作不验收质量，也不自动更新 Decision-maker。

真实后端须让推理侧能够解析新版本，并兑现重试提交的幂等性、轮询和取消约定。

## 本地 RL 与整项任务委托

真实训练结果的应用需要向 DefaultExplorationHarness 注入 `model_activator`，确认目标版本可用于推理后才更新角色绑定；仅产生 checkpoint 不足以启用。接口与重试约定见[共享协议](../../../contracts/README.md#作业与版本)。适配器可读取外部 ArtifactRef，框架不要求复制权重或数据到本地存储。版本校验和已应用任务去重仍由 Harness 保留。

默认 `rl` 仍由 Harness 顺序执行本地 rollout → learner 训练 → 应用版本 → 下一轮 rollout；只有 rollout 调用 ActorPolicy.act 和 Actor Action Space。SFT 与 RL 的训练阶段直接调用 learner。

可选 `train` 需要配对注册 [training_directive()](../../../decision_making/action_space/directives.py) 和 [TrainingCommandHandler](command.py)，并注入 learner。handler 将 TrainSpec 与 Directive.input_refs 转为 ActorTrainingRequest，保留 origin；base 缺省时先使用 origin.initial_model，无 origin 才捕获角色模型。单个后端作业独立负责整个任务，不启动本地 RL 循环。提交重试、轮询、取消、证据记录及可选应用复用 Harness 的作业流程。装配方法见[使用指南](../../../../../../README.md#接入整项训练任务)。

[backend_training.py](../../../../../../examples/backend_training.py) 演示独立子进程中的真实 CPU 标量回归优化与产物回传，不是 LLM 训练，也不是 verl 的现成适配；该示例不演示 RL rollout。

显式 origin 可从与当前角色不同的 initial_model 训练，但 `apply_result=True` 的直接角色替换要求初始化与当前角色相同，并在 prepare 时检查。跨初始化的候选应单独评测，再由 SystemRegistry 检查实验父版本并应用，不能绕过 stale-parent 校验。
