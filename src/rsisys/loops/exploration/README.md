# Exploration loop

负责指令执行、环境探索、学习和候选评测；跨 loop 协调见[系统架构](../../../../docs/ARCHITECTURE.md)。

## 生命周期

[loop.py](loop.py) 对外提供 accept / advance / inspect / cancel / acknowledge，委托 [commands.py](commands.py) 管理单条指令。指令类型与终态约定见[共享协议](../contracts/README.md)。

`accept` 复制并校验请求；`advance` 推进阶段（execute 会同步跑完整个 episode）；`inspect` 不推进；`cancel` 标记取消，后续 `advance` 处理；终态反馈交付后 `acknowledge` 释放占用。返回进度为副本，终态由是否携带 Feedback 判断。

## 可替换组件

- [policy](policy/README.md)：选择行动。
- [action_space](action_space/README.md)：声明并校验行动。
- [environment](environment/README.md)：执行行动与产生观察。
- [learner](learner/README.md)：SFT/RL 训练后端。
- [evaluation](evaluation/README.md)：候选评测后端。

通过 `ExplorationLoop(policy, space, environment, experience, learner=..., evaluator=...)` 注入。

## 本地控制

本地入口是 Python API：`run(Directive(..., mode="execute"))`、`submit_learning` / `poll_learning` / `cancel_learning` / `apply_learning`，以及 `submit_evaluation` / `poll_evaluation` / `cancel_evaluation`。提交训练、评测须提供 `idempotency_key`；完整示例见 [loop_internals.py](../../../../examples/loop_internals.py)。

[rl.py](rl.py) 定义 RLPlan / RLState；`start_rl` / `step_rl` / `cancel_rl` 驱动 rollout → submitting → training → 下一次 rollout，达到更新次数后结束。每次成功应用的新版本用于下一次采样，每次 rollout 使用独立 ID；跨 loop 的 RL 反馈汇总采样证据、训练记录和已应用的候选。

跨 loop 指令未 acknowledge 前，禁止本地执行、提交、应用模型及 RL 控制；已有 job 的 poll / cancel 仍可调用。本地 RL 占用执行、训练提交与模型更新，评测可独立提交。SFT 指令或 RL 取消与成功竞速时不自动应用本次候选，已应用的更新不回滚。当前 RL 顺序采样/训练，不支持并行采样或异步权重广播。
