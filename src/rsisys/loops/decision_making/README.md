# Decision-making loop

负责准备、选择改进策略、下发指令并消费反馈。跨 loop 的交付规则见[系统架构](../../../../docs/ARCHITECTURE.md)。

## 流程

由 [loop.py](loop.py) 编排：读取目标和可见经验 → policy 选择决策 → 执行准备操作或下发探索指令 → 消费反馈后继续，直到结束。

同步准备完成后可继续推进；异步准备等待外部回填，不自动轮询或重做。下发指令后等待反馈，不直接执行 Actor，也不能在反馈到达前作出下一次决策；新指令必须使用未用过的 ID。

有界运行在下发、等待、结束或步数上限处停止。接纳反馈后清空本轮准备结果并保留最新反馈；仅最近一次已确认且内容完全一致的交付可幂等重试。

## 组件导航

- [policy](policy/README.md)：决策上下文、输出及模型策略参数。
- [decision_space](decision_space/README.md)：准备操作、参数及异步回填。
- [learner](learner/README.md)：决策者训练任务及模型应用边界。

[decision.py](decision.py) 定义决策与运行结果，[state.py](state.py) 保存流程状态；经验读写见 [Experience](../../experience/README.md)。决策者自身训练独立于探索指令，不自动完成 Actor 任务，也不按时间或 token 自动触发。
