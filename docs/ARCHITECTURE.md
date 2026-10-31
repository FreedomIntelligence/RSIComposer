# 系统架构

框架负责两个 loop 的生命周期与协同，具体能力由组件实现。本文维护跨组件关系；组件细节从[根 README](../README.md)进入，启动方法见[使用指南](GUIDE.md)。

## 模块关系

```text
src/rsisys/
├── loops/
│   ├── decision_making/   # loop + policy / memory / decision_space / learner
│   ├── exploration/      # loop + policy / memory / action_space / environment / learner / evaluation
│   └── contracts/        # 两个 loop 共用的消息、引用和公开端口
├── experience/           # 共享系统反馈库
├── snapshot/             # 独立保存与恢复
├── runtime/              # 公共后端与协同推进
├── application.py        # 本地持久化装配
├── config.py             # 内存演示装配
├── cli.py                # 命令逻辑
├── __main__.py           # python -m rsisys 的薄入口
└── __init__.py           # 包初始化
```

policy、space、learner、memory 等实现归所属 loop，通过构造函数注入。Decision memory 保存 research 准备历史、工作记忆与决策记录；Actor memory 保存任务轨迹与工作记忆。原始 test-time history 留在各自 memory，不进入共享 Experience。

Experience 收集 runtime_feedback / training_run / evaluation / improvement / insight。scope 仅标记来源，不限制可见性，也不指定优化对象：Decision policy 可检索全部来源，自行选择优化目标与策略；Actor policy 默认读取自身 memory/history，不注入系统经验。improvement / insight 是可记录接口，不代表已实现自动系统自修改。

Runtime 提供公共能力并通过公开端口协调，不选择改进策略。Snapshot 独立保存两个 loop 的 memory 状态与共享 Experience；当前保存恢复仅支持内置实现。

每个 loop 目录的组件整体组成对应角色的 harness：policy 负责模型调用与策略，context + memory 管理上下文与历史，space 提供工具定义与校验，environment 执行 Actor 任务，loop/state 管理运行循环与生命周期；learner/evaluation 是 RSI 学习与评测扩展。Decision 的准备操作由 decision_space 执行。memory 是 harness 子模块，无需另建 harness 目录或重复包装类；Runtime 提供底层模型、作业和 sandbox/进程能力，Runner 协调两个角色。

模型可共享或独立绑定，部署在框架外；角色引用相同模型不表示训练结果自动同步。

## 双 loop 协同

1. Decision-making 完成本地准备，policy 发出 Directive。
2. Exploration 接收请求并按指令推进环境执行、学习或评测。
3. Runner 遇到等待返回调用方，下次推进继续原任务。
4. Exploration 形成终态 Feedback；Decision-making 通过 receive_feedback 接收，将策略结果记为 improvement，供下一次决策使用。
5. 反馈交付成功后确认指令，释放探索流程占用。

消息与公开端口定义见[共享协议](../src/rsisys/loops/contracts/README.md)；两边内部流程见 [Decision-making](../src/rsisys/loops/decision_making/README.md) 和 [Exploration](../src/rsisys/loops/exploration/README.md)。

当前一次只有一个未完成交付的指令；一条 RL 指令内部可以多次采样和更新 Actor。自动监控频率、时间/token 优化触发、多 Actor 并发及共享模型更新事务尚未实现。

Actor 每个执行 episode 记录一条 runtime_feedback；Feedback.evidence_refs 引用系统记录。RL 请求的 trajectory_refs 另从 Actor 本地 history 获取；训练终态反馈仍写入 Experience，原始轨迹不复制到系统反馈库。

## 交付与失败边界

- Runner.step 最多调用一次决策并推进一个探索阶段；run 遇到 waiting 返回，不在内部连续轮询。等待状态可以没有 JobRef，rounds 统计已交付反馈的指令数。
- 新决策使用新 ID；同 ID 同内容重复接收返回原进度，内容不同则拒绝。指令交付确认前独占探索流程。
- 取消只提出请求，后续推进确认终态；取消与训练成功竞速时不自动应用候选。同步环境调用不能由下一 tick 中途抢占。
- 提交响应丢失后保留请求与幂等键，不重新生成任务。后端需兑现幂等提交；非终态查询或记录异常允许后续推进重试。
- Runner 保留待交付反馈，Decision-making 使用稳定记录 ID。receive_feedback 接受最近一次相同交付的重试；写入失败只重试交付，不重跑 Actor。
- Actor 已执行但系统反馈写入失败时，保留结果和稳定记录 ID，下一次推进仅补写；结果保存前不交付终态，也不重新执行 rollout。执行异常仍向调用方抛出，清理异常附加到原异常。
- 准备操作的记录失败同样保留待写结果，重试完成 memory 与 Experience 写入后再继续。回执和待写区仅在进程内，不保证崩溃后的恢复。

`completed` 表示终态反馈已交付，任务结果看 Feedback；是否有质量提升由评测确定。模型绑定、服务部署与候选验收由各自组件或策略负责。

## 图示

![系统流程](diagrams/system-flow.svg)

![文件架构](diagrams/file-architecture.svg)

修改生成器后运行 `python docs/diagrams/build_diagrams.py`。只维护 SVG，不保留重复 PNG 预览或独立图集文档。

## 参考

- [Anthropic：Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)：跨 session 上下文与执行衔接。
- [LangChain：The Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness)：state、tools、orchestration、memory 与底层 sandbox 的职责。
