# SFT / RL 方法接入后的架构检查

结论：保留 **RSI 决策器 → Lab → 反馈 → RSI 决策器** 两模块边界。
这次 SEAL 嵌套 SFT 和 TTRL GRPO 均通过注册及同一个运行入口完成两轮真实训练，
没有为训练方法增加第三个顶层模块，也没有让 launcher 或维护者替算法挑候选。
但不能把“能注册”理解为“已经提供了完整训练平台的运行保障”。

## 实跑证据与范围

| 方法 | 决策器负责 | Lab 内部负责 | 实际证据 |
| --- | --- | --- | --- |
| SEAL | 请求生成、内层适配、top-1 阶段、外层训练及模型交接 | 8 卡推理、48 次内层 SFT、判分、8 卡外层 DDP | [结果](../../experiments/seal/RESULTS.md) |
| TTRL | 无标签 rollout、reward、GRPO 更新与下一轮起点 | 多数奖励、token likelihood、初始 KL reference、8 卡 DDP | [结果](../../experiments/ttrl/RESULTS.md) |

SEAL 的 top-1 是决策器选择的 `select` 算法阶段，由 Lab 按固定规则执行；维护者不读取
分数后手选。内层试训同样是执行动作，其反馈决定外层训练数据，属于算法明确规定的嵌套。
两次小规模实跑均没有得到性能收益，因此只证明接口与执行闭环，不证明论文指标或通用收益。

## 本轮修复

| 问题 | 观察或故障模式 | 修复及验证 |
| --- | --- | --- |
| 注册成功不保证能力匹配 | TTRL 决策器可被配置到没有 rollout/reward 的 Lab，原本到执行时才发现 | 可选 `required_operations()`，装配时检查命名操作；错误组合的测试保证在执行前失败 |
| 生命周期与算法结果混在一起 | 两轮训练的 host `rounds_completed` 仍为 0；失败后 finish 也可能显示 finished，`training_performed` 对自定义装配为 null | 可选 `report()`，正式协议在 src；`Application.controller_report()` / CLI 透传独立算法状态、轮数和真实训练信息；不更改旧 host 字段语义 |
| 字符串引用无法证明正确交接 | 同一轮不同父模型的数据可能被误交给后续阶段；目录名改变不证明权重改变 | 新 Lab 保存 round、producer、parent 和 SHA256，拒绝错轮、错父模型及被修改的数据；实跑后逐张量比较发布权重 |
| adapter 改变不等于发布权重改变 | 很小的 LoRA 更新可能在 BF16 合并时消失 | 新 worker 发布前检查目标 dtype 中仍有真实变化；CPU 张量测试覆盖舍入丢失和实际 merge 一致性 |
| 组件替换可能破坏真实性标记 | 自定义 engine 如果没有 simulated/version，容易产生含糊训练声明 | engine 在创建工作目录前验证接口与标记；SEAL scorer / TTRL verifier 可通过库外工厂注入 |
| GPU 配置可能越过上层分配 | 子进程重设 CUDA_VISIBLE_DEVICES，可能使用调用方没有分配的卡 | 新 engine 校验物理 GPU indices 不得越过继承的 mask，含 CPU-only mask 测试 |

`required_operations` 只检查操作名，不声称预先证明整个输入 schema 或训练语义兼容；
运行时已有 schema 验证仍然生效。`controller_report` 是方法提供的事实，不是 host 根据
分数推断出来的“科研成功”。模拟更新与真实更新分别记录，质量变化另看评测。

上述补强是运行基础设施和证据约束，不修改 reward、优势、候选、top-1 或 checkpoint 选择规则。
原始两次实验使用冻结 v1 源码，后续库修复没有热更新进运行。最终控制器重放原回执，
逐条得到相同请求；默认算法路径保持一致。发布前 dtype 检查由后置权重审计及 CPU 测试补验。

## 仍需作为底层能力建设的部分

1. **恢复是当前首要缺口。** 新决策器的阶段、pending、模型引用和 Lab 的 artifact 所有权
   仍在进程内。虽然每阶段保存回执和 checkpoint，本配方明确不支持自动恢复。需要将 controller
   state adapter、Lab 账本及外部作业 reconciliation 接入已有恢复机制；不能直接重放 train。
2. **长作业的异步和预算语义尚未打通。** 当前通过同步 Lab 和独立 worker timeout 完成。
   host 在一次长 execute 内不能按研究预算即时重排；需要可持久化的 submit/poll/cancel、
   心跳、结果绑定以及训练耗时/GPU 成本接入。不能把“提交成功”当“训练成功”。
3. **GPU 资源归属仍由配置方负责。** 继承 mask 已保护，但多 Lab 同时选择同一 GPU 没有
   统一 lease / 容量调度。本轮两个实验依次运行，退出后确认全部显存为零。多实验并行还需要
   基础资源管理，而不是继续在算法插件里复制 GPU 管理器。
4. **模型和数据引用仍非全框架统一契约。** 新 Labs 做了内容哈希和 lineage，但不能替所有
   第三方 Lab 保证 checkpoint、tokenizer、奖励协议及数据 split 兼容。后续可标准化可选
   artifact manifest；不要让决策器理解每种训练库的内部 checkpoint 目录。
5. **参考训练器不替代成熟分布式后端。** 当前 GRPO 为独立 LoRA 参考实现。扩展到正式
   大规模 RL 时，应注册采用 verl / FSDP 等后端的 engine/Lab，复用相同决策器；不能把本轮
   张量测试等同于已经完成 verl 集成。

这些缺口不要求推翻两模块设计。注册层继续负责组合和早期兼容检查；持久化、调度、资源、
取消和公共 artifact 语义属于底层库；方法内部训练策略、奖励和领域执行留在组件与 Lab。

## 验证

最终 Python 全量回归 2054 项，74 项可选依赖/外部环境测试跳过，其余通过。
b200 的最终独立源码上另跑 19 项针对性测试全部通过，包含 4 项真实 CPU 张量 / PEFT 检查。
新增文件 Ruff E9/F/I、修改的核心入口 E9/F 及 diff whitespace 检查通过。
wheel 构建并在干净环境安装验证：两方法可注册，不导入 torch / transformers / math_verify，
派生 SEAL 代码的许可声明包含在 wheel 中。前端没有改动。
