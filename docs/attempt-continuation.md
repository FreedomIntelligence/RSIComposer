# Feedback、Memory 与跨 attempt 交接

同目录 `--resume` 和新 attempt 的成果继承是两个协议。同目录恢复继续严格核对
source、数据、配置、journal 和外部效果；代码或数据变更后不能通过修改 checkpoint
身份来续跑。新 attempt 使用宿主明确选择的不可变 continuation bundle。

## 职责与组件边界

| 层 | 本次实现 | 替换组件时保留的约束 |
| --- | --- | --- |
| 通用运行时 | `runtime.continuation` 的交接包、来源校验、兼容轴、产物 manifest、部分提交标记 | 不导入执行状态，不重放动作，不重置账本；保留公开/封闭边界 |
| 默认 meta harness | AgentMemory 显式 CRUD、任务、原始来源；continuation 适配器；可复用 PeriodicReflection | 自定义 harness 提供自己的导入/上下文/状态端口，不依赖对象图猜测 |
| 通用证据投影 | 最近失败索引；压缩后保留评分、数据、协议、运行时和权重身份 | 投影视图不能更改原始记录；完整原文必须有授权分页入口 |
| 医疗 Lab | 候选打包、结构预检、官方评分、公开输出诊断、封闭聚合回执 | 不把模型自产解析/日志当成宿主分数；不放宽路径或封闭数据隔离 |
| RSI 优化器装配 | 定时 reflect 的触发、预算和允许端口；默认仅 meta.prompts/meta.harness | 可以有证据地 no_change；不把采纳、激活与效果混为一谈 |

这里既有通用协议缺口，也有具体实现问题。过去复用调用账本，并不意味着复用了
研究知识；把 reflect 只写在医疗示例中、允许旧引用反复通过，也是组件装配缺口。
组件内部如何训练、推理、检索、何时提出改动，仍可替换。

## 显式交接

宿主先确认原 controller/Lab 已停止，原 journal 无未知 mutation，所属进程均有
终态回执且精确身份不再存活。医疗适配器还持有原 controller lock，防止导出时并发续跑。
只选择已通过公开接口交付给 meta-agent 的历史记录、memory/task 的来源闭包，以及
操作者指定的 Lab 产物目录。它不读取封闭数据、私有 gateway 日志或模型请求日志。

```sh
PYTHONPATH=src:. python -m examples.medical.continuation \
  --root /operator/stopped-attempt \
  --destination /operator/campaign/handoff-01 \
  --artifact trained-adapter=baseline/lora_ckpt \
  --artifact inference=infer
```

新 attempt 在原有启动参数之外传入 `--continuation /operator/campaign/handoff-01`。
此参数属于 host，不提供给 meta-agent 自行绕过停止/隔离检查。只支持同一 campaign，
医疗适配器同时要求共用原 evaluation ledger，并从源 attempt 的调用上限扣掉已尝试调用，
把新 attempt 的上限夹紧到剩余额度；绝对截止只可提前。调用原账本保留，handoff 不代替
全部 provider usage 总账。历史实验到期后，这个入口不会自动启动新实验。

宿主获得用户新的运行窗口授权后，可以为**新 attempt** 提供
`--continuation-authorization` JSON：schema 为 `rsisys-host-continuation-renewal/v1`，
绑定 `handoff_sha256`、`campaign_id`、`target_attempt_id`、`previous_deadline_epoch`，
并记录新的 `deadline_epoch`、`authorized_at` 和 `reason`。仅允许已到期窗口的明确续期；
不增加剩余调用额度、不换评测账本，也不修改旧 attempt 或交接包。授权随新 attempt
冻结并在恢复时核验，修复或恢复本身不会延长窗口。

导出包包含来源 attempt、source/data/protocol/runtime/account 兼容轴、原始记录、
明确写过的笔记与任务，以及逐文件 SHA/大小。未写过 Memory 时仍保留已有回执与指定产物，
不会合成笔记或伪造 CRUD。导入时：

1. 给历史记录和历史 decision ID 加命名空间，避免与新 attempt 的 decision-1 等冲突。
2. 重新绑定 Memory 引用并保留原来源摘要、旧 revision；笔记仍是 agent assertion。
3. 保留 task 的 open/closed、固定来源与历史锚点，原始反馈继续可分页读取。
4. 将产物复制到新 Lab 的 `inherited/<name>`，核对 SHA。副本可继续修改，原交接包保留。
5. 在第一次 durable checkpoint 前写入交接回执，注册独立状态端口。恢复时只核验，不重复导入。

不导入 pending actions、当前 decision ledger、job/key 的执行绑定、PID/supervisor 所有权、
采纳状态或计费预留。历史原文中出现的 job/PID 仅是历史证据。部分导入留下 `.applying`
标记并阻止自动重试，由宿主核对效果后处理。完整回执的 history SHA 不符同样拒绝恢复。

`research_agenda.continuation` 提供少量历史测量/失败的机械事实索引和原文入口；
`memory_read` 可以追到重绑定后的来源。数据、协议、运行时发生变化时，旧测量保留为历史，
是否可比逐条核对。继承权重既不表示训练得到验证，也不表示新系统已经加载、评测或采纳。
已激活过的 harness 候选同样需要明确的组件注册/激活流程，不在交接时偷偷激活。

## Feedback 与 Memory

单一 meta-agent 自己解释反馈和写笔记，没有独立整理模型。最近失败索引不会强制写 Memory，
也不把“没有任务”解释成“没有问题”。最新两个失败在即时反馈窗口中保留授权原文入口，
显式任务的固定原文按原生命周期保留。第三方工具可以使用相同的观察适配器，无需伪造
`medical_*` 名字。旧 harness 的额外窗口保持兼容，避免复制查询回执撑破既有预算。

Memory 来源校验按写入时的 history_id 与 SHA 定位，重复读取相同 `result:<sha>` 不再
把旧笔记错误绑定到新记录。直接共用一个 Memory 文件却缺失来源归档会失败；应使用交接协议。

## Reflect

通用实现位于 `harness.reflection.PeriodicReflection`，医疗旧入口仅保留兼容导入。
报告至少覆盖上次报告后的最新普通操作，以及最近一次失败。每条引用仍须在本窗口读完
完整原始 archive 分页；旧记录可作补充，不能替代近期失败。

每个 interval 默认最多占用 16 个已执行 review operation；8 步后收窄到归档与报告。
耗尽后以 `incomplete_budget_exhausted` 明示该轮未完成，并释放普通研究动作。下一周期再给
有界审查机会；不生成虚假的 no_change，不要求一定改动，也不重置原 cadence。
报告仍绑定实际激活的组件版本；`improvement_verified` 始终不由自报告置为 true。

## 验证边界

回归覆盖新 attempt 的真实 CLI 装配/暂停/恢复、Memory CRUD、原文分页、产物复制/防篡改、
来源冲突、部分导入拒绝、近期反思覆盖、review 预算退出、公开诊断复算以及封闭回执隔离。
这些属于框架验收。模板的 tokenizer/collator 可以在已安装环境做 CPU 验证；并不等于
完成新的模型训练或真实医学评测。本轮修复不重启已结束实验。
