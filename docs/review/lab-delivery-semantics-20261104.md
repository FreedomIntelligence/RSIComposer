# Lab 失败结果与反馈事件交付

用户已确认：可由模型修正的工具错误作为失败结果回传并继续 decision；
同一终态结果的每次收集都归档，默认只整理一次 memory，显式请求可再整理。
权限拒绝、身份冲突和基础设施故障仍交给 host。
此前 [执行审查](lab-execution-review-20261104.md) 与复现数据保留为修复前证据。

## 工具失败

[LabToolError](../../src/rsisys/loops/contracts/execution.py) 是可选的 provider 错误类型。
Lab 可直接返回失败 `OperationResult`，或抛出此类型；
[LabExecutionTools](../../src/rsisys/adapters/execution.py) 只将此类型转换成失败结果。
默认 harness 记录、归档并把错误交付给下一次 decision，不自动重做原动作。
通用 describe / execute 两方法入口保持不变。

[LocalProcessLab](../../src/rsisys/adapters/local_lab.py) 明确分类：

| 模型可修正，返回失败结果 | 仍抛给 host |
| --- | --- |
| 缺少或写错参数、空 argv、未知 owned job、非法读取窗口、非文本文件内容 | host 额度/超时上限、关闭后新增工作、权限拒绝、workspace 路径越界 |
| workspace 内缺失文件或目录 | workspace 本身缺失、符号链接访问拒绝、产物/日志存储失败 |
| 找不到指定可执行文件、文件不可执行格式 | 其他未分类 provider/backend 异常 |
| 并发槽暂满；修改参数却复用已有提交 key | 模型/进程/服务身份不符、同一终态事件的内容冲突 |

夜间实测补充：提交 key 冲突仍在副作用前拒绝，不执行新参数，也不覆盖原 key。
错误作为失败结果提示 decision 改用新 key；并发槽忙则先查询或取消既有任务。
这类可修参数错误与已经登记的进程、模型或终态结果身份冲突分别处理，后者仍交 host。

程序自身非零退出仍由原有失败结果交付，不需要抛 LabToolError。
输入校验在副作用前完成；已经发生效果的工具错误不承诺无副作用，更不自动重放。
直接调用 Lab.execute 的接入方可以捕获 LabToolError；默认装配已由适配器处理。
核心不会宽泛捕获所有 ValueError/OSError，也不会匹配异常文字猜测是否可继续。

## 收集回执与结果事件

可选 [FeedbackDelivery](../../src/rsisys/loops/contracts/delivery.py)：

```python
FeedbackDelivery(event_id="producer:run:job:terminal", reprocess=False)
```

`OperationResult.feedback_delivery` 或 `ExecutionReceipt.feedback_delivery` 携带这份元数据。
event_id 由执行设施提供，属于 Lab/run 作用域；未提供作用域时应自行保证唯一命名。
它表示一个不可变的生产结果，与收集 decision 的 ID 分开。
结果业务字段仍是自由格式；同名 output/data 字段不能影响处理或去重。
没有该元数据的模块保持每回执处理，无需补充这个可选窗口。

[交付路由](../../src/rsisys/loops/decision_making/feedback/delivery.py) 校验同事件的完整原始结果，
仅排除交付元数据本身；不按文字相同或时间相近归组，不解释 stdout 或工具名称。
引用与状态也属于不可变结果，同事件重新生成不同引用应当被拒绝或使用新的事件身份。
收集参数和收集 decision 可变化，不改变生产结果身份。

| 场景 | 原始捕获/归档 | 后续处理 |
| --- | --- | --- |
| 首次收集 | 独立回执 | 排队，按原有 host 准入处理 |
| 重复收集同一结果 | 新回执，记录 duplicate_of | archive_only，跳过 context observer、memory processor 与可选提案 |
| 同一回执交付重试 | 同 ID、同内容幂等 | 不新增工作 |
| 显式 reprocess | 新收集回执 | 新处理尝试，保留同生产事件身份，仍受 host 限额约束 |
| 同事件却结果变化 | 原始决策/操作历史保留 | 身份冲突，交给 host，不吞掉或覆盖 |
| 不同事件但文字相同 | 各自捕获 | 分别处理 |
| 只读查询归档 | 各自归档 | 不占用首次处理机会 |

“默认一次”是一次被捕获的处理尝试；队列/输入准入拒绝、worker 失败或结果未知时，
重复读取也不自动重试。需要再处理时显式请求，由 host 再次准入。
journal 原子写成功后才建立去重关系；落盘失败后不会丢掉唯一的整理机会。
重载 feedback journal 后保留去重，但不承诺恢复 Lab 进程、重放程序或重置预算。
只归档的读取不占用首次整理机会，但仍校验同一事件的结果不可变。
去重按捕获顺序指向首次整理任务，不依赖 journal 对象键的读取顺序。

LocalProcessLab 的 lab_result 增加可选 `reprocess` 参数，事件 ID 由 session + process + terminal 构成。
lab_poll 继续只观察，lab_start 幂等键继续防重复启动。
一个程序执行一次、普通收集两次、显式再整理一次，得到三份结果收集轨迹和两次整理。
每次结果仍直接进入 decision context；去重只影响后台整理。

## 兼容与验证

OperationResult 新字段默认 None，旧 snapshot 解码允许缺少该字段。
无元数据的回执/history payload 不写这个新字段，保持既有记录的幂等内容。
带元数据的新 snapshot 由显式类型注册表编解码，不动态加载类型。
接入方保存或计算 OperationResult 归档校验值时使用 `result.payload()`；
`dataclasses.asdict` 是完整对象展开，不能替代这一兼容序列化入口。
旧原文及其校验值保持原样，不通过忽略字段或规范化原文放宽身份校验。

新增 [22 项回归](../../tests/runtime/test_lab_delivery_semantics.py)，涵盖实际 policy 请求中的错误、
真实 CPU 程序重复收集及显式重整、失败程序、相同文字的不同程序、journal 重载与写失败、
真实 host 计费路径的确定性模型 fixture、额度拒绝、原始业务字段隔离及旧快照兼容。
全量核对还发现：历史回执省略默认 None 后，证据投影使用完整 asdict 计算指纹会匹配失败，
进而把长原文误当成没有归档的结果保留。现已在记账、compact、保护选择和证据卡中统一使用
OperationResult.payload 作为归档指纹来源；带元数据和不带元数据的长结果都有有界投影回归。
相关测试构造的 operation history 同样使用此公开入口，保留完整归档找回、live 依赖保护和
重复读取 compact 的原有断言。
相关 108 项检查全部通过；全量 1607 项通过，其中 12 项跳过。
测试/来源 SHA 与修复后复现见 [validation](repros/lab-delivery-semantics-validation.json)。

没有真实模型推理或训练效果结论。事件 ID 不自动证明原始程序的完整代码、权重、部署、
评测协议或采纳；收集回执 issuing_model 仍是收集 decision 的模型。
LocalProcessLab 的受信本机执行、无 OS 隔离和不支持 durable Lab 恢复的边界保持不变。
