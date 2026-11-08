# Lion 夜间实测后的链路修复

本轮按用户确认的五项方案实现；正式评测随后按用户要求固定在医疗 Lab。候选程序、训练和推理实现仍可自由修改。

## 实时状态窗口

`LabExecution` 的必需方法仍只有 `describe()` 和 `execute(call)`。
新增可选 `runtime_context() -> str`，提供者自行定义文本内容和内部状态结构。
默认 harness 在 context processors 完成后刷新到 `research_agenda.runtime_context`。
历史压缩不折叠这个当前状态窗口；无能力或空文本时不增加提示字段。

默认本地 Lab 提供全部活跃任务和最近八个终态任务，另有 `lab_status` 工具。
busy 回执的原生 data 包含占用任务，`LabToolError.data` 经适配器保留。
同 key、同参数重读返回原任务 ID、当前状态和 `replayed=true`；参数冲突仍拒绝，不自动重放动作。

## 持久日志

`LocalProcessBackend` 在 `root/logs/` 下创建独立日志，关闭 workspace 后仍保留。
每流默认持久前缀额度为 64 MiB，可由 host 的 `max_log_bytes` 配置；另保留不超过
`max_output_bytes` 的尾部窗口，医疗部署为 128000 字节。
回执使用有界首尾预览，`lab_read_log` 提供字节游标或末尾读取。
超额中间段明确返回 gap 和下一个可读 offset；引用包含范围和摘要。
管道继续排空，Python 默认 `PYTHONUNBUFFERED=1`。旧 run 已丢失的日志无法补回。

## LangMem 准入

完整原生 messages/tools 组装后才计量。超限先尝试原有证据打包器的更小投影，保留原始
归档和精确引用；检索按完整 memory 记录做有界选择。工具、patch 对话、模型身份和
输出上限不变；最终完整请求仍须通过相同计量。

无法容纳只记录一次本地 `input_too_large`，不派发推理。私有 `NativeAdmissionAbort`
穿过 Trustcall 通用 `except Exception` 修复路径，只在 LangMem 工作边界捕获为终态失败。
普通调用者仍收到 `FeedbackCapacityError`。不增加调用额度、graph guard 或格式重试，
本地拒绝仍按原有保守方式记账，失败不提交 staged memory。
非法 patch、截断和自由文字语义错误仍可能发生，未宣称全部解决。

## 可替换的入队策略

可选 `FeedbackAdmissionPolicy.admit(event, pending) -> FeedbackAdmission` 位于队列限额之前。
只能选择本条是否整理、合并哪些 queued 事件；不能改变 running job 或预算。
合并关系持久记录，原始回执逐条保留，context observer 和轨迹归档照常工作。
不配置策略时保持通用框架原行为。

医疗部署选择 `LocalLabMemoryAdmission`：成功写文件、提交、取消确认和 busy 等回执进入
context/归档，terminal evidence 和可修正错误进入整理器；相同请求的等价排队错误可合并。
策略不替换 LangMem，也不关闭必须激活的 context/memory。

## 固定医疗评测

`FixedMedicalLab` 增加 `medical_evaluate(key, candidate_id, argv)`，仍使用同一 owned
process、并行槽、超时、任务总限额和清理路径。正式代码位于 host 的
`experiments/medical/fixed_evaluation.py`，不复制进可编辑 workspace。
启动前核验源码 SHA，固定 development 文件逐项核验内容 SHA。

候选 stdin 为 `{protocol, items:[{benchmark_id,id,messages}]}`：question/options 固定构造，
请求不含 gold。候选 stdout 为 `{predictions:[{benchmark_id,id,text}]}`，诊断写 stderr。
示例 `candidate.py` 使用现有 HTTP 服务；候选也可使用自己的权重和推理实现。
工具不接受 gold、parser、scorer、数据集路径或评分协议参数。

Lab 固定 exact-option-letter-v1 的答案解析与 gold 匹配，按样本 ID 计算 accuracy/n。
`lab_result` 再次核验数据、协议、候选声明、源码、逐项正确性和聚合值，保存独立 artifact
及 `data.fixed_evaluation`。自写评分脚本仍是原生实验输出，不能代替正式分数。
此入口不自动采纳候选，不证明完整权重身份，开发集分数不是医学泛化结论。
本地 Lab 是 trusted execution，不宣称同 UID 任意代码下的 OS 防篡改隔离。

## 验收与范围

`tests/runtime/test_lion_followup_repair.py` 覆盖真实 CPU 子进程、压缩后的实时状态、busy/replay、
日志尾部和 quota gap、跳过整理后的完整 context/轨迹、queued/running 合并边界、
固定 gold/完整输出解析、正式候选执行，以及真实 LangMem/Trustcall 的容量中止与证据投影。
合成 fixture 不代表模型训练或医疗效果验证。

测试日志和源码清单位于 `.local/lion-followup-repair-20261105/`。
lion 检查副本为 `/home/jovyan/projects/rsisys-deployments/lion-followup-repair-check-20261105/`，
只执行 CPU 测试；旧冻结源码、旧 campaign、预算和 GPU 任务均未修改。

验证结果：本地完整回归 1690 项、44 项跳过；lion 完整回归 1690 项、12 项跳过，均无失败。
全量后最后补充的评测验证失败清理分支，由本地 41 项定向检查（2 项跳过）和 lion 56 项
定向检查（无跳过）覆盖，均通过。全量版本与最终版本的差异及日志保留在 `validation.json`。
最终 168 个相关源码/提示/测试文件的 SHA 已逐项核验本地与 lion 一致。
