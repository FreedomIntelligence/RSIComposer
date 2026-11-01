# Decision tools

[默认 Decision Harness](../README.md) 的执行绑定，接收 Action Space 已校验的本地 research 操作。tools 是默认实现细节；替代 Harness 无需暴露独立 tools 接口。

## 接口与实现

[DecisionTools](__init__.py) 提供 `execute(call: OperationCall) → OperationResult | JobRef`。`BoundDecisionTools(bindings)` 将操作名映射到接收 arguments 的处理函数，复制参数并检查返回类型；它不负责描述 Action Space 或调度系统指令。

[LocalDecisionTools](local.py) 接收 FileArtifactStore：prepare_data 保存 text 为 input_data；search_literature 对已显式导入、kind=literature 的 UTF-8 文本作不区分大小写的子串检索，最多 20 条，保存 literature_results；store_strategy 保存 strategy。三者均为同步本地操作，simulated=False，不包含联网检索或 PDF 解析。

`DemoDecisionTools(sandbox, artifacts)` 内置 prepare_data：在 sandbox 中执行 echo 并轮询一次，成功后保存 demo_input，simulated 取自后端；失败抛异常，最终释放 sandbox。Demo/Local tools 均接受可选 `extra_bindings`，不能覆盖内置操作。

## 共用执行与记录

默认 Harness 的普通 step 与运行中 review 都使用本模块的 `execute_operation`：通过当前 Action Space 校验参数、检查权限，执行 tools，验证返回类型并合并 simulated 标记。返回 `OperationExecution`，保存独立复制的调用参数、结果、耗时和异常；工具对输入的修改不会改写回执。失败产生可记录的 OperationResult，由调用方决定抛出异常或交给 policy 继续分析。

`record_operation` 使用固定记录 ID 写本地 history，普通决策同时写系统反馈。review 的原始结果留在本地，由最终复查记录引用。写入失败后 Harness 保留回执，只补写、不重复调用工具。

这条共用路径不接管调度：普通决策收到 JobRef 后等待 resume；review 将其作为 submitted 回执继续研究，不宣称后台任务已经完成。

## 只读查询

[QueryTools](queries.py) 只通过显式 `QueryProvider.describe_queries()` / `query(request)` 访问提供者，不反射其内部状态。`QueryTools(provider, artifacts=None)` 暴露 `specs`、`definitions` 和 `bindings`；装配把 definitions 加入 Action Space、bindings 加入原 tools，保留 `tools.query_tools` 供绑定校验。没有查询能力的外部 Harness 不广告查询。

默认提供 `query_exploration_status/metrics/tasks/logs/candidates/model/artifacts/experience`。参数为 `filters`、`limit`（1–100，默认20）和 `cursor`（上一页返回的 offset，最高10000）；未知字段或能力会拒绝。全部定义 `read_only=True`，不提交、poll、取消作业或应用模型。任务列表是进程内回执；logs 只提供已有本地 history 摘要，不能访问远端后端日志时明确说明。Experience 返回有限摘要和证据引用，不展开任意原始配置。

OperationResult.data 包含完整 QueryResult：`name/source/observed_at/version/data/refs/next_cursor/simulated`。`observed_at` 是查询读取时间；`data.items[].job_progress.observed_at` 才是后端样本时间，未知时为 None。exploration 的 version 是返回页内容摘要，不是全系统版本。分页查询实时视图，不保证跨页快照一致；数据窗口和来源未知不能推断为完整历史。返回 JSON 最大128 KiB，refs 最多100个。

传入 artifacts 后额外提供 `query_exploration_artifact_read`，filters 必须携带已知引用的 artifact_id/kind/uri，以及存在的 version/digest。只由注册 store 验证和读取，不打开任意远程 URI。文本按1000字符分块，每页最多16块；二进制或超过1 MiB的产物只返回引用和不可用原因。FileArtifactStore 使用公开 resolve 验证（摘要扫描为流式）和有界读取；通用 get-only store 的 get 可能全量载入，输入内存边界由 store 负责，适配器只限制返回内容。artifact_read 的 version 是内容摘要。

## 扩展与回填

新增操作需在平级 [Action Space](../../action_space/README.md) 提供定义和校验，再向工具绑定注入处理函数。异步操作可返回 JobRef，由外部通过 `loop.harness.resume(job, result)` 回填终态结果；Harness 不自动轮询、取消或重做准备操作。记录失败的补写边界见 Harness README。
