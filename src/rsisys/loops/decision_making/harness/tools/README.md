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

## 通用只读工作区

[WorkspaceTools](workspace.py) 让研究者自主列目录、读日志/数据/配置/代码和搜索文本，不依赖固定 metrics 接口，也不要求把全部信息预注入上下文。外部服务实时状态仍由上面的 `QueryProvider` 获取；文件观察不能替代实时服务查询。

```python
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.capabilities import Capabilities
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.harness.tools.workspace import WorkspaceTools

workspace = WorkspaceTools(
    roots={"run": run_directory, "code": repository_directory},
    max_bytes=64 * 1024,              # 单次 read / search 每文件读取上限
    max_output_bytes=128 * 1024,      # OperationResult.data 序列化 JSON 上限
    scan_budget_bytes=1024 * 1024,    # 单次 search 实际读取总字节上限
    max_entries=1000, max_depth=8, max_results=100,
)
capabilities = Capabilities.from_provider(workspace)  # 或 workspace.capabilities
space = ModularDecisionActionSpace(capabilities.definitions)
tools = BoundDecisionTools(capabilities.bindings)
# 已有装配可合并 workspace.definitions / workspace.bindings，或合并 Capabilities。
```

操作均要求 `root` 为配置中的名称，`path` 默认为 `.`、相对所选 root，最长256字符。schema 广告 root 名称，不公开主机绝对路径。所有操作的 `OperationSpec.read_only=True`，普通决策与 review 的发现/执行权限沿用 `ActionPermissions`；没有第二套工具访问名单。roots 只是显式文件作用域。直接调用 bindings 与其他工具一样，不代替 Harness/Action Space 的权限检查。

| 操作 | 参数与分页 |
| --- | --- |
| `workspace_list` | `root, path, offset=0, limit=20`。只列一层；offset 为文件系统枚举条目位置，使用返回的 `next_offset` 继续。最多返回 max_results 条。 |
| `workspace_read` | `root, path, offset=0, max_bytes`。offset / next_offset 为字节位置，可直接 seek 到大型日志尾部附近；仅有界读取当前窗口。 |
| `workspace_search` | `root, path, query, offset=0, limit=20, max_bytes, scan_budget_bytes, max_entries, max_depth`。UTF-8、区分大小写、单行字面子串搜索，每个匹配行返回一个摘录；可限定到单个文件或目录。offset / next_offset 为本次有界扫描中的匹配行位置。 |

请求可调小各预算，不能超过构造时的上限。`max_output_bytes` 至少4096；read 为 JSON 转义预留空间，所以实际读取可能小于 max_bytes。search 只读取各文件前 max_bytes 字节，扫描总量由 scan_budget_bytes 限制；长行只返回匹配附近的有限摘录及字节位置，不全量读取巨型文件。目录扫描不排序、不全量收集条目，深度从所选目录的0开始计算。list 的 max_entries 包含跳过 offset 的条目，因此超过该扫描窗口时需缩小 path 或由部署提高预算。search 翻页重新执行同一有界扫描；预算外区域不保证能由 next_offset 抵达，应该缩小 path、调用 read 的字节 offset 或由部署提高预算。

`OperationResult.data` 包含 `source="workspace:<root>" / root / path / observed_at / read_only / available / truncated / reasons / offset / next_offset`，另有操作对应的 text、items、size_bytes、bytes_read 或 scanned_bytes/scanned_entries/scanned_files/skipped_files 等事实。`observed_at` 为本次观察开始时间，不是文件修改时间。达到页数、字节、条目、深度或输出预算时明确返回 `truncated=True` 与原因；`next_offset=None` 不意味着搜索完整。search 摘录的 `items[].truncated` 单独表示行摘录被裁剪。不存在、越界、不可读取和非普通文件返回 `available=False` 与 `unavailable_reason`；未知参数、root 和非法相对路径在校验时拒绝。二进制（控制字节或非 UTF-8）不返回文本；read 的 UTF-8 字节页边界用替换字符解码，并明确返回 `decoding_errors="replace"`。二进制判断只针对实际读取窗口，搜索跳过的二进制文件计入 skipped_files。

只访问使用者明确配置的 roots；不自动加入 home、凭据目录或其他主机文件，也不会隐藏已被使用者明确纳入 root 内的配置文件。拒绝绝对路径、`..` 和解析后越出所选 root 的软链，即使目标属于另一个已配置 root。支持 root 内软链的显式读取；目录搜索不递归跟随目录软链，避免循环，并报告扫描不完整。解析后逐级使用 descriptor-relative `O_NOFOLLOW` 打开，阻止解析到打开之间将组件替换为软链的逃逸；不具备该能力的平台在构造时拒绝。所有文件观察都发生在实时文件系统，分页不是一致快照，文件改动/重命名会影响结果。

**只读工具不等于 OS 沙箱。** 真正的权限隔离依赖部署、容器/沙箱、文件系统权限与挂载配置；使用者应只挂载和配置允许研究者读取的目录。工具不防御已配置目录中的硬链接、挂载映射或恶意并发目录搬移，不进行凭据识别/脱敏。

## 扩展与回填

新增操作需在平级 [Action Space](../../action_space/README.md) 提供定义和校验，再向工具绑定注入处理函数。异步操作可返回 JobRef，由外部通过 `loop.harness.resume(job, result)` 回填终态结果；Harness 不自动轮询、取消或重做准备操作。记录失败的补写边界见 Harness README。
