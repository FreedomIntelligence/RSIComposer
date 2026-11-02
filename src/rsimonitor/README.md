# RSISys 实验监控库

Python 标准库 HTTP 服务 + 原生 HTML/CSS/JavaScript。无需安装前端依赖、构建或发布站点；不启动训练、不修改运行文件。源码位于 `src/rsimonitor/`，拥有独立的 `pyproject.toml` 和 `rsimonitor` distribution，不依赖 `rsisys`，要求 Python 3.12+；HTTP 服务运行仅依赖 Python 标准库。

## 启动

在仓库根目录单独安装监控（不安装 RSI 核心）：

```sh
cd RSISys
python3.12 -m pip install ./src/rsimonitor
rsimonitor --runs-dir runs
```

打开 [http://localhost:8765](http://localhost:8765)。默认绑定 `127.0.0.1:8765`，Ctrl-C 停止。`--runs-dir` 相对当前工作目录解析，也可使用绝对路径：

```sh
rsimonitor --runs-dir /path/to/runs --port 8765
```

源码环境也可使用 `PYTHONPATH=src python3.12 -m rsimonitor.server --runs-dir runs`。

## 独立分发与数据协议

```sh
python3.12 -m pip wheel --no-deps ./src/rsimonitor --wheel-dir dist
# 在另一台机器或干净环境中，仅安装监控：
python3.12 -m pip install /path/to/rsimonitor-0.1.0-py3-none-any.whl
rsimonitor --runs-dir /path/to/runs
```

源码可编辑安装使用 `python3.12 -m pip install -e ./src/rsimonitor`。命令以仓库根目录为起点；进入
`src/rsimonitor/` 后也可直接 `pip install .`。目前提供本地源码和 wheel 安装，尚未发布到 PyPI。
根目录安装仅包含 `rsisys`，不会安装监控；监控安装仅包含 `rsimonitor` 和静态资源，无运行时依赖。

双方只通过 `schema_version: 1` 的 `monitor.json` 对接。RSISys 或其他实验框架负责生成公开快照，
监控独立读取、校验和投影；不导入 RSI 的 Decision、Experience、Harness 或训练后端类型。
目录名是 `run_id`，生产端以临时文件加原子替换发布；可在任意框架中写入以下最小快照：

```json
{
  "schema_version": 1,
  "project": {"project_id": "my-research", "name": "我的研究"},
  "experiment_name": "第一次实验",
  "status": "running",
  "updated_at": "2026-11-02T00:00:00Z",
  "decisions": []
}
```

保存为 `/path/to/runs/experiment-001/monitor.json` 即可显示。决策、尝试、依赖、采纳和训练指标
使用下述可选协议字段；协议外字段不会自动显示。不支持的版本明确拒绝，不猜测转换；
同版本新增可选字段保持旧快照可读。共享目录的挂载或远程快照传输由使用者负责。

## Python 库 API

其他实验代码可直接复用读取、投影及 HTTP 服务，无需仓库顶层目录：

```python
from rsimonitor import RunStore, MonitorServer, project_snapshot

store = RunStore("/path/to/runs")
projects = store.projects()     # 项目及运行摘要
runs = store.list()             # 运行摘要
snapshot = store.read("run-001") # 读取并投影已有快照
public = project_snapshot({"schema_version": 1, "status": "running"}, "run-002")

with MonitorServer(("127.0.0.1", 8765), "/path/to/runs") as server:
    server.serve_forever()      # 阻塞；嵌入线程时由调用方负责 shutdown()
```

`project_snapshot(raw, run_id)` 接受快照字典与权威运行 ID，返回公开字段投影；不读取或写入文件。
`RunStore(root)` 只读已有 `monitor.json`；`MonitorServer(address, runs_dir)` 使用同一投影并提供 UI 与 API。
需要嵌入 CLI 时可调用 `rsimonitor.server.main(argv=None)`，传入参数列表；缺省解析进程参数。
wheel 内含五个运行资源（HTML、CSS、三个 JavaScript 文件），无需前端构建。前端测试与 demo fixture
位于 `tests/monitor_frontend/`，不打入 wheel，也不通过 HTTP 发布。

**一次完整 RSI 系统运行就是一个实验，对应一个 `run_id`；一个项目可以包含多个实验。** 一次实验包含 decision-making loop 与 exploration loop 的多轮交互、训练、评测和调整，单个 decision 或训练 job 不会产生新实验。 API 仅按快照 `project.project_id` 的显式归属分组，项目 ID 使用与运行 ID 相同的有效字符和长度规则。没有有效归属的实验统一列于 `project_id: "unassigned", name: "未归属"`。不按运行名称、目录前缀、`group` 或 `system_id` 推断归属，也不强制 `project_id=run_id`。历史 Qwen / 医疗快照共享的有效项目 ID 会自然聚合；明确声明 `project_id=run_id` 的新实验保持各自独立。历史运行文件保持原样。

本轮不提供项目注册表或 `--projects-file` 参数；归属以已有 `monitor.json` 的 `project` 为权威，不读取其他文件建立历史映射，不创建空注册项目。`unassigned` 是保留 ID，即使快照显式使用该 ID，也固定输出名称“未归属”，忽略其项目名称及其他元信息。

项目名称取 `project.name`，兼容顶层 `project_name`，缺省为有效项目 ID；这些名称都不能单独建立归属。实验名称单独输出为 `experiment_name`，依次取快照的 `experiment_name`、`display_name`、`name`；仅当 `project.project_id` 等于目录 `run_id` 时兼容旧 `project.name`，最后回退目录 `run_id`。只接受非空字符串，去掉首尾空白并限制为 1000 字符，不自动附加 ID。同项目名称冲突时按目录 `run_id` 排序，首个不同于项目 ID 的有效名称胜出；其他项目元信息取首个 run，单个 run 的 `project` 保留自身上报值。服务仅读取 `monitor.json`；生产端须将项目归属和实验名称写入该快照。

首页读取真实接口，无快照时显示空态；页头不提供示例切换入口。开发 fixture 不会写入 runs，也不会出现在真实运行列表中。

## 界面语言 / Interface language

右上角「中 / EN」切换完整界面语言，默认中文。选择保存在浏览器 localStorage 的 `rsisys-language`（`zh` / `en`），重载后恢复；存储不可用时保留当前页面内的选择。按钮、状态、空态、指标名、图表提示、流程和详情标题均跟随语言；日期使用相应 locale。

切换只用已缓存的项目与运行数据重新渲染，不触发 API 请求，保留当前页面、所选实验和决策、展开记录及图谱平移、缩放和节点位置。模型 `reason` / `strategy` / `objective`、反馈和日志原文、项目名称与说明、技术 ID 和协议字段保持原样。已知指标显示友好名称，原始 key 放在 title；自定义指标保留可见 key，便于区分。

无外部依赖或翻译 API。页面以 `defer` 依次加载 `/i18n.js`、`/decisions.js` 和 `/app.js`。全局 `RSII18n` 提供 `t(key, params)`、`getLanguage()`、`setLanguage("zh" | "en")`、`applyStatic(document)` 和 `subscribe(listener)`；订阅返回取消函数。`messages` 为只读双语字典，`storageKey` 为存储键。动态文案显式调用 `t()`，参数使用 `{p0}` 等模板；静态 UI 仅通过 `data-i18n` / `data-i18n-aria-label` / `data-i18n-title` / `data-i18n-placeholder` 标记翻译，不遍历替换模型文本。计数与标签分开标记，以保留动态内容。

Use **中 / EN** to switch the interface. The preference persists locally. Switching reuses cached data and preserves the current page, selected experiment and decision, graph scroll and expanded records. Reported model text and logs stay verbatim.

## 统一监控入口

正式入口统一为 **`http://127.0.0.1:18777/`**，通过 SSH 访问 modelmate 的
`127.0.0.1:8877`。旧地址 `http://localhost:8767/` 的兼容隧道已停用，不能作为入口。

modelmate 部署使用安装 wheel 的包目录 `RSISys/monitor-ui/package`，数据目录是 `RSISys/runs/`。
前端与服务端源码均以本仓库 `src/rsimonitor/` 为准；发布时构建并安装 wheel 到该目录，
使用其中的 `rsimonitor.server:main`，参数为 `--runs-dir /path/to/RSISys/runs --port 8877`。
例如 `python3.12 -m pip install --upgrade --target /path/to/RSISys/monitor-ui/package /path/to/rsimonitor-0.1.0-py3-none-any.whl`
（wheel 文件名使用实际构建产物），然后从已安装目录启动：

```sh
PYTHONPATH=/path/to/RSISys/monitor-ui/package python3.12 -m rsimonitor.server --runs-dir /path/to/RSISys/runs --port 8877
```

服务或静态资源变化时只重启该监控进程。`monitor-ui/version.json` 记录来源版本与文件摘要，
`monitor-ui/process.json` 记录实际 PID；重启前核验身份并保留旧包备份。

lion 的 17 条历史实验监控记录已汇总到 modelmate，每条导入记录附有
`monitor-import.json`，保留原始主机、目录和文件摘要。这是历史监控元数据的导入，
不是模型权重、训练代码或运行进程的迁移，也不是自动双向同步。后续新实验需把监控
输出发布到统一数据目录，不能默认其他主机的新增 run 会自动出现。

研究运行的 `runs/<run>/code` 是冻结代码，不作为前端发布源。更新监控不覆盖冻结代码、
重启模型或重置预算。`8769` / `8770` 等临时预览端口不作为会话间共享入口。

## 数据与交互

读取 `runs/<run-id>/monitor.json`，协议为 `schema_version: 1`。支持 `workflow`、`current`、`decisions`、`jobs`、`candidates`、`round_results` 和 `events`；旧快照与缺省字段保持兼容。字段缺失显示未报告，不补零、不补造执行结果。

- `/` 为项目管理页，按大项目分块，每个项目下以卡片列出实验名称、状态、更新时间和已报告决策数。项目名称与实验名称独立，缺少归属的实验归入“未归属”。首页只读取清单，不自动进入实验。点击实验进入 `?project=<project-id>&run=<run-id>`；仅指定项目时仍停留管理页，指定 run 时由其真实归属确定上级项目。返回按钮和浏览器前进 / 后退保持两页导航。
- 实验详情分别显示实验名和所属项目名；项目导航与 run 隔离保持不变。主图仅显示「研究尝试 / Attempts」：优先按该 run 的 `research_attempts` 模型整理视图合并，一次尝试一个节点，标题取整理模型原文；未提供新视图的旧实验才沿用 `research_attempt_id` / `research_attempt_title` 显式分组。首次进入不自动选择主节点或展开详情；点击主节点后显示完整内部决策子图，逐条选择查看现有理由、提交内容、反馈和指标。尝试可以包含调整提示词、评测、比较、采用等多条决策；所有子决策（包括 wait 和 finish）分别保留。尝试没有独立的完成回执，界面不根据子步骤成功推断整个尝试成功，也不从标题推断结果。
- 「当前任务 / Active tasks」仅在有明确活动状态的任务时显示紧凑卡片，包括已关联和未关联任务，以及所有并行任务；不展开历史指标。运行已完成、失败、取消或停止时不把残留 running 等任务当成当前执行，冲突以简短提示保留，原始状态仍可在关联节点或未关联任务历史中查看。没有活动任务时不显示空任务区，运行状态及读取错误仍可见。
- 旧实验没有有效尝试身份时显示「未记录研究尝试」，不按候选、动作、时间或相邻顺序猜测分组。没有「全部记录」并行主图或切换入口。辅助记录及所有未归属决策只在默认折叠的「实验日志 / Experiment logs」中检查，即使它们涉及依赖、采纳或来源关系，也不会成为主图节点。日志保留当前公开快照的每条决策（API 上限 300），逐条展示理由、执行反馈、进度、结果，并提供折叠的原始决策证据；未归属记录明确标注「未记录研究尝试」。
- `decisions.js` 沿用已记录的 `depends_on_decision_ids`、`adopted_decision_ids`、`source_decision_ids` 和兼容字段 `parent_decision_id`。尝试主图只收缩两个端点均有明确尝试归属的真实跨组关系，不经未归属记录补造中转或传递边，保留并行分支和多结果采纳汇合；同一对节点只画最强关系（采纳 > 依赖 > 引用），原始关系类型与端点保留在前端模型的 `edge.types` / `edge.decisions`。依赖用灰色实线，已确认采纳用强调色实线，来源引用用虚线。布局起点只是画布锚点，不表示依赖。尝试主图不补造顺序边。完整决策投影仍保留原始关系，供内部子图与日志核对。
- 有效原始链 `A1 → B1 → A2` 可能在收缩后成为 `A → B → A`；这种情况仍显示尝试节点及真实跨尝试依赖与采纳边，并在折叠诊断中提示收缩成环；不展开为决策主图，也不按收缩后的环拒绝或删除原始关系。尝试详情和实验日志保留全部成员及原始关系证据。原始数据中的自引用、循环结构关系、未知或歧义引用继续按既有校验规则过滤，并通过小型折叠提示说明；其余有效外部父边和后继边保留。
- 跨实验的 `parent_run_id` 和 `source_run_ids` 留在「来源记录」中，以链接打开来源实验；不进入实验内部的决策图谱。
- 每 5 秒静默刷新：管理页只更新清单，实验页更新清单与当前实验。隐藏页面暂停轮询，切换实验取消旧请求并拒绝迟到响应。缓存按实验隔离；不显示连接状态、手动刷新或导出工具栏。
- 刷新保持详情容器挂在原位置，不把页面高度短暂折叠，并保留页面滚动位置。刷新无数据变化时复用运行详情 DOM；有新反馈时，按稳定尝试与决策身份保留所选尝试、内部子决策、展开记录、两个画布视角和节点位置；尝试主图与实验日志各自保留显式选择。所选节点消失时清空选择，不自动换到其他记录；未选择时刷新或中英切换也不补选。中英切换同样保留状态。拖动或布局动画期间延后刷新画布；结束交互后更新反馈。切换实验、返回管理页或隐藏标签页会取消动画并释放图谱监听，重新进入时恢复布局。保留实验日志、未关联任务历史、候选结果和来源记录折叠区的展开状态。
- 执行流程按已报告时间从早到晚排列；未知时间保留快照顺序列于末尾，不补造因果。尝试内部子图与实验日志保留当前快照中的全部决策。每次展示一个步骤的**决策理由、提交内容、实际反馈、进度及结果**。只有“执行已完成”等通用状态时不显示反馈块；缺失的理由不补造。`evaluate` 显示为“验证改动效果”，明确评测与采用改动是两个步骤。训练步骤带损失 / 学习率曲线与历史点，评测带基线 / 候选对比。重复 finish 也分别保留；“结束本轮研究”不代表整个运行已停止。
- 已知动作使用中英友好名称，例如 `configure_prompt` → 调整提示词、`define_recipe` → 定义训练方案；自定义动作直接显示原名，只有没有动作字段时才显示“未命名决策”。模型理由、策略及反馈原文不改写。与动作名称完全重复的 strategy 不再重复展示；常规详情不展示技术 ID、版本哈希或协议字段列表；原始端点仍可通过监控 API 和运行文件核对，不在看板展开 JSON。
- 只按明确唯一的 `job_id`、`candidate_id` 或指令引用关联反馈；找不到指定 job 或存在歧义时不回退猜测。复查时指标与关联任务当前结果分开展示。原始事件留在快照和日志中，不进入阅读界面。
- 已移除重复的“指标与记录”区域。选择执行步骤后呈现决策与结果，已移除“记录详情”“运行详情”“运行事件”和任务原始字段列表。未关联且不在当前任务概览的任务统一放入默认折叠的「未关联任务历史 / Unlinked task history」，保留状态、进度、指标及已报告错误；`unknown`、`cancel_requested` 等不确定或非终态记录也保留真实标签，不声称正在运行或已结束。历史任务不猜测归属到图节点；未关联候选继续折叠。旧实验没有尝试字段时仍显示活动任务并提供实验日志。技术字段仍保留在公开 API 和运行文件中；底层运行数据不删除。
- 百分比只使用有效的 step / total_steps；训练 100% 不代表整个运行完成。曲线只绘制真实历史点，无历史不补造。候选差值只表示候选减基线，不自动判断优劣。

决策的可选 `changes` 为 `{field, after, before?}` 数组，最多 20 项，字段名最多 100 字符、值最多 4000 字符；仅输出生产端明确选择的公开参数，包括提交给 Actor 的提示词。`before` 必须来自真实记录，不从状态推断。`feedback_summary` 和 `error` 保存实际执行反馈；不读取模型隐藏推理、系统提示词或训练样本。旧记录没有理由时保持缺失，后续模型决策可记录简短公开 `reason`。

### 研究尝试字段

每条决策可包含两个公开字符串字段：

```json
{
  "decision_id": "recorded-decision-id",
  "research_attempt_id": "recorded-attempt-id",
  "research_attempt_title": "简短提示词能提高基准准确率",
  "operation": "configure_prompt",
  "depends_on_decision_ids": []
}
```

身份由生产端记录，前端只读取这些字段，不生成新尝试实体或写回历史。相同 ID 的全部决策属于一次尝试；不同 ID 即使同名也分别显示。缺少两字段表示旧的未分组记录。仅标题没有有效非空字符串 ID 时仍未分组，并提示无效身份；有效 ID 缺标题时显示「未命名研究尝试」。同 ID 标题冲突时采用按记录时间排序后的首个非空标题，保留各条原始值并提示核对，不据此拆分身份。技术 ID 保留在载荷、测试及实验日志的折叠原始证据中，不作为主图或常规详情字段显示。生产端应导出全部成员的明确归属与关系；前端无法恢复 API 截断之外的历史。

### 可配置指标看板协议

`monitor.json` 的顶层可增加以下三个可选字段；外层 `schema_version` 仍为 `1`。
没有这些字段的旧快照继续使用已有任务和决策指标，生产端无需迁移历史文件。

```json
{
  "metric_definitions": {
    "accuracy": {"label": "准确率", "unit": "percent", "direction": "higher"},
    "loss": {"label": "训练损失", "unit": "number", "direction": "lower"}
  },
  "metric_points": [
    {"metric": "loss", "value": 0.3, "step": 2, "elapsed_seconds": 1.5, "task_id": "train-1"},
    {"metric": "accuracy", "value": 0.75, "decision_id": "evaluate-1", "attempt_id": "attempt-1"}
  ],
  "dashboard": {
    "schema_version": 1,
    "panels": [
      {"id": "training", "title": "训练损失", "type": "line", "metrics": ["loss"], "x": "step", "group_by": "task"},
      {"id": "quality", "title": "准确率", "type": "scalar", "metrics": ["accuracy"], "x": "evaluation", "group_by": "attempt"}
    ]
  }
}
```

- `metric_definitions`：最多 **100** 项，以指标名为键；每项仅包含 `label`、`unit`、`direction`。
  `unit` 为 `number`、`percent`、`seconds` 或 `tokens_per_second`；`direction` 为
  `higher`、`lower` 或 `neutral`，仅表达显式的比较方向。
- `metric_points`：最多保留最近 **10000** 个上报点。必需字段为 `metric` 和有限数值 `value`；
  可选 `step` 为非负整数，`elapsed_seconds` 为非负有限数值，`timestamp`、`task_id`、
  `decision_id`、`attempt_id` 为字符串。布尔值不算数值；负指标值允许，负坐标不允许。
  时间戳只作为已报告字符串传递，不推断时间或身份；缺失坐标保持缺失，重复上报点保持原顺序。
- `dashboard`：独立的 `schema_version: 1`，`panels` 最多 **24** 项；每项必需字段为
  `id`、`title`、`type`、`metrics`、`x`、`group_by`。`type` 为 `line`、`scalar` 或 `table`；
  `metrics` 最多 **12** 个指标名；`x` 为 `step`、`elapsed_seconds` 或 `evaluation`；
  `group_by` 为 `task`、`attempt` 或 `none`。空面板列表、空指标列表均允许。
  面板 ID 必须唯一：生产端拒绝重复 ID，主机投影仅保留同 ID 的首个有效面板。
  两端均在校验及数量限制后对面板内指标名去重，保留首次出现的顺序，不修改调用方输入。
- 指标名限 100 字符，标签、面板 ID/标题及关联 ID 限 200 字符，时间戳限 100 字符；
  字符串必须非空且不能全为空白。超长身份不截断，避免合并不同实体。
- **只有显式 `unit: "percent"` 表示比例**：值必须在 **0–1**，如 `0.75` 显示为 `75%`。
  `75` 不是合法比例；未定义的 `score: 75` 仍是原始数值 `75`，不自动缩放或推断单位。
  指标无需先定义即可上报，面板也可引用尚未上报的指标。

主机通过 `GET /api/runs/<id>` 输出这三个可选字段；列表 API 继续返回简短运行摘要。
投影只输出上述白名单字段，移除配置、凭据、任意嵌套对象等附件；无效定义或面板被丢弃，
一个点的必需或已提供可选字段无效时丢弃整个点，不把 NaN、Infinity、null、字符串或布尔值
转换成零或缺失坐标。未知看板版本或无效看板容器被省略；无效定义/点容器投影为空字典/数组。
超限输入取前 100 个定义、前 24 个面板和各面板前 12 个指标，点取最近 10000 个后再校验；
整个快照仍受 4 MiB 文件上限约束。读取和投影不修改原文件，也不补造任务、决策或历史点。

独立库的公共投影函数位于 `rsimonitor.server`：`metric_definitions_only(value)`、
`metric_point_only(row, metric_definitions=None)`、`metric_points_only(value, metric_definitions=None)`、
`dashboard_only(value)`。单点无效返回 `None`，无效看板返回 `None`；百分比校验使用已投影的定义。
这些函数和 `project_snapshot` 仅依赖标准库，不导入 `rsisys`。

Qwen 示例的生产端提供两个暂存 API，均不立即写文件：

```python
from pathlib import Path
from examples.qwen_toy.monitor import Monitor
from rsisys.loops.contracts.model_service import ModelRef

run_dir = Path("runs/dashboard-example")
run_dir.mkdir(parents=True, exist_ok=True)
monitor = Monitor(run_dir, ModelRef("toy", "model", "v1"), "研究目标")
monitor.configure_dashboard(
    [{"id": "quality", "title": "准确率", "type": "line", "metrics": ["accuracy"],
      "x": "evaluation", "group_by": "attempt"}],
    metric_definitions={
        "accuracy": {"label": "准确率", "unit": "percent", "direction": "higher"},
    },
)
monitor.log_metric("accuracy", 0.75, step=0, elapsed_seconds=1.5,
                   timestamp="2026-11-02T00:00:00Z", task_id="evaluation-1",
                   decision_id="evaluate-1", attempt_id="attempt-1")
monitor.write()  # 下一次显式 write 或已有事件触发的 write 原子发布暂存内容
```

`Monitor.configure_dashboard(panels, *, metric_definitions=None)` 深拷贝已校验的轻量配置。
省略定义时保留已有定义，显式传入字典时替换；`{}` 清空定义。配置不得把已记录值重新定义成
不合法的百分比。生产端对超限配置、未知字段和无效输入抛出 `ValueError`，失败不改暂存状态；
`log_metric(name, value, *, step=None, elapsed_seconds=None, timestamp=None, task_id=None,
decision_id=None, attempt_id=None)` 将 `None` 可选参数省略，保留最近 10000 点。
生产端不导入可选的 `rsimonitor` 包，两端只共享 JSON 协议；配置编辑、本地存储和面板渲染由界面负责。

### 训练历史协议

`jobs[].metric_history` 是可选数组，每项为 `{step, elapsed_seconds, metrics: {数值指标}}`，快照最多保留最近 **1000 点**。当前 toy worker 的实际训练日志配置为 **logging_steps=2**，因此日志并非每一步都产生一个点。完整 worker `<job_id>.metrics.jsonl` 可供离线分析；UI 只读取公开快照，不直接读取 JSONL。

```json
{
  "job_id": "training-example",
  "kind": "train",
  "metrics": {"loss": 0.12, "learning_rate": 0.00002},
  "metric_history": [
    {"step": 2, "elapsed_seconds": 1.5, "metrics": {"loss": 0.3, "learning_rate": 0.00001}},
    {"step": 4, "elapsed_seconds": 3.0, "metrics": {"loss": 0.12, "learning_rate": 0.00002}}
  ]
}
```

上例仅为协议示意。前端只接受有限数值；按真实步数绘图，步数不可用时尝试真实耗时；同一横坐标保留最后上报点，不按数组序号伪造步数，不补中间点。旧真实 run 的 `metrics` 没有完整 history 时，只显示当前数值和诚实空态，不能从末次 loss / lr 还原训练曲线。曲线可逐点查看；公开快照仍保留原始历史数组。

真实 worker 的 flat metrics 可以直接使用：

```json
{
  "candidate_id": "candidate-007",
  "status": "accepted",
  "base_revision": "actor-r006",
  "revision": "actor-r007",
  "metrics": {
    "base_nll": 1.942,
    "candidate_nll": 1.781,
    "delta_nll": -0.161,
    "eval_samples": 128
  }
}
```

`base_nll` / `candidate_nll` 显示为 NLL 的基线／新版本数值；`delta_nll`、`eval_samples` 保留在「已报告指标」中。对比表的「变化」仅为新版本减基线，不判断改善方向，不用红绿颜色表达数值优劣。很小的非零指标使用科学记数，避免把细微 NLL 显示成零。也支持 `{ "nll": { "baseline": 1.942, "candidate": 1.781 } }` 和按 `baseline` / `candidate` 分组的指标。缺少一侧时显示 `—`，不会补零或伪造评测。

## HTTP 接口

`GET /api/projects`：返回按显式 `project_id` 聚合的 `projects` 数组，每项包含一个或多个运行摘要，不包含完整快照。项目按项目 ID、内部运行按目录 `run_id` 升序返回；同名但不同 ID 的项目不会合并。

```json
{
  "schema_version": 1,
  "projects": [{
    "project_id": "medical-rsi",
    "name": "医疗自主研究",
    "description": "…",
    "system_id": "medical-system",
    "runs": [
      {"run_id": "run-001", "experiment_name": "首轮实验", "objective": "…", "status": "running", "updated_at": "…", "project": {"project_id": "medical-rsi", "name": "医疗自主研究", "description": "…", "system_id": "medical-system"}},
      {"run_id": "run-002", "experiment_name": "第二轮实验", "status": "completed", "project": {"project_id": "medical-rsi", "name": "医疗自主研究"}}
    ]
  }]
}
```

`GET /api/runs`：每行包含独立的 `experiment_name` 和 `project` 归属元信息。


```json
{
  "schema_version": 1,
  "runs": [
    {
      "schema_version": 1,
      "run_id": "run-001",
      "experiment_name": "首轮实验",
      "objective": "…",
      "status": "running",
      "updated_at": "2026-11-01T00:00:00Z",
      "project": {"project_id": "medical-rsi", "name": "医疗自主研究"},
      "counts": {"jobs": 1, "candidates": 2, "events": 5}
    }
  ]
}
```

只列出存在可读取监控文件的运行，忽略无 `monitor.json` 的目录（包括 `runs/monitor-preview/`）。缺失 runs 目录返回空列表且不创建目录；损坏 JSON / 不支持版本 / 超限快照保留为带 `error` 的 unavailable 清单项，`experiment_name` 暂用目录 ID，归入“未归属”。快照恢复有效后自动重新读取显式项目归属，不缓存猜测，不写回文件。

运行列表及快照可包含可选 `parent_run_id` 字符串与 `source_run_ids` 字符串数组；其中 `parent_run_id` 定义续接来源，`source_run_ids` 定义资料引用；两者均不改变项目身份。

`GET /api/runs/<id>`：返回协议字段的只读投影，始终包含 `experiment_name` 和 `project: {project_id, name}`；有效归属可额外包含公开的 `description`、`system_id`，缺失时不补造。快照可包含与决策分离的 `research_attempts` 模型整理视图（见下文）。历史决策可包含 `research_attempt_id`、`research_attempt_title` 字符串，以及 `parent_decision_id`、`depends_on_decision_ids`、`adopted_decision_ids`、`source_decision_ids` 等已记录关系。目录名是路由和返回 `run_id` 的权威来源，即使快照内 `run_id` 不一致也不改变；项目归属则来自快照的有效 `project.project_id`。不跟随快照内的路径或文件引用。无快照返回 404，损坏或写入中的 JSON 返回 503（带 `Retry-After: 2`），不支持的协议版本返回 422，文件超过 4 MiB 返回 413。

建议生产端在同一目录写临时文件，再原子替换 `monitor.json`。服务不会监听或读数据库、训练样本、配置、日志、模型文件。

## 读取边界

- 运行 ID 仅允许 1–128 个 ASCII 字母、数字、点、下划线、连字符，首字符必须为字母或数字。
- 通过目录文件描述符和 `O_NOFOLLOW` 打开 run 目录及固定的 `monitor.json`；拒绝路径穿越、符号链接和非普通文件。仅适用于提供这些接口的 macOS / Linux。
- 静态资源采用明确路径白名单，无目录列表；不会暴露 README、源码或截图。
- API 仅输出协议字段，移除任意配置／凭据／训练数据附件。metrics 只保留有界的数值字典，过滤敏感键；`eval_samples` 等数值计数允许通过。事件 details 仅允许公开的指令/任务/候选/版本身份、状态、复查动作、显式决策说明及进度等元数据。列表最多返回最近 300 条，文本字段最多 1000 字符；原始完整历史仍保留在对应 Harness 中。
- `objective`、`summary` 等文本作为纯文本渲染；生产端仍需确保这些公开摘要不含秘密或训练样本。服务不尝试从自然语言中识别所有秘密。
- 所有响应 `no-store`，不启用 CORS，拒绝跨站请求和非本机 Host；页面有严格 CSP。仅支持 GET/HEAD，写请求返回 405。

## 验证

```sh
PYTHONPATH=src python3.12 -m unittest discover -s tests -p 'test_monitor*.py' -v
node --check src/rsimonitor/static/app.js
node --test tests/monitor_frontend/*.test.js
```

服务测试覆盖显式项目归属与多实验分组、同名不同 ID 隔离、实验名称优先级、跨项目来源、历史文件不变、快照投影、读取边界、错误恢复与只读接口。前端验证重点为独立项目导航与深链接、来源链接、双语与原文保留、按需读取与并发取消、刷新保留展开状态和稳定步骤身份、训练历史及反馈关联。`attempts.test.js` 额外覆盖显式尝试身份、假设标题下调整提示词 / 评测 / 比较 / 采用的逐步选择、并行尝试与多结果汇合、关系优先级和原始证据、收缩成环时保持尝试主图、原始坏数据校验、历史空态与实验日志、双画布交互及刷新保留。`overview.test.js` 覆盖默认不展开详情、并行活动任务、终态运行与陈旧任务冲突、未关联历史及不确定状态、显式历史结果、刷新与双语保留选择和折叠状态、节点移除清空选择、旧实验日志入口及严格任务关联。Node 只用于检查与测试，服务运行不依赖 Node。

指标协议测试另覆盖生产端写入至主机投影的往返、配置深拷贝与延迟发布、四项集合上限、
附件白名单、无效输入及非有限数值整点拒绝、显式百分比、旧快照兼容、原文件不变，
以及生产端缺少 `rsimonitor`、主机缺少 `rsisys` 时各自可独立运行。

界面采用独立的实验管理页与实验详情页，使用暖白纸面、深棕文字与低饱和陶土橙选择态；标题使用 Georgia／本机宋体，正文使用本机系统无衬线字体，无外部字体请求或品牌 logo。

浏览器 QA 截图放在临时目录 `/tmp/rsisys-*`，不写入 runs、不作为源码入库。

## 生产端与历史记录

[Qwen toycase 的 monitor.py](../../examples/qwen_toy/monitor.py) 从模型的结构化 Decision / ReviewDecision 和实际 Runner 结果生成上述摘要；[worker](../../examples/qwen_toy/worker.py) 在 Trainer callback 中报告真实 global_step、max_steps、耗时及已记录的 loss / learning_rate。UI 从 monitor.json 读取，不直接访问训练服务。

已完成的旧 toycase 可以显式补导出已有决策记录：

```sh
PYTHONPATH=src python examples/qwen_toy/monitor.py --run-dir runs/YOUR_COMPLETED_RUN
```

导出只从该 run 的 decision_memory.json、result.json 和原监控事件生成公开字段，并原子更新 monitor.json；不改训练结果、时间戳或模型文件，不启动训练。source=history 表示真实历史记录，缺失的时间与中间步数保持未知。这个导出器属于 toycase；其他 Harness 自行生产同样的轻量监控字段，不需要适配统一训练数据格式。

监控生产端保留系统生成的 `decision_id`。采纳仅从成功 `OperationResult` / `Feedback` 的 `adopted_candidate_ids` 与已成功产出的候选来源对应；评测不会变成候选生产者，失败或未知来源不画采纳边。离线导出沿用历史中的明确身份和回执。参考端到端示例 `PYTHONPATH=src:. python examples/decision_graph.py --run-dir runs/decision-graph-demo`；它使用模拟任务验证并行、组合采纳和后续消费，不代表真实模型训练。

项目管理页按项目内最近一次实验的 `updated_at` 倒序排列；项目内实验同样按更新时间倒序。缺失或无效时间置后，相同时间保留清单原顺序；自动刷新本身不改变更新时间。

评测进度与最终结果使用同一后端 `job_id`；初始基线任务以 `role: "baseline"` 标识并单独折叠显示。历史医疗评测的 `evaluation:<candidate_id>` 别名仅在候选身份吻合且指令对应唯一任务时关联到决策；任意未知任务编号不回退匹配。无法确认的记录保留在折叠的关联诊断中。


### 模型整理的研究尝试

`research_attempts` 是与原始 `decisions` 分离的增量解释视图：`schema_version: 1`、`authority: "model_interpretation"`，包含 `items`、`revisions`、`ungrouped_decision_ids`。每个条目记录 `attempt_id`、`title`、`summary`、`member_decision_ids`、`source_refs`、`revision`、`provenance`（job、模型身份、输入摘要）和 `simulated`。生产端通过 `Monitor.sync_attempt_interpretations(view)` 复制后端已校验的快照；此方法不更改原始决策字段，随下一次 `write()` 发布。

前端优先展示这份视图，点击节点展开摘要、成员决策及反馈；整理来源和版本历史保留在数据协议中，不在看板展示。空 `items` 表示尚未整理出尝试，保留实验日志，不回退猜测或旧分组。没有新视图的历史实验继续支持旧的显式归组。分组仅表达模型对研究工作的整理，不宣告执行成功、方案采纳或控制执行。主图连线仍来自真实 Decision 依赖与采纳回执；时间顺序和整理模型的摘要不会产生新的关系。

快照只保留近期决策时，Attempt 和其成员 ID 不因详情缺失而消失，节点明确提示部分详情不在当前快照，证据引用保留在 API 中供核对。异步整理尚未关联的 Decision 留在日志中，不作为错误。模型摘要作为纯文本显示，不执行模型输出。

节点选择及详情重建保留页面滚动位置；鼠标选择不强制焦点，键盘选择保留可访问焦点并使用主题色圆环，无浏览器蓝色矩形框。图内滚轮沿用浏览器页面滚动，不拦截或缩放图谱；拖动平移视图，缩放使用图旁按钮。

研究尝试标题栏不显示分组数量或解释性声明；整理来源保留在监控 API 与运行文件中。自动刷新在请求超过8秒时保留现有数据并报告监控连接超时，随后自动重试；此状态不代表实验执行失败。

看板只展示研究语义：隐藏无可读名称的哈希引用、内部资料编号、原始决策 JSON 和整理审计附件；短训练参数使用自适应多列排列，提示词等长正文单独占行。模型理由和实际反馈不通过哈希规则改写，监控 API 与底层运行文件保持完整。

实验日志按时间顺序使用整行条目，列出序号、时间、动作和状态；选中条目在原位置展开理由、改动和反馈，可再次点击收起，刷新保留选择。Benchmark 概览按已记录评测顺序绘制各 benchmark 准确率趋势，纵轴固定为 0–100%，详情中的准确率/F1 使用百分比，比较差值使用百分点（pp）。仅明确标记且包含相应指标的唯一基线任务可产生基线虚线；没有基线不推断，缺失指标不连线，最新评测不等于已采纳结果。训练损失等非比例指标保持原单位。

指标看板的「配置看板」支持选择指标、折线图／数值卡／对比表、横轴和任务／研究尝试分组，以及增删、排序面板。横轴按真实训练步数、耗时或记录顺序显示；不同单位分开绘制。配置按实验保存在当前浏览器，优先于生产端默认布局；「恢复默认」恢复生产端布局或自动生成的布局，不修改实验记录。编辑期间继续采集刷新数据，关闭或保存后再更新视图，避免打断输入。
