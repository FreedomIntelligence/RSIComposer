# RSISys 本地只读监控

Python 标准库 HTTP 服务 + 原生 HTML/CSS/JavaScript。无需安装前端依赖、构建或发布站点；不启动训练、不修改运行文件。Python 3.9+ 可独立运行，推荐使用项目现有 Python 环境。

## 启动

在 RSISys 项目目录运行：

```sh
cd RSISys
python3 web/server.py --runs-dir runs
```

打开 [http://localhost:8765](http://localhost:8765)。默认绑定 `127.0.0.1:8765`，Ctrl-C 停止。`--runs-dir` 相对当前工作目录解析，也可使用绝对路径：

```sh
python3 web/server.py --runs-dir /path/to/runs --port 8765
```

**一次完整 RSI 系统运行就是一个实验，对应一个 `run_id`；一个项目可以包含多个实验。** 一次实验包含 decision-making loop 与 exploration loop 的多轮交互、训练、评测和调整，单个 decision 或训练 job 不会产生新实验。 API 仅按快照 `project.project_id` 的显式归属分组，项目 ID 使用与运行 ID 相同的有效字符和长度规则。没有有效归属的实验统一列于 `project_id: "unassigned", name: "未归属"`。不按运行名称、目录前缀、`group` 或 `system_id` 推断归属，也不强制 `project_id=run_id`。历史 Qwen / 医疗快照共享的有效项目 ID 会自然聚合；明确声明 `project_id=run_id` 的新实验保持各自独立。历史运行文件保持原样。

本轮不提供项目注册表或 `--projects-file` 参数；归属以已有 `monitor.json` 的 `project` 为权威，不读取其他文件建立历史映射，不创建空注册项目。`unassigned` 是保留 ID，即使快照显式使用该 ID，也固定输出名称“未归属”，忽略其项目名称及其他元信息。

项目名称取 `project.name`，兼容顶层 `project_name`，缺省为有效项目 ID；这些名称都不能单独建立归属。实验名称单独输出为 `experiment_name`，依次取快照的 `experiment_name`、`display_name`、`name`；仅当 `project.project_id` 等于目录 `run_id` 时兼容旧 `project.name`，最后回退目录 `run_id`。只接受非空字符串，去掉首尾空白并限制为 1000 字符，不自动附加 ID。同项目名称冲突时按目录 `run_id` 排序，首个不同于项目 ID 的有效名称胜出；其他项目元信息取首个 run，单个 run 的 `project` 保留自身上报值。服务仅读取 `monitor.json`；生产端须将项目归属和实验名称写入该快照。

首页默认读取真实接口。无快照时显示空态，点击「查看示例」才加载明确标记「示例数据」的静态 fixture；「返回真实运行」恢复接口。示例不会写入 runs，也不会出现在真实运行列表中。

## 界面语言 / Interface language

右上角「中 / EN」切换完整界面语言，默认中文。选择保存在浏览器 localStorage 的 `rsisys-language`（`zh` / `en`），重载后恢复；存储不可用时保留当前页面内的选择。按钮、状态、空态、指标名、图表提示、流程和详情标题均跟随语言；日期使用相应 locale。

切换只用已缓存的项目与运行数据重新渲染，不触发 API 请求，保留当前页面、所选实验和决策、展开记录及图谱平移、缩放和节点位置。模型 `reason` / `strategy` / `objective`、反馈和日志原文、项目名称与说明、技术 ID 和协议字段保持原样。已知指标显示友好名称，原始 key 放在 title；自定义指标保留可见 key，便于区分。

无外部依赖或翻译 API。页面以 `defer` 依次加载 `/i18n.js`、`/decisions.js` 和 `/app.js`。全局 `RSII18n` 提供 `t(key, params)`、`getLanguage()`、`setLanguage("zh" | "en")`、`applyStatic(document)` 和 `subscribe(listener)`；订阅返回取消函数。`messages` 为只读双语字典，`storageKey` 为存储键。动态文案显式调用 `t()`，参数使用 `{p0}` 等模板；静态 UI 仅通过 `data-i18n` / `data-i18n-aria-label` / `data-i18n-title` / `data-i18n-placeholder` 标记翻译，不遍历替换模型文本。计数与标签分开标记，以保留动态内容。

Use **中 / EN** to switch the interface. The preference persists locally. Switching reuses cached data and preserves the current page, selected experiment and decision, graph scroll and expanded records. Reported model text and logs stay verbatim.

## 数据与交互

读取 `runs/<run-id>/monitor.json`，协议为 `schema_version: 1`。支持 `workflow`、`current`、`decisions`、`jobs`、`candidates`、`round_results` 和 `events`；旧快照与缺省字段保持兼容。字段缺失显示未报告，不补零、不补造执行结果。

- `/` 为项目管理页，按大项目分块，每个项目下以卡片列出实验名称、状态、更新时间和已报告决策数。项目名称与实验名称独立，缺少归属的实验归入“未归属”。首页只读取清单，不自动进入实验。点击实验进入 `?project=<project-id>&run=<run-id>`；仅指定项目时仍停留管理页，指定 run 时由其真实归属确定上级项目。返回按钮和浏览器前进 / 后退保持两页导航。
- 实验详情分别显示实验名和所属项目名；缓存按 run ID 隔离，同项目下不同实验各自保存选中决策、展开状态和图谱视角与节点位置。顶部的研究脉络只包含该 run 内的决策节点：固定起点、圆点和直线，自左向右排列。点击节点后，下方显示对应决策的理由、提交内容、反馈和指标。图谱使用当前快照中的全部决策（API 上限 300），不再分页列出步骤按钮。
- `decisions.js` 将明确的 `parent_decision_id` 绘成父子分支，`source_decision_ids` 绘成引用虚线；没有明确父关系时按记录顺序连线，仅表示先后，不代表因果。缺失、歧义或循环关系不会生成虚假依赖。图谱恢复圆点、细直线和舒展的动态布局，采用 420px 高画布。拖动节点时其他节点弹性调整，实验根节点固定；拖动空白可自由平移，Ctrl/⌘ + 滚轮缩放。右下角提供缩小、放大、适应和重排，详细说明放在按钮提示中。节点标签使用页面同款系统中英文字体、16px 自然字形，不用 SVG 强制压缩或拉伸；长标签最多两行、22px 行距，完整名称保留在悬停提示中。
- 跨实验的 `parent_run_id` 和 `source_run_ids` 留在「来源记录」中，以链接打开来源实验；不进入实验内部的决策图谱。
- 每 5 秒静默刷新：管理页只更新清单，实验页更新清单与当前实验。隐藏页面暂停轮询，切换实验取消旧请求并拒绝迟到响应。缓存按实验隔离；不显示连接状态、手动刷新或导出工具栏。
- 刷新保持详情容器挂在原位置，不把页面高度短暂折叠，并保留页面滚动位置。刷新无数据变化时复用运行详情 DOM；有新反馈时，按稳定决策身份保留所选决策、展开记录、图谱视角和节点位置。中英切换同样保留状态。拖动或布局动画期间延后刷新画布；结束交互后更新反馈。切换实验、返回管理页或隐藏标签页会取消动画并释放图谱监听，重新进入时恢复布局。保留候选结果和来源记录折叠区的展开状态。
- 执行流程按已报告时间从早到晚排列；未知时间保留快照顺序列于末尾，不补造因果。图谱显示当前快照中的全部决策。每次展示一个步骤的**决策理由、提交内容、实际反馈、进度及结果**。只有“执行已完成”等通用状态时不显示反馈块；缺失的理由不补造。`evaluate` 显示为“验证改动效果”，明确评测与采用改动是两个步骤。训练步骤带损失 / 学习率曲线与历史点，评测带基线 / 候选对比。重复 finish 聚合并保留全部原始记录；“结束本轮研究”不代表整个运行已停止。
- 已知动作使用中英友好名称，例如 `configure_prompt` → 调整提示词、`define_recipe` → 定义训练方案；自定义动作直接显示原名，只有没有动作字段时才显示“未命名决策”。模型理由、策略及反馈原文不改写。与动作名称完全重复的 strategy 不再重复展示；页面不展示技术 ID、版本哈希或协议字段列表。
- 只按明确唯一的 `job_id`、`candidate_id` 或指令引用关联反馈；找不到指定 job 或存在歧义时不回退猜测。复查时指标与关联任务当前结果分开展示。原始事件留在快照和日志中，不进入阅读界面。
- 已移除重复的“指标与记录”区域。执行步骤直接呈现决策与结果，已移除“记录详情”“运行详情”“运行事件”和任务原始字段列表。未关联的任务 / 候选只保留状态及指标摘要。技术字段仍保留在公开 API 和运行文件中；底层运行数据不删除。
- 百分比只使用有效的 step / total_steps；训练 100% 不代表整个运行完成。曲线只绘制真实历史点，无历史不补造。候选差值只表示候选减基线，不自动判断优劣。

决策的可选 `changes` 为 `{field, after, before?}` 数组，最多 20 项，字段名最多 100 字符、值最多 4000 字符；仅输出生产端明确选择的公开参数，包括提交给 Actor 的提示词。`before` 必须来自真实记录，不从状态推断。`feedback_summary` 和 `error` 保存实际执行反馈；不读取模型隐藏推理、系统提示词或训练样本。旧记录没有理由时保持缺失，后续模型决策可记录简短公开 `reason`。

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

`GET /api/runs/<id>`：返回协议字段的只读投影，始终包含 `experiment_name` 和 `project: {project_id, name}`；有效归属可额外包含公开的 `description`、`system_id`，缺失时不补造。决策可包含 `parent_decision_id` 及 `source_decision_ids` 字符串数组。目录名是路由和返回 `run_id` 的权威来源，即使快照内 `run_id` 不一致也不改变；项目归属则来自快照的有效 `project.project_id`。不跟随快照内的路径或文件引用。无快照返回 404，损坏或写入中的 JSON 返回 503（带 `Retry-After: 2`），不支持的协议版本返回 422，文件超过 4 MiB 返回 413。

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
node --check web/static/app.js
node --test web/static/*.test.js
```

服务测试覆盖显式项目归属与多实验分组、同名不同 ID 隔离、实验名称优先级、跨项目来源、历史文件不变、快照投影、读取边界、错误恢复与只读接口。前端验证重点为独立项目导航与深链接、来源链接、双语与原文保留、按需读取与并发取消、刷新保留展开状态和稳定步骤身份、训练历史及反馈关联。Node 只用于检查与测试，服务运行不依赖 Node。

界面采用独立的实验管理页与实验详情页，使用暖白纸面、深棕文字与低饱和陶土橙选择态；标题使用 Georgia／本机宋体，正文使用本机系统无衬线字体，无外部字体请求或品牌 logo。

浏览器 QA 截图放在临时目录 `/tmp/rsisys-*`，不写入 runs、不作为源码入库。

## 生产端与历史记录

[Qwen toycase 的 monitor.py](../examples/qwen_toy/monitor.py) 从模型的结构化 Decision / ReviewDecision 和实际 Runner 结果生成上述摘要；[worker](../examples/qwen_toy/worker.py) 在 Trainer callback 中报告真实 global_step、max_steps、耗时及已记录的 loss / learning_rate。UI 从 monitor.json 读取，不直接访问训练服务。

已完成的旧 toycase 可以显式补导出已有决策记录：

```sh
PYTHONPATH=src python examples/qwen_toy/monitor.py --run-dir runs/YOUR_COMPLETED_RUN
```

导出只从该 run 的 decision_memory.json、result.json 和原监控事件生成公开字段，并原子更新 monitor.json；不改训练结果、时间戳或模型文件，不启动训练。source=history 表示真实历史记录，缺失的时间与中间步数保持未知。这个导出器属于 toycase；其他 Harness 自行生产同样的轻量监控字段，不需要适配统一训练数据格式。
