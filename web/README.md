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

为现有 toy runs 启用显式项目注册：

```sh
python3 web/server.py --runs-dir runs --projects-file configs/monitor-projects.json --port 8769
```

`--projects-file` 相对当前工作目录解析，格式如下（`runs` 是运行目录 ID 的数组）：

```json
{
  "projects": [{
    "project_id": "qwen-parity-rsi",
    "name": "Qwen 自主研究",
    "description": "以奇偶判断任务验证 RSI 决策、训练、评测与模型应用链路。",
    "system_id": "qwen-parity-system",
    "runs": ["qwen-default-30m-20261101T042500Z", "qwen-autonomous-20261101T024815Z", "qwen-two-round-20261101T003827Z", "qwen-cpu-003", "qwen-toy-003"]
  }]
}
```

注册文件在启动时读取，修改后须重启服务。重复项目 ID、同一运行重复归属（包括同一项目内重复）、无效 ID 会在启动时报错。`unassigned` 是保留 ID。注册文件优先于 `monitor.json` 内的 `project: {project_id, name, description, system_id}`；无注册记录时使用快照的显式归属，两者都没有则统一为 `project_id=unassigned`、`name=未归属`。不按运行名称、目标或模型名推断项目。注册但尚无快照的项目显示空运行入口。

首页默认读取真实接口。无快照时显示空态，点击「查看示例」才加载明确标记「示例数据」的静态 fixture；「返回真实运行」恢复接口。示例不会写入 runs，也不会出现在真实运行列表中。

## 界面语言 / Interface language

右上角「中 / EN」切换完整界面语言，默认中文。选择保存在浏览器 localStorage 的 `rsisys-language`（`zh` / `en`），重载后恢复；存储不可用时保留当前页面内的选择。按钮、状态、空态、指标名、图表提示、流程和详情标题及导出报告均跟随语言；日期使用相应 locale。

切换只用已缓存的项目与运行数据重新渲染，不触发 API 请求，不重置所选运行、执行步骤、分页、图谱缩放、平移或节点位置。拖拽与 RAF 松弛期间，图谱沿用原有延后重绘机制。模型 `reason` / `strategy` / `objective`、反馈和日志原文、项目名称与说明、技术 ID 和协议字段保持原样。已知指标显示友好名称，原始 key 放在 title；自定义指标保留可见 key，便于区分。

无外部依赖或翻译 API。`/i18n.js` 与 `/lineage.js` 必须先于 `/app.js` 以 `defer` 加载，并加入静态服务白名单。全局 `RSII18n` 提供 `t(key, params)`、`getLanguage()`、`setLanguage("zh" | "en")`、`applyStatic(document)` 和 `subscribe(listener)`；订阅返回取消函数。`messages` 为只读双语字典，`storageKey` 为存储键。动态文案显式调用 `t()`，参数使用 `{p0}` 等模板；静态 UI 仅通过 `data-i18n` / `data-i18n-aria-label` / `data-i18n-title` / `data-i18n-placeholder` 标记翻译，不遍历替换模型文本。计数与标签分开标记，以保留动态节点。

Use **中 / EN** to switch the interface. The preference persists locally. Switching reuses cached data and preserves the selected run, execution step, graph view. Reported model text and logs stay verbatim. Exported reports follow the selected interface language.

验证语言覆盖、持久化、原文保留、无额外请求、错误与空态、图工具、导航和 RAF 期间切换：

```sh
node --test web/static/i18n.test.js web/static/project.test.js
```

## 数据与交互

读取 `runs/<run-id>/monitor.json`，协议为 `schema_version: 1`。支持 `workflow`、`current`、`decisions`、`jobs`、`candidates`、`round_results` 和 `events`；旧快照与缺省字段保持兼容。字段缺失显示未报告，不补零、不补造执行结果。

- 主图只显示项目与运行。项目作为最左侧固定的 `0 · 项目名` 节点；点击运行后，下方显示该次运行的执行流程。decision、job、candidate 不进入主图。旧 `?run=<id>` 与 `?project=<id>&run=<id>` 深链接仍有效。
- **树的层级代表明确的续接关系。** 新运行的 `parent_run_id` 指向沿用的父运行：如 run6、run7 都从 run5 继续，它们就是 run5 的两个子分支。首个独立运行的迭代深度为 0，后续每条有效父链加 1；它不等于训练次数或已验证提升次数。生产端需要写入真实来源，前端不会根据分数、名称、时间或模型 revision 自动认定采纳。
- `source_run_ids` 只表示资料引用，以细虚线展示，不增加树深度；`parent_run_id` 使用带箭头直线。没有父运行的分支连接项目根；有效子运行只连父运行，不重复连根。缺失、歧义、自引用和环不会建立虚假父关系，异常链保留在根下，页面只提示来源不完整，原始诊断仍在数据中。来源关系使用当前项目内的唯一 run ID。
- `lineage.js` 是无依赖纯数据模块：`RSILineage.build(nodes, edges)` 生成有效树、深度与分支位置。应用将 API 中的项目归属和完整快照作为权威输入。`networkLayout` 再进行有限步松弛：同深度目标 x 相同，相邻深度相距 180；分支按树行错开。根坐标固定 `(70,245)`。运行编号与同层顺序使用 `updated_at` / ID 稳定排序，不表示采纳。
- 运行节点可拖动，其他运行节点通过 RAF 自适应，根始终固定。每帧 2 步，移动补充 40 帧热度，松手后 18 帧停止；无常驻动画。拖动或缓动时延后合并刷新。切换项目、示例或隐藏页面会取消动画。支持空白平移、适应、缩放、恢复 100%、重排和键盘选择。自动适应最低缩放为 85%，较大的树允许滚动，避免文字被缩成小点；正文 16px，节点标签 15px，节点点击区域 32px。v5 位置缓存按真实 / 示例及项目隔离；忽略旧布局缓存。
- 首屏只读取 `/api/projects`，选择运行后才读取 `/api/runs/<id>`。缓存隔离每个项目和运行，切换时取消旧请求并拒绝迟到响应。默认每 5 秒刷新当前选中运行，可暂停；隐藏页面暂停轮询。已有快照时静默刷新，不展示“正在刷新”。失败保留旧数据并提供重试。
- 刷新无数据变化时复用运行详情 DOM；有新反馈时，按稳定决策身份保留所选步骤、展开记录及流程横向滚动。中英切换同样保留状态。保留候选结果折叠区的展开状态。
- 执行流程按已报告时间从早到晚排列；未知时间保留快照顺序列于末尾，不补造因果。首批显示 30 个步骤，可继续加载。每次展示一个步骤的**动作、理由 / 策略、反馈、进度及结果**。训练步骤带损失 / 学习率曲线与历史点，评测带基线 / 候选对比。重复 finish 聚合并保留全部原始记录；“结束本轮研究”不代表整个运行已停止。
- 已知动作使用中英友好名称，例如 `configure_prompt` → 调整提示词、`define_recipe` → 定义训练方案；自定义动作直接显示原名，只有没有动作字段时才显示“未命名决策”。模型理由、策略及反馈原文不改写。与动作名称完全重复的 strategy 不再重复展示；页面不展示技术 ID、版本哈希或协议字段列表。
- 只按明确唯一的 `job_id`、`candidate_id` 或指令引用关联反馈；找不到指定 job 或存在歧义时不回退猜测。复查时指标与关联任务当前结果分开展示。原始事件留在快照和日志中，不进入阅读界面。
- 已移除重复的“指标与记录”区域。执行步骤直接呈现决策与结果，已移除“记录详情”“运行详情”“运行事件”和任务原始字段列表。未关联的任务 / 候选只保留状态及指标摘要。技术字段仍保留在公开 API 与导出报告中；底层运行数据不删除。
- 百分比只使用有效的 step / total_steps；训练 100% 不代表整个运行完成。曲线只绘制真实历史点，无历史不补造。候选差值只表示候选减基线，不自动判断优劣。

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

上例仅为协议示意。前端只接受有限数值；按真实步数绘图，步数不可用时尝试真实耗时；同一横坐标保留最后上报点，不按数组序号伪造步数，不补中间点。旧真实 run 的 `metrics` 没有完整 history 时，只显示当前数值和诚实空态，不能从末次 loss / lr 还原训练曲线。曲线可逐点查看；公开快照仍保留原始历史数组，导出报告保留数值汇总。

### 导出报告

「导出报告」直接在浏览器将当前已读取的公开快照下载为 UTF-8 Markdown（`.md`）。报告包括目标、运行状态、快照与导出时间、聚合决策、对应反馈、训练与评测指标、历史点数及首末值/范围、候选与轮次结果和可展开的原始记录。排队、未应用、未报告、无历史均如实表达；示例报告明确标记示例，断连导出注明使用旧快照。导出全部当前快照内容，不受页面搜索和筛选影响。

导出不调用模型、不新增后端、不触发训练或云上传；与可选的 SwanLab 离线导出 CLI 相互独立。快照不是完整运行档案，报告也不宣称覆盖完整实验历史。

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

`GET /api/projects`：返回项目与运行元信息，不包含所有运行快照。

```json
{
  "schema_version": 1,
  "projects": [{
    "project_id": "qwen-parity-rsi",
    "name": "Qwen 自主研究",
    "description": "…",
    "system_id": "qwen-parity-system",
    "runs": [{"run_id": "run-001", "objective": "…", "status": "running", "updated_at": "…", "project": {"project_id": "qwen-parity-rsi", "name": "Qwen 自主研究"}}]
  }]
}
```

`GET /api/runs`：每行同样含 `project` 元信息。


```json
{
  "schema_version": 1,
  "runs": [
    {
      "schema_version": 1,
      "run_id": "run-001",
      "objective": "…",
      "status": "running",
      "updated_at": "2026-11-01T00:00:00Z",
      "counts": {"jobs": 1, "candidates": 2, "events": 5}
    }
  ]
}
```

只列出存在可读取监控文件的运行，忽略无 `monitor.json` 的目录（包括 `runs/monitor-preview/`）。缺失 runs 目录返回空列表且不创建目录；损坏 JSON / 不支持的版本保留为带 `error` 的 unavailable 清单项。

运行列表及快照可包含可选 `parent_run_id` 字符串与 `source_run_ids` 字符串数组；其中 parent_run_id 定义续接父节点，source_run_ids 仅定义资料引用，不增加迭代深度。

`GET /api/runs/<id>`：返回协议字段的只读投影，包含 `project: {project_id, name, description, system_id}`（可选描述和系统 ID 缺失时不补造）。决策可包含 `parent_decision_id` 及 `source_decision_ids` 字符串数组。目录名是路由和返回 `run_id` 的权威来源；不跟随快照内的路径或文件引用。无快照返回 404，损坏或写入中的 JSON 返回 503（带 `Retry-After: 2`），不支持的协议版本返回 422，文件超过 4 MiB 返回 413。

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
python3 -m unittest discover -s tests -p test_monitor.py -v
node --check web/static/app.js
node --test web/static/*.test.js
```

服务测试覆盖快照投影、读取边界、错误恢复与只读接口。前端测试覆盖命名 / 双语、显式父链与引用、分叉 / 环 / 歧义 / 长链、根固定、拖拽 RAF 与计算预算、按需读取与并发取消、刷新保留原生展开状态和稳定步骤身份、训练历史和反馈关联。Node 只用于检查与测试，服务运行不依赖 Node。

视觉参考：[Zotero Style 关系图谱演示](https://zotero-chinese.com/user-guide/plugins/style#关系图谱)，仅参考小圆点、极细连线和下方标签的视觉形式，未引入其代码。保留暖白纸面、深棕文字与低饱和陶土橙选择态；标题使用 Georgia／本机宋体，正文使用本机系统无衬线字体，无外部字体请求或品牌 logo。

浏览器 QA 截图与下载报告放在临时目录 `/tmp/rsisys-*`，不写入 runs、不作为源码入库。

## 生产端与历史记录

[Qwen toycase 的 monitor.py](../examples/qwen_toy/monitor.py) 从模型的结构化 Decision / ReviewDecision 和实际 Runner 结果生成上述摘要；[worker](../examples/qwen_toy/worker.py) 在 Trainer callback 中报告真实 global_step、max_steps、耗时及已记录的 loss / learning_rate。UI 从 monitor.json 读取，不直接访问训练服务。

已完成的旧 toycase 可以显式补导出已有决策记录：

```sh
PYTHONPATH=src python examples/qwen_toy/monitor.py --run-dir runs/YOUR_COMPLETED_RUN
```

导出只从该 run 的 decision_memory.json、result.json 和原监控事件生成公开字段，并原子更新 monitor.json；不改训练结果、时间戳或模型文件，不启动训练。source=history 表示真实历史记录，缺失的时间与中间步数保持未知。这个导出器属于 toycase；其他 Harness 自行生产同样的轻量监控字段，不需要适配统一训练数据格式。
