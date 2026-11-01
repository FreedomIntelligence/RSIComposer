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

首页默认读取真实接口。无快照时显示空态，点击「查看示例」才加载明确标记「示例数据」的静态 fixture；「返回真实运行」恢复接口。示例不会写入 runs，也不会出现在真实运行列表中。

## 数据与交互

读取 `runs/<run-id>/monitor.json`，协议为 `schema_version: 1`。支持 `workflow`、`current`、`decisions`、`jobs`、`candidates`、`round_results` 和 `events`；旧快照与缺省字段保持兼容。字段缺失显示未报告，不补零、不补造执行结果。

- 主视图按决策时间倒序排列，每项并排展示中文动作与对应反馈。直接使用决策的公开 `metrics`，同时兼容按 `job_id` / `directive_id` 精确关联任务指标；存在但未找到的 job_id 不回退关联其他任务，多任务歧义时不猜测。复查记录不把任务当前指标冒充复查时点指标。
- 连续、同理由、同轮次、同状态和版本的重复 finish 聚合为一项，保留数量及每条原始记录。finish 统一称为「结束本轮研究」，不推断整个运行已经停止。
- 动作、指标名称使用中文友好映射。未知操作显示「其他操作」，原始名称保留在详情；技术 ID、模型绑定与事件收进详情。reason / strategy 原文保留在可展开区，不替换 candidate、baseline 或协议名称，不调用模型翻译。
- 当前执行只在 step / total_steps 均有效时显示百分比；训练 100% 不代表评测、应用、整个运行或目标验证完成。
- 训练任务展示原生 SVG 损失与学习率图，无图表库或外部资源。支持鼠标查看最近采样点、聚焦图表后用左右键逐点查看，Home / End 跳到首尾；一个历史点只画一个点，无历史时明确显示当前指标或空态。
- 候选比较、轮次结果与原始事件默认折叠，保留状态筛选、搜索、加载更多和指标详情。对比差值只表示候选减基线，不自动判断优劣。
- 默认每 5 秒刷新，可暂停或手动刷新；后台页面暂停轮询。请求超时为 8 秒；断连保留旧快照并明确提示，活跃运行超过 2 分钟未更新会提示过期。切换运行清除上一运行数据；刷新保留筛选和详情展开状态。

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

上例仅为协议示意。前端只接受有限数值；按真实步数绘图，步数不可用时尝试真实耗时；同一横坐标保留最后上报点，不按数组序号伪造步数，不补中间点。旧真实 run 的 `metrics` 没有完整 history 时，只显示当前数值和诚实空态，不能从末次 loss / lr 还原训练曲线。原始历史和全部数值指标可在任务详情中查看。

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

`GET /api/runs`：

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

`GET /api/runs/<id>`：返回协议字段的只读投影。目录名是路由和返回 `run_id` 的权威来源；不跟随快照内的路径或文件引用。无快照返回 404，损坏或写入中的 JSON 返回 503（带 `Retry-After: 2`），不支持的协议版本返回 422，文件超过 4 MiB 返回 413。

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
```

服务测试覆盖读取摘要、缺省字段、flat worker metrics、空目录、损坏快照恢复、协议版本、大小限制、路径穿越、目录／文件符号链接、FIFO、字段过滤、Host／跨站防护及只读方法。Node 只用于可选的 JavaScript 语法检查，运行服务不依赖 Node。

视觉参考：[Claude 官方博客](https://claude.com/blog)、[Anthropic News](https://www.anthropic.com/news)。使用暖白纸面、深棕文字、低饱和陶土橙；标题使用 Georgia／本机宋体，正文使用本机系统无衬线字体，无外部字体请求或品牌 logo。

浏览器 QA 截图与下载报告放在临时目录 `/tmp/rsisys-*`，不写入 runs、不作为源码入库。

本次用已有两轮 GPU 运行的真实记录检查决策列表、复查理由、版本归因和轮次结果；进行中的训练进度另用明确标记的示例与 callback/HTTP 测试验证，不把历史运行补导出当作新训练。

## 生产端与历史记录

[Qwen toycase 的 monitor.py](../examples/qwen_toy/monitor.py) 从模型的结构化 Decision / ReviewDecision 和实际 Runner 结果生成上述摘要；[worker](../examples/qwen_toy/worker.py) 在 Trainer callback 中报告真实 global_step、max_steps、耗时及已记录的 loss / learning_rate。UI 从 monitor.json 读取，不直接访问训练服务。

已完成的旧 toycase 可以显式补导出已有决策记录：

```sh
PYTHONPATH=src python examples/qwen_toy/monitor.py --run-dir runs/YOUR_COMPLETED_RUN
```

导出只从该 run 的 decision_memory.json、result.json 和原监控事件生成公开字段，并原子更新 monitor.json；不改训练结果、时间戳或模型文件，不启动训练。source=history 表示真实历史记录，缺失的时间与中间步数保持未知。这个导出器属于 toycase；其他 Harness 自行生产同样的轻量监控字段，不需要适配统一训练数据格式。

## 前端改造交付与验证（2026-11-01）

本次前端修改仅涉及 `web/static/index.html`、`style.css`、`app.js`、`demo.json` 和本 README；未修改 Python、未提交。保留暖白纸面、serif 标题与陶土色，将决策与反馈设为主视图，移除英文装饰 caption、内部符号和重复说明。示例 fixture 中的六点曲线和三条重复 finish 明确属于静态演示，不用于补全真实运行。

已验证：

- `node --check web/static/app.js` 通过。
- 在用户已启动的 `http://127.0.0.1:8768` 上使用真实 Chrome / Playwright 验证：真实运行连接成功，199 条决策聚合成 3 项，其中 197 条重复 finish 完整保留于详情；旧训练只有当前指标时没有伪造 SVG 曲线。
- 示例 loss / lr 两条 SVG 曲线各有 6 个历史点；Home、方向键、End 可查看真实采样点；坐标没有 NaN / Infinity。单点历史只显示单点，零学习率有效。
- 真实与示例报告均通过浏览器下载事件实际保存为 Markdown，并检查重复数量、无历史提示、示例标识、历史摘要与尚未开始执行的说明。临时文件为 `/tmp/rsisys-real-report.md`、`/tmp/rsisys-demo-report.md`。
- 390px 移动视口无横向溢出；搜索、筛选、未知操作原名展开、精确关联、最多 1000 历史点、非法/缺失指标与文本安全检查通过；本轮浏览器验证没有 pageerror。
- 后端回归由 server agent / 用户单独完成；用户报告全量 **625 tests 通过、3 skip**，本前端交付不重复运行后端回归。

尚未同步远程 8767；同步和余下浏览器验收由用户接手。前端没有引入构建步骤或运行时依赖。
