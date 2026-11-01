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

读取 `runs/<run-id>/monitor.json`，协议为 `schema_version: 1`。字段包括 `run_id`、`objective`、`status`、`updated_at`、`loops`、`models`、`jobs`、`candidates`、`events`；新增可选 `workflow`、`current`、`decisions`、`round_results`，旧快照仍可读取；可选字段缺省或为 null 时显示未报告／空态。未知状态按原值显示，筛选中的「其他」可以查看未归类状态。

- 运行切换、双循环状态与摘要、当前 decision maker / actor 绑定。
- 当前执行显示轮次、流程阶段、已完成指令数、当前指令/任务、耗时与复查设置。只有后端报告有效 step / total_steps 时才显示百分比；训练 100% 不代表评测、应用或整个运行完成。
- 模型决策区区分执行指令、运行复查和结束决策，显示显式 strategy / reason、来源、轮次、模型版本及执行状态。可搜索、筛选和展开关联任务、候选、配方与评测协议；不展示原始 prompt、模型完整响应或隐藏推理。
- 轮次结果显示父版本到候选版本、是否应用、Actor 实际版本、评测指标和复查次数。
- 任务状态筛选，候选状态筛选，基线与新版本并排对比。
- 事件按时间倒序，支持类型筛选、搜索摘要／详情、展开详情；每次显示 30 条，可加载更多。
- 默认每 5 秒自动刷新，可暂停或手动刷新；后台页面暂停轮询，回到页面后恢复。
- 8 秒请求超时；断连时明确标记旧快照并保留上次成功的数据，不把失败显示为空运行。运行中快照超过 2 分钟未更新会提示过期。
- 切换运行时清除上一运行数据，避免错配；自动刷新保留筛选条件与已展开事件。

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

QA 截图保存在已 gitignore 的 `runs/monitor-preview/`，不作为 web 源码重复入库。

本次用已有两轮 GPU 运行的真实记录检查决策列表、复查理由、版本归因和轮次结果；进行中的训练进度另用明确标记的示例与 callback/HTTP 测试验证，不把历史运行补导出当作新训练。

## 生产端与历史记录

[Qwen toycase 的 monitor.py](../examples/qwen_toy/monitor.py) 从模型的结构化 Decision / ReviewDecision 和实际 Runner 结果生成上述摘要；[worker](../examples/qwen_toy/worker.py) 在 Trainer callback 中报告真实 global_step、max_steps、耗时及已记录的 loss / learning_rate。UI 从 monitor.json 读取，不直接访问训练服务。

已完成的旧 toycase 可以显式补导出已有决策记录：

```sh
PYTHONPATH=src python examples/qwen_toy/monitor.py --run-dir runs/YOUR_COMPLETED_RUN
```

导出只从该 run 的 decision_memory.json、result.json 和原监控事件生成公开字段，并原子更新 monitor.json；不改训练结果、时间戳或模型文件，不启动训练。source=history 表示真实历史记录，缺失的时间与中间步数保持未知。这个导出器属于 toycase；其他 Harness 自行生产同样的轻量监控字段，不需要适配统一训练数据格式。
