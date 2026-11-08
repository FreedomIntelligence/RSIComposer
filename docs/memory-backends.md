# 显式 Memory 与 ReMe 检索后端

RSISys 的单 meta-agent 路径支持可选的 ReMe BM25 检索。写入、审阅和删除仍由行动中的
meta-agent 决定；不启用独立整理模型、embedding、Auto Memory 或 Auto Dream。

## 使用

在新的 controller 环境显式安装可选依赖：

```sh
python -m pip install '.[reme]'
```

`pyproject.toml` 固定 ReMe 源码 commit
`e3fade0a781d080a4be7fb83648a3b2e6d477522`，对应 `reme-ai==0.4.2.3`。
依赖声明包含 upstream 的 `as` extra，因为该版本的包导入依赖 AgentScope；接入层只构建
`BM25Index` 和 `RegexTokenizer`，不会构建 AgentScope agent 或模型客户端。

通用装配与默认 Harness、DSH Harness 共用同一组 Memory 工具：

```python
from rsisys.research import ResearchAssembly

research = ResearchAssembly.reme(require_memory_review=True)
# 等价：ResearchAssembly.single_meta(memory_backend='reme', require_memory_review=True)
# 将 research 传给 Application、ModuleAssembly 或 assemble_research 的现有入口。
research.prepare()  # 可在申请资源前检查依赖；装配时也会检查。
```

医疗 Open Lab 的新实验使用：

```sh
python -m experiments.medical.open_lab_experiment \
  --feedback-routing agent_memory --memory-backend reme \
  --root NEW_RUN --task TASK_JSON --model-path MODEL \
  --python LAB_PYTHON --endpoint META_ENDPOINT --run-id NEW_RUN_ID
```

传统医疗 experiment 入口使用 `--memory-provider reme`。两条入口在创建新 run 前检查
可选依赖。Open Lab 的 `reme` 与 legacy feedback routing 不能混用。

默认仍为 `literal`，旧 journal 的配置和检索语义不变。不要为已有 run 原地切换后端；
通过新的显式 continuation attempt 继承已提交记忆和经 SHA 核验的证据，再选择 ReMe。
同目录恢复继续核对原来的 assembly 配置，不允许后端配置漂移。

## 边界与行为

| 层 | 实现与职责 |
| --- | --- |
| 原始证据 | 既有执行 archive；Memory 的描述不升级为实验事实。 |
| 已提交记忆 | `agent-memory.json` 的原子写入、修订历史、source_refs 和 provenance SHA。 |
| 可替换检索 | 原生 ReMe BM25/RegexTokenizer；索引当前有效的 memory_id、content、applicability。 |
| 研究任务 | 原有 open/update/pin/close/read 和 native receipt 验收，不交给检索器判定成功。 |

这次复用的是 **ReMe 的检索组件**，不是其完整文件管理应用。没有第二份持久化记忆库、
服务、watcher 或外部写入事务。索引只缓存于当前 Memory 实例内，笔记改变后重建；
重启、restore、seed 和 continuation 后根据有效记忆重建。检索不加载 ReMe pickle，
不写文件，不引入需重放的外部 mutation。

显式 `memory_search` 和自动 `memory_context` 共用 ReMe。自动召回结合 objective、最新反馈
和未关闭任务的 title/next_step。BM25 是关键词检索，支持词序变化、部分重合和中文字符；
**不承诺没有共同词的同义改写检索**。未命中时返回空，不用最近笔记冒充命中。
空 query 按修订时间列出笔记；精确 memory_id 优先，评分相同时按修订和 ID 稳定排序。

每次检索附带当前语料的 `corpus_sha256`、方法、匹配数量以及零模型/embedding 调用声明。
分页仍受全局 revision 约束；更新和删除的旧版本只可通过历史 `memory_read` 读取，
不会作为当前知识召回。检索错误会抛出并进入既有故障策略，不静默退回旧索引。
是否允许 context 使用 last_valid 仍由显式 FeedbackFailurePolicy 控制。

工具说明随选定后端一起进入模型上下文。模型仍需用 `memory_read` 读取完整笔记和来源，
在后续行动中说明适用性。`usage_verified=false` 表示命中不等于实际应用，也不证明科研提升。

## 验证

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m unittest tests.memory.test_reme_memory -v
```

测试使用真实 ReMe 组件，包含上一轮失败的查询形式、ID 查询、负查询、修订分页、
更新/删除、无文件写入、作用域隔离、活动 event loop、来源 SHA、任务回执验收、
默认 Harness/DSH 的上下文与工具，以及原 continuation 契约。
安装了可选依赖时这些测试必须执行；未安装时配置/错误边界仍测试，真实组件测试显式 skip。

查询样例使用标注的测试笔记，没有把历史私有题目或完整轨迹复制进测试。
DSH 和决策消费者使用脚本 fixture，验证上下文和工具链路，不代表实际模型已学会复用经验。
本次没有启动新医学实验，也不据此声称重复错误减少或医学准确率提升。

上游参考：

- ReMe: https://github.com/agentscope-ai/ReMe/tree/e3fade0a781d080a4be7fb83648a3b2e6d477522
- BM25: `reme/components/keyword_index/bm25_index.py`
- Tokenizer: `reme/components/tokenizer/regex_tokenizer.py`
