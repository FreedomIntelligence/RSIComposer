# 框架运行就绪检查

日期：2026-11-04。范围：当前本地工作树；延续统一模块装配，没有启动远端实验、模型推理或训练。

## 判断

核心框架已经可以进行端到端运行，并具备开展一次短时真实服务联调的条件。本次完成的是本地运行、接口组合、反馈投影、状态恢复和异常收尾的验证。真实模型的长期自主探索、摘要语义、训练后部署与收益仍需单独验收。

现有模块通过 `ModuleAssembly → assemble_runner` 组合；完整研究链路需显式配置 `ResearchAssembly` 和所需 Lab。默认 mock/demo 能运行，不表示已经连接实际领域模型、训练服务或第三方 memory。

## 发现并修复

### 执行器关闭失败后丢失所有权保护

原 Application 把共享 sandbox 的 `close_all` 放在 `ExitStack`。当关闭失败，ExitStack 仍继续关闭数据库和工作区锁，且丢弃失败回调；再次 `close()` 无法重试这个执行器。

独立复现记录 `.local/framework-readiness-20261104/cleanup-before.json` 显示：一个 owned session 仍存在，store 与锁已经释放，再次关闭仍留下 session。

现在共享执行器和默认 exploration environment 的关闭均登记为 owned cleanup。失败保留回调和依赖，禁止继续执行，后续显式 `close()` 可以重试；确认成功后才释放数据库与锁。demo 的共享 sandbox 也沿用同一语义。

回归测试使用真实 CPU 子进程与故障注入验证：关闭失败时 PID 仍由原 owner 持有、另一持有者拿不到锁、数据库仍可访问；重试后进程退出、session 清空并释放锁。目录删除失败另行验证 environment 引用保留与重试。

### 医疗入口缺少统一收尾与关闭后推理拦截

医疗 CLI 原先运行完直接返回，没有调用 `MedicalAssembly.close()`。现在用 context manager 覆盖运行、结果持久化与异常路径；原始运行异常保留，额外关闭异常附加为 note。

`AssemblyRunner.closed` 暴露统一 owner 的关闭状态。`MedicalAssembly.run()` 在建立 anchor、调用模型前检查该状态，避免关闭后先发生推理、随后才在 step 被拒绝。

Python API 的 step budget 耗尽仍返回 `needs_input`，由持有装配的调用方决定继续或关闭；CLI 为一次性调用，会关闭装配。插件或外部训练资源仍须由其 owner 注册正确的 cleanup，接口组合不自动取得外部资源所有权。

## 实际验证

| 验证 | 结果与范围 |
| --- | --- |
| 默认 demo | 本地模拟双 loop 跑通 |
| CLI run → inspect → resume | local_process 运行成功；快照检查、在新目录恢复成功；保留同一 run_id，Experience 从 3 条延续到 6 条 |
| `examples/components/open_lab.py` | 实际 CPU 子进程完成写代码、执行、结果读取与收尾；原生输出 value=5；没有模型推理或训练 |
| `examples/components/feedback_pipeline.py` | 模拟回执、轨迹归档与提案引用闭环；没有验证 harness 实际应用 |
| `examples/components/decision_policy_lab.py` | 轨迹到数据任务再到模拟训练候选闭环；没有真实优化权重或切换 live policy |
| 医疗 CLI | 使用本地 HTTP fixture 验证正常完成、步数耗尽、运行异常、结果持久化异常、关闭异常及关闭后零推理 |
| 默认 feedback 连续整理 | 120 条不同原生事件；每 20 条检查实际 Decision 请求中的 memory 投影；最早原始输出仍能完整分页读取 |
| 完整回归 | 命令、通过/跳过数、耗时与日志 SHA 见 `repros/framework-readiness-validation.json` |

120 条反馈测试中，memory 常驻文本为 11,463–11,746 UTF-8 bytes；请求 prompt 为 27,714–28,006 bytes。归档记录持续增长，常驻投影没有随之线性增长。这里的 bytes 不是 tokenizer tokens；测试没有调用摘要模型，也不证明任意 memory 插件、任意长度研究或自由文本总结语义正确。

## 接入真实运行时仍需明确的事项

- **领域执行与身份**：配置实际后端、设备、数据、权限、资源 owner 和 cleanup；本地 CPU 进程实现适用于可信代码执行，不提供宿主安全隔离。
- **模型与 memory**：用真实后端确认 tool calling、完整请求 token、compact 阈值与摘要内容；候选、协议、测量值和采纳状态必须与原始回执对齐。
- **训练与部署**：本次没有验证新权重、优化器、真实 SFT/RL、部署服务就绪或训练收益；提案与模拟候选不等同于已应用改进。
- **恢复支持**：自定义组件仍需提供自己的 state adapter / recovery contract。统一接口不自动赋予某个第三方 SDK 或远端任务恢复能力；当前替换发生在装配阶段。
- **安装范围**：使用仓库现有 Python 环境验证源码入口；未安装新包、未构建发布 wheel、未验证干净机器安装。

本次保留既有未提交修改，未改冻结 run、历史 campaign、远端服务或自动化。基线文件完整性对比及最终源码清单保存在 `.local/framework-readiness-20261104/`。
