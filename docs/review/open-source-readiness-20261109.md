# 开源前的架构与发布审查

日期：2026-11-09。基线：`aca55bca545f6090d393720cd011de9b0b7bec4f`，
加本轮 `codex/open-source-readiness` 改动。范围覆盖核心、默认实现、独立 monitor、
库外注册、实验组织、持久化/资源边界、发行包与公开文档；没有重新启动 GPU 实验。

## 判断

适合先发布 **0.1 研究框架**，再在同一接口上开发自己的 RSI 算法。
保留 **决策器 → Lab → 反馈 → 决策器**，不因 SFT、RL 或 meta-agent 增加第三个顶层模块。
首次开源应以真实能力和清楚的限制为准，不把完整生产训练平台作为隐含承诺。
公开仓库、合并分支、创建版本标签和上传 PyPI 是不同动作；本轮代码准备不自动切换仓库可见性。

## 架构检查

| 边界 | 检查与结论 |
| --- | --- |
| 公开协议 | `rsisys.api` 提供决策、Lab、注册与装配类型；正式协议位于 `src`，运行包不导入 tests/examples/experiments。内置旧名是实现/兼容细节。 |
| 注册与装配 | `Registry → ModuleAssembly → Application` 复用统一运行时；controller/Lab 的验证、版本和配置独立；插件显式受信任，方法装配不是第三类执行对象。 |
| 决策归属 | 固定算法与 meta-agent 复用同一端口；候选、reward、选择和训练起点由方法规定，host 不读取分数替算法决策。 |
| Lab 与组件 | 可替换 engine/verifier/scorer；Lab 自己持有数据和领域执行。修复 SEAL/TTRL 同类多实例目录冲突，显式隔离产物所有权。 |
| Host 与资源 | 已有操作授权、回执、作业交付和 owned cleanup 保留；可信本地进程边界不等于 OS sandbox；任意插件内部资源仍须自己声明/释放。 |
| 持久化 | 不支持的注册组合明确拒绝自动恢复；修复 SQLite 初始化失败前未交出所有权却泄漏连接的问题。 |
| 模型与证据 | 训练与评测边界、产物 SHA/父模型、真实/模拟标记继续保留；小规模 SEAL/TTRL 无收益如实报告。 |
| 独立监控 | `rsimonitor` 不依赖核心、ML 或 Node 运行环境；发行包包含前端静态资源，只读与无鉴权边界写入 SECURITY。 |
| 代码组织 | examples 只放入门/接口，experiments 放模型与数据配方，tests 保留正式回归；不为发布做大范围内部重写。 |

默认 harness 中仍有较大实现文件，双端点命名与历史兼容设施也仍存在；它们是逐步维护的
技术债，不需要在第一次公开前改名或再造一套运行器。未来拆分应以明确的故障/扩展需求
和现有回归为依据。

## 本轮补齐

- Apache-2.0 主许可，核心与 monitor 的包元数据/许可文件，以及 SEAL 等第三方归属说明。
- 简明英文首页、中文高级指南、文档导航、支持矩阵、版本策略、发布流程与路线图。
- CONTRIBUTING、SECURITY、社区规范、软件引用、CHANGELOG、Issue/PR 模板。
- Linux/macOS × Python 3.12/3.13 的离线 CI、Node 前端检查、sdist/wheel 与隔离安装检查。
  Actions 固定到已核对的 commit；workflow 只读权限，不含部署或上传凭据。
- `Registry.describe()` / `rsisys catalog` 与 `--version`，查询注册不构造运行资源。
- ReMe 保持已验收的源码 pin，移动至独立 `requirements/reme.txt`；不再把 VCS URL 写入
  distribution 依赖元数据，也不声称不存在的 PyPI 版本已可安装。
- SEAL/TTRL 的安全 `output_subdir` 配置，覆盖同类 Lab 的注册装配、跨所有者产物拒绝和
  不可覆盖行为；该隔离不提供 GPU lease。
- SQLite 初始化失败时显式关闭连接，并修复一处测试把数据库事务上下文误当关闭上下文。
  Python 3.13 的资源警告用于定位此问题，不通过关闭警告来掩盖。
- b200 Linux 验收发现隔离执行的解释器路径保留 `..` / 父目录 alias，而挂载路径已被
  canonicalize，导致命令先经过未挂载目录。现在只解析可执行文件父目录，保留 venv
  的 `bin/python` 叶子入口；不增加挂载权限、不退回裸执行，并补充真实进程回归。

## 发布验收记录

| 验收 | 结果与范围 |
| --- | --- |
| macOS / Python 3.12 | 2,061 项，68 项可选环境测试跳过，其余通过。 |
| macOS / Python 3.13 | 2,061 项，68 项跳过，其余通过；修复后无 SQLite ResourceWarning。 |
| b200 / Linux / Python 3.12 | 2,061 项，68 项跳过，其余通过；包含隔离执行和医疗评测的真实 CPU 进程路径。 |
| 前端 | 本地与 b200 各 147 项通过；本轮未修改前端资源。 |
| 文档 | 538 条维护中文档的本地文件/标题链接通过；历史 review 不当作当前文档验收。 |
| 发行包 | 核心和 monitor 均从 sdist 构建 wheel；四份包通过 twine strict；解包后的核心源码文档/脚本完整性通过。 |
| 仓库外安装 | 两个全新独立 venv，只安装本地 wheel，无额外运行依赖；CPU ReST^EM、外部 controller/Lab、catalog、monitor 及许可/静态资源通过。 |
| 源码一致性 | Linux 验证副本的 448 个 Python 源码/测试/fixture 文件与本地 SHA256 一致；远端原工作树未变更。 |
| 静态检查 | Ruff E9/F63/F7/F82 与 diff whitespace 通过。 |

原始日志只存于忽略的 `.local/open-source-readiness-20261109/`，不把本机环境、运行日志
或权重加入发行包。GitHub hosted CI 的配置已准备；是否执行通过以 PR 的实际 checks 为准，
不将本机结果冒充 GitHub 运行结果。实验效果沿用已记录的真实运行，本轮没有新增效果声明。

Git 历史模式扫描覆盖准备提交之前全部本地可达 refs 的 2,424 个 blob，合计
41,691,699 bytes：未命中所检查的私钥/GitHub/Hugging Face/AWS/API key 模式，
没有超过 5 MB 的 blob。这是特定模式检查，不是对数据权利或个人信息的全面保证。
历史 review 含原始请求/回执与本机路径，不应当成面向用户的使用契约；公开前维护者仍须
确认贡献和历史材料的披露权限，代码检查无法替代这项确认。

## 后续核心里程碑

1. 持久化 controller/Lab 状态和外部作业 reconciliation，验证不会重复训练。
2. 长作业 submit/poll/cancel、心跳与 host 预算/成本语义。
3. 跨 Lab/run GPU lease 与可选公共 artifact manifest。
4. 注册成熟 verl/FSDP engine，并单独验收算法、后端与恢复行为。

这些能力尚未实现，已在[支持范围](../compatibility.md)和[发布指南](../releasing.md)明确。
可以先开源边界清楚的研究版；自己的算法继续通过注册接入，用实际研究需求推动上述核心能力。
