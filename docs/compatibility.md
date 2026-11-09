# 支持范围与兼容策略

## 首次公开版本的范围

定位是可扩展的 RSI 研究框架，公开接口为 controller 与 Lab；不是所有训练后端、
作业系统与任意插件的自动适配器。当前 0.1 版本提供以下范围：

| 能力 | 当前边界 |
| --- | --- |
| 核心 / CPU 入门 | Python 3.12、3.13；Linux、macOS；核心零运行依赖。Windows 未支持，持久化实现使用 POSIX 文件锁与进程设施。 |
| 库外扩展 | `rsisys.api`、`register_controller/register_lab`、JSON 选择、`alias.operation` 路由；插件是受信任代码。 |
| 内部组件 | 可在工厂中注入 harness、memory、engine、verifier/scorer；不强制统一所有内部图。 |
| 运行保障 | 已接入的操作授权、参数检查、反馈交付、证据、所有权清理与有界 host 推进。任意插件内部调用须自行遵守边界。 |
| 恢复 | 内置持久化/快照及显式 state adapter / recovery contract；普通注册不自动恢复 controller 内存或远端作业。 |
| ReST^EM / SEAL / TTRL GPU recipes | Linux/CUDA、本地模型与数据；实验文档记录实际版本。SEAL/TTRL 为同步作业，不支持原目录自动续跑。 |
| RL | TTRL 的独立 LoRA GRPO 参考实现，以及可选 `ttrl_verl` 官方全参数后端；后者需要固定上游源码和独立 GPU 依赖环境，见 [TTRL 配方](../experiments/ttrl/README.md)。 |
| FT-Agent / AIDE | 可选上游源码与独立依赖环境。FT-Agent 为本地 GSM8K / LoRA 执行适配；AIDE 为原生 Agent 的数学程序任务适配。两者都有真实模型验证要求，不承诺原 benchmark 收益、搜索恢复或 meta-agent 权重训练。 |
| GPU 调度 | 显式设备配置和继承 mask 检查；没有跨 Lab / 跨 run 的全局 lease。并行实验由操作者分配互不重叠的资源。 |
| Monitor | 独立零依赖 distribution；只读公开投影；无鉴权、多租户与远程作业控制。 |

`rsisys catalog` 返回当前注册名称/版本，不启动模型、训练器或作业；显式 `--plugin`
会执行该受信任文件的注册代码。目录中出现某组件不表示其可选依赖已安装，
也不表示任意 controller / Lab 组合的 schema 与语义兼容。

## 扩展和修改的分界

一般新算法在库外实现 controller 和 Lab；组件替换通过工厂装配。
若需要新的调度、回执交付、恢复、预算中断或资源分配语义，扩展底层库并补充回归。
框架不要求把这些运行保障复制到每个算法注册文件。

多个 SEAL/TTRL 实例应分别设置 Lab 配置的 `output_subdir`，如 `seal-a` / `seal-b`。
默认目录仍为 `seal` / `ttrl`，不能共用已有产物目录。该配置只隔离产物，不分配 GPU。

## 版本与迁移

0.x 期间接口仍可能调整；长期实验固定 Git commit、配置、模型/数据 revision 和依赖。
`rsisys.api` 是推荐导入入口，`src` 内各 Protocol 是对应行为的正式定义。
内部 helper、默认 harness 的私有状态以及历史审查脚本不提供稳定性承诺。

公共接口、配置语义与产物格式发生变化时，在 CHANGELOG 说明影响及迁移步骤。
旧 run 只能用兼容实现和原始组件身份恢复；不能通过改版本字段或盲重放 train 迁移。
CLI `resume` 恢复可移交快照；Python `Application.resume` 使用已有恢复契约；
`--parent-run-id` 只声明来源关系，不继承模型、memory 或算法状态。

`finished` 描述 host 生命周期，`controller_report` 描述方法报告的执行结果。
`simulated`、真实优化器更新和质量提升分别验收，论文收益另按固定评测协议判定。
可选测试的 skip 不是对应硬件/服务分支通过。
