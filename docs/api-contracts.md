# 正式接口与测试的边界

正式接口定义属于 `src/rsisys`。外部插件只需安装 RSISys，不需要 examples、experiments 或 tests。

| 接口 | 正式定义 | 测试职责 |
| --- | --- | --- |
| 决策器：`decide(DecisionInput) -> Decision` | [decision.py](../src/rsisys/loops/contracts/decision.py)、[公开别名](../src/rsisys/loops/contracts/rsi.py) | 验证合法决策、输入隔离、反馈及失败处理 |
| Lab：`describe()`、`execute(LabRequest) -> LabFeedback` | [execution.py](../src/rsisys/loops/contracts/execution.py)、[回执](../src/rsisys/loops/contracts/coordination.py) | 验证操作描述、执行、反馈真实性和生命周期 |
| 模块注册与装配 | [registry.py](../src/rsisys/registry.py) | 验证配置、插件加载、模块替换与库外运行 |
| 可复用数学校验器与训推引擎 | [labs](../src/rsisys/labs/README.md) | 验证各组件遵守库中定义的行为 |

统一导入入口为 `rsisys.api`，开发指南见 [registration](registration.md)。
`tests/support` 只提供测试数据、假后端和测试辅助；不承担正式接口或运行逻辑。
测试中的 contract/schema 命名表示“验证契约或 schema 的行为”，不表示契约定义位于测试中。

临时排查代码和验证输出放入被 Git 忽略的 `.local/`；正式测试保留在 `tests/`。
历史审计记录位于 `docs/review`，其中的探针记录不作为库接口或用户运行依赖。
