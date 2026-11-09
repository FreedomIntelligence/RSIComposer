# 文档导航

第一次使用：先运行[首页 quickstart](../README.md#quickstart)，再复制
[库外注册示例](../examples/registered/README.md)。算法通过注册接入框架，
无需从内部双端点实现或医学实验开始阅读。

| 目标 | 文档 |
| --- | --- |
| 理解两个模块和框架职责 | [统一架构](architecture.md) |
| 选择算法、确认原生适配程度与实跑证据 | [方法支持与验证范围](methods.md) |
| 开发自己的 controller / Lab | [注册与配置](registration.md)、[正式 API](api-contracts.md) |
| 确认支持范围、恢复和版本兼容 | [兼容与支持](compatibility.md) |
| 替换内部组件 | [组件组合](component-composition.md)、[Lab 组件](../src/rsisys/labs/README.md)、[Memory 后端](memory-backends.md) |
| 了解默认 meta-agent、reflect 与旧端点兼容 | [默认实现与高级架构](advanced-architecture.md) |
| 接入模型、权限、后端和快照 | [高级使用指南](usage.md) |
| 选择真实模型实验 | [实验导航](../experiments/README.md) |
| 监控已有运行 | [独立 monitor](../src/rsimonitor/README.md) |
| 修改和验证源码 | [贡献指南](../CONTRIBUTING.md)、[测试导航](../tests/README.md) |
| 准备公开版本 | [发布流程](releasing.md)、[最新开源整理](review/open-source-refresh-20261109.md) |
| 理解信任与执行边界 | [安全说明](../SECURITY.md) |

`review/` 的[审查记录与归档说明](review/README.md)以及 `development-history.md` 是带日期的历史记录，可能保留旧术语、路径和
当时的支持范围，不是当前使用契约。以这里的架构、注册指南和实际发布版本为准。
临时排查脚本、原始运行证据和本机环境放入忽略的 `.local/`。
