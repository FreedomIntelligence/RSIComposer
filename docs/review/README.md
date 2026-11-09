# 审查记录与历史证据

这里保留带日期的设计讨论、故障归因和验收结论。
历史记录可能包含旧端点、当时的命令和本机路径，不是当前 API 规范；
开发新方法以[公共架构](../architecture.md)、[注册指南](../registration.md)和
[支持范围](../compatibility.md)为准。

| 主题 | 入口 |
| --- | --- |
| 最新开源整理与剩余限制 | [2026-11-09 再审视](open-source-refresh-20261109.md) |
| 首轮发布准备与验收 | [开源准备](open-source-readiness-20261109.md) |
| 方法来源、适配程度与证据索引 | [方法支持表](../methods.md) |
| SEAL / TTRL 本地偏差归因 | [复现审查](weight-reproduction-diagnosis-20261109.md) |
| 原生训练后端的适配验证 | [论文协议适配](weight-paper-adaptation-20261109.md) |
| FT-Agent / AIDE 的接入边界 | [meta-agent 接入审查](meta-agent-integration-20261109.md) |

## Archived evidence

2026-11-09 从当前源码树移出了 `repros/`、`prompt-snapshots/`、`audit-20261104/`
中的 131 个文件，共 1,343,952 bytes：原始请求、响应、日志、临时探针及其验证输出。
长期维护的回归保留在 `tests/`；实验总结和当前引用的复现诊断继续保留。
未来临时证据统一写入忽略的 `.local/`，不进入 Git 或发行包。

移出前对原 `docs/review` 的 171 个文件做了完整归档，并逐文件校验 SHA256：

- 本地归档：`.local/open-source-refresh-20261109/review-original-6a411a7.tar.gz`。
- 文件清单：同目录的 `review-original-manifest.json`，包含每个文件的大小和 SHA256。
- 归档 SHA256：`83bd376b915530a41ddf2721203c5ca44683c101d07907d28c6307418aea15a4`。
- 原始 Git 位置：`6a411a762568ab94f36ed657590c217bf2717ddc` 的 `docs/review/`。

本地归档不随公开仓库分发。保留的历史笔记中，原始材料链接改指本节，
链接文字、当时的命令与审查结论仍保留；需要完整来源时按清单或原 Git 版本定位。

此次只整理当前树，没有改写 Git 历史。原始材料仍可能存在于拟公开的旧 commits 中；
维护者公开历史前仍需审查其披露范围，删除当前文件并不清除历史。
