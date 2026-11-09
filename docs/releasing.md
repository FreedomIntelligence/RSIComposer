# 发布流程

首次公开可以作为 **0.1 研究版**：公开注册框架、可组合组件、CPU 入门、真实模型配方
与如实标注的结果。现有 runtime 提供部分异步作业和显式恢复设施；
把这些保障接入所有同步 GPU 方法、恢复原生搜索状态，以及全局 GPU 调度仍是后续工作。
不应把当前版本描述成已具备这些通用保证的生产训练平台。

## 可重复的发布验收

在目标 commit 的干净工作树执行。Python 3.12/3.13，Node 22：

```sh
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m unittest discover -s tests -t . -q
node --test tests/monitor/frontend/*.test.js
ruff check --select E9,F63,F7,F82 src tests examples experiments tools
python tools/check_docs.py
python -m build --outdir dist/core .
python -m build --outdir dist/monitor src/rsimonitor
python -m twine check --strict dist/core/* dist/monitor/*
python tools/smoke_wheels.py --core-wheel dist/core/rsisys-0.1.0-py3-none-any.whl \
  --monitor-wheel dist/monitor/rsimonitor-0.1.0-py3-none-any.whl \
  --core-sdist dist/core/rsisys-0.1.0.tar.gz
git diff --check
```

默认 `build` 同时构建 sdist，并从 sdist 构建 wheel。安装验收在仓库外的全新 venv 中
运行，清除 `PYTHONPATH`，只安装给定本地 wheel；验证零额外运行依赖、注册目录、
CPU ReST^EM、库外自定义 controller / Lab、monitor 独立安装及许可/静态资源打包。
CI 对 Linux/macOS × Python 3.12/3.13 跑离线回归，并单独执行包验收。
可选 ML/服务集成没有被这些零依赖 CI 替代。

## 公开前的仓库步骤

1. 确认当前 PR/commit 的 CI 实际通过；历史 run 的通过不替代本次检查。
2. 维护者确认有权以 Apache-2.0 发布拟公开的代码，保留第三方 notices；不要把该许可
   扩展到模型、外部数据或无权重新许可的贡献。
3. 检查当前树和拟公开 Git 历史：凭据、私有数据、原始请求、内部端点、用户信息与
   大文件。历史审查材料要单独看；删除当前文件不能清除历史。不要公开未审查的
   `.local/` 证据、模型或数据。凭据若曾提交须先撤销，单纯删文件不够。
   原始审查材料已移出当前树，备份与历史定位见[归档说明](review/README.md#archived-evidence)。
4. 合并已验证的发布分支，核对 README、版本、CHANGELOG 和 GitHub 默认分支。
5. 在 GitHub 启用私人漏洞报告；检查 Issues、Actions 和分支保护设置。公开可见性
   由维护者在发布时明确切换，本地准备和分支推送不会自动执行该动作。
6. 创建明确指向已验收 commit 的标签和 GitHub release，列出范围、限制、依赖和
   实验差异。尚未标记发布的准备提交不称为已经发布。

GitHub 公开仓库足以启动社区研究，不要求同时发布 PyPI。若发布 PyPI，先确认包名
归属、凭据/Trusted Publishing 与目标平台；当前 CI 只构建和验证，不上传包。
ReMe 的固定 Git runtime 独立位于 `requirements/reme.txt`，不写入包依赖元数据。

## 后续工作顺序

- 优先：为原生搜索与同步训练方法接入持久化状态、作业 reconciliation，以及不重复训练的恢复。
- 随后：将同步 GPU Labs 接到 submit/poll/cancel 与心跳边界，补齐 host 预算对长训练的中断和统一成本记录。现有进程作业能力可复用，但尚未覆盖这些方法。
- 再后：跨 Lab/run GPU lease、可选公共 artifact manifest，以及超出 TTRL 专用桥接的可复用 verl/FSDP 引擎。
- 新算法在同一注册边界开发；遇到共同的运行保障缺口再补核心，避免为新算法增加平行运行器。

这些工作不阻塞定位清楚的首次研究版；若承诺无人值守长程可靠恢复或多租户训练，
则须先完成并验收相应能力。
