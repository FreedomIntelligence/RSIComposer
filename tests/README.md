# 测试导航

这里是长期维护的自动化验证代码。正式接口定义在 [src/rsisys](../docs/api-contracts.md)，
`tests/support` 仅提供测试辅助，库及示例不依赖 tests。

从仓库根目录运行完整离线回归（Python 3.12+）：

```sh
PYTHONPATH=src python3.12 -m unittest discover -s tests -t . -q
node --test tests/monitor/frontend/*.test.js
```

按模块运行，例如：

```sh
PYTHONPATH=src python3.12 -m unittest discover -s tests/controllers -t . -v
PYTHONPATH=src python3.12 -m unittest tests.registration.test_rsi_modules -v
```

| 目录 | 职责 |
| --- | --- |
| controllers、labs、datasets | 算法决策器、数学 Lab 及数据划分契约 |
| registration | 注册、装配及库外插件边界 |
| decision、runtime、feedback、memory | 通用框架行为与组件契约 |
| monitor | 监控服务与 frontend 下的 Node 测试 |
| regressions | 跨模块审计与故障回归 |
| experiments | 医疗及 Qwen 场景的装配回归 |
| integration | quickstart 端到端回归，以及可选真实模型 / 服务验证 |
| support | 测试专用数据与辅助组件，不从示例插件获取库测试夹具 |

CPU LoRA 验证需要已安装 PyTorch、Transformers、PEFT，使用本地随机小模型、不下载权重：

```sh
CUDA_VISIBLE_DEVICES= RSISYS_REST_EM_CPU_INTEGRATION=1 PYTHONPATH=src \
  python3.12 -m unittest tests.integration.test_rest_em_cpu -v
```

符号校验集成通过 RSISYS_MATH_VERIFY_INTEGRATION=1 启用；真实服务回归的环境变量见
[integration/test_live_generation.py](integration/test_live_generation.py)。未显式启用的外部依赖测试会跳过。
目录迁移保留了原有 2025 项测试；新增算法继续在对应模块目录补充行为测试。
