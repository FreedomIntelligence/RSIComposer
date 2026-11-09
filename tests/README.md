# 测试导航

这里是长期维护的自动化验证代码。正式接口定义在 [src/rsisys](../docs/api-contracts.md)，
`tests/support` 仅提供测试辅助，库及示例不依赖 tests。
契约测试验证这些接口的行为，属于开源仓库的一部分；临时探针和运行日志放在忽略的 `.local/`。

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
测试数量会随开发变化，以本次运行输出为准；新增算法在对应模块目录补充行为测试。
skip 表示所需环境未启用，不表示对应分支通过。

SEAL / TTRL 的决策回执及注册运行验证位于 `controllers/test_weight_evolution.py` 和
`integration/test_weight_framework.py`；各 Lab 的泄漏、组件和产物检查位于 `labs/`。
`integration/test_weight_objectives.py` 在安装 PyTorch / PEFT 后可在 CPU 上验证真实梯度及
BF16 合并舍入；未安装时跳过，不下载模型。

FT-Agent / AIDE 的离线注册与边界验证位于 `registration/test_ft_agent.py` 和
`registration/test_aide.py`。真实上游 FT 策略测试需要可选 PyYAML/Jinja2 及固定 checkout，
可通过 `RSISYS_FT_UPSTREAM` 指定路径；AIDE 原生协议测试使用
`AIDE_UPSTREAM_ROOT` / `AIDE_WORKER_PYTHON`，并要求 Linux bubblewrap。
安装与运行条件见 [FT-Agent](../experiments/ft_agent/README.md) 和
[AIDE](../experiments/aide/README.md)。这些协议测试不替代真实模型实验。
