# ReST^EM quickstart

在仓库根目录使用 Python 3.12+ 执行：

```sh
PYTHONPATH=src python3.12 -m rsisys run --config examples/quickstart/rest_em.json
```

无需 GPU、下载模型、模型服务或插件。配置直接选择库内的 `rest_em` 决策器和 `math` Lab，
执行基线评测及两轮 Generate → Filter → Train → Evaluate，共产生 9 份操作回执。
数据为自编算术题；fixture 引擎模拟生成和训练，回执标记 `simulated=true`、
`training_performed=false`，结果用于检查流程，不表示真实模型能力。

`data_root` 相对执行命令时的工作目录；本配置要求从仓库根目录运行。
默认输出到 `runs/rest-em-quickstart`；再次运行请指定新的输出目录：

```sh
PYTHONPATH=src python3.12 -m rsisys run --config examples/quickstart/rest_em.json --data-dir /absolute/path/to/new-run
```

下一步：[注册自己的模块](../registered/README.md)、[真实模型实验](../../experiments/rest_em/README.md)。
