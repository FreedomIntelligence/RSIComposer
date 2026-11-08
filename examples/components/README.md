# 内部组件演示

这些脚本演示调度、反馈、训练后端及并行机制，不是独立的论文算法。
在仓库根目录运行，例如：

```sh
PYTHONPATH=src python3.12 examples/components/coordination.py
PYTHONPATH=src python3.12 examples/components/decision_graph.py --run-dir runs/decision-graph
```

| 机制 | 脚本 |
| --- | --- |
| 调度和循环 | coordination.py、loop_internals.py、rsi_methods.py |
| 决策及研究装配 | decision_policy_lab.py、modular_research.py、decision_graph.py |
| 反馈 | feedback_pipeline.py、feedback_learning.py |
| 执行后端 | backend_training.py、open_lab.py、parallel_experiments.py |
| DSH 适配检查 | dsh_smoke.py |

各脚本头部说明具体依赖、参数及模拟范围。首次使用请先运行 [quickstart](../quickstart/README.md)。
