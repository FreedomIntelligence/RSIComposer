# Lab 与内部组件

系统边界为 `Lab.describe()` 和 `Lab.execute(LabRequest) -> LabFeedback`。
这些公开名称复用现有 LabExecution / OperationCall / OperationResult 契约，原有实现继续可用。
反馈包含完成/失败状态、指标、产物和日志引用；任务提交回执不表示训练已经完成。

`Registry.register_lab(name, factory, version=...)` 注册一个 Lab 工厂。用户可以直接实现
整个 Lab，也可以在工厂内部组装本目录的组件。校验器、训练引擎和推理引擎不增加顶层注册对象。

## 数学 Lab

`MathLab` 提供 generate / filter / train / evaluate 四种能力。它不自己推进 RSI 轮次，
也不替决策器选择训练模型；请求中的 initial_model 是该次 SFT 的起始快照。
ReST^EM 决策器负责每轮都指定最初的快照，以及选择本轮数据和下一轮生成模型。
Lab 检查数据来源、训练与留出集重叠、产物完整性，并返回结构化反馈。
当前接口为同步执行；后台作业调度需要另一个实现现有任务契约的 Lab。

内置注册名为 `math`，配置 `engine=fixture|transformers|vllm`；旧 `mode` 字段兼容。
`data_root` 默认为启动命令所在目录，推荐 Python 装配时传入明确的 `input_root`，或在配置中
填写绝对 `data_root`。所有相对模型/数据路径在该根目录解析。产物位于 run 的 `math-lab/`。
一个运行内装配多个数学 Lab 时，分别配置不同的 output_subdir，避免产物目录冲突。

可复用组件：

| 组件 | 入口 | 责任 |
| --- | --- | --- |
| `MathLab` | `rsisys.labs.math` | 操作与反馈、数据和产物检查 |
| `StrictNumericVerifier` | `rsisys.labs.math_verifiers` | 严格数值答案解析与二值正确性判断 |
| `MathVerifyVerifier` | `rsisys.labs.math_verifiers` | 可选 Math-Verify，提取最后一个 boxed 答案并做符号等价判断 |
| `FixtureMathEngine` | `rsisys.labs.math_engines` | 明确标记的 CPU 协议模拟 |
| `ParallelMathEngine` | `rsisys.labs.math_parallel` | 离线 vLLM 多卡推理与 DDP LoRA SFT |
| `TransformersMathEngine` | `rsisys.labs.math_engines` | 本地推理与从给定初始快照开始的 LoRA SFT |

`MathVerifier` 要求 `name`、`parse(text)` 和 `is_correct(completion, answer)`；
`MathEngine` 要求 `name`、布尔 `simulated`、`generate(ref, rows, count, round_number, sample)`
和 `train(ref, rows, output, round_number)`。train 成功返回代表权重已完成保存；组件应自行
清理单次调用资源。内置引擎不持有后台任务；自定义工厂获得额外资源时须立即用
`context.add_owned_cleanup(...)` 注册退出清理。异步或共享资源管理需要显式适配。

例如替换校验器并复用执行实现：

```python
from pathlib import Path
from rsisys.labs.math import MathLab
from rsisys.labs.math_engines import TransformersMathEngine

# MyVerifier 由用户实现，需按自己的任务验证准确性。
def build_lab(context, config):
    root = Path(config['data_root']).resolve()
    engine = TransformersMathEngine(config, input_root=root)
    return MathLab(context, config, engine=engine,
                   verifier=MyVerifier(), input_root=root)

def register(registry):
    registry.register_lab('my_math', build_lab, version='1')
```

组件 name 应包含版本；实际运行 manifest 记录引擎和校验器名称。Python 实现及其依赖的版本
仍由注册方固定，不能仅凭同名假设行为相同。模型调用或远程引擎的计费需接宿主账本，
任意第三方 SDK 调用不会自动被框架拦截。

数值校验器适合作为 GSM8K 接入起点，不覆盖完整 MATH 的符号等价判断。
MATH 配置 `verifier=math_verify`、`answer_format=latex`，需安装 `rsisys[math]`。
可选符号依赖只在实例化该校验器时加载；默认数值 Lab 和框架主体不依赖它。
符号校验拒绝无合法最终 box 的答案，不能用推导中的中间表达式代替最终答案。
Transformers 引擎目前每次训练重新加载指定初始快照、新建 adapter 和 optimizer；不支持
从上一轮 LoRA 继续训练。支持批量生成、训练 microbatch 和梯度累积，默认各为 1；
配置键为 generation_batch_size / train_batch_size / gradient_accumulation_steps。
prompt_format 默认为 chat；Base 模型可设为 completion，并通过 few_shot_data 提供固定
JSONL 示范（prompt / answer）。示范不能与目标问题重叠。两种格式都仅对生成 completion 计损失。
answer_format 默认为 numeric，latex 时提示模型输出 boxed LaTeX 最终答案。
需要其他训练语义时替换引擎。Qwen3.5-9B 已完成单卡真实 GPU 预跑，
实际质量收益需查看独立运行报告，不能用 fixture 或预跑推断。

多卡配置与独立依赖环境见 `experiments/rest_em/configs/math-8gpu.json` 和该目录 README。

## 固定权重的 harness 实验

`harness_math` Lab 执行 Python harness 并测量固定模型；RRSI 决策器负责提出、审查和选择候选。
Lab 不决定采用哪个版本，也不更新模型权重。可替换的模型访问和隔离执行组件位于
[harness_runtime.py](harness_runtime.py)，实验依赖及运行方式见
[RRSI 配方](../../../experiments/rrsi/README.md)。
