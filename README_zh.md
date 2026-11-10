# RSISys

**以完整方法为接入单元的递归自改进（RSI）实验运行框架。**

[English](README.md) · [快速开始](#快速开始) · [已接入方法](#已接入方法) · [注册自己的方法](#注册自己的方法)

在一个方法包中定义决策器与 Labs，注册后通过同一个实验闭环运行：

**Controller → Lab 执行 → 反馈 → Controller**

方法可以搜索推理 harness、通过 SFT/RL 更新模型权重，也可以由 LLM meta-agent
决定下一项实验。方法拥有自己的研究策略；RSISys 提供执行生命周期、操作路由与证据记录。

首版研究框架 **0.1.0**，采用 [Apache-2.0](LICENSE)。CPU 入门流程无需 GPU、API key
或模型服务；真实模型实验需要另外准备模型、数据与执行环境。

## 为什么使用 RSISys

- **完整方法一次注册。** 决策器、Labs、workers 和配置放在同一个方法包，支持库内
  或库外开发；方法私有组件无需分别全局注册。通过 `register(registry)` 接入新方法，
  无需在核心 runner 中添加算法专属分支。
- **研究决策归方法所有。** Controller 可以是状态机、搜索策略、LLM meta-agent
  或混合实现，由它决定实验、候选选择和停止。框架不要求所有方法遵循固定的
  propose–train–evaluate–adopt 流程。
- **不同改进机制共用执行框架。** 已接入的 harness 搜索、SFT、RL 方法都通过
  controller/Lab 边界执行。Lab 自定义领域操作，调用原生训推栈；共享 harness、memory、
  backend 和 verifier 按需使用。
- **用证据区分运行与收益。** 方法身份、操作回执、产物引用和方法报告记录实际发生的
  执行过程。`finished` 表示宿主循环结束；训练是否发生、候选是否有效、收益是否成立，
  需要结合方法证据判断。

框架的扩展要求是：**注册完整方法，并运行其决策—实验—反馈闭环**。内部如何拆分与
注入依赖由方法作者决定，不要求任意组件在所有方法之间自由互换。

适合希望保留算法自身实验协议，并通过统一宿主运行 RSI 方法的研究者。
如果只需要一次训练或推理任务，可以直接使用相应原生引擎。

## 快速开始

核心框架和 CPU 示例支持 Linux/macOS、**Python 3.12 或 3.13**。
可选上游集成建议使用 **Python 3.12**，并遵循各自依赖要求。从源码安装：

```sh
git clone https://github.com/satsuki2486441738/RSISys.git
cd RSISys
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
rsisys catalog
rsisys run --config src/rsisys/algorithms/rest_em/configs/cpu/rest_em.json
```

这会运行 ReST^EM 的调度流程，使用**模拟算术 Lab**，不进行 LLM 训练。
命令输出 JSON 运行摘要，并写入 `runs/rest-em-quickstart/result.json`。
结合 `controller_report` 和 Lab 回执判断方法状态；宿主 `status` 本身不代表训练或收益。
核心包没有必装的第三方运行依赖，目前从源码或本地构建的 wheel 安装，不假定已发布到 PyPI。

每次运行使用新的输出目录，例如：

```sh
rsisys run --config src/rsisys/algorithms/rest_em/configs/cpu/rest_em.json \
  --data-dir runs/rest-em-another-run
```

命令行选项可通过 `rsisys --help` 和 `rsisys run --help` 查看。

## 已接入方法

当前 catalog 包含 **8 个完整方法注册项**。`ttrl` 与 `ttrl_verl` 是 TTRL 的两种执行
profile；`meta_agent` 是本框架的方案骨架，不属于论文复现。
`reference`、`paper` 是适配器配置类别，不代表完整论文复现声明。各方法的适配范围如下。

| 方法 | 接入范围 | 配置 |
| --- | --- | --- |
| `rest_em` | 生成、筛选、每轮从初始模型重新训练；轻量 LoRA 实现 | [GSM8K](src/rsisys/algorithms/rest_em/configs/gsm8k-8gpu.json)、[MATH](src/rsisys/algorithms/rest_em/configs/math-8gpu.json) |
| `rrsi` | 固定权重的 harness 进化；保留正则化与 critic，移植到 MATH | [配置](src/rsisys/algorithms/rrsi/configs/math.json) |
| `seal` | self-edit、内层适配、选择、外层 SFT；reference/paper 两种 profile | [参考配置](src/rsisys/algorithms/seal/configs/8gpu.json)、[论文配置](src/rsisys/algorithms/seal/configs/paper.json) |
| `ttrl` | 多数奖励与独立 LoRA GRPO 参考后端 | [配置](src/rsisys/algorithms/ttrl/configs/8gpu.json) |
| `ttrl_verl` | 固定官方全参数后端，连续保持优化器与 reference 状态 | [配置](src/rsisys/algorithms/ttrl/configs/paper.json) |
| `ft_agent` | 部分原生 RD-Agent 策略与本地 GSM8K/LLaMA-Factory 适配；未接完整 CoSTEER 链路 | [配置](src/rsisys/algorithms/ft_agent/configs/gsm8k.json) |
| `aide` | 原生 Agent/search/review/Journal，任务改为固定模型推理程序搜索 | [配置](src/rsisys/algorithms/aide/configs/gsm8k.json) |
| `meta_agent` | 模型提出 harness、读取搜索反馈并锁定最终留出选择；不训练权重，self-reflection 尚未实现 | [配置](src/rsisys/algorithms/meta_agent/configs/math.json) |

### 准备真实模型运行

**GPU/模型服务配置是模板，不是已经准备好的环境。** 配置含示例本地路径、服务模型名
与设备分配。复制配置后编辑 JSON；CLI 不支持任意点号路径的配置覆盖参数。

| 方法类别 | 需要准备 |
| --- | --- |
| ReST^EM | 本地初始权重、训练/评测数据、训推解释器与 GPU 分配 |
| RRSI / meta-agent | Linux、bubblewrap、固定任务模型服务、搜索/留出数据；meta-agent 还需决策模型服务 |
| SEAL / TTRL | 本地权重、方法专用数据与独立训练环境；论文 profile 另需固定版本的上游代码 |
| FT-Agent / AIDE | Linux、bubblewrap、固定版本的干净上游 checkout、模型服务与对应 worker 环境 |

以准备 ReST^EM MATH 配置为例：

```sh
mkdir -p .local/configs
cp src/rsisys/algorithms/rest_em/configs/math-8gpu.json .local/configs/rest-em-math.json
# 编辑 JSON 中的模型/数据路径、训推解释器与 gpu_ids。
rsisys run --config .local/configs/rest-em-math.json --data-dir runs/rest-em-math
```

该配置需修改 `framework.method.config.controller` 下的 `initial_model`、`train_data`、
`eval_data`，以及 `framework.method.config.labs.math` 下的 `few_shot_data`、`gpu_ids`、
`training_python`、`inference_python` 等。训推使用不同环境时应显式填写解释器路径，
未指定时由引擎采用配置或默认解释器。真实训练需明确选择 `engine`，CPU 入门配置用的是 `fixture`。

复制配置时注意路径规则：

- 顶层 `data_dir` 相对 **JSON 文件所在目录** 解析；CLI 的 `--data-dir` 相对
  **当前工作目录** 解析，并覆盖配置值。
- 方法内部路径由各方法处理，移动配置不会自动重定位。内置配置按仓库工作目录编写；
  自己的模型、数据、解释器与上游代码建议使用绝对路径。ReST^EM 也支持 Lab 的
  `data_root`，作为输入路径基准。
- `rsisys-fixed` 等端点模型名必须与实际服务的 served model alias 一致，
  不代表库中附带或会自动下载的权重。

训练需另装 PyTorch/Transformers/PEFT，使用 vLLM 时另备对应推理环境。
框架可选依赖安装到实际导入它们的环境，例如使用 Math-Verify 时执行
`python -m pip install -e '.[math]'`。SEAL/TTRL 论文后端在独立环境中使用
[requirements/seal-ttrl.txt](requirements/seal-ttrl.txt)，SEAL 语义判分另需配置的服务。
FT-Agent/AIDE 分别使用 [FT-Agent](requirements/ft_agent.txt) /
[AIDE](requirements/aide.txt) 依赖；这些文件对应特定 worker 栈，不应视作所有方法共用的一套环境。

数据格式由各 Lab 定义，框架不强制统一 schema。ReST^EM 可参考
[合成 JSONL fixture](src/rsisys/algorithms/rest_em/configs/cpu/data/train.jsonl) 与
[Lab 实现](src/rsisys/algorithms/rest_em/lab.py)；RRSI/meta-agent 使用包含
`id`、`prompt`、`answer` 的 JSONL，搜索集和留出集需分开。
其他格式、所需上游 revision 与接受的参数在对应适配代码中定义。
注册不会自动获取数据、安装 GPU 栈或启动外部服务。

## 注册自己的方法

从可运行的 [_template](src/rsisys/algorithms/_template/__init__.py) 开始：

```text
my_method/
  __init__.py       # register(registry)：绑定完整方法
  controller.py     # 根据反馈决定下一步
  lab.py            # 执行实验并返回回执
  configs/run.json  # 选择方法并设置参数
```

内部布局由方法作者决定，简单方法也可只用一个文件。常规注册直接绑定接收
`(context, config)` 的私有工厂：

```python
from rsisys.api import MethodDefinition
from .controller import MyController
from .lab import MyLab

def register(registry):
    registry.register_method("my_method", MethodDefinition.from_factories(
        controller=MyController,
        labs={"experiment": MyLab},
        version="1",
    ))
```

配置选择完整方法，controller 与 Lab 参数仍分别设置：

```json
{
  "framework": {
    "method": {
      "name": "my_method",
      "config": {
        "controller": {},
        "labs": {"experiment": {}}
      }
    }
  }
}
```

这里仅展示方法选择片段；目标、输出目录和预算等请参考模板的完整运行配置。
可以先运行内置模板，无需 GPU 或服务：

```sh
rsisys run --plugin src/rsisys/algorithms/_template \
  --config src/rsisys/algorithms/_template/configs/run.json \
  --data-dir runs/custom-prompt
```

模板通过字面 prompt 指令检查展示反馈交付与完整闭环，不调用 LLM 或执行训练。

在库外开发一个方法包：

```sh
cp -R src/rsisys/algorithms/_template ./my_method
# 修改 my_method/__init__.py 和 my_method/configs/run.json 中的 custom_prompt 方法名。
rsisys run --plugin ./my_method --config ./my_method/configs/run.json \
  --data-dir runs/my-method
```

复制后保留原名也能运行；向同一 catalog 添加第二个副本前需改名，避免重名。
使用 `--data-dir` 可避免继承模板原先相对仓库布局的输出路径。
方法的 controller、Lab 和内部组件都留在这个目录，无需额外全局注册。

也可放入 `src/rsisys/algorithms/my_method/`，由内置发现机制加载；无需改核心注册表
或 `api.py`。`_template` 本身不参与自动发现。插件执行宿主 Python，只加载可信方法包。

Controller 实现 `decide(DecisionInput)`，返回 `invoke`、`wait` 或 `finish`。
Lab 实现 `describe()` 与 `execute(call)`，返回包含状态、数据和证据引用的
`OperationResult`。多个 Lab 通过 `别名.操作名` 路由；可选配置校验器在构造前拒绝错误参数。
需要自定义装配时，实现 `RSIMethod.build(config)` 并使用
`MethodDefinition.from_method(...)`，或提供返回 `ModuleAssembly` 的工厂。
公共契约见 [rsisys.api](src/rsisys/api.py)。

### 按需复用组件

方法可直接引用共享代码，也可以在自己的工厂内解析已注册组件；私有组件无需额外注册。

| 角色 | 内置注册名 | 实现 |
| --- | --- | --- |
| `harness` | `python_step` | 隔离执行 `(harness_path, problem, history)`；可配置 timeout，需要 bubblewrap |
| `backend` | `chat_pool`、`lora` | 推理服务池、接受已准备输入的 LoRA 引擎 |
| `memory` | `agent_memory` | agent 自己维护的记忆与研究任务，由方法绑定 harness |
| `verifier` | `strict_numeric`、`math_verify` | 最终答案验证；后者需要安装 `rsisys[math]` |

通过 `registry.resolve_component(kind, name, context, config)` 解析；解析不会自动安装
依赖或把组件绑定到方法。自定义共享组件使用 `register_component`。
DSH 可通过 `rsisys.harnesses.decision.dsh.dsh_decision_factory` 和 `rsisys[dsh]`
显式接入；ReMe 检索使用 [requirements/reme.txt](requirements/reme.txt) 与
`ResearchAssembly.reme()`，LangMem 是独立可选 extra。

独立 controller/Lab 注册 API 用于显式自定义组合，默认独立注册 `single_loop` 与
`local_process`。私有组件留在各自方法包中；具体方法类从各自包导入，框架契约从
`rsisys.api` 导入。

## 架构与职责

![RSISys 架构：完整方法包含 Controller 与 Labs，通过决策和反馈构成闭环，按需复用共享组件，由宿主 Runtime 支持执行。](assets/architecture.png)

| 归属 | 职责 |
| --- | --- |
| 方法 | Controller 策略、Lab 操作、候选选择、停止与采纳规则 |
| Lab | 领域校验、训推执行、指标与产物 |
| 宿主 | 注册、运行生命周期、授权、路由、回执交付、资源归属与显式恢复集成 |

```text
src/rsisys/
  algorithms/       # 完整方法包、开发模板与共享实现
  loops/            # 契约与循环机制
  harnesses/        # 可选决策/actor 执行、记忆与反馈服务
  labs/             # 通用进程 Lab 和可复用场景
  backends/         # 训推引擎
  runtime/          # 宿主调度、执行、证据与存储
  snapshot/         # 捕获与显式恢复集成
  api.py            # 公共框架入口
  methods.py        # 方法声明
  registry.py       # 注册目录与受信任插件加载
```

Controller 与 Lab 不必使用 LLM 或内置 harness。共享组件可选，核心负责生命周期语义，
研究策略由方法决定。有后台资源的 Lab 应提供幂等 `close()` 与 `shutdown(mode) -> bool`；
异步 Lab 可暴露 start/poll/result/cancel，由方法驱动其协议，注册不会自动轮询。

`runs/` 与 `runtime-output/` 保存本地输出并由 Git 忽略。
源码中的 `runtime/`、`snapshot/` 是运行机制，不是已保存的实验产物。

### Self-reflection：后续方向

[Self-Reflection Lab](src/rsisys/labs/scenes/self_reflection/__init__.py) 是可选架构占位
（`0-placeholder`），目标是通过积累的决策—Lab 反馈提升 meta-agent 自己的决策能力。
预期流程为经验积累 → 候选构建 → 长程验证，再由 Controller 决定是否采纳；
验证可以保持待完成或证据不足。当前合法操作返回 `self_reflection_not_implemented`，
不实现经验持久化、训练、验证或决策器切换。
可运行的 `meta_agent` 方法搜索的是任务 harness，尚未实现上述自身改进闭环。
Self-reflection 是后续研究方向，当前版本尚未实现。

## 监控

可选只读查看器，与核心包独立安装：

```sh
python -m pip install ./src/rsimonitor
rsimonitor --runs-dir runs
```

默认仅监听本机，不提供鉴权或多租户隔离。

## 发布范围与限制

- 当前是首版研究框架，0.x 接口可能调整；实验需固定 commit、模型/数据 revision 与依赖。
- 快照恢复需要显式状态适配。注册方法不会自动恢复任意 controller 内存或外部任务。
  FT-Agent/AIDE 原生搜索和同步训练配方需要新的运行目录。
- GPU 由操作者分配；目前没有跨 Lab/run 的全局 GPU lease，长作业计费与预算接管尚不完整。
- 插件是受信任宿主 Python。普通 local-process Lab 不是 OS 沙箱；不可信生成程序需显式
  隔离执行器。自定义 Lab 负责外部访问、计费与资源退出。凭据放环境变量，分享前检查
  提示与产物中的私有内容。

## 参与贡献

```sh
python -m pip install -e '.[dev]'
ruff check --select E9,F63,F7,F82 src
python -m build --outdir dist/core .
python -m build --outdir dist/monitor src/rsimonitor
python -m twine check --strict dist/core/* dist/monitor/*
```

通过 [GitHub Issues](https://github.com/satsuki2486441738/RSISys/issues) 提交问题、功能建议
或方法接入疑问。报告问题时提供 commit/包版本、Python/操作系统、脱敏最小配置、
失败命令与错误文本，不附私有数据或凭据。

贡献新方法时，将实现与配置放入一个算法包，说明适配边界与上游来源，并给出别人可执行的
最小命令。PR 说明行为变化与相应验证；上方命令可在公开仓库中执行，完整回归测试暂不公开。
仓库不接收实验结果、过程记录、下载数据、权重或原始运行日志。
漏洞优先使用 GitHub 私密报告；不可用时请求私密联系渠道，不公开漏洞细节或凭据。

## 许可

Copyright 2026 RSISys contributors，采用 [Apache-2.0](LICENSE)。
第三方来源在对应适配代码旁标注；SEAL 改编代码的 MIT 文本保留在
[seal/lab.py](src/rsisys/algorithms/seal/lab.py)。依赖、模型权重与数据集分别遵循其条款。
