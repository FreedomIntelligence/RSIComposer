# RSI 决策器与 Labs：注册、配置和运行

框架主链路为 **RSI 决策器 → Lab 执行 → 反馈 → RSI 决策器**。
用户只需实现自己的决策器和领域 Lab，再用库外注册文件及 JSON 配置选择运行。
Host 统一提供操作授权、回执记录、作业推进和资源生命周期；meta-agent 是一种可选决策器实现。

## 两个核心对象与内部组件

公开边界为 `RSIController.decide(DecisionInput) -> Decision` 和
`Lab.execute(LabRequest) -> LabFeedback`（另有 describe 能力描述）。公开名称复用
现有 DecisionPolicy / LabExecution / OperationCall / OperationResult 类型，保持原有
序列化、身份和校验。LabFeedback 是状态、指标、产物与日志引用的结构化回执。

register_controller / register_lab 是常规注册入口。方法表示这两个对象的选择与装配，
不是第三种执行对象。harness、memory、验证场景、校验器与训推引擎属于模块内部，
可以复用库内实现，也可以自行实现；不要求所有算法都使用相同的内部组件图。

框架运行时提供决策路由、反馈交付、运行状态与资源清理，并通过已接入的公共端口记录证据。
改变并发调度、反馈交付或恢复等运行语义时，需要修改或扩展底层实现并测试。
注册任意 Python 实现不会自动接管其内部网络调用、计费、后台任务或恢复。

## 从库外运行

安装 Python 3.12+ 的 RSISys 后，将 `examples/registered/` 复制到自己的项目目录：

```sh
rsisys run --config ./single-loop.json --plugin ./plugin.py
rsisys run --config ./custom-controller.json --plugin ./plugin.py
```

这两个配置使用同一个计数 Lab，分别验证固定迭代和根据反馈终止，全程不调用模型。
需要具体 LLM 自训练算法时，使用 [ReST^EM 示例](../experiments/rest_em/README.md)，包含可替换的数学 Lab、CPU fixture，以及 Qwen2.5-7B Base 的 GSM8K / MATH 配置。
它们是 CPU 接口示例；数学后训练可复用内置 math Lab，或注册自己的执行实现。
运行目录相对配置文件解析，每次新 run 使用空目录；也可传 `--data-dir /path/to/new-run`。
`max_decision_steps` 可写在配置里或通过 CLI 覆盖。注册入口默认最多 64 次决策，
达到上限是 `step_limit`，不代表目标完成。后台任务结束与资源清理由实际 Lab 兑现。

## 最小公开接口

所有接入类型可从 `rsisys.api` 导入，不需要导入库内 harness 或修改源码。

```python
from rsisys.api import Decision, LabRequest, LabFeedback, OperationSpec

class MyController:
    def __init__(self, context, config):
        self.target = config["target"]

    def decide(self, input):
        if input.feedback is not None:
            if input.feedback.error:
                return Decision("finish", reason="实验失败，保留回执")
            if input.feedback.data["score"] >= self.target:
                return Decision("finish", reason="观测到目标分数")
        return Decision("invoke", operation=LabRequest("math.iteration", {}))

class MathLab:
    def __init__(self, context, config):
        # 在这里接入自己的执行器、数据和算法状态；构造时不启动实验。
        ...

    def describe(self):
        return (OperationSpec("iteration", "运行一次训练与开发集评测", {
            "type": "object", "properties": {}, "additionalProperties": False
        }),)

    def execute(self, call):
        # 校验领域参数，执行自己的算法，返回真实结果和产物引用。
        # LabFeedback(..., data={"score": measured_score})
        raise NotImplementedError("接入实际训练与评测后再运行")

def register(registry):
    registry.register_controller("my_rsi", MyController, version="1")
    registry.register_lab("my_math", MathLab, version="1")
```

工厂接收 `(AssemblyContext, config)`，可使用 `context.artifacts`、`data_dir`、已装配
后端和执行设施。模型调用须沿用宿主提供的调用及计费边界。配置参数为独立复制的 JSON；
可通过注册时的 `validate_config` 在构造资源前校验，校验函数成功返回 `None`。

`DecisionInput` 提供 `objective`、`feedback`、`last_operation`、`capabilities`、
`context` 与 `references`。`feedback` 是最近操作的结构化回执，`last_operation` 是其操作；
输入副本的修改不影响宿主记录。大型结果可能按现有上下文策略投影，需要沿证据引用读取原文。
`context` 是默认 harness 的补充视图，普通算法可只使用结构化反馈和自己的状态。

`Decision("invoke", operation=...)` 请求 Lab 操作；`wait` 让出当前推进机会；`finish`
表示决策器结束，不自动证明质量提升。操作进入现有授权与回执路径，插件不得直接绕过它执行
其向框架提交的动作。需要替换整个 harness 的高级方法见下文。

## 独立配置

```json
{
  "objective": "Improve mathematical reasoning within the allocated budget",
  "data_dir": "runs/math-001",
  "max_decision_steps": 64,
  "framework": {
    "controller": {"name": "my_rsi", "config": {"target": 0.7}},
    "labs": {"math": {"name": "my_math", "config": {}}}
  }
}
```

运行命令为 `rsisys run --config experiment.json --plugin my_registration.py`。
可重复传 `--plugin` 加载多个宿主插件。JSON 仅选择已注册名称，不从配置执行任意 import。
插件是显式受信任的宿主 Python 代码，必须提供 `register(registry)` 并返回 `None`。
运行身份保留选择项、版本、参数及注册文件 SHA；插件引用的其他代码和资源版本由接入方管理。
重复注册、未知名称或无效参数在执行前失败。

`labs` 可配置多个独立实例，工具统一命名为 `别名.原生操作名`，例如 `math.train`、
`math.evaluate`、`data.prepare`；Lab 收到的操作恢复为自己的原生名称。
`--permissions permissions.json` 使用既有 ActionPermissions 配置，操作授权名称使用完整别名。
工具参数的领域校验由 Lab 在副作用前执行。

Lab 若拥有后台资源，应实现 `shutdown(mode) -> bool` 与可重试、幂等的 `close()`。
`shutdown` 只有确认自有工作退出才返回 `True`。异步执行推荐暴露 start/poll/result/cancel
操作并返回 `OperationResult`；回执完成只表示此次调用完成，作业提交不是训练结束。
若直接返回 `JobRef`，接入方仍须安排既有 `resume_operation` 的终态交付，注册表不自动轮询。

## 内置方案

| 名称 | 类型 | 行为 |
| --- | --- | --- |
| `rrsi` | 决策器 | 固定模型的 harness 进化、质量/成本/稳定性筛选，见 [实验配方](../experiments/rrsi/README.md)。 |
| `rest_em` | 决策器 | 生成 → 二值筛选 → 从初始模型训练 → 评测；详见 [ReST^EM 示例](../experiments/rest_em/README.md)。 |
| `single_loop` | 决策器 | 重复调用指定 Lab 操作，读取回执后再进入下一次迭代；不需要模型。 |
| `meta_agent` | 决策器 | 使用宿主 `decision_maker` 模型选择动作；默认安装显式 memory，可用 `config.memory=false` 关闭。 |
| `math` | Lab | 由数学校验器与 fixture / Transformers 引擎组成；见 [Lab 组件](../src/rsisys/labs/README.md)。 |
| `local_process` | Lab | 复用已有本地程序 Lab，提供代码文件与 start/poll/result/cancel 操作；需宿主 `execution=local_process`。 |

`single_loop` 参数为 `operation`、`iterations=1`、`arguments={}`、`stop_on_failure=true`。
它是单循环控制模板，Lab 的一次 iteration 可以封装某种自训练算法；框架不将这个模板
标成 ReST、STaR 或其他已验证算法实现。失败后结束仍保留失败回执。

`meta_agent` 需要配置 `backends` 与 `roles.decision_maker`，见 `examples/registered/meta-agent.json`。
修改 endpoint、model_id、revision 为实际部署；不要求单独的 actor 模型角色。
同一个 Lab 可以直接从固定算法切换到 meta-agent，不改变执行接口。
`local_process` 使用可信本地进程边界；生产隔离 Lab 可通过同一注册接口接入自己的执行器。

## 选择已适配的 harness / 组装 Lab 内部组件

`meta_agent` 配置支持 `harness=default|dsh`，默认 default。DSH 使用库内现有适配器，
仍接同一 Lab 和宿主运行时；不需要用户再实现一个适配库。选择 dsh 时必须提供原生配置：

```json
{
  "name": "meta_agent",
  "config": {
    "harness": "dsh",
    "memory": true,
    "dsh": {
      "home": "/absolute/path/to/dsh-home",
      "workspace": "/absolute/path/to/workspace",
      "model": "your-dsh-model-route",
      "session_id": "unique-session-id"
    }
  }
}
```

需安装可选依赖 `rsisys[dsh]`，配置实际 provider / endpoint / 凭据环境变量，并提供宿主
roles.decision_maker 模型身份。该片段不是可直接运行的完整配置。DSH 原生 session 的
压缩与恢复有独立契约；memory=true 仍表示已有 ResearchAssembly.single_meta 接线，
不是任意 memory 组件的无条件兼容开关。具体限制见 harness README。

数学 Lab 的 engine、verifier 可以在 Lab 工厂中注入，见 [Lab 组件说明](../src/rsisys/labs/README.md)。
内置数学 Lab 无需插件；[quickstart](../examples/quickstart/README.md) 直接选择库内实现，库外注册入口见 `examples/registered/plugin.py`。

## 高级兼容：整套装配

常规新算法优先注册决策器和 Lab。已有方法需要原生 harness、状态适配或组件图时，
可以继续使用现有低层装配协议；这不是第三类运行对象，也不保证任意运行机制免适配：

```python
from rsisys.api import MethodDefinition

def register(registry):
    registry.register_method("my_complete_method", MethodDefinition(build_modules, "1"))

# build_modules(config) 返回现有 ModuleAssembly。
```

此时配置 `"framework": {"method": {"name": "my_complete_method", "config": {}}}`，
不再同时指定 controller/labs。原生 `MethodCatalog`、`ComponentGraph` 和 Python
`Application(..., modules=...)` 仍是可用的高级接口。

## 结果、状态与验收边界

运行保存 `result.json`、操作账本及装配身份。通用入口不推断领域任务是否真实训练或模拟，
这两个汇总字段为 `null`，须查看原始回执与训练证据。
内置注册组合当前不自动序列化任意算法状态或第三方作业，报告 `snapshot_supported=false`；
恢复须通过整套方法的 `state_factory` / recovery contract 显式实现，并使用现有 Python 恢复 API。
注册成功、CPU 例子通过、模型产生决策，分别是不同层次的验收，不等同于训练收益或递归研究能力提升。

正式接口定义与测试的职责划分见 [API 边界](api-contracts.md)。
