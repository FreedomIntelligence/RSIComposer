# Decision space

定义并执行决策者可用的准备操作，结果返回所属 [loop](../README.md)。

## 接口

[DecisionSpace](__init__.py) 提供 `describe() → tuple[OperationSpec, ...]` 和 `execute(call: OperationCall) → OperationResult | JobRef`。OperationCall 包含 name、arguments；OperationResult 包含 summary、artifacts、simulated、status（默认 COMPLETED）和可选 error。结果必须为 COMPLETED / FAILED / CANCELLED；FAILED 必须有 error，COMPLETED 不能有 error。

execute 是同步调用，可返回异步任务 JobRef。调用方通过 `loop.resume(job, result)` 回填终态 OperationResult，job 必须匹配当前等待任务；loop 不自动轮询或取消，也不重复执行。操作抛异常时，loop 尝试记录失败经验后重抛原异常。

`ModularDecisionSpace(operations)` 接收操作元组，每项提供 spec 和 `execute(arguments)`，名称必须唯一。它校验调用类型、参数为 dict、操作存在及返回类型；具体参数校验与副作用由操作负责，spec 中的 JSON Schema 仅用于描述。

## 当前实现

[local.py](local.py) 的 `LocalDecisionSpace(artifacts: FileArtifactStore)` 组合三个同步操作，均只接受表中单个字符串参数，结果标记 simulated=False：

| 操作 | 参数与行为 |
| --- | --- |
| prepare_data | text（可为空）：保存为 input_data 产物。 |
| search_literature | query（不可全为空白）：对已导入、kind=literature 的 UTF-8 文本作不区分大小写的子串检索，最多 20 条；返回 literature_results JSON 产物。 |
| store_strategy | strategy（不可全为空白）：保存为 strategy 产物。 |

`DemoDecisionSpace(sandbox, artifacts)` 仅支持 prepare_data(text)，在注入的 sandbox 执行 echo 并轮询一次；未成功则抛异常，最终关闭 sandbox。成功时保存 demo_input 产物，simulated 取自执行结果，并非固定为模拟。

[literature.py](literature.py)、[data.py](data.py)、[strategy.py](strategy.py) 是专项协议，不是实际联网/清洗实现。

## 扩展

新增操作后注入 ModularDecisionSpace，或替换整个 DecisionSpace。PDF 解析、在线检索、数据生成与质量筛选由具体操作实现。
