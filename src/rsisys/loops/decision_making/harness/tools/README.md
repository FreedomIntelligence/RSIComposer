# Decision tools

[默认 Decision Harness](../README.md) 的执行绑定，接收 Action Space 已校验的本地 research 操作。tools 是默认实现细节；替代 Harness 无需暴露独立 tools 接口。

## 接口与实现

[DecisionTools](__init__.py) 提供 `execute(call: OperationCall) → OperationResult | JobRef`。`BoundDecisionTools(bindings)` 将操作名映射到接收 arguments 的处理函数，复制参数并检查返回类型；它不负责描述 Action Space 或调度系统指令。

[LocalDecisionTools](local.py) 接收 FileArtifactStore：prepare_data 保存 text 为 input_data；search_literature 对已显式导入、kind=literature 的 UTF-8 文本作不区分大小写的子串检索，最多 20 条，保存 literature_results；store_strategy 保存 strategy。三者均为同步本地操作，simulated=False，不包含联网检索或 PDF 解析。

`DemoDecisionTools(sandbox, artifacts)` 只支持 prepare_data：在 sandbox 中执行 echo 并轮询一次，成功后保存 demo_input，simulated 取自后端；失败抛异常，最终释放 sandbox。

## 扩展与回填

新增操作需在平级 [Action Space](../../action_space/README.md) 提供定义和校验，再向工具绑定注入处理函数。异步操作可返回 JobRef，由外部通过 `loop.harness.resume(job, result)` 回填终态结果；Harness 不自动轮询、取消或重做准备操作。记录失败的补写边界见 Harness README。
