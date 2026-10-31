# Environment

拥有任务语义和执行副作用，接收校验后的 Action，产生下一次观察。所属流程见 [默认 Exploration Harness](../README.md)。

## 接口

[Environment](__init__.py) 提供 reset(Directive) → observation、step(Action) → Transition、close()。Transition 包含 observation、terminated、simulated、可选 reward、产物和 error。

环境终止、步数耗尽和执行错误分别产生 completed、partial、failed Feedback；reward 保留在 Actor memory 的本地轨迹中，缺失时保持 None；原始观察与行动不写入 Experience。默认 Harness 在异常路径也调用 close，清理异常不会覆盖原始执行异常。

## 当前实现

- SandboxEchoEnvironment：要求 sandbox 立即完成 echo，模拟标记取自后端；应用默认配 MockSandboxBackend。
- [LocalProcessEnvironment](local.py)：真实执行 echo / execute，每个任务执行一次行动即终止。输入放入 `inputs/<artifact_id>`，合并保存 stdout/stderr；成功时回收声明的 output_paths，缺失输出视为失败。

Feedback 的 simulated 汇总指令、policy 和环境标记；真实本地进程不保证整轮反馈为非模拟。

执行进程由 [Runtime](../../../../runtime/README.md) 提供；环境定义任务规则，执行后端管理进程资源。

## 替换

实现三个方法并注入 DefaultExplorationHarness 的 environment 参数，配套注入支持的 ActionSpace。内置环境调用是同步的；长执行需设置后端超时，跨 loop 的下一 tick 无法中途抢占当前同步调用。
