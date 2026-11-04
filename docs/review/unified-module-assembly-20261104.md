# 统一模块装配与状态接入

日期：2026-11-04。范围：本地工作树；继承已有的 decision、feedback、Lab 接口调整。

## 本次实现

| 边界 | 统一实现 | 模块自己的责任 |
|---|---|---|
| 装配 | `ModuleAssembly` → `assemble_runner` | 工厂只构造，不执行研究 |
| Decision / Exploration / Lab | 现有公开端口和工厂 | 自己的算法、工具、模型与领域协议 |
| Feedback | 同一 `ResearchAssembly / assemble_research` | context/memory 必须激活，其他产物按配置启用 |
| 状态 | `state_factory` → `ComponentState` → `AssemblyRecovery` | 完整导出内部状态、依赖、身份、配置与已有任务引用 |
| 快照 | 同一状态端口，额外 `validate_snapshot()` | 确认空闲、内容可迁移、外部引用可解析 |
| 收尾 | `AssemblyRunner / AssemblyContext` | 自有资源关闭可重试，不操作别人的任务 |

Application、内存 demo、医疗 `build` 和 Qwen 控制器的 Runner 构造均使用 `assemble_runner`。
医疗长实验仍在领域代码里配置 ResearchPolicy / DSH、固定评测协议和训练初始化器，再通过相同的反馈入口绑定到装配 owner；收尾调用统一 owner。
Qwen 保留其特定的监控与 worker 协议，不会因迁移入口而自动获得医疗规则或反馈模型调用。
历史 `.local` supervisor、隧道和冻结 campaign 没有迁移或重启。

## 使用入口

```python
from rsisys.application import Application
from rsisys.assembly import ModuleAssembly
from rsisys.research import ResearchAssembly

modules = ModuleAssembly(
    decision_factory=make_decision,
    lab_factory=make_lab,
    research=ResearchAssembly(context_memory=my_memory),
    feedback_adapter_factory=make_feedback_adapter,
    state_factory=make_state_adapters,
)
with Application(config, modules=modules, durable=True) as app:
    app.runner.step(config.objective)

with Application.resume(config.data_dir, modules=modules) as app:
    app.runner.step(config.objective)
```

`Application` / `build_demo` 的旧工厂参数继续兼容；不能同时传 `modules=` 和旧模块参数。
`feedback_options` 可传反馈调用限制、model、path 等实例参数，不能覆盖 owner/backend/adapter 等绑定。
持有自有资源的工厂应在取得资源后立即向 context 注册 cleanup。

## 状态端口

`make_state_adapters(context, decision, exploration)` 返回具名适配器：

```python
class MyState:
    state_id = 'my-component/v1'

    def configuration(self):       # 可选，参与恢复兼容性比对
        return {'mode': 'my-mode'}

    def capture(self):
        return my_component.export_state()  # 只读；经框架显式 JSON codec

    def restore(self, state):
        my_component.restore_state(state)   # 校验/接回已有状态，不提交工作

    def validate_snapshot(self):   # 仅需要可迁移快照时实现
        my_component.assert_quiescent_and_portable()
```

必须显式覆盖 `decision`、`exploration`，以及已启用的 `lab_execution` 和 `feedback`；
无状态组件也提供明确的空状态适配器。可增添其他具名端口。
顶层适配器可管理其 policy、工具、memory、后端、权限及提示词配置等内部依赖，也可将这些依赖拆成额外端口。
框架不规定这些内部状态的组织形式，不遍历任意 Python 对象，也不复制模块未知目录。
私有文件、内存数据库、计数器、幂等键和回执若要迁移，必须由所有者导出或提供稳定引用。
适配器必须将代码/配置变化体现到 state_id、configuration 或自身严格的恢复校验；框架源码指纹不会替第三方插件做版本管理。

完整 inventory、全部适配器版本与配置先比对，再调用 restore。未注册组件或适配器在运行中被替换，下一次 durable 边界会在执行前拒绝。
组件内部依赖的一致性由其状态适配器校验，host 不解析插件私有结构。
原有全组合 `RecoveryContract` 仍可使用；显式契约现在也能接入 Lab 原地恢复，取消了原先即使有契约也一律拒绝的限制。

## 恢复与快照

`durable=True` 在提供 `state_factory` 时自动选择 `AssemblyRecovery`；没有适配器的默认组合继续使用原内置恢复。
未提交 intent 保持不确定状态，必须 `reconcile(..., evidence=...)` 后才能继续。
恢复不重新发起工具、训练、推理或外部任务。预算、outcomes、计时和已有失败记录延续，停机时间计入原预算。
Runner 最近一次 execution shutdown 错误也进入状态，旧记录缺少这个新增字段时兼容为 None。
workspace 的 `module_assembly.json` 防止恢复时漏传模块装配而静默回落默认实现。

```python
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot

manifest = LocalSnapshotCapture(app).capture('idle')
snapshot = app.config.data_dir / 'snapshots' / manifest.snapshot_id
restored = restore_snapshot(snapshot, new_workspace, modules=modules)
```

同一端口可用于 `component_quiescent` 快照，但所有适配器都必须实现 `validate_snapshot()`。
存在在途工作、不确定 journal、不能迁移的私有资源或缺失能力时拒绝捕获；不能把有 capture/restore 当成可迁移证明。
组件快照和默认快照共用文件清单与 SHA 检查、源码/Python 校验、原子发布、锁与 inode 所有权保护。
恢复失败且 owned cleanup 仍未确认时保留目标 workspace；不删除仍属于未收尾资源的目录。
组件快照保存同一 run 身份；任意插件内部的来源引用不能通用改写，因此通用路径拒绝 `run_id=` 分叉。

## 验证与范围

`tests/test_module_assembly_state.py` 用完全没有默认 policy/state/memory 字段的外部 harness 和原生 Lab 验证：

- 构造无模型调用、无执行提交；统一 close 不重复关闭已完成资源。
- 原地恢复计数、既有 Lab 回执、私有 memory；恢复不重新提交工作。
- 版本/配置不符时，在任何 restore hook 前拒绝；缺失 Lab/feedback 端口明确失败。
- 未注册组件替换在执行前阻断；中断边界必须核对，不能通过 snapshot 绕过。
- 新目录快照恢复产物和私有 memory；缺失可迁移声明、活跃工作或篡改字节均拒绝。
- 恢复保留原时间预算、清理失败证据与 run 身份；失败清理未完成时保留 workspace。

领域回归沿用医疗、Qwen CPU/HTTP fixtures，包括固定初始化、候选评测/采纳、调用限额、停止和取消等行为。
这些测试不表示完成真实 SFT/RL、NPU 运行或验证某个外部系统的 session 恢复。
原 LangMem/DSH/进程训练适配器不会仅因接口统一就自动变成可迁移组件，仍需它们各自的完整状态适配器。
当前支持装配时替换；任意运行中热替换、并行 durable 恢复、外部权重/优化器迁移不在本次承诺内。

本轮测试结果与 SHA 见 [验证记录](repros/unified-module-assembly-validation.json)。
