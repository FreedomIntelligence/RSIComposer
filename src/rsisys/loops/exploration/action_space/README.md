# Exploration Action Space

仅定义 Actor 可在环境中执行的动作，如 echo / execute。系统 Directive 的 sft / rl / evaluate / configure_prompt 由 [Exploration Harness](../harness/README.md) 分派，不属于这里的动作；SFT、prompt 配置和评测绕过 ActorPolicy.act，RL 只有 rollout 使用本 Action Space。

## 接口

[ActionSpace](__init__.py) 提供 `describe() → tuple[ActionSpec, ...]` 和 `validate(proposal, allowed_actions) → Action`。Action 携带通用 arguments，text 字段供 echo 使用。校验不执行副作用，实际行动交给 Harness 的 [Environment](../harness/environment/README.md)。

## 实现与替换

同文件提供 EchoActionSpace 与 ModularActionSpace。后者接收带 spec 和 `validate(arguments)` 的 ActionDefinition，检查名称唯一、行动获授权、校验后名称与提案一致。声明 schema 不会自动产生执行能力或运行通用 JSON Schema 校验。

[CommandActionSpace](../harness/environment/local.py) 支持 echo 和 execute，与 LocalProcessEnvironment 配套。execute 要求非空字符串列表 argv，可选 output_paths 为工作区内相对路径，`Directive.allowed_actions` 必须包含 execute。这里的 execute 是进程动作，与系统指令 mode=execute 分属不同层次。

通过 `ExplorationLoop(harness, action_space)` 注入；新增行动需同时提供校验器和 Harness 的环境执行支持。
