# Action space

负责行动描述和校验，实际副作用由 [Environment](../environment/README.md) 执行。

## 接口

[ActionSpace](__init__.py) 提供 `describe() → tuple[ActionSpec, ...]` 和 `validate(proposal, allowed_actions) → Action`。Action 携带通用 arguments；text 字段供 echo 行动使用。

## 实现与替换

同文件提供 EchoActionSpace 与 ModularActionSpace。后者接收带 spec 和 validate(arguments) 的 ActionDefinition，检查名称唯一、行动获授权、校验后名称与提案一致。

本地命令使用 [CommandActionSpace](../environment/local.py)，支持 echo 和 execute，与 LocalProcessEnvironment 配套，通过 loop 的 space 参数注入。execute 要求非空字符串列表 `argv`，可选 `output_paths` 为工作区内相对路径；`Directive.allowed_actions` 必须包含 execute。

增加行动需同时提供校验器和环境执行支持。声明 schema 不会自动产生执行能力，也不会自动运行通用 JSON Schema 校验。
