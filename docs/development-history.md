# RSISys 开发历史

记录框架开发过程中遇到的问题、采用的方案、验证结果和仍未解决的事项。首版覆盖2026年11月4日至11月6日首轮半小时实测的归档结果；文中日期和时间均为北京时间。

这段开发的主线是让meta-agent逐步接管研究过程，同时补齐框架的执行、反馈、恢复和验证边界，并让研究策略、harness及训推策略与后端可独立组装。开放流程中，自主加载、推理、调试和评测已经实际运行；11月7日新增真实LoRA训练及权重保存证据，11月7日晚至8日已出现实际加载后的探索性公开对照、真实记忆/任务使用及封闭聚合；正式采纳、独立确认和研究能力收益仍需分别验收。

## 开发节点

| 时间与节点 | 遇到的问题 | 采用的方案 | 验证结果与遗留问题 |
| --- | --- | --- | --- |
| 11月4日至5日，开放Lab执行接口 | 训练、加载、推理和评测容易固化在框架流程里，模型难以自行实现和替换方案。 | 保留通用程序执行、文件读写、日志与状态查询入口；模型自行编写候选程序并部署Lab推理，host管理资源、预算、身份和截止时间。 | 后续实测运行了自主加载、推理、修错和固定评测。接口可用不等于模型已经选择训练，新增流程中的自主训练仍未验证成功。 |
| 11月4日至5日，执行错误和清理边界 | 参数、路径和并发槽等可修正错误可能终止整个研究；部分关闭异常会跳过其他清理，取消请求也可能被误当成退出确认。 | 用明确的`LabToolError`返回原生失败反馈，让模型继续修正；基础设施、权限与身份故障继续交给host。逐项处理owned清理，保留失败项及其依赖，确认进程退出后才释放资源。 | 通过真实CPU子进程、故障注入和决策请求回归。已发生效果或结果未知的动作不自动重放。详见执行生命周期和失败交付记录。 |
| 11月5日，路径说明和决策格式恢复 | 数据说明中的路径与实际`data/`目录不一致；一次非法JSON即可终止运行；一次压缩只剩35个输出token，连续三次生成均截断。 | 统一工作区相对路径及数据根目录；对明确可重试的格式错误最多增加两次生成，全部计入原调用账本和截止；调用摘要模型前预留配置要求的输出空间。 | 修复环境描述、有限格式恢复和预算准入。被拒绝的响应不执行Lab动作。输出空间充足不保证摘要内容正确，记忆和科研结论仍需核验。 |
| 11月5日，原目录暂停与恢复 | 重启可能重复执行程序、重置调用账本，或将旧PID误当成仍属于当前实验的进程。 | 保存checkpoint、journal、提交身份、原配置和绝对截止；使用controller与工作区互斥；恢复只重建可核验的终态读取视图，未知副作用不自动重放。 | CPU验收覆盖暂停、恢复、去重、预算保留和到期不新增调用。该能力不等于任意硬崩溃透明恢复，也不等于恢复训练的optimizer状态。 |
| 11月5日至6日，原始反馈和单一agent记忆 | 摘要可能遗漏或误述失败；记忆中的结论可能有引用但缺少事实支持；单独的整理流程使责任和证据路径更复杂。 | 原始反馈进入下一次决策；长结果机械分页并归档；用显式task保留跨步原件；同一个meta-agent决定如何整理和写入`agent_memory`。 | 合成验收实测了原始失败保留、分页重组、任务关闭和带来源的memory写入，没有独立整理模型。医疗运行中task和memory仍未实际使用，不能以合成通过代替领域实测。 |
| 11月6日，长运行暴露的错误分类遗漏 | 过期归档游标抛出`ValueError`，原六小时运行在552次调用后中断；修复后的一个运行段又因笔记新增项缺少`text`在514次调用后中断。 | 分别将可修正的归档游标拒绝、笔记内容冲突和过期笔记读取游标转为原生失败反馈；存储等host故障继续上抛。保留原运行、journal、调用和截止证据。 | 修复后分别通过合成验收。原连续六小时运行仍记为失败；最后修复段完成不能覆盖原失败，也不能算连续六小时通过。 |
| 11月6日，查询与可选等待 | 修复段有646次`lab_poll`，其中614次返回running，最长连续查询150次。即时查询之外缺少可选的有限等待。 | 保留即时查询，增加可选`wait_seconds`，默认最多30秒；进程终止、等待耗尽或host暂停、停止、截止时返回。等待不启动或取消程序，不自动收集终态结果。 | 本地和远端均完成1787项回归。新的半小时实测使用41次等待，截止中断和清理正常。该修改没有引入固定训练流程，也尚未证明训练收益或研究效率提升。 |
| 11月6日，空白资源池与八卡试跑 | 入口围绕`candidate.py`及Qwen3.5推理，可能引导反复调试推理；旧8001还占用第12卡。 | 用户要求将空闲卡交给同一个meta-agent自由研究。新增默认空白资源池，不生成候选脚本、不预选模型或启动程序；经用户授权停止旧8001，将8–15号卡交给Lab。 | 双端24项针对性测试通过，548文件SHA一致；10:45–11:15实跑105次调用对账、清理通过。模型自行查询环境、修错并选择单卡推理，无已验证训练或多卡使用；低分后来查明候选实现异常，不能解释为模型能力下降。 |
| 11月6日，低分审查与正式评测边界收紧（实现中） | 两轮MedXpertQA均为11/128且全选H。CPU复现发现候选解析器将`The correct answer is B.`解析成H；同时存在默认thinking配32-token上限、十选项任务却提示只选A–E等变化。固定数据与计分不足以固定实际推理链路。 | 用户要求只提供正式评测入口。独立工作区已接入宿主checkpoint/adapter评测，统一tokenizer、完整选项、生成参数和严格解析，留存原始生成及权重/推理配置身份；探索性程序继续由模型自由编写。 | 原记录计分复算一致，历史未留提取前生成原文，无法逐题还原截断。新入口相关回归已报告118项（2跳过），远端微型随机模型及LoRA的真实库CPU生成通过；尚无最终整仓、真实医疗运行或部署验收；旧结果不能直接作为新profile基线。 |
| 11月6日下午，评测协议与提交权限 | 上午只允许提交权重的方案限制了推理策略研究；同UID本地执行无法证明封闭集不可见。 | 用户改为提交checkpoint/LoRA及推理脚本；host固定数据、严格计分、预算和受理。公开集可自测，封闭集由独立gateway执行，只回汇总；全campaign共享至少1800秒一次的持久额度。冻结完整bundle，受理后失败仍占额度，相同key不重跑。 | `3650eb7`实现协议/客户端/账本及脱敏，合成CPU验证通过；独立封闭评测服务未部署，当前封闭提交禁用。测量的是完整候选方案；训练单项收益需固定推理做对照。此决定取代上午“只提交权重”的限制。 |
| 11月6日下午，唯一版本与旧部署收口 | GitHub虽只剩main，本地及远端仍有旧worktree、克隆和源码快照，最新改动尚在集成副本。 | 用户要求各端与GitHub对齐。先归档核对独有改动，补齐已squash历史，再快进主目录、迁移旧看板并清理冗余源码；模型权重、环境及实验记录保留。 | `8a18a35`时三端558文件一致，本地4/lion10/modelmate12份旧源码副本已清理；本地1810项（44跳过）、远端各49项无失败。之后加入b200，最新各端回执为`0da61f2`。modelmate旧27B精确身份清理完成，8000/8001关闭为当时回执。 |
| 11月6日下午，b200工作区与训推环境 | 用户决定转到b200的xinyuan工作区；初始没有GitHub认证及可用训推依赖，推理检查又遇PATH和CUDA开发库布局问题。 | 建立`/home/snchen/xinyuan/RSISys`；分离train/serve环境并锁版本，验证账号已有仓库权限。补验证PATH和项目内CUDA SDK的lib64/链接布局，保留前两次失败日志。 | 八卡BF16反向与NCCL、DeepSpeed ZeRO-2合成LoRA一步且权重变化、量化及vLLM/FlashInfer真实4-token生成通过；40项核心回归通过，37份小证据本轮SHA匹配。测试进程退出，未启动医疗研究或常驻模型服务；环境验收不算自主科研训练成果。 |
| 11月6日下午，按配置组合研究组件 | 原装配主要替换整个factory，细粒度策略/引擎缺少统一依赖、版本及恢复身份，未达到用户“搭积木”要求。 | `ComponentGraph`通过host显式实现目录组装训练/推理策略、引擎、RSI策略与harness，构造前检查契约/依赖/循环，记录组合；补权限接线、owned逆序清理和有状态组件恢复约束。 | `0da61f2`：15项新增测试覆盖32种组合及真实子进程接线；完整1825项（44跳过）无失败，b20098项通过。四个工作区与GitHub同提交；不证明任意原生GPU组合、医疗效果或组件级OS隔离，尚无新研究运行。 |
| 11月6日晚，整套方法与共同演化边界 | 组件图仅解决静态装配，复杂方法仍需整体接入；权限和联合候选的定义不够明确。 | `aac10fd`新增版本化整套RSI方法入口。用户统一meta-agent/Lab共同演化，reflect按改进对象归meta侧、触发归RSI优化器；研究尽量开放，仅约束敏感操作，正式数据与计分归host。 | 整套/组合两种CPU方法接同一入口，1837项含44跳过无失败、b200110项通过；第三方算法与真实收益仍需各自验收。触发不要求每轮生成候选。 |
| 11月6日晚，执行隔离与真实GPU验收 | 允许lab_start仍可越界写宿主文件，冻结候选可读外部可变依赖；权限恢复、多模型联合版本、优化器接线和计费存在缺口。 | `647f1a5`修执行敏感边界、依赖/运行时身份、可恢复权限、多目标版本及演化/计费接线；`380dbb2`/`a64afc8`补隔离运行时和venv/PATH/host环境。 | 1864项含44跳过无失败；八卡BF16/NCCL、未授权GPU拒绝、合成LoRA训练保存重载及冻结GPU评测通过。只证明登记边界和合成链路，不证明任意组件、SDK或医学效果。 |
| 11月6日夜至7日，官方评测与周期反思 | 自写答案解析和不完整题面影响比较；反思提示到期后未执行，修复attempt重置周期，报告重试又禁止读取缺失原文。 | 用户授权双27B四卡+四卡、半小时harness反思且禁meta训练。`cf9a8f8`引入官方口径，`307001f`/`91d0414`/`6df7219`修到期边界、原cadence与有界归档重试；`e707ed9`按用户要求固定公开500及两封闭集各500。 | 夜间7attempt共888calls到10:00、清理完成；训练128step/2epoch保存约319MB适配器，但提交路径越界拒绝，无训练后成绩。原失败保留，反思报告通过不等于系统变更；新500与旧1268不横比。 |
| 11月7日上午，压缩目标与实际容量分离 | compact已成功2次，随后最低保留40984token高于40k目标而终止；扩大服务后后端准入仍错用56k。 | `cbd0433`将软压缩目标与真实输入硬容量分开，允许有界放宽；服务改200k、host输入190784。`43a7e15`让compactor和AccountedModelBackend共用预算。 | 原检查点复现，修复后保留原始反馈；服务实测180042token输入，后续生产请求超过56k正常。旧失败不抹除，容量放宽不是无限长程或摘要语义验收。 |
| 11月7日上午，反馈、成果交接与有限反思 | 新attempt重复基线，调用账本连续却未继承知识/权重；Memory重复结果来源错绑，反思可反复引用旧记录并占满研究。 | `64d7d19`引入host显式continuation、命名空间/来源闭包/产物SHA、近期失败与任务投影；反思覆盖近期操作和失败、最多16步后明示未完成。加医疗打包/预检及公开输出诊断。`c129fac`仅允许新授权为到期窗口开启新attempt。 | 本地1907项=1863通过/44跳过、Linux72通过；67条原件与318835672字节LoRA交接验收。原Memory/task为0如实保留；不导入执行状态、PID或采纳，不重放、不清账。 |
| 11月7日下午，真实LoRA与效果归因断点 | 三epoch训练超时；一epoch训练成功但推理脚本仍直载基模，提示修改也曾被官方system覆盖。重复回执容易误算新评测或训练收益。 | 模型自行调试/缩短训练；host分别核验optimizer/checkpoint、最终权重、实际推理代码和逐题预测，保留提交/record身份与输出分布。 | 新六小时公开500题从异常129题恢复418题，无效降0；LoRA最终512tensor有限、256个LoRA_B非零，但未加载，500预测与基模相同。无封闭成绩、采纳或训练增益；memory/task0、最终6次no_change。 |
| 11月7日下午，审计环境漂移与截止终态 | CPU审计生成两个torch pyc，冻结环境检查拒绝评测并终止；首轮恢复又因RunPolicy模块身份不同被拒。到期守护终止controller而无新final result。 | 保留两次失败，精确恢复原运行时SHA并禁审计写缓存；确认提交未受理后按原`python -m`语义同源恢复，从call580继续。截止不顺延，精确核对owned身份、supervisor及GPU清理。 | 本地/Linux各29回归通过，无动作重放/账本重置。11:30:40–17:30:40窗口682calls全部对账，campaign1573/10000；终态failed/absolute_deadline_without_fresh_final_result。八卡清空、errors/remaining_owned为空不改写为运行成功。 |
| 11月7日18:00，Memory与任务组件验收完成 | 记忆检索仅围绕静态目标，当前相关经验可能被无关新笔记挤出；任务缺进度/关闭结论，后续反馈保留依赖固定工具名。 | `bf8a245`结合目标/近期反馈/打开任务检索；补任务update、完成条件、结论和历史分页查询；按生产者/终态/测量识别反馈，continuation继承任务字段和重绑定来源，失败建议不自动写任务。 | 81项本地定向通过；完整1917项=1878通过/39跳过，Linux controller/train各80通过/1跳过。已推origin/main，b200独立冻结603文件；9份源码SHA本轮匹配。真实Qwen采用与医学收益未验证，未重启实验。 |
| 11月7日18:08–21:08，继承LoRA实际加载与公开配对 | 先前已训练却未加载，旧83.6%不能直接作为新推理基线；反思连续验证失败，到期仍无新final。 | 在bf8a245冻结源上继续用户授权窗口，模型修打包/加载；host按相同推理、执行策略、公开样本核对adapter SHA与成对结果，保留截止失败。 | 基模393/500、一epoch437/500，+8.8pp（60纠正/16回退）；仅探索性公开集、未收集/采纳，无新训练确认。运行failed、448次研究调用中末次usage未知；Memory0、仅开1任务、54次报告失败，清理通过。 |
| 11月7日晚，反思时限和Memory审阅修订 | 操作次数限制不足以覆盖原文，失败审阅可反复卡住；Memory持续空白。 | 按用户要求2588c46取消操作次数上限，显式开始且最长3600秒/host截止；持久化待审原文快照，Memory inbox要求明确处置但不强制写笔记，不由host生成经验。 | revision2离线回放29记录/68页/41待审来源通过；当地初验1934项1890通过/44跳过、最终相关95通过/1跳过。原回执的未提交身份按时点保留，后续提交另记；离线fixture不是模型采用。 |
| 11月8日00:14–09:14，九小时实测与三类运行故障 | tokenizer token列表误受生成响应字节上限，反思尚未开始时归档不可用，长TMPDIR触发DataLoader AF_UNIX路径超长；任务在评测前即关闭。 | 56ac520分离tokenizer容量；5b732cd保留反思准备期归档；264a83b使用隔离私有/tmp，按新attempt交接产物、记忆与账本。分别保留旧失败与修复回归。 | 最新04在截止后验收completed/清理完成；两条真实Memory与读后引用、task更新/pin出现。继承一epoch封闭MedQA445/500、MedXpertQA95/500，无基模对照。新两epoch1024条/128step权重变化，但bundle缺candidate_adapter使评测加载前失败；科研未完成，3次no_change/第4次未完成。 |
| 11月8日上午，研究闭环修订15d3f03 | 工具正确也被当作无需行为改进，归档反复首页，任务缺验收约束，缺依赖直到全量评测才暴露。 | 加behavioral_review及reflect_read原文下一页；任务区分resolved/abandoned/inconclusive/interrupted，resolved校验声明的原生成功回执谓词/SHA；正式评测前执行同bundle smoke，Memory强调路径/依赖/操作经验。 | 本地1950项1905通过/45跳过，Linux128通过、14文件SHA一致；旧toy150ms测试偶发失败复跑与全量通过，未放宽。smoke不评分、不独立证明加载声明权重；验收谓词仍可能被agent写得过弱。 |
| 11月8日09:57起，新三小时效果对照与暴露限制纠正 | 先前两epoch84.2%与旧87.4%口径不齐；模型反复称无可编辑组件，不能仅凭工具名归责模型。 | 15d3f03冻结实跑修打包并重评一epoch，逐项核protocol/runtime/inference/dataset/budget；核实际请求和修改端口实现，保留旧解释并补纠正。 | 截至11:44 running、12:57:53截止；匹配公开500一epoch87.4%、两epoch84.2%，低3.2pp。Memory5、3任务关闭均inconclusive，4次no_change/无激活；默认空组件投影及缺当前内容/schema/示例，harness仅4整数限制。改善暴露与并行副本仍为建议，未实现。 |
| 11月8日12:57，医疗三小时终态与口径纠正 | 缺最终result，看板仍running；“官方协议身份一致”不足以证明候选忠实使用官方提示；Memory组合查询全部空。 | 截止后核精确进程与清理，独立复算三候选、归一化推理AST并审计system/答案提取；冻结历史复现literal连续整段匹配缺口，旧失败和原解释不覆盖。 | failed、368调用对账/campaign4269，83身份退出/21作业终态、八卡清空；37小证据SHA一致，本轮监控删除。78.6%/87.4%/84.2%仅共享自定义CoT/解析公开对照，严格官方口径未过。6次no_change无激活、Memory6/6空搜索、3任务inconclusive；新训练缺脚本失败。 |
| 11月8日下午，ReMe可选检索接入 | literal不适配关键词组合查询，存有笔记仍不能可靠召回；用户要求复用成熟后端。 | 4e70b5f接入真实ReMe BM25/RegexTokenizer供search和context共用，保留现有原子记忆、来源/修订、任务/交接；同一meta-agent管理，无独立整理模型/embedding/Auto Dream，默认literal不变。 | 完整1976总项1937通过/39跳过；原1974项2error夹具缺feedback_routing失败保留。11文件对提交SHA一致；实现回执610已提交文件本地/GitHub/b200对齐，b200未装ReMe依赖或验证真实效用，不能称完整ReMe系统已部署。 |
| 11月8日下午，公开注册与决策器/Lab边界收口 | 示例/机制/实验混杂、算法与默认meta-agent耦合；“任意方法完全库外”弱化了框架运行机制责任。 | 用户确认两个核心对象及内部组件复用，新机制可改核心；4e70b5f上未提交API/registry/CLI、LabGroup、rest_em与math Lab，复用原授权/回执/资源路径，DSH可配置，示例统一导航。 | module-boundary25文件同步，本地2011项1942通过/69跳过，b2001948通过/63跳过、57专项含极小CPU真实训练；安装wheel库外3配置运行。源码清单而非新提交标识，无GPU质量/任意插件恢复保证。 |
| 11月8日下午至18时，数学后训练与ReST^EM首跑 | 用户转向通用LLM后训练，要求同模型起点与GSM8K/MATH；需要固定算法训练闭环作为公开基线。 | 注册rest_em：当前模型生成、答案二值筛选、每轮从initial新建LoRA训练再评测；math拆校验器/引擎，准备Qwen3.5-9B/GSM8K、批量生成/梯度累积。 | 实现会话报告9B预跑8候选筛4、训练保存重载复测完成，37专项通过；两轮pilot调batch8→32后run-b32仍进行中，缺本地最终result/正式测试成绩。原配置及runtime-sync存本地，未提交；非论文指标复现、非meta-agent研究能力递归提升，完整MATH未验收。 |

11月6日上午的查询审计还确认：当时修复段的29项诊断与评测提交均为新提交，没有幂等key重放。重复基线分数不能据此归因为取回旧提交。查询是允许的研究动作，查询次数本身不能确定模型为何没有选择训练。

## 关键修复提交

| 提交 | 对应改变 |
| --- | --- |
| `0b994d1` | 默认由meta-agent自行部署Lab推理。 |
| `3890f39` | 保留原生Lab反馈并验证候选输入路径。 |
| `60786e6` | 医疗Open Lab原目录暂停与恢复。 |
| `c094d42` | 将反馈和显式memory管理合并到行动中的meta-agent。 |
| `b6f1ed4` | 明确拒绝不支持的可搬迁agent-memory快照。 |
| `f8a7db4` | 过期归档游标返回可恢复反馈。 |
| `ff90a19` | 可修正的研究笔记拒绝返回原生反馈。 |

可选等待补丁及首轮半小时实测使用`ff90a19fc891bfd0c34213d0f14ead21575211bb`上的已测试未提交修改，以完整文件SHA清单标识源码；不将该补丁表述为已经提交或已合入所有checkout。实现工作区为`.local/single-meta-integration-20261106`。

## 合成验收与真实运行

| 验收 | 实际观察 | 可支持的结论 |
| --- | --- | --- |
| 单一meta-agent CPU验收 | 14次真实调用的请求、响应和provider usage全部匹配；原始错误跨请求保留，程序修复、合成诊断和task/memory操作完成。 | 反馈、调试和记忆接口可以协同运行；合成诊断不是医学提升。 |
| 归档游标修复验收 | 5次调用；原始失败在5份请求可见；3页重组为原件；来源核验后的memory写入完成。 | 归档读取和恢复反馈通过。原报告曾因tuple与JSON list比较给出错误布尔值，后续规范JSON审计纠正测量，原报告保留，没有重复推理。 |
| 笔记修复验收 | 6次调用；原始错误在6份请求保留，active反馈在5份请求保留；两条笔记修正，memory revision达到3，task关闭。 | 笔记拒绝可由同一agent继续修正，并将证据用于显式记忆；仍为合成场景。 |
| 原六小时campaign及修复段 | 原连续运行因实现故障失败；最后repair-03段在原截止前完成，累计账本达到5957次。医疗agent_memory revision为0，没有task。 | 最后修复段完成、清理完成与原连续六小时失败可以同时成立。没有验证新的训练或医学提升。 |
| 首轮半小时实测 | 新授权窗口内102次调用，正式评测记录4份；运行到期，清理和证据同步完成。 | 运行链路验收通过；训练、task/memory使用和医学提升仍未通过真实领域验收。 |

## 2026年11月6日半小时运行验收

运行标识：`medical-modelmate-single-meta30m-20261106T013539Z`。

绝对窗口为09:37:31至10:07:31。控制器报告研究运行1789.16秒；提交窗口与控制器的研究计时分别记录。本轮提供固定的1024条MedQA训练数据；正式评测使用128条MedXpertQA和64条MedQA开发数据，评分协议为`exact-option-letter-v1`。

- 模型自行写入8版候选代码，提交7次输入诊断和7次正式评测，共14项新提交，没有幂等重放。
- 102次模型调用全部与provider usage匹配；累计调用从5957增加到6059。
- 99个完整反馈窗口和1个分页窗口与原始记录逐字匹配；71份请求带有上下文摘要。
- 模型自行选择41次30秒上限的等待：13次因任务终止返回、27次等待到期、1次因host中断返回。即时查询能力保留，本轮模型未选择即时poll。
- 4份完成的固定评测记录均可独立复算。两份达到MedXpertQA 25/128和MedQA 46/64；其余分别为24/128与46/64、0/128与0/64。没有超过本轮基线，没有采纳记录。
- 当前长评测由模型改为启用思考、batch size为4、每题最多1024个新token；到截止仍未完成，按中断记录，没有补记分数或延长窗口。
- 提交代码主要执行模型加载和生成。没有核验到optimizer步骤、loss记录或更新权重；`agent_memory` revision为0，没有task操作。
- controller、Lab及supervisor精确owned身份退出，清理回执无错误、无remaining_owned；7张实验卡无计算进程，既有8000和8001服务身份及健康状态正常。
- 同步32份允许导出的小型证据，逐文件SHA及归档、manifest SHA一致；完整日志、请求、医学样本和权重保留远端。

可选等待版本的回归为本地1787项、跳过45项，远端1787项、跳过13项，两端均无失败或错误。远端核验547个源码文件，未发生内容变化。

## 2026年11月6日资源池实测与评测审查

运行标识：`medical-modelmate-resource-pool30m-20261106T024421Z`；窗口10:45:24至11:15:24，累计调用6059→6164，本轮105次全部与provider usage匹配。资源池与等待补丁是`ff90a19`上的已测试未提交修改，548文件manifest SHA256为`a67b21f067307d66e00da126c5e4109fb4d22a0954d3af9f558c5291ff052d5e`。

- 起始无候选程序或预置Lab服务，8–15号卡均交给Lab。模型自行查环境、权重及库，处理导入/编译和适配器错误，最终自行选择单卡Qwen3.5推理，未验证训练、多卡使用或新权重。
- 93个完整原始反馈窗口及11个分页窗口核验一致；没有task/memory操作，memory revision为0。三次正式评测提交仅两次完成，最后一次到截止中断，未补分。
- 两份记录为MedXpertQA 11/128、MedQA 23/64及22/64，均未采纳。controller、Lab和supervisor精确身份退出，清理无remaining_owned，8张实验卡无计算进程；旧8001按授权保持停止，8000身份及健康检查通过。这是收尾时点观察。

11:30事后只读审查确认，固定数据清单、评分器和分数复算没有变化，候选推理/答案提取却发生了实质变化：逐字符扫描A–J会从普通英文中提取H；两轮MedXpertQA实际128题全部H；默认thinking配32-token、错误的A–E提示及从整段题面扫描选项也是已定位的问题。解析错误已在CPU复现；thinking导致逐题截断只是可能触发因素，因历史没有保留提取前原文，不能全部还原。先前运行验收通过只证明生命周期、调用和反馈交付，不证明答案链路正确。

用户随后要求把正式评测交还宿主。实现工作区`.local/single-meta-integration-20261106`已增加`CheckpointMedicalLab`，正式入口只接收已登记底座及checkpoint或LoRA adapter；宿主统一加载、模板、非thinking/greedy/32-token/batch1、严格完整字母解析，记录原始答案、token和输入/权重身份。只有`profile_sha256`和`dataset_sha256`均一致才直接比较。当前为未提交实现及相关CPU回归；实现会话另报告远端微型随机模型/LoRA真实库加载生成通过，无新医学基线。用户最新要求修完先不跑、清理modelmate的Qwen3.8-27B部署并准备转lion，清理尚待终态回执。尚无真实医学评测验收；此处不提前列为已完成修复。仍为可信本地执行契约，不是同UID程序的OS隔离。

## 2026年11月6日下午：评测协议、环境与组件组装

上午的宿主固定推理方案已完成119项相关回归（2跳过）和微型CPU加载检查，但用户随后明确允许提交推理脚本。因此当前以[评测协议](evaluation-protocol.md)为准：候选提交权重及推理代码，host固定公开/封闭数据身份、输入输出、计分、资源和受理规则。公开集作为开发材料；封闭集至少间隔1800秒受理，整个campaign共账，重启不清零、受理失败不退额度、未知副作用不自动重放。封闭gateway只返回允许字段；没有独立隔离服务就拒绝封闭提交。CPU gateway fixture仅证明接线与脱敏，不能证明隔离部署或真实泛化。

版本统一先完成`3650eb7`协议修改，再以`8a18a35`补齐已合入历史；558文件一致及旧源码清理均有回执。已删除的集成worktree不再作为当前开发位置，历史产物保留归档。随后用户指定b200，`a3ed56d`登记其标准工作区；训推依赖16:11验收，GPU实际识别为NVIDIA B20Z，不能仅由SSH别名推断型号。GitHub测试为SSH认证/fetch/push dry-run，不是修改账号权限或额外发布内容。

本次b200真实计算属于合成安装验证：DeepSpeed LoRA更新一步、loss及参数变化有记录，vLLM输出4token；并非meta-agent在医疗研究中自主训练。先前PATH缺ninja、CUDA wheel的lib布局导致的两次推理检查失败保留，项目内SDK布局修复后第三次通过。八卡空闲是16:11收尾观察，本同步未远端复核实时状态。

[组件组合说明](component-composition.md)将策略、harness及后端分别定义，`0da61f2`已配置化组装。JSON只能选择host显式目录中的实现，不自动安装或任意导入；声明版本/契约不是自动证明代码语义。组件可替换不等于模型有修改权，通用组件文件沙箱尚未提供。同UID程序不能保护封闭数据。恢复时在构造前拒绝版本、参数、依赖或根端口变化；有状态组件仍须提供真实state adapter，不可用空状态冒充恢复。

17:29回执记录本地、b200、lion、modelmate及GitHub为`main@0da61f23f5e36cf934c0a306ce06e9d477817eec`，当时工作树干净。完整CPU1825项含44跳过，b20098项无跳过；组合示例和进程fixture均不执行医疗训练。当前仍按用户要求不提交新实验；真实训推组合、医疗效果、task/memory行为和独立封闭评测服务待分别验收。

## 2026年11月6日晚至7日：真实研究、容量修复与成果继承

**架构与执行边界。** 用户把reflect定义为meta-agent自身的改进，Lab侧改进与其共同构成coevolution；触发和搜索可以整体封装在RSI优化器中。用户明确否定逐项默认deny，改为已装配范围内开放、敏感边界由host兑现。整套方法入口`aac10fd`与组件图并存。随后审查复现了执行越界和冻结依赖缺口，修复`647f1a5`及GPU follow-up通过隔离运行时、权限持久化、分目标版本和统一资源端口落实边界。GPU验收中的三步合成LoRA、重载logits误差0、八卡通信和未授权设备拒绝均属工程验收，不能与后续医疗训练混计。

**夜间运行与反思故障。** 运行`medical-b200-overnight-20261106T155811Z`在b200使用meta0–3/Lab4–7，两个模型均固定Qwen3.8-27B revision `8c0b3d30f3ae96f5c7f774de704bf0ada83e422a`。官方评分身份改为`medxpertqa-official-zero-shot-ao-v1`；开始公开1268、封闭2450/1273，后经用户要求固定各500，训练1024，不能把不同数据/评分协议连为收益曲线。7attempt原失败分别保留：到期仅提示未强制调度、修复后cadence重置、report阶段禁archive导致无法补齐证据。10:00终态清理回执确认888calls全对账、截止后无新请求、全部owned退出。真实训练128step/2epoch保存318835672字节LoRA，但以越界路径提交而被拒；训练本身已发生，效果闭环未完成。

**容量故障及测量纠正。** 04轮已完成2次compact，下一次输入52611token，按16k/28k/40k压缩后最低保留40984导致失败。当时“正常预算耗尽/未确认实现故障”的判断由后续新证据纠正：40k是软目标，服务仍有容量，host缺安全降级。`cbd0433`分离目标/硬预算并在200k服务实测180042token输入；05轮又因账本后端仍限56000而在39calls失败，`43a7e15`补生产factory接线为190784。原失败checkpoint、journal及累计调用保留，不将两次修复写作原段连续成功。此处只记录本地回执，不将实现会话查询的外部模型规格或DSH资料冒充本同步重新核实。

**跨attempt交接与反馈。** 用户要求修复成果继承后，`64d7d19`提供[显式交接协议](attempt-continuation.md)：host确认旧段停止且无未知mutation，再选择允许交付的原件、memory/task来源闭包和产物；导入历史命名空间/原文，核对SHA，不导入pending动作、PID、执行绑定或采纳。约319MB旧LoRA及67条记录真实交接验收，旧Memory/task为空不补造。恢复继承与到期新授权分开，`c129fac`新增绑定handoff/campaign/新attempt及截止的host续期凭据。反思统一为通用`PeriodicReflection`，近期失败原文不可用旧引用代替，16步后以incomplete显式退出而不伪造no_change。1907项本地测试含44跳过、Linux72项通过属于工程验证。

**六小时医疗结果。** 新窗口11月7日11:30:40–17:30:40，运行`medical-b200-sixhour-20261107T033040Z-01`冻结`c129fac`，保留同campaign账本与五组继承产物。推理实现将公开500从129/500恢复到418/500，参数未变；此前25.8%含485C/15无效，不能把恢复正常输出写为超越健康基模。一轮LoRA成功（1088秒、loss0.05116），最终权重SHA为`144334256cffaae8db60bc44284c1fc61a484bc224106d9b84bc5cd3999dd69a`，512tensor有限、256个LoRA_B全非零；先前三epoch第256步保存optimizer但后来超时，是另一段证据。冻结`infer_fast.py`没有PEFT/adapter加载路径，500预测与基模完全相同；同record_sha返回亦不证明新增推理。最终仍无封闭成绩、训练增益或采纳，6次有效反思均no_change、版本initial，Memory/task操作0。

**审计干扰与真实终态。** CPU读取优化器的审计生成两个torch serialization pyc，使已冻结运行时身份变化，579calls时正式评测被拒且未受理。保存干扰文件证据、精确恢复原SHA并禁止写缓存；首次恢复将原`__main__.AbsoluteHostPolicy`当包模块加载而失败，未增调用/改变checkpoint。改回原入口语义后显式恢复，失败反馈进入call580，本地/Linux各29项通过，无重放或账本重置。截止守护结束controller，未留下fresh final result；17:39终态包如实记failed/absolute_deadline_without_fresh_final_result，682calls全对账、累计1573/10000。旧失败result保留，不能冒充本段最终结果。精确owned、supervisor与Lab进程退出、八卡无计算进程，cleanup errors/remaining_owned为空；provider close hook未在恢复段观察到，清理依据是外部逐身份检查。以上为收尾回执时点，不保证后续实时资源状态。

**仍在修复的Memory/任务能力。** 用户17:22要求继续修复；实现会话已核对真实请求有工具，并复现Memory只按静态总目标检索的缺口，正补当前反馈检索、任务进度/完成条件/结论和未跟踪失败提示。17:42本地为`c129fac`上的未提交修改，涉及`src/rsisys/loops/decision_making/harness/`下的`agent_memory.py`、`observations.py`、`policy/prompts/agent_memory.txt`、`reflection.py`，以及`src/rsisys/runtime/continuation.py`及相关测试。当前turn仍inProgress，不列为已验收，不把接口改动当成真实领域使用。后续以提交或源码SHA及回归、真实行为证据补记。

本轮核对夜间128份、六小时50份本地manifest文件SHA均匹配；六小时终态包在巡检过程中17:39同步后补核。本同步未SSH、执行训练、改源码或重复外部调研，也不修改实验冻结源或另一个会话的未提交文件。

## 2026年11月7日18:00：Memory与任务修复验收

承接17:42的未提交节点，默认可复用组件与交接协议现已提交`bf8a2458e332784c17d8b23178355440bf3e7f84`，18:00回执确认推送origin/main。Memory仍由同一meta-agent显式写入，确定性字面检索改用总目标、当前反馈与打开任务；任务支持更新下一步/验收条件、关闭时保存outcome、按ID/状态分页读取。continuation保留任务进度与结论并重绑定原件；原始后续反馈按FeedbackDelivery、终态job、测量或失败字段识别，自定义Lab不必伪造工具名。host给未跟踪失败提供带record_id/revision的动作建议，不自动创建任务、写Memory或重放旧操作。

首次全量1916项有一项DSH结果卡片8KB断言失败。定位为`examples/dsh_smoke.py`夹具未使用生产规范回执序列化，归档身份不符导致完整原件保留；修正夹具而不放宽8KB断言，定向通过，随后完整1917项1878通过/39跳过。81项定向及真实decision/tool/accounting边界的脚本化任务→进度→来源Memory→关闭→跨attempt继承通过；Linux controller/train各80通过/1跳过。原失败日志与后续通过日志均保留，不将脚本化后端称为Qwen自主采用。

本轮直接核对9份源码SHA与回执一致；b200603文件独立源码冻结来自source-freeze回执，非主机canonical checkout全部更新证明。旧医疗冻结源保持`c129fac`，无截止后训练/生成/评测。六小时监控已于17:49按实现会话回执删除，51份本地证据manifest匹配；研究同步仍继续。下一节点需真实模型Memory/task使用与训练产物实际加载评测证据，既有零操作和未评测LoRA的历史不改写。

## 2026年11月7日晚至8日：效果证据与研究闭环修订

**实际加载后的首个公开配对与失败收尾。** `medical-b200-threehour-20261107T100807Z-01`冻结`bf8a245`。继承一epoch适配器SHA `144334256cffaae8db60bc44284c1fc61a484bc224106d9b84bc5cd3999dd69a`实际加载成功；与基模仅adapter加载/日志不同，同样候选内部CoT/答案提取、host固定评分，393→437/500。旧83.6%运行身份不同，不是此处基线。60纠正/16回退仅为适应性复用公开集的信号，无独立确认或封闭增益。末次分数截止前已完成却未被模型收集/采纳；21:08:07无fresh result.json，仍failed。448研究调用中447有终态、末次被截止中断且token未知，累计2024/10000；journal447 started而checkpoint446 committed，未知动作不重放。两次有效no_change、54次报告失败；Memory0、1任务仍open，资源清理有独立回执，不能由此声称provider close hook已执行。

**反思和审阅规则的后续决定。** 用户取消16步限制，允许构建、测试与激活贯穿窗口，最长一小时且受实验截止约束。`2588c46`落实显式开始、持久时限与待审失败原文快照，允许归档准备；Memory producer inbox要求显式写/更新/引用既有知识或说明证据不足/无可复用结论，不设笔记数量和host经验生成。两版repair回执起初均是`bf8a245`上的未提交修改，源码清单与离线回放身份保留；不能把后续提交冒充原验收时已部署。revision2回放29条记录/68页/41项Memory待审来源，证明协议可执行，非Qwen采用证据。并行能力研究副本仅为讨论方案，尚未实现。

**九小时多attempt：基础设施恢复与科研失败分别验收。** `medical-b200-ninehour-20261107T161415Z-04`最终冻结`264a83b`。先后修复tokenize大响应被8MiB生成响应限制拒绝（`56ac520`，同合成输入真实tokenizer1800012token通过、生成上限不变）、反思准备期归档缺失（`5b732cd`，787记录离线重建45150token/最新失败保留）、隔离进程长临时目录AF_UNIX错误（`264a83b`，原场景可复现，私有短/tmp后Linux25项通过）。没有据此跳过执行权限或延长窗口。09:14:15到期，09:15保存包确认completed、owned终止/八卡清空、监控删除；终态以回执时点为限。

研究中确实写出两条Memory，读取及至少一次读后决策引用已有核对，task更新/pin/close也已出现；最后attempt两条为继承，3读/0新写/0search，不能混成最后attempt新增两条。继承一epoch候选封闭记录`82f73ac0004490483972a67a6788e5b1a29211c7d2200d922bffe5cf267dbae4`为MedQA445/500、MedXpertQA95/500，无匹配封闭基模。新两epoch1024条/128optimizer步确有权重变化，但最终bundle遗漏`candidate_adapter`，评测在加载前失败，故无新准确率。3次反思均no_change、第4次未完成；反复首页83次（此前314次）及任务未评测先关闭仍是失败模式，不以Memory数量证明修复有效。

**11月8日闭环修订及边界。** `15d3f03`保留单一meta-agent，增加行为反思计划、`reflect_read(window_id)`按已提交历史确定下一缺失原文页（保持hash，不生成摘要），任务关闭结果分类及resolved验收来源。通用`acceptance_checks`验证operation/JSON Pointer/expected与原生非模拟成功回执/SHA，跨attempt重绑定；agent可声明过弱条件，故不等于自由文字目标被证明。`medical_smoke_test`在同环境执行同冻结bundle最多2个公开输入、最长300秒，不算准确率、不访问封闭数据、不占封闭slot；正式评测默认需匹配回执，host可信配置可显式兼容关闭、agent无开关。回执`model_load_verified=false/scored=false`，仍需独立核实际权重加载与全量结果。完整本地1950项1905通过/45跳过、Linux128通过，14文件SHA一致。旧toy150ms截止夹具偶发失败单独复跑/完整回归通过，原失败不抹除，未修改测试或放宽断言。

**后续实跑及旧解释纠正。** 用户要求“跑一下看看”后新三小时`medical-b200-closedloop3h-20261108T015753Z-01`冻结`15d3f03`；11:44快照仍running，截止12:57:53，不能写已完成。补齐一epoch重评后，一/两epoch的protocol `cf51d3b77223410a1d21be2a8e7477a09842d658f681327dcc18851c58f39e44`、dataset、runtime、inference和预算全部一致，record分别`f6b49b0f2084359f77f11da38b51c07cb3b5bd122b1582030abd4d84489ae88c`与`677c0998b756f32f0b3e6623b3bd2c3679ac6de2f8af5c900f140103adaf2049`：437/500与421/500，观察到两epoch低3.2pp；单次公开比较不证明一般训练规律、显著性或封闭收益。Memory5条含继承2/新增3、3任务均以inconclusive关闭、receipt_checked_closed=0，4次no_change，无激活。

早期审计曾以请求中存在`system_status/register/activate`将“无可编辑组件”完全归于模型判断；后续审计与源码限定该解释：默认`initial/components={}`缺已装配端口清单，查询只有元信息，未完整展示当前值/格式/示例。`open_lab_evolution.py`的`meta.prompts`可换模板，`meta.harness`仅`experience_limit/memory_limit/review_tool_limit/max_outstanding`四个整数，不能任意修改Memory检索或Python调度。工具入口存在与修改对象充分暴露分别判断；没有host拒绝证据也不足以排除设计缺口。改善发现入口、可编辑候选工作区和能力验证目前为助手建议，未见提交，不因该解释纠正改写原反思失败或声称必须SFT。

本轮只核本地小证据：三个新运行41/84/32份manifest SHA匹配，两版离线replay SHA匹配；完整请求、预测、权重及私有样本未读取，未SSH。证据记录能支持上述阶段结论，不能补足独立科研效果验收。

## 2026年11月8日午后：医疗终态、成熟检索与数学算法注册

**截止失败与提示口径的新纠正。** 早先11:44 running记录按当时快照保留；`medical-b200-closedloop3h-20261108T015753Z-01`已在12:57:53到期，缺fresh result，终态`failed/absolute_deadline_without_final_result`。独立身份审计83项、Lab21作业全部终态，组件cleanup errors/remaining_owned为空、八卡无计算进程，本轮监控`b200-rsi`已删除；看板因缺result仍running属于陈旧展示，不是运行未止。368请求/响应对账，campaign4269/10000，历史call448未知usage不补零；无截止后请求或重启。本同步只读这些收尾回执，未SSH。

后续基模record `f52f40e8b9bc49034cb7ef192b9360ad7d46b18c6d89e675f935a51c21f11893`完成393/500，连同一/两epoch437/421，host复算与AST归一化支持共享推理流程下差值+8.8/+5.6/−3.2pp。新增`official_prompt_fidelity_audit`同时确认三个程序覆盖“Please answer only.”为医疗CoT系统提示，再预提取答案返回；因此协议SHA一致、分数复算正确不证明严格官方提示/原始生成一致。保留原比较与原“官方口径”说法作为测量历史，本节点明确降格为共享自定义提示/解析的探索性公开结果，不能称严格官方指标或医学泛化。继承适配器比较也不是本轮新训练收益；新命令`train_lora_v7.py`不存在，执行前退出2。6反思全no_change、无system_status/register/activate，3关闭任务inconclusive/receipt_checked_closed=0。

**Memory缺口到ReMe薄适配。** 终态6条记忆/3次显式读取，6次组合词搜索全空；已按冻结历史复现literal要求整个query连续匹配。检索易用性与修改对象暴露是独立设计缺口，不能以笔记已写或工具存在归责模型。用户同意复用成熟后端；`4e70b5f516ba85a41e7057af5526d91619627dcd`固定上游依赖commit `e3fade0a781d080a4be7fb83648a3b2e6d477522`，仅调用ReMe BM25/RegexTokenizer供显式搜索及上下文召回。`agent-memory.json`继续为原子持久来源，索引为可重建缓存，无第二写入存储/服务、不加载pickle；source SHA、历史修订、task原生回执与continuation保留。默认literal及旧run身份不变，新的选择应在新attempt完成，不热改冻结实验。无embedding、自动提炼/Auto Dream或独立整理模型；无共同词同义召回、记忆正确性与行为改善均未证明。第三方机制细节来自实现方固定源码核对回执，本同步未重新外部核验论文/仓库。

初次整仓1974项出现2个错误：旧SimpleNamespace夹具未含feedback_routing，新装配入口报AttributeError；原日志保留，后续完整1976总项=1937通过/39跳过。实现会话“1976项通过”应按原日志这一计数解释，不把跳过算通过。本轮核对11文件与Git提交SHA一致。用户要求对齐b200后，实现会话报告610已提交文件本地/GitHub/b200一致、远端86项含24跳过；这仅为源码同步，未在b200安装ReMe依赖或开展其真实效用实验。

**两个核心对象与未提交注册工程。** 用户希望框架可注册算法、复用harness/memory/校验器/训推引擎，并明确超出接口的底层机制可以修改核心。最终核心注册对象是`RSIController.decide(DecisionInput)->Decision`与`Lab.execute(LabRequest)->LabFeedback`；方法表示装配，不强制第三执行对象，也不要求固定内部图。统一运行时负责循环推进、授权、反馈、状态、清理及证据；注册不会自动追踪插件私有调用、计费或创造异步/恢复语义。保留较早完整方法入口为兼容，不把“任意方法纯库外实现”当当前硬门槛。

`4e70b5f`上未提交新增`src/rsisys/api.py`、`registry.py`、`loops/contracts/rsi.py`、`controllers/`、`labs/`、`adapters/lab_group.py`，并改装配、CLI、DSH接线及示例。`registration-20261108`初版24文件远端验收1991项1929通过/62跳过；后经用户边界修订，`module-boundary-20261108/sync.json`25文件标识最终该阶段源码，完整本地2011项1942通过/69跳过、b2001948通过/63跳过，57专项含微型模型真实CPU LoRA训练/保存/加载。wheel `75da04a42b07d8cf151f7e79c102b7ae79c7c9c6f06b311b2e7efc7624f1620f`库外运行rest_em builtin/fixture各10步、自定义controller4步。没有新Git提交；25文件中22当前匹配，3文件在下一pilot before快照匹配，不能把较早验收套给后续变化。

**算法闭环与进行中的9B实跑。** `rest_em`固定状态机通过既有调用/回执接线：先评初始模型，每轮当前模型采样→二值成功筛选→从同一initial重新初始化LoRA/优化器训练→评测，筛选为空不训练。Lab验证完整请求回显、样本/模型引用与数据身份，训练仅使用被接受的生成答案而非参考答案作目标。`single_loop`仅调度模板，`meta_agent`为另一可选决策器；固定ReST^EM不是自主修改科研策略。数学Lab当前同步终态、不接裸JobRef、无断点恢复；普通数值verifier适配GSM8K形式，不涵盖完整MATH符号等价，不能称GSM8K+MATH均验收。

用户随后授权“先复现一下ReST^EM”。18:07所读`01a1ba8a`仍inProgress：实施方报告b200 Qwen3.5-9B预跑8候选筛4个，LoRA实际更新/保存/重载及后评测通过，37相关测试通过；两轮GSM8K pilot已启动，因吞吐调整生成batch8→32，原进程停止后新`run-b32`运行。尚无两轮完成/最终测试成绩。预跑原始result、batch32配置与完整训练产物尚未同步本机；当前证据为直接会话报告，本地`.local/rest-em-pilot-20261108T094737Z/experiment.json`仍是batch8原配置，runtime-sync记录增量，不能冒充本轮独立远端验收。准备入口固定Qwen3.5-9B/data revision、train64/dev128及官方test单独导出；实际拆分验收和最终测试仍待原件。论文模型/LoRA配置不同，定位为算法移植闭环而非论文数值复现。MATH-500、研究者与候选同步切换、开源首发门槛为讨论建议，不因代码可运行就成为已交付承诺。

## 保留的未解决问题

1. **数学两轮实跑尚未完成验收。** 9B预跑成功来自实施会话，run-b32仍进行中，最终基线/候选同解码对照与test原件未同步；MATH校验器/完整MATH结果未交付，不能外推论文指标或研究者递归收益。
2. **注册与pilot仍为未提交修改。** module-boundary25文件验收及pilot后续增量分别标识；接口、wheel及极小CPU通过不代表任意第三方运行语义。同步ReST^EM无断点恢复，插件内部计费/异步需显式接线。
3. **ReMe检索已接入，真实效用未验收。** 当前b200源码对齐但未安装ReMe依赖；6空搜索是旧literal实验事实。回归召回不等于真实行为少犯错，也非完整ReMe自动记忆系统部署。
4. **医疗口径与终态保留。** 新三小时failed且清理完成，看板running陈旧；78.6%/87.4%/84.2%只支持共享自定义提示/预解析探索性对照。严格官方提示一致性失败、匹配封闭基模/独立确认/正式采纳仍缺。
5. **自修改与任务验收仍有边界。** 6反思无激活、3任务inconclusive；修改对象/当前内容/schema暴露不足仍未修。resolved只验证agent声明谓词，smoke不独立证明声明权重加载，不把这些接口当科研目标已证明。
6. **开源与多领域目标待分别交付。** 用户当前转数学，医疗历史保留；research preview发布门槛及MATH-500等为助手建议，未见发布/许可CI全验收。仅按确证进展归档，不因计划扩张执行或延长实验。

## 后续记录方式

每个开发节点保留问题、定位、修复和验证的完整关系。后续纠正用新条目说明，不覆盖原失败，不清调用账本，不把新的运行段改写成旧运行连续成功。

自2026年11月6日起，由研究同步会话在定期检查时增量维护。沿用上表粒度，将同一问题的连续调试归并为开发节点；只补充有来源的新调整、关键故障及阶段验收，无实质变化不新增巡检记录。按证据区分用户决定、方案建议、已实现和待验收；重要节点保留提交或运行标识及证据入口，并随确证进展更新遗留问题。当前全局研究状态另记于项目根目录的`RESEARCH_CONTEXT.md`。

```text
日期与版本
问题及触发条件
定位证据与被排除的假设
采用的方案及原因
回归结果与真实运行结果
仍未解决的问题
提交、运行记录及证据位置
```

持续遵循的原则：模型能修正的问题，带着原始证据交回模型；基础设施和身份故障由host处理。调用成功、程序成功、评测有效、训练有效和结果被采纳分别证明。接口存在与模型实际使用分别记录。运行停止与资源确认退出分别验收。

## 证据入口

仓库中的设计与复现记录：

- [开放Lab执行](review/open-lab-execution-20261104.md)
- [执行生命周期修复](review/execution-contract-fixes-20261104.md)
- [失败结果与反馈事件交付](review/lab-delivery-semantics-20261104.md)
- [路径与有限格式恢复](review/open-lab-path-response-fix-20261105.md)
- [摘要输出空间修复](review/open-lab-summary-headroom-fix-20261105.md)
- [原目录暂停与恢复](review/open-lab-recovery-20261105.md)

以下为本机`.local`中的实现及操作证据，不保证随仓库克隆分发：

- [查询与可选等待设计](review/open-lab-wait-20261106.md)
- [原六小时campaign终态](../.local/modelmate-single-meta6h-20261105T174020Z/summary.json)
- [归档游标故障](../.local/modelmate-single-meta6h-20261105T174020Z/cursor-fault.json)及[规范JSON验收纠正](../.local/modelmate-single-meta6h-20261105T174020Z/cursor-acceptance-audit.json)
- [笔记故障](../.local/modelmate-single-meta6h-20261105T174020Z/patch-fault.json)及[修复验收](../.local/modelmate-single-meta6h-20261105T174020Z/patch-acceptance-audit.json)
- [半小时运行终态](../.local/modelmate-single-meta30m-20261106T013539Z/summary.json)
- [半小时科学证据审查](../.local/modelmate-single-meta30m-20261106T013539Z/scientific-review.json)
- [等待行为与原始失败保留审计](../.local/modelmate-single-meta30m-20261106T013539Z/behavior-audit.json)
- [两端回归验证](../.local/modelmate-single-meta30m-20261106T013539Z/acceptance-validation.json)
- [最终证据同步验证](../.local/modelmate-single-meta30m-20261106T013539Z/evidence-verification.json)

- [资源池准备与双端针对性回归](../.local/modelmate-resource-pool-prep-20261106T022658Z/remote-regression-result.json)
- [八卡资源池终态](../.local/modelmate-resource-pool30m-20261106T024421Z/summary.json)及[最终证据核验](../.local/modelmate-resource-pool30m-20261106T024421Z/evidence-verification.json)
- [低分与答案解析事后审查](../.local/modelmate-resource-pool30m-20261106T024421Z/posthoc-evaluation-audit-20261106T033019Z.json)
- [宿主checkpoint阶段实现](../examples/medical/checkpoint_evaluation.py)及[当前评测入口说明](../examples/medical/README.md)

- [当前评测协议](evaluation-protocol.md)及[组件组合与权限边界](component-composition.md)
- [旧27B部署清理回执](../.local/modelmate-qwen27b-retirement-20261106T040028Z/retirement.json)
- [版本统一与源码清理回执](../.local/version-unification-20261106T051207Z/final-receipt.json)
- [b200部署](../.local/b200-deploy-20261106T065320Z/DEPLOYMENT.json)及[训推环境验收](../.local/b200-runtime-setup-20261106/RUNTIME.json)
- [组件组合检查与五处对齐](../.local/component-composition-check-20261106/receipt.json)

首轮半小时运行源码manifest SHA256为`ed4a79d88cc23f14361974b6a1d890fd03fa7217db5c636444c58648ba1b8e99`。最终小型证据归档SHA256为`4c3b6539ad0c97d1bed9bdebc47592a1b5a878c9a97462ed5f2a5f9ad1289064`，证据manifest SHA256为`556d2ea0d2140b09ef2d4edc361f8bada985ce989aaed84edb5de26ef2763754`。

本轮新增证据入口：

- [整套方法接入验收](../.local/rsi-method-check-20261106/receipt.json)
- [架构审查原始复现](../.local/architecture-review-20261106T122642Z/review.md)、[六项修复](../.local/architecture-fixes-20261106T150010Z/receipt.json)及[隔离GPU验收](../.local/architecture-gpu-check-20261106T151200Z/receipt.json)
- [官方评测与周期反思协议](medical-overnight-protocol.md)
- [夜间终态](../.local/b200-medical-overnight-20261106T155811Z/summary.json)、[训练/评测审查](../.local/b200-medical-overnight-20261106T155811Z/model-attempts-audit.json)及[清理回执](../.local/b200-medical-overnight-20261106T155811Z/terminal-cleanup-receipt.json)
- [压缩容量纠正](../.local/b200-medical-overnight-20261106T155811Z/capacity-repair-receipt.json)及[后端准入修复](../.local/b200-medical-overnight-20261106T155811Z/admission-repair-receipt.json)
- [跨attempt交接验证](../.local/continuation-repair-20261107/validation.json)及[真实旧产物交接](../.local/continuation-repair-20261107/continuation-artifact-validation-receipt.json)
- [六小时终态与科学限制](../.local/b200-medical-sixhour-20261107T033040Z/summary.json)、[最终LoRA权重审查](../.local/b200-medical-sixhour-20261107T033040Z/training-final-adapter-audit.json)、[运行时恢复](../.local/b200-medical-sixhour-20261107T033040Z/runtime-recovery-receipt.json)及[截止清理](../.local/b200-medical-sixhour-20261107T033040Z/terminal-cleanup-receipt.json)

- [Memory/任务修复验收](../.local/agent-memory-task-repair-20261107T092325Z/repair-receipt.json)、[b200独立源码冻结](../.local/agent-memory-task-repair-20261107T092325Z/source-freeze-receipt.json)及[Linux两环境回归](../.local/agent-memory-task-repair-20261107T092325Z/linux-targeted-receipt.json)
- [DSH首次全量失败](../.local/agent-memory-task-repair-20261107T092325Z/full-regression.log)、[修正后完整回归](../.local/agent-memory-task-repair-20261107T092325Z/full-regression-v2.log)及[实验监控终结](../.local/b200-medical-sixhour-20261107T033040Z/monitor-finalization.json)

- [晚间三小时配对及失败终态](../.local/b200-medical-threehour-20261107T100807Z/summary.json)及[截止清理](../.local/b200-medical-threehour-20261107T100807Z/terminal-cleanup-receipt.json)
- [反思/Memory首版离线修复](../.local/framework-loop-repair-20261107/repair-receipt.json)、[用户规则修订版](../.local/framework-loop-repair-20261107/revision2/repair-receipt.json)及[修订版回放](../.local/framework-loop-repair-20261107/revision2/real-trajectory-replay-receipt.json)
- [九小时终态与科研限制](../.local/b200-medical-ninehour-20261107T161415Z/summary.json)、[tokenizer容量修复](../.local/b200-medical-ninehour-20261107T161415Z/tokenize-repair-receipt.json)、[归档准备期修复](../.local/b200-medical-ninehour-20261107T161415Z/reflection-context-repair-receipt.json)及[临时目录修复](../.local/b200-medical-ninehour-20261107T161415Z/tmpdir-repair-receipt.json)
- [行为反思、任务验收及运行检查](review/research-closed-loop-20261108.md)及[15d3f03回归](../.local/b200-medical-closedloop3h-20261108T015753Z/regression-receipt.json)
- [新三小时11:44快照](../.local/b200-medical-closedloop3h-20261108T015753Z/summary.json)、[同口径对照与暴露限制纠正](../.local/b200-medical-closedloop3h-20261108T015753Z/poll-detail-audit.json)及[当前修改端口源码](../examples/medical/open_lab_evolution.py)

- [医疗新三小时终态](../.local/b200-medical-closedloop3h-20261108T015753Z/summary.json)、[提示一致性与Memory审计](../.local/b200-medical-closedloop3h-20261108T015753Z/poll-detail-audit.json)及[精确清理](../.local/b200-medical-closedloop3h-20261108T015753Z/terminal-cleanup-receipt.json)
- [ReMe接入范围](memory-backends.md)、[工程验收](../.local/reme-memory-integration-20261108/summary.json)、[提交源码清单](../.local/reme-memory-integration-20261108/source-manifest.json)、[首次失败](../.local/reme-memory-integration-20261108/initial-full-suite-failure.log)及[最终回归原始计数](../.local/reme-memory-integration-20261108/final-full-suite.log)
- [注册指南](registration.md)、[初版注册验收](../.local/registration-20261108/validation.json)、[两模块阶段验收](../.local/module-boundary-20261108/validation.json)及[未提交源码同步清单](../.local/module-boundary-20261108/sync.json)
- [ReST^EM范围及限制](../examples/rest_em/README.md)、[控制器实现](../src/rsisys/controllers/rest_em.py)、[9B原batch8实验配置](../.local/rest-em-pilot-20261108T094737Z/experiment.json)及[pilot源码增量同步清单](../.local/rest-em-pilot-20261108T094737Z/runtime-sync.json)。9B预跑/运行中batch32状态来自关联会话01a1ba8a，原始远端回执尚待同步。
