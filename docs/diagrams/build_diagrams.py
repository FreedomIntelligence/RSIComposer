#!/usr/bin/env python3
"""Generate editable SVGs for two loop-local Harnesses and shared Experience."""
from pathlib import Path
from html import escape
import unicodedata
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
INK = '#14243B'
MUTED = '#52637B'
LINE = '#8493A8'
BLUE = '#2365CB'
GREEN = '#178164'
PURPLE = '#7445C0'
AMBER = '#A86B13'
FAMILY = "'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif"
MONO = "'SFMono-Regular', 'Menlo', 'Consolas', 'PingFang SC', monospace"


class SVG:
    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title>',
                      f'<desc id="desc">{escape(description)}</desc>', '<defs>']
        for name, color in [('gray', LINE), ('blue', BLUE), ('green', GREEN), ('purple', PURPLE), ('amber', AMBER)]:
            self.parts.append(f'<marker id="arrow-{name}" markerWidth="10" markerHeight="10" refX="8.7" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="{color}"/></marker>')
        self.parts += ['</defs>', f'<rect width="{width}" height="{height}" fill="#F6F8FC"/>']
        self.text_count = 0

    def rect(self, x, y, w, h, fill='#FFFFFF', stroke='#D5DEEA', rx=18, sw=1.5, dash=None):
        assert x >= 0 and y >= 0 and x+w <= self.width and y+h <= self.height
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def text(self, x, y, text, size=24, color=INK, weight=400, anchor='start', mono=False, max_width=None):
        # Conservative width check complements the rendered visual inspection.
        units = sum(1.0 if unicodedata.east_asian_width(c) in ('W', 'F') else .62 if mono else .57 for c in text)
        width = units*size
        if max_width is not None:
            assert width <= max_width, (text, round(width), max_width)
        assert 0 <= x <= self.width and 0 <= y <= self.height
        fam = MONO if mono else FAMILY
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{escape(fam, quote=True)}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(text)}</text>')
        self.text_count += 1

    def path(self, d, color=LINE, width=2.5, arrow='gray', dash=None):
        marker = f' marker-end="url(#arrow-{arrow})"' if arrow else ''
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{marker}{extra}/>')

    def label(self, x, y, text, color=MUTED, size=18, width=None):
        width = width or sum(1 if unicodedata.east_asian_width(c) in ('W','F') else .58 for c in text)*size+20
        self.rect(x-width/2, y-size+1, width, size+10, fill='#F6F8FC', stroke='none', rx=6, sw=0)
        self.text(x, y, text, size, color, anchor='middle', max_width=width-8)

    def chip(self, x, y, w, label, fill, color, size=18):
        self.rect(x, y, w, 34, fill=fill, stroke='none', rx=17, sw=0)
        self.text(x+w/2, y+24, label, size, color, 600, 'middle', max_width=w-18)

    def node(self, ident, x, y, w, h, title, lines=(), color=INK, fill='#FFFFFF', stroke='#D5DEEA', title_size=26, tag=None):
        self.parts.append(f'<g id="{ident}">')
        self.rect(x,y,w,h,fill,stroke,rx=16,sw=1.8)
        if tag:
            self.text(x+24,y+29,tag,15,color,700,max_width=w-48)
            title_y=y+64
        else:
            title_y=y+40
        self.text(x+24,title_y,title,title_size,color,650,max_width=w-48)
        for i,line in enumerate(lines):
            if line:
                self.text(x+24,title_y+34+i*29,line,20,MUTED,max_width=w-48)
        self.parts.append('</g>')

    def save(self, name):
        self.parts.append('</svg>')
        s='\n'.join(self.parts)+'\n'
        ET.fromstring(s)
        (HERE/name).write_text(s, encoding='utf-8')
        print(f'{name}: {self.width}x{self.height}, {self.text_count} editable text elements')


def system_flow():
    s = SVG(2400, 1920, 'RSISys 系统流程图',
            '两个 Loop 分别组合可整体替换的 Harness 与独立 Action Space，共享 Experience。'
            'Directive 由 Decision policy 或外部系统发给 Exploration Harness，不是 Actor 动作。'
            'SFT、prompt 配置与评测绕过 Actor.act；RL 只有 rollout 使用 Actor 动作。'
            'Runtime 提供公共服务；Snapshot 保存内置 Harness 状态。')
    s.text(100, 80, 'RSISys / 系统流程', 42, weight=750)
    s.text(100, 126, '两个 Loop · 每个 Loop = Harness + Action Space · 一个共享 Experience', 26, MUTED)
    s.chip(1990, 52, 310, '概念架构 / 模块职责', '#E3F3EC', GREEN, 20)

    for x, title, color, fill, stroke in [
        (100, '01  Decision-making Loop', BLUE, '#EFF5FF', '#BCD1F2'),
        (1360, '02  Exploration Loop', GREEN, '#EFF8F3', '#B9DDCF'),
    ]:
        s.rect(x, 210, 940, 1050, fill, stroke, rx=24)
        s.text(x+40, 260, title, 31, color, 700)
        s.text(x+40, 303, 'loop.py / 薄端点与有界推进', 23, MUTED)
        s.text(x+40, 344, '业务编排在 Harness；Action Space 独立描述与校验', 22, MUTED)

    # Only the two public composition boundaries are expanded inside each loop.
    s.node('decision-harness', 140, 390, 860, 360, 'Harness / 决策执行边界', (
        '默认 engine.py：policy / 上下文 / 记忆 / 工具 / learner',
        '读取共享反馈，选择目标、research 操作与改进策略',
        '本地操作：Action Space 校验 → harness/tools 执行',
        '系统指令：policy 生成 Directive → 等待 Feedback',
        'receive_feedback 接纳结果，记录改进结论',
        'resume 与本地学习控制通过 loop.harness 访问',
        '可整体替换；memory / tools 不是框架强制接口',
    ), BLUE, stroke='#9FBEEB', title_size=29)
    s.node('exploration-harness', 1400, 390, 860, 360, 'Harness / 系统指令执行边界', (
        '默认 engine.py：policy / 上下文 / 记忆 / 环境 / learner / evaluator',
        '接收、校验并分派 Directive，汇总终态 Feedback',
        'execute：Actor.act → 动作校验 → 环境执行',
        'sft / configure_prompt / evaluate：绕过 Actor.act',
        'rl：仅 rollout 使用 Actor 动作；训练直接调用 learner',
        '推理评测由 evaluator 负责；默认训练 / 评测仍为 Mock',
        '可整体替换，包括不含 memory 的实现',
    ), GREEN, stroke='#9ACBB6', title_size=29)

    s.node('external-system', 1070, 215, 260, 120, '外部系统', (
        '也可直接提交指令',
    ), AMBER, '#FFF5DD', '#E4C47F', title_size=25)
    s.node('directive', 1070, 390, 260, 225, 'Directive', (
        'execute / sft / rl',
        'evaluate',
        'configure_prompt',
        '不是 Actor 动作',
    ), AMBER, '#FFF5DD', '#E4C47F', title_size=26)
    s.path('M 1200 335 V 390', AMBER, arrow='amber')
    s.path('M 1000 460 H 1070', BLUE, arrow='blue')
    s.path('M 1330 460 H 1400', BLUE, arrow='blue')
    s.node('feedback', 1070, 670, 260, 160, 'Feedback', (
        '状态 / 指标 / 候选',
        'evidence_refs',
        '引用共享系统记录',
    ), AMBER, '#FFF5DD', '#E4C47F', title_size=26)
    s.path('M 1400 700 H 1360 V 740 H 1330', AMBER, arrow='amber')
    s.path('M 1070 740 H 1040 V 700 H 1000', AMBER, arrow='amber')
    s.text(1200, 886, 'Runner 协调交付', 22, AMBER, 650, anchor='middle')
    s.text(1200, 922, '反馈确认后释放占用', 20, MUTED, anchor='middle')

    s.node('decision-action-space', 140, 900, 860, 240, 'Action Space / 决策动作定义与校验', (
        '本地 research：prepare_data / search_literature / store_strategy',
        '系统指令：execute / sft / rl / evaluate / configure_prompt',
        'describe / validate / validate_directive；directive_modes 列出模式',
        '纯定义与校验；实际执行位于 Harness 的 tools 或指令分派',
        '系统指令中的 execute 与 Actor 的 execute 动作分属不同层次',
    ), BLUE, '#FFFFFF', '#9FBEEB', title_size=27)
    s.node('exploration-action-space', 1400, 900, 860, 240, 'Action Space / Actor 环境动作', (
        '例如 echo / execute；仅供 execute episode 与 RL rollout 使用',
        'describe / validate；Directive.allowed_actions 限制可用动作',
        'ActorPolicy.act 生成提案，校验后交给 Harness 内的环境',
        '不包含 sft / rl / evaluate / configure_prompt 系统指令',
        '新增动作需同时提供校验定义与环境执行支持',
    ), GREEN, '#FFFFFF', '#9ACBB6', title_size=27)
    for x, color, arrow in [(570, BLUE, 'blue'), (1830, GREEN, 'green')]:
        s.path(f'M {x} 900 V 750', color, arrow=arrow)
        s.label(x, 831, '提供描述与校验', color, 20)
    s.text(140, 1190, 'Default Harness 封装内部组件；Loop 不暴露旧属性别名。', 21, MUTED)
    s.text(1400, 1190, 'PromptSpec(instructions) 配置 Actor 指令文本，不训练权重。', 21, MUTED)
    s.text(1400, 1226, 'Mock 不执行真实梯度更新或真实推理打分。', 21, MUTED)

    for x in (570, 1830):
        s.path(f'M {x} 1260 V 1370', PURPLE, arrow='purple')
        s.label(x, 1317, '提交系统 / 训练 / 评测反馈与改进结论', PURPLE, 20)
    s.rect(100, 1370, 2200, 290, '#F2EBFB', '#B89CDC', rx=24)
    s.text(140, 1416, 'Experience / 两个角色共享的系统反馈', 30, PURPLE, 700)
    s.text(140, 1453, '默认 Decision Harness 查询全部来源的反馈，policy 自行选择优化目标与策略。', 22, PURPLE)
    s.path('M 100 1450 H 55 V 470 H 140', PURPLE, arrow='purple', dash='9 7')
    for ident, x, title, lines in [
        ('system-feedback', 140, '系统运行反馈', ('Decision-making + Exploration', '耗时 / token / errors')),
        ('training-evaluation-feedback', 860, '训练与评测反馈', ('训练结果 / 评测指标 / 版本关联', 'Mock 指标不表示真实质量提升')),
        ('improvement-evidence', 1580, '改进结论与证据引用', ('方案 / 结论 / 来源关联', 'Feedback.evidence_refs → 系统记录')),
    ]:
        s.node(ident, x, 1475, 680, 130, title, lines, PURPLE,
               stroke='#D7C5EC', title_size=25)
    s.text(140, 1637, '原始 history / Actor 轨迹留在默认 Harness 内；RL trajectory_refs 与系统 evidence_refs 不可互换。', 22, MUTED)

    s.node('runtime', 100, 1720, 1050, 145, 'Runtime / 底层共享服务', (
        'Model / Sandbox / Artifacts / Jobs；Runner 协调两个公开端点',
        '业务编排在 Harness，loop.py 保留薄端点与有界推进',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=28)
    s.node('snapshot', 1190, 1720, 1110, 145, 'Snapshot / 内置实现的保存与恢复', (
        '保存 Harness 状态、模型绑定、内部记忆、Experience 与产物',
        '仅支持内置实现的空闲边界；替代 Harness 需显式集成',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=28)
    s.text(100, 1900, '实线：执行或数据流向；紫色虚线：查询共享反馈。高层只展开 Harness 与 Action Space，内部组件见各目录 README。', 21, MUTED)
    s.save('system-flow.svg')


def file_architecture():
    s = SVG(2400, 1900, 'RSISys 文件架构图',
            '两个 Loop 均含 loop.py、harness/ 与 action_space/。默认业务编排在 harness/engine.py，'
            'policy、上下文、memory、tools、learner 与 evaluator 属于默认 Harness 内部。'
            'Experience 为共享系统反馈库，Runtime 和 Snapshot 保留独立职责。')
    s.text(100, 80, 'RSISys / 文件架构', 42, weight=750)
    s.text(100, 126, 'src/rsisys/ · 两个 Loop + 一个共享 Experience · Harness 与 Action Space 平级', 25, MUTED)
    s.chip(1990, 52, 310, '目录归属 / 关键模块', '#E3F3EC', GREEN, 20)

    for ident, x, title, color, fill, stroke in [
        ('decision-dir', 100, 'loops/decision_making/', BLUE, '#EFF5FF', '#BCD1F2'),
        ('exploration-dir', 870, 'loops/exploration/', GREEN, '#EFF8F3', '#B9DDCF'),
    ]:
        s.parts.append(f'<g id="{ident}">')
        s.rect(x, 210, 730, 910, fill, stroke, rx=24)
        s.text(x+40, 260, title, 29, color, 700, mono=True, max_width=650)
        s.text(x+40, 303, 'Loop(harness, action_space)', 25, color, 650, mono=True)
        s.text(x+40, 349, 'loop.py  ·  薄端点与有界推进', 23, MUTED)
        if x == 100:
            lines = (
                '__init__.py：DecisionHarness 协议',
                'engine.py：DefaultDecisionHarness / 业务编排',
                'policy/：策略与模型调用',
                'memory/ · tools/ · learner/：默认内部组件',
                'decision.py · state.py：决策、上下文与状态',
                '',
                '读取共享反馈，执行准备操作，发布 Directive',
                '本地操作执行绑定在 harness/tools/',
                '可整体替换，不要求独立 memory / tools 接口',
            )
            space_lines = (
                '__init__.py：DecisionActionSpace',
                'research 操作 + 系统指令模式',
                'execute / sft / rl / evaluate / configure_prompt',
                'describe / validate / validate_directive',
                'directive_modes 列出模式；不执行工具',
            )
        else:
            lines = (
                '__init__.py：ExplorationHarness 协议',
                'engine.py：DefaultExplorationHarness / 编排',
                'policy/ · environment/：Actor 与环境执行',
                'memory/ · learner/ · evaluation/：默认内部组件',
                'commands.py · rl.py · state.py：分派与状态',
                '',
                '接收系统 Directive，直接调度学习 / 评测',
                'Actor.act 仅用于 execute episode / RL rollout',
                '可整体替换，包括不含 memory 的实现',
            )
            space_lines = (
                '__init__.py：ActionSpace',
                '仅 Actor 环境动作，例如 echo / execute',
                'describe / validate；环境负责执行',
                'SFT / prompt 配置 / 评测绕过 Actor.act',
                '系统 Directive 不属于 Actor 动作',
            )
        s.node(ident+'-harness', x+25, 390, 680, 395, 'harness/  ·  可整体替换',
               lines, color, '#FFFFFF', stroke, title_size=28)
        s.node(ident+'-action-space', x+25, 840, 680, 235, 'action_space/  ·  独立定义与校验',
               space_lines, color, '#FFFFFF', stroke, title_size=26)
        s.parts.append('</g>')

    s.parts.append('<g id="experience-dir">')
    s.rect(1640, 210, 660, 910, '#F2EBFB', '#B89CDC', rx=24)
    s.text(1680, 260, 'experience/', 31, PURPLE, 700, mono=True)
    s.text(1680, 303, '共享系统反馈 / 改进记录', 25, PURPLE)
    for ident, y, title, lines in [
        ('experience-records', 390, 'records.py / provenance.py', ('反馈、改进与来源引用类型', 'scope 标记来源，不指定优化对象')),
        ('experience-retrieval', 605, 'retrieval.py', ('record / retrieve；默认决策查询全部来源', '系统 evidence_refs 与原始轨迹引用分离')),
        ('experience-storage', 820, 'storage.py / sqlite.py', ('内存与 SQLite 存储实现', '不复制原始 history / Actor 轨迹')),
    ]:
        s.node(ident, 1680, y, 580, 165, title, lines, PURPLE,
               stroke='#D7C5EC', title_size=25)
    s.text(1680, 1053, '原始任务上下文由各 Harness 实现管理。', 22, MUTED)
    s.parts.append('</g>')

    s.rect(100, 1170, 2200, 235, '#FFF8E8', '#E4C47F', rx=24)
    s.text(140, 1217, 'loops/contracts/  ·  跨 Loop 消息与公开端口', 28, AMBER, 700)
    s.text(140, 1269, 'Directive / Feedback / coordination', 26, AMBER, 700)
    s.text(140, 1308, 'Directive：decision policy 或外部系统 → Exploration Harness', 23, MUTED)
    s.text(140, 1347, 'requests.py：SFT / RL / Evaluation / PromptSpec(instructions)', 23, MUTED)
    s.text(1280, 1269, '作业、模型与产物引用', 26, AMBER, 700)
    s.text(1280, 1308, 'jobs.py · model_service.py · artifacts.py', 23, MUTED, mono=True)
    s.text(1280, 1347, '定义共享消息；具体后端由 runtime 或自定义实现提供', 23, MUTED)

    s.node('runtime-dir', 100, 1455, 1040, 325, 'runtime/  ·  底层公共服务', (
        'model.py：模型调用与绑定',
        'sandbox.py / local_process.py：执行后端',
        'artifacts.py / files.py：产物与文件',
        'runner.py / jobs.py：端点协调与作业记账',
        '',
        '各 Harness 的 engine.py 负责业务编排',
        '默认训练与评测为 Mock，无真实梯度或推理评分',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=29)
    s.node('snapshot-dir', 1180, 1455, 1120, 325, 'snapshot/ + application.py  ·  装配与持久化', (
        'snapshot/：捕获 / 编解码 / 保存 / 恢复内置实现',
        'application.py：本地装配；config.py：内存演示装配',
        '',
        '通过 loop.harness 访问控制状态与模型绑定',
        '内置记忆分别持久化；Experience 独立保存',
        '空闲边界捕获，禁止在途作业或待写结果',
        '自定义 Harness 需显式集成，不自动获得快照支持',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=29)
    s.text(100, 1833, '每个 Loop 只展开 Harness 与 Action Space；内置细节通过 loop.harness 访问，不保留 loop.policy / memory / space 旧别名。', 23, MUTED)
    s.text(100, 1876, '组件 README 保留默认实现细节；框架允许整体替换 Harness，包括不含 memory 的实现。', 22, MUTED)
    s.save('file-architecture.svg')


if __name__ == '__main__':
    system_flow()
    file_architecture()
