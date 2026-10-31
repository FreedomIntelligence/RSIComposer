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
        (HERE/name).write_text(s)
        print(f'{name}: {self.width}x{self.height}, {self.text_count} editable text elements')


def system_flow():
    s = SVG(2400, 1920, 'RSISys 系统流程图',
            '两个 Loop 各有一套 Harness 和本地 memory，共享 Experience 系统反馈。'
            'Actor 读取自身 history，RL 读取 Exploration memory 的原始轨迹；'
            'receive_feedback 是 Loop 执行控制。Runtime 提供底层服务，Snapshot 分别保存两份 memory 和 Experience。')
    s.text(100, 80, 'RSISys / 系统流程', 42, weight=750)
    s.text(100, 126, '两个 Loop · 各自一套 Harness + RSI 扩展 · 一个共享 Experience', 26, MUTED)
    s.chip(1990, 52, 310, '概念架构 / 模块职责', '#E3F3EC', GREEN, 20)

    # Each Harness is a grouping of existing loop components, not a new layer.
    for x, title, color, fill, stroke in [
        (100, '01  Decision-making Loop', BLUE, '#EFF5FF', '#BCD1F2'),
        (1360, '02  Exploration Loop', GREEN, '#EFF8F3', '#B9DDCF'),
    ]:
        s.rect(x, 210, 940, 1050, fill, stroke, rx=24)
        s.text(x+40, 260, title, 31, color, 700)
        s.text(x+40, 299, 'Loop = Harness + RSI 扩展；loop.py 负责执行编排', 22, MUTED)
        s.rect(x+20, 330, 900, 670, '#FFFFFF', stroke, rx=20)
        s.text(x+40, 366, 'Harness / policy · context + memory · space + tools · 执行控制',
               21, color, 650, max_width=860)
        s.text(x+40, 1045, 'RSI 扩展', 21, color, 700)

    s.node('decision-policy', 140, 395, 860, 135, 'Decision Policy / Context', (
        '生成结构化 Decision：invoke / explore / wait / finish',
        '读取本地记忆及两个 Loop 的系统反馈，制定改进方案',
    ), BLUE, stroke='#9FBEEB')
    s.node('decision-memory', 140, 700, 410, 200, 'Harness-local memory/', (
        'test-time history',
        'working memory / 长期任务知识',
        '自身 Harness 的上下文与任务记忆',
    ), BLUE, '#EFF5FF', '#9FBEEB', title_size=24)
    s.node('decision-space', 590, 700, 410, 200, 'Decision Space / Tools', (
        'describe / execute',
        '文献 / 数据 / 策略操作',
        '操作结果与 ArtifactRef',
    ), PURPLE, '#F2EBFB', '#B89CDC', title_size=25)
    s.path('M 345 700 V 530', BLUE, arrow='blue')
    s.label(345, 650, '读取自身 history / 任务记忆', BLUE, 18)
    s.path('M 900 530 V 700', BLUE, arrow='blue')
    s.label(900, 663, 'invoke', BLUE, 18)
    s.path('M 795 900 V 955 H 345 V 900', BLUE, arrow='blue')
    s.label(570, 955, '操作结果写入自身 memory', BLUE, 18)
    s.node('decision-learner', 140, 1070, 860, 145, 'Decision Learner', (
        '训练作业控制 / 模型绑定',
        '训练反馈与改进结论进入共享 Experience',
    ), BLUE, stroke='#9FBEEB')
    s.text(140, 1244, 'Harness 管执行与上下文；RSI 扩展负责策略改进。', 19, MUTED)

    s.text(1070, 346, '跨 Loop contracts', 21, AMBER, 650, max_width=260)
    s.node('directive', 1070, 385, 260, 145, 'Directive', (
        '目标 / 输入 / 策略',
        '执行 / 学习 / 评测',
    ), AMBER, '#FFF5DD', '#E4C47F', title_size=26)
    s.path('M 1000 440 H 1070', BLUE, arrow='blue')
    s.label(1035, 420, 'explore', BLUE, 16)
    s.path('M 1330 440 H 1340 V 410 H 1400', BLUE, arrow='blue')
    s.node('feedback', 1070, 560, 260, 150, 'Feedback', (
        '结果 / 状态 / 指标',
        'evidence_refs',
        '引用系统记录',
    ), AMBER, '#FFF5DD', '#E4C47F', title_size=26)
    # Feedback advances control state; it is not a separate cognitive module.
    s.path('M 1070 630 H 1020 V 570 H 980 V 530', AMBER, arrow='amber')
    s.text(590, 580, 'loop.py · receive_feedback', 22, BLUE, 650, max_width=410)
    s.text(590, 614, '接收反馈 / 推进控制状态', 20, MUTED, max_width=410)

    s.node('exploration-control', 1400, 385, 860, 95, '执行控制 / loop.py', (
        '接收指令、推进状态；分派执行 / 学习 / 评测，汇总 Feedback',
    ), GREEN, stroke='#9ACBB6', title_size=25)
    s.path('M 1400 455 H 1355 V 630 H 1330', AMBER, arrow='amber')
    s.node('actor-policy', 1400, 540, 390, 140, 'Actor Policy / Context', (
        'Context ← 本地 history',
        'Action Space / Tools',
    ), GREEN, stroke='#9ACBB6', title_size=24)
    s.node('environment', 1870, 540, 390, 140, 'Environment / 执行', (
        'reset / step / close',
        '执行行动并返回结果',
    ), GREEN, stroke='#9ACBB6', title_size=24)
    s.path('M 1595 480 V 540', GREEN, arrow='green')
    s.path('M 1790 610 H 1870', GREEN, arrow='green')
    s.label(1830, 591, '行动', GREEN, 18)
    s.node('exploration-memory', 1400, 780, 860, 170, 'Harness-local memory/', (
        'test-time history / working memory / 长期任务知识',
        '原始 Actor trajectories 留在本地；供 RL 读取',
        '自身 Harness 的上下文、任务记忆与执行轨迹',
    ), GREEN, '#EFF8F3', '#9ACBB6', title_size=26)
    s.path('M 1510 780 V 680', GREEN, arrow='green')
    s.label(1510, 736, 'Actor 读取 history', GREEN, 19)
    s.path('M 2065 680 V 780', GREEN, arrow='green')
    s.label(2065, 736, '写入 history / 原始轨迹', GREEN, 19)
    s.node('actor-learner', 1400, 1070, 390, 145, 'Actor Learner / RL', (
        '从本地 memory 读取轨迹',
        'SFT / RL 作业与版本更新',
    ), GREEN, stroke='#9ACBB6', title_size=24)
    s.node('evaluation', 1870, 1070, 390, 145, 'Evaluation', (
        '候选模型 / 评测协议',
        '产出评测反馈与证据引用',
    ), GREEN, stroke='#9ACBB6', title_size=25)
    s.path('M 1680 950 V 1070', GREEN, arrow='green')
    s.label(1680, 1025, 'RL 读取轨迹', GREEN, 19)
    s.text(1400, 1244, '执行、训练与评测结果由 loop.py 汇总为 Feedback。', 19, MUTED)

    # Only system feedback and improvement knowledge cross into Experience.
    for x in (570, 1830):
        s.path(f'M {x} 1260 V 1370', PURPLE, arrow='purple')
        s.label(x, 1317, '提交系统 / 训练 / 评测反馈、改进方案与结论', PURPLE, 19)
    s.rect(100, 1370, 2200, 290, '#F2EBFB', '#B89CDC', rx=24)
    s.text(140, 1416, 'Experience / 两个 Harness 共享的系统反馈', 30, PURPLE, 700)
    s.text(140, 1453, 'Decision Policy 可查询两个 Loop 的系统反馈，形成改进方案与结论。', 22, PURPLE)
    s.path('M 100 1450 H 55 V 470 H 140', PURPLE, arrow='purple', dash='9 7')
    for ident, x, title, lines in [
        ('system-feedback', 140, '系统运行反馈', ('Decision-making + Exploration', '耗时 / token / errors')),
        ('training-evaluation-feedback', 860, '训练与评测反馈', ('训练结果 / 评测指标 / 版本关联', '支持比较、分析与改进判断')),
        ('improvement-evidence', 1580, '改进方案、结论与证据引用', ('方案 / 结论 / 来源关联', 'Feedback.evidence_refs → 系统记录')),
    ]:
        s.node(ident, x, 1475, 680, 130, title, lines, PURPLE,
               stroke='#D7C5EC', title_size=25)
    s.text(140, 1637, 'Experience 不持有 memory，也不保存原始 Actor 轨迹；history、working memory 与长期任务知识由各 Harness 管理。', 22, MUTED)

    s.node('runtime', 100, 1720, 1050, 145, 'Runtime / 底层共享服务', (
        'Model / Sandbox / Artifacts；Runner / Jobs 提供运行协调',
        '执行编排仍在各 Loop；策略、上下文与 memory 属于各 Harness',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=28)
    s.node('snapshot', 1190, 1720, 1110, 145, 'Snapshot / 分区保存与恢复', (
        '分别保存 Decision memory / Exploration memory / Experience',
        '同时组织控制状态、模型绑定与产物引用；由 application 装配',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=28)
    s.text(100, 1900, '实线：执行或数据流向；紫色虚线：Decision Policy 查询共享反馈。Harness 沿用 Loop 内模块划分，无额外目录或平行类。', 21, MUTED)
    s.save('system-flow.svg')


def file_architecture():
    s = SVG(2400, 1900, 'RSISys 文件架构图',
            '沿用 loops 目录：每个 Loop 的 Harness 包含 policy/context、memory、space/tools 和执行控制；'
            'learner/evaluation 为 RSI 扩展。contracts/memory.py 和 loops/_memory.py 仅共享协议和工具，'
            '不构成全局 memory。Experience 仅保存系统反馈与改进记录，Snapshot 分区保存两份 memory 与 Experience。')
    s.text(100, 80, 'RSISys / 文件架构', 42, weight=750)
    s.text(100, 126, 'src/rsisys/ · 两个 Loop + 一个共享 Experience · Harness 复用现有模块划分', 25, MUTED)
    s.chip(1990, 52, 310, '概念归属 / 关键模块', '#E3F3EC', GREEN, 20)

    for ident, x, title, color, fill, stroke in [
        ('decision-dir', 100, 'loops/decision_making/', BLUE, '#EFF5FF', '#BCD1F2'),
        ('exploration-dir', 870, 'loops/exploration/', GREEN, '#EFF8F3', '#B9DDCF'),
    ]:
        s.parts.append(f'<g id="{ident}">')
        s.rect(x, 210, 730, 910, fill, stroke, rx=24)
        s.text(x+40, 260, title, 29, color, 700, mono=True, max_width=650)
        s.text(x+40, 299, 'Loop = Harness + RSI 扩展', 23, MUTED)
        s.rect(x+25, 330, 680, 540, '#FFFFFF', stroke, rx=20)
        s.text(x+50, 376, 'Harness / 该 Loop 内的组件整体', 26, color, 700)
        s.text(x+50, 424, 'policy/ + context  ·  策略与上下文构建', 23, INK)
        s.node(ident+'-memory', x+50, 455, 630, 170, 'memory/  ·  Harness-local', (
            '每个 Loop 自有模块与默认实现',
            'test-time history / working memory / 长期任务知识',
            'Decision：决策上下文' if x == 100 else 'Exploration：Actor history + 原始轨迹 → RL',
        ), color, fill, stroke, title_size=25)
        if x == 100:
            s.text(x+50, 674, 'decision_space/ + tools', 24, color, 650, mono=True)
            s.text(x+50, 710, '文献 / 数据 / 策略操作与产物引用', 22, MUTED)
            s.text(x+50, 763, 'loop.py / state.py  ·  执行编排与控制状态', 23, INK)
            s.text(x+50, 802, 'receive_feedback：接收反馈、推进控制状态', 22, MUTED)
            s.text(x+50, 838, 'Policy 可查询两个 Loop 的共享系统反馈', 22, MUTED)
            s.node(ident+'-rsi', x+25, 910, 680, 170, 'RSI 扩展 / learner/', (
                'Decision Learner：训练作业控制 / 模型绑定',
                '改进方案、训练反馈与结论 → Experience',
            ), color, stroke=stroke, title_size=26)
        else:
            s.text(x+50, 674, 'action_space/ + tools + environment/', 24, color, 650, mono=True)
            s.text(x+50, 710, '行动定义、工具调用与环境执行', 22, MUTED)
            s.text(x+50, 763, 'loop.py / state.py  ·  执行编排与控制状态', 23, INK)
            s.text(x+50, 802, '指令生命周期：接收 / 推进 / 检查 / 取消', 22, MUTED)
            s.text(x+50, 838, 'Actor 从自身 memory 读取 history', 22, MUTED)
            s.node(ident+'-rsi', x+25, 910, 680, 170, 'RSI 扩展 / learner/ + evaluation/', (
                'RL 从 Exploration memory 读取原始轨迹',
                '训练 / 评测反馈与结论 → Experience',
            ), color, stroke=stroke, title_size=25)
        s.parts.append('</g>')

    s.parts.append('<g id="experience-dir">')
    s.rect(1640, 210, 660, 910, '#F2EBFB', '#B89CDC', rx=24)
    s.text(1680, 260, 'experience/', 31, PURPLE, 700, mono=True)
    s.text(1680, 299, '两个 Harness 共享的系统反馈', 25, PURPLE)
    for ident, y, title, lines in [
        ('experience-runtime-feedback', 350, '系统运行反馈', ('两个 Loop 的耗时 / token / errors', '供 Decision Policy 分析和制定改进')),
        ('experience-learning-feedback', 535, '训练与评测反馈', ('训练结果 / 评测指标 / 版本关联', '支持效果比较与改进判断')),
        ('experience-improvement', 720, '改进方案与结论', ('方案 / 结论 / 证据引用 / 来源关联', 'Feedback.evidence_refs 引用系统记录')),
    ]:
        s.node(ident, 1680, y, 580, 150, title, lines, PURPLE,
               stroke='#D7C5EC', title_size=27)
    s.text(1680, 927, '记录 / 检索 / 存储实现', 25, PURPLE, 650)
    s.text(1680, 971, '不拥有 memory，不保存原始 Actor 轨迹。', 23, MUTED)
    s.text(1680, 1012, '原始轨迹保留在 Exploration memory。', 23, MUTED)
    s.text(1680, 1053, '各 Harness 的任务知识由自身 memory 管理。', 22, MUTED)
    s.parts.append('</g>')

    s.rect(100, 1170, 2200, 235, '#FFF8E8', '#E4C47F', rx=24)
    s.text(140, 1217, '跨 Loop 共享契约与复用工具 / 不持有全局 memory 实例', 28, AMBER, 700)
    s.text(140, 1269, 'loops/contracts/', 26, AMBER, 700, mono=True)
    s.text(140, 1308, 'Directive / Feedback / coordination；模型、训练与引用协议', 23, MUTED)
    s.text(140, 1347, 'memory.py：共享 memory 数据类型与协议', 23, MUTED)
    s.text(1280, 1269, 'loops/_memory.py', 26, AMBER, 700, mono=True)
    s.text(1280, 1308, '复用存储工具，由各 Loop 的 memory/ 默认实现使用', 23, MUTED)
    s.text(1280, 1347, '协议与工具不构成第三套 Harness 或共享 memory 系统', 23, MUTED)

    s.node('runtime-dir', 100, 1455, 1040, 325, 'runtime/  ·  底层公共服务', (
        'Model：模型调用与绑定',
        'Sandbox：执行环境 / 进程隔离',
        'Artifacts：产物保存与引用',
        'Runner / Jobs：运行协调与作业记账',
        '',
        '各 Loop 的 loop.py 保留执行编排职责',
        'Policy、Context、Memory、Space / Tools 归所属 Harness',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=29)
    s.node('snapshot-dir', 1180, 1455, 1120, 325, 'snapshot/ + application 装配与持久化', (
        'Snapshot 分别保存 / 恢复两边 memory 和共享 Experience',
        '同时组织控制状态、模型绑定与产物引用',
        '',
        'Decision Harness memory → decision_memory.json',
        'Exploration Harness memory → exploration_memory.json',
        'Experience → 独立保存的反馈、方案、结论与证据引用',
        'application 负责装配与分别持久化，不合并两份 memory',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=29)
    s.text(100, 1833, 'Harness 是 Loop 内现有组件的整体称谓；沿用 loop 目录，不增加 harness/ 目录或平行 Harness 类。', 24, MUTED)
    s.text(100, 1876, '本图展示模块归属与数据边界；memory 的共享协议 / 存储工具与各 Loop 拥有的 memory 实例分别标示。', 22, MUTED)
    s.save('file-architecture.svg')


if __name__ == '__main__':
    system_flow()
    file_architecture()
