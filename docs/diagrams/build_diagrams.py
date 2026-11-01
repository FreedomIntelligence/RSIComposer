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
        width = width or round(sum(1 if unicodedata.east_asian_width(c) in ('W','F') else .58 for c in text)*size+20, 3)
        self.rect(x-width/2, y-size+1, width, size+10, fill='#F6F8FC', stroke='none', rx=6, sw=0)
        self.text(x, y, text, size, color, anchor='middle', max_width=width-8)

    def chip(self, x, y, w, label, fill, color, size=18):
        self.rect(x, y, w, 34, fill=fill, stroke='none', rx=17, sw=0)
        self.text(x+w/2, y+24, label, size, color, 600, 'middle', max_width=w-18)

    def node(self, ident, x, y, w, h, title, lines=(), color=INK, fill='#FFFFFF', stroke='#D5DEEA', title_size=26, tag=None, dash=None):
        self.parts.append(f'<g id="{ident}">')
        self.rect(x,y,w,h,fill,stroke,rx=16,sw=1.8,dash=dash)
        if tag:
            self.text(x+24,y+25,tag,15,color,700,max_width=w-48)
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


def legend(s):
    s.rect(100, 158, 42, 26, '#EAF0F6', LINE, rx=6)
    s.text(156, 179, '框架骨架', 20, MUTED)
    s.rect(350, 158, 42, 26, '#F4F8FF', BLUE, rx=6, dash='8 5')
    s.text(406, 179, '整体可换 / 实现可换（遵守接口）', 20, MUTED)
    s.rect(980, 158, 42, 26, '#FFF8E8', AMBER, rx=6, dash='8 5')
    s.text(1036, 179, '可换 · 需适配', 20, MUTED)
    s.text(2300, 179, '蓝 / 绿：两个 Loop    紫：共享 Experience', 20, MUTED, anchor='end')


def replaceable(s, ident, x, y, w, h, title, lines=(), color=BLUE,
                fill='#FFFFFF', title_size=27, whole=False, adapt=False):
    s.node(ident, x, y, w, h, title, lines, color, fill, color,
           title_size=title_size,
           tag='可换 · 需适配' if adapt else '整体可换' if whole else '实现可换',
           dash='8 5')


def system_flow():
    s = SVG(2400, 1920, 'RSISys 系统流程图',
            '两个 Loop 的 Harness 和 ActionSpace 可整体替换；图中展示默认 Harness 的可替换内部实现。'
            'Decision Policy 自主选择科研步骤，Actor Policy 选择环境动作；外部权限过滤与执行检查独立约束能力。'
            'Decision step / review 共享操作执行；默认 Exploration Harness 仅协调，内部执行组件显式接收依赖。'
            'Experience、模型、训练、环境、沙箱和产物后端可替换。Snapshot 实现可替换但自定义模块需显式适配。'
            'Loop protocol 与 Runner 协同是框架骨架，保留单 inflight 约束。')
    s.text(100, 80, 'RSISys / 系统流程', 42, weight=750)
    s.text(100, 126, '模型自主科研流程 · 框架协议协同 · 可替换的执行实现', 26, MUTED)
    legend(s)

    for x, w, title in [(100, 930, 'Decision-making Loop'), (1320, 980, 'Exploration Loop')]:
        s.rect(x, 230, w, 1150, '#EAF0F6', LINE, rx=24)
        s.text(x+35, 280, title, 32, INK, 700)
        s.text(x+w-35, 280, 'Loop protocol · 骨架', 19, MUTED, anchor='end')

    s.parts.append('<g id="decision-harness">')
    s.rect(130, 330, 870, 780, '#EFF5FF', BLUE, dash='8 5')
    s.text(170, 379, 'Decision Harness', 29, BLUE, 650)
    s.chip(815, 352, 145, '整体可换', '#DCE9FC', BLUE)
    replaceable(s, 'decision-policy', 170, 425, 790, 155, 'Decision Policy', (
        'decide(context) → Decision · review(context)',
        '模型自主选择操作、指令、复查、等待或结束',
    ), title_size=30)
    s.node('decision-execution', 170, 680, 790, 140, 'step / review · 共享操作执行', (
        'tools.execute_operation → 校验外部权限 / 执行 / 回执',
        'step 等待异步作业；review 接收提交回执；指令 → Exploration',
    ), BLUE, stroke='#9FBEEB', title_size=28)
    s.path('M 565 580 V 680', BLUE, arrow='blue')
    s.label(565, 637, 'Decision / ReviewDecision', BLUE, 22)
    s.path('M 170 750 H 150 V 505 H 170', BLUE, arrow='blue', dash='7 6')
    for ident, x, w, title, line in [
        ('decision-tools', 170, 250, 'tools', '现有操作绑定'),
        ('decision-memory', 440, 250, 'memory', '本地工作记忆'),
        ('decision-learner', 710, 250, 'learner', '显式自身学习'),
    ]:
        replaceable(s, ident, x, 910, w, 130, title, (line,), title_size=26)
    s.path('M 295 820 V 910', BLUE, arrow='blue')
    s.parts.append('</g>')

    replaceable(s, 'decision-action-space', 130, 1180, 870, 170, 'Decision ActionSpace', (
        '能力：显式定义 / 外部 provider 发现；外部权限过滤与校验',
        'review 默认只读；可开放全部操作或 allowlist',
    ), whole=True, title_size=28)
    s.path('M 945 1180 V 1110', BLUE, arrow='blue')
    s.label(775, 1152, '描述 / 校验 / 执行检查', BLUE, 20)

    s.parts.append('<g id="exploration-harness">')
    s.rect(1350, 330, 920, 780, '#EFF8F3', GREEN, dash='8 5')
    s.text(1390, 379, 'Exploration Harness', 29, GREEN, 650)
    s.chip(2085, 352, 145, '整体可换', '#D7EEE2', GREEN)
    s.node('directive-dispatch', 1390, 410, 840, 105, '默认 Harness · 仅装配与指令协调', (
        'Directive → execute / SFT / RL / prompt / evaluate',
    ), GREEN, stroke='#9ACBB6', title_size=28)
    s.path('M 1810 515 V 540', GREEN, arrow='green')
    s.parts.append('<g id="exploration-execution">')
    s.rect(1390, 540, 840, 285, stroke=GREEN, dash='8 5')
    s.text(1414, 580, '默认内部执行职责', 25, GREEN, 650, max_width=600)
    s.text(2206, 580, '实现可换', 18, GREEN, 600, anchor='end')
    for y, flow in [
        (630, 'EpisodeExecution → Actor Policy / Environment → 轨迹'),
        (680, 'LearningExecution → learner → 候选 / 应用'),
        (730, 'EvaluationExecution → evaluator → 指标 / 证据'),
        (780, 'LocalRLExecution：rollout → 训练 → 应用'),
    ]:
        s.text(1414, y, flow, 23, GREEN, 500, max_width=792)
    s.parts.append('</g>')
    replaceable(s, 'exploration-policy', 1390, 850, 390, 110, 'Actor Policy', (
        'act(context) → 选择环境动作',
    ), GREEN, title_size=27)
    replaceable(s, 'system-execution', 1840, 850, 390, 110, 'learner / evaluator', (
        '训练 / 评测后端',
    ), GREEN, title_size=25)
    replaceable(s, 'environment', 1390, 985, 390, 110, 'Environment', (
        '环境后端：动作 → 观察',
    ), GREEN, title_size=27)
    replaceable(s, 'exploration-memory', 1840, 985, 390, 110, 'memory', (
        '本地工作记忆',
    ), GREEN, title_size=26)
    s.path('M 1585 825 V 850', GREEN, arrow='green')
    s.path('M 2035 825 V 850', GREEN, arrow='green')
    s.path('M 1585 960 V 985', GREEN, arrow='green')
    s.path('M 1390 1040 H 1370 V 920 H 1390', GREEN, arrow='green', dash='7 6')
    s.parts.append('</g>')

    replaceable(s, 'exploration-action-space', 1350, 1180, 920, 170, 'Actor ActionSpace', (
        '环境动作的定义与校验', '供 execute episode / RL rollout 使用',
    ), GREEN, whole=True, title_size=28)
    s.path('M 1450 1180 V 1110', GREEN, arrow='green')
    s.label(1620, 1152, '描述 / 校验', GREEN, 20)

    s.node('external-system', 1060, 230, 230, 100, '外部控制器', (
        '可直接提交指令',
    ), INK, '#EAF0F6', LINE, title_size=25)
    s.node('directive', 1060, 410, 230, 180, 'Directive', (
        'execute · sft', 'rl · evaluate', 'prompt 配置',
    ), INK, '#EAF0F6', LINE, title_size=26)
    s.path('M 1175 330 V 410')
    s.path('M 960 710 H 1045 V 465 H 1060', BLUE, arrow='blue')
    s.path('M 1290 465 H 1390', BLUE, arrow='blue')
    s.node('runner', 1060, 680, 230, 180, 'Runner', (
        '框架协同 / 交付', '单 inflight', '推进·复查·取消',
    ), INK, '#EAF0F6', LINE, title_size=26)
    s.node('feedback', 1060, 960, 230, 120, 'Feedback', (
        '状态 / 指标', '候选 / 证据引用',
    ), INK, '#EAF0F6', LINE, title_size=26)
    s.path('M 1350 1020 H 1290')
    s.path('M 1060 1020 H 1020 V 505 H 960')

    for x in (600, 1810):
        s.path(f'M {x} 1380 V 1450', PURPLE, arrow='purple')
        s.label(x, 1424, '系统反馈', PURPLE, 20)
    replaceable(s, 'experience', 100, 1450, 2200, 150, 'Experience / 共享系统反馈与改进证据', (
        '运行反馈 · 训练与评测 · 改进结论与证据引用',
        'Experience 实现 / 检索 / 存储后端可换；两个 Loop 各自保留工作记忆',
    ), PURPLE, '#F2EBFB', title_size=30)
    s.path('M 100 1525 H 55 V 505 H 170', PURPLE, arrow='purple', dash='9 7')
    s.label(55, 1410, '检索', PURPLE, 18)

    replaceable(s, 'runtime-backends', 100, 1660, 1050, 165, 'Runtime / 共享后端', (
        'Model · Sandbox · Artifact 后端',
        '通过公开端口接入；Runner 协同见框架骨架',
    ), BLUE, '#EFF5FF', title_size=28)
    replaceable(s, 'snapshot', 1190, 1660, 1110, 165, 'Snapshot / 保存与恢复', (
        '内置实现在空闲边界捕获与恢复',
        '自定义模块需显式适配保存 / 恢复 / 引用一致性',
    ), AMBER, '#FFF8E8', title_size=28, adapt=True)
    s.text(100, 1875, '虚线框内部展示默认组合；整体替换 Harness 不要求保留相同内部结构。默认训练 / 评测后端为 Mock。', 22, MUTED)
    s.save('system-flow.svg')


def file_architecture():
    s = SVG(2400, 1840, 'RSISys 文件架构图',
            '两个 Loop 的 Harness 与 ActionSpace 可整体替换，Policy、tools、memory、learner、evaluator 等实现可替换。'
            'Decision ActionSpace 下的 access.py 负责外部权限过滤和执行检查，capabilities.py 负责显式定义或外部 provider 发现。'
            'step / review 共用 tools 中的操作执行；默认 Exploration Harness 仅协调，四项执行职责位于现有文件并显式接收依赖。'
            '跨 Loop 协议与 Runner 是框架骨架。Experience 与服务后端可换，Snapshot 自定义模块需显式适配。')
    s.text(100, 80, 'RSISys / 文件架构', 42, weight=750)
    s.text(100, 126, 'src/rsisys/ · 默认组合与替换边界', 26, MUTED)
    legend(s)

    for ident, x, title, policy, interface, color, fill in [
        ('decision-dir', 100, 'loops/decision_making/', 'Decision Policy',
         'DecisionPolicy · decide / review', BLUE, '#EFF5FF'),
        ('exploration-dir', 870, 'loops/exploration/', 'Actor Policy',
         'ActorPolicy · act(context)', GREEN, '#EFF8F3'),
    ]:
        s.parts.append(f'<g id="{ident}">')
        s.rect(x, 230, 730, 1100, '#EAF0F6', LINE, rx=24)
        s.text(x+35, 280, title, 29, INK, 700, mono=True, max_width=660)
        s.text(x+35, 327, 'loop.py · 协议端点 / 框架骨架', 23, MUTED)
        s.rect(x+30, 390, 670, 600, fill, color, dash='8 5')
        s.text(x+60, 438, 'harness/', 29, color, 650, mono=True)
        s.chip(x+510, 411, 150, '整体可换', '#FFFFFF', color)
        s.text(x+60, 480, '__init__.py：Harness 协议', 22, MUTED)
        replaceable(s, ident+'-policy', x+60, 515, 610, 130, 'policy/ · '+policy,
                    (interface,), color, title_size=28)
        if ident == 'decision-dir':
            s.text(x+60, 700, 'engine.py · decision.py · state.py', 23, MUTED)
            replaceable(s, ident+'-internals', x+60, 740, 610, 150, 'tools/ · learner/ · memory/', (
                'tools/__init__.py：step / review 共享操作执行',
                'execute_operation · record_operation',
            ), color, title_size=26)
            s.text(x+60, 945, 'step 等待作业；review 接收提交回执', 22, color)
            lines = ('access.py：外部权限过滤 / 执行检查',
                     'capabilities.py：显式定义 / 外部 provider 发现',
                     'directives.py · queries.py：指令与查询定义',
                     'review：默认只读 / 全部操作 / allowlist')
        else:
            s.text(x+60, 685, 'engine.py · commands.py：仅装配与指令协调', 22, MUTED, max_width=610)
            s.text(x+60, 721, 'state.py · memory/；下列执行实现可换', 21, color, max_width=610)
            s.parts.append('<g id="exploration-dir-internals">')
            s.rect(x+60, 740, 610, 225, '#FFFFFF', color, rx=16, dash='8 5')
            for y, source, execution, dependencies in [
                (770, 'environment/__init__.py', 'EpisodeExecution',
                 'policy · environment · memory · Experience · state'),
                (822, 'learner/__init__.py', 'LearningExecution',
                 'policy · memory · Experience · state · jobs · activator'),
                (874, 'evaluation/__init__.py', 'EvaluationExecution',
                 'Experience · state · JobBook(evaluator)'),
                (926, 'rl.py', 'LocalRLExecution',
                 'EpisodeExecution · LearningExecution · state'),
            ]:
                s.text(x+80, y, f'{source} → {execution}', 19, color, 600, max_width=570)
                s.text(x+80, y+25, dependencies, 16, MUTED, max_width=570)
            s.parts.append('</g>')
            lines = ('__init__.py：ActionSpace 协议 / 动作定义',
                     'local.py：本地命令动作',
                     '只定义与校验；环境负责执行')
        replaceable(s, ident+'-action-space', x+30, 1050, 670, 220, 'action_space/',
                    lines, color, whole=True, title_size=28)
        s.parts.append('</g>')

    s.rect(1640, 230, 660, 1100, '#F2EBFB', PURPLE, rx=24, dash='8 5')
    s.text(1680, 280, 'experience/', 31, PURPLE, 700, mono=True)
    s.chip(2110, 253, 150, '实现可换', '#FFFFFF', PURPLE)
    s.text(1680, 327, '共享反馈与证据 · ExperienceSystem', 23, MUTED)
    s.node('experience-records', 1680, 390, 580, 160, 'records.py · provenance.py', (
        '记录类型与来源关联',
    ), PURPLE, stroke='#D7C5EC', title_size=26)
    replaceable(s, 'experience-retrieval', 1680, 610, 580, 160, 'retrieval.py', (
        'Experience 写入与检索实现',
    ), PURPLE, title_size=27)
    replaceable(s, 'experience-storage', 1680, 830, 580, 160, 'storage.py · sqlite.py', (
        '存储后端；内置内存 / SQLite',
    ), PURPLE, title_size=27)
    s.text(1680, 1105, '共享系统反馈', 25, PURPLE, 650)
    s.text(1680, 1150, '运行 / 训练 / 评测 / 改进证据', 22, MUTED)
    s.text(1680, 1200, '工作记忆保留在两个 Harness 内', 22, MUTED)

    s.node('framework', 100, 1380, 2200, 140, '框架骨架 / Loop protocol + Runner 协同', (
        'loops/contracts/：Directive · Feedback · coordination · queries · requests · jobs · model_service · artifacts · candidate · training',
        'runtime/runner.py：交付 / 推进 / 复查 / 取消；单 inflight；科研决策由 Harness 内的 Policy 负责',
    ), INK, '#EAF0F6', LINE, title_size=28)
    replaceable(s, 'runtime-dir', 100, 1570, 1050, 170, 'runtime/ · 共享后端', (
        'model.py · sandbox.py · local_process.py · artifacts.py',
        'Model / Sandbox / Artifact 实现可换；files.py · jobs.py 配套',
    ), BLUE, '#EFF5FF', title_size=28)
    replaceable(s, 'snapshot-dir', 1190, 1570, 1110, 170, 'snapshot/ · 保存与恢复', (
        'capture.py · restore.py · storage.py / local.py · codec.py · manifest.py',
        '内置实现限空闲边界；自定义模块需显式适配保存 / 恢复 / 引用一致性',
    ), AMBER, '#FFF8E8', title_size=28, adapt=True)
    s.text(100, 1798, '装配入口：application.py · config.py · assembly.py    |    默认内部结构可随 Harness 整体替换；默认训练 / 评测为 Mock。', 22, MUTED)
    s.save('file-architecture.svg')


if __name__ == '__main__':
    system_flow()
    file_architecture()
