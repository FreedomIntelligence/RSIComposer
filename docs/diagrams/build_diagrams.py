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
    s = SVG(2400, 1750, 'RSISys 系统流程图',
            '默认 Decision Harness 包含 DecisionPolicy，默认 Exploration Harness 包含 Exploration Policy（ActorPolicy）。'
            'DecisionPolicy 选择系统指令，由 Exploration Harness 分派；ActorPolicy 仅选择环境动作。'
            '两者共享 Experience，Action Space 分别约束决策与环境动作。')
    s.text(100, 80, 'RSISys / 系统流程', 42, weight=750)
    s.text(100, 126, '两个 Loop · 默认 Harness 内的 Policy 与执行流程', 26, MUTED)

    for x, w, title, color, fill, stroke in [
        (100, 930, 'Decision-making Loop', BLUE, '#EFF5FF', '#BCD1F2'),
        (1320, 980, 'Exploration Loop', GREEN, '#EFF8F3', '#B9DDCF'),
    ]:
        s.rect(x, 170, w, 1030, fill, stroke, rx=24)
        s.text(x+35, 220, title, 32, color, 700)

    s.parts.append('<g id="decision-harness">')
    s.rect(130, 270, 870, 700, '#FFFFFF', '#9FBEEB')
    s.text(170, 315, 'Decision Harness', 29, BLUE, 650)
    s.node('decision-policy', 170, 365, 790, 130, 'Decision Policy', (
        'DecisionPolicy.decide(context) → Decision',
        '选择准备操作、系统指令、等待或结束',
    ), BLUE, '#F4F8FF', '#9FBEEB', title_size=30)
    s.node('decision-execution', 170, 620, 790, 150, '决策执行', (
        '本地准备 → tools',
        '系统指令 → Exploration Harness',
    ), BLUE, stroke='#9FBEEB', title_size=28)
    s.path('M 565 495 V 620', BLUE, arrow='blue')
    s.label(565, 562, 'Decision', BLUE, 22)
    s.path('M 170 695 H 150 V 430 H 170', BLUE, arrow='blue', dash='7 6')
    s.text(170, 875, '自身学习：Decision Learner（显式调用）', 23, MUTED)
    s.parts.append('</g>')

    s.node('decision-action-space', 130, 1030, 870, 120, 'Decision Action Space', (
        '定义与校验：准备操作 + 系统指令',
    ), BLUE, stroke='#9FBEEB', title_size=28)
    s.path('M 900 1030 V 770', BLUE, arrow='blue')
    s.label(900, 1000, '描述 / 校验', BLUE, 20)

    s.parts.append('<g id="exploration-harness">')
    s.rect(1350, 270, 920, 700, '#FFFFFF', '#9ACBB6')
    s.text(1390, 315, 'Exploration Harness', 29, GREEN, 650)
    s.node('directive-dispatch', 1390, 350, 840, 100, '指令分派', (
        '接收 Directive，选择执行流程',
    ), GREEN, stroke='#9ACBB6', title_size=28)
    s.node('exploration-policy', 1390, 535, 390, 130, 'Exploration Policy', (
        'ActorPolicy.act(context)',
        '选择环境动作',
    ), GREEN, '#F3FAF6', '#9ACBB6', title_size=27)
    s.node('system-execution', 1840, 535, 390, 130, '训练 / 配置 / 评测', (
        'learner · prompt · evaluator',
        '由 Harness 直接调用',
    ), GREEN, stroke='#9ACBB6', title_size=25)
    s.path('M 1585 450 V 535', GREEN, arrow='green')
    s.label(1585, 496, 'execute / RL rollout', GREEN, 20)
    s.path('M 2035 450 V 535', GREEN, arrow='green')
    s.label(2035, 496, 'SFT / prompt / evaluate', GREEN, 20)
    s.node('environment', 1390, 750, 390, 110, 'Environment', (
        '执行动作，返回观察',
    ), GREEN, stroke='#9ACBB6', title_size=27)
    s.node('command-results', 1840, 750, 390, 110, '结果与候选', (
        '训练结果 / 评测指标 / 配置',
    ), GREEN, stroke='#9ACBB6', title_size=27)
    s.path('M 1585 665 V 750', GREEN, arrow='green')
    s.label(1585, 712, '动作', GREEN, 20)
    s.path('M 1390 805 H 1370 V 600 H 1390', GREEN, arrow='green', dash='7 6')
    s.path('M 2035 665 V 750', GREEN, arrow='green')
    s.path('M 1780 810 H 1815 V 630 H 1840', GREEN, arrow='green', dash='7 6')
    s.label(1815, 715, 'RL 轨迹', GREEN, 19)
    # Applied model/prompt changes affect subsequent policy calls, not environment actions.
    s.path('M 2035 860 V 920 H 1795 V 680 H 1745 V 665', GREEN, arrow='green', dash='7 6')
    s.label(2010, 924, '训练 / prompt 按配置应用', GREEN, 19)
    s.parts.append('</g>')

    s.node('exploration-action-space', 1350, 1030, 920, 120, 'Actor Action Space', (
        '定义与校验：环境动作（execute episode / RL rollout）',
    ), GREEN, stroke='#9ACBB6', title_size=28)
    s.path('M 1450 1030 V 860', GREEN, arrow='green')
    s.label(1450, 1000, '描述 / 校验', GREEN, 20)

    s.node('external-system', 1060, 170, 230, 100, '外部控制器', (
        '可直接提交指令',
    ), AMBER, '#FFF5DD', '#E4C47F', title_size=25)
    s.node('directive', 1060, 350, 230, 180, 'Directive', (
        'execute · sft',
        'rl · evaluate',
        'prompt 配置',
    ), AMBER, '#FFF5DD', '#E4C47F', title_size=26)
    s.path('M 1175 270 V 350', AMBER, arrow='amber')
    s.path('M 960 650 H 1045 V 405 H 1060', BLUE, arrow='blue')
    s.path('M 1290 405 H 1390', BLUE, arrow='blue')
    s.node('feedback', 1060, 850, 230, 120, 'Feedback', (
        '状态 / 指标',
        '候选 / 证据引用',
    ), AMBER, '#FFF5DD', '#E4C47F', title_size=26)
    s.path('M 1350 890 H 1290', AMBER, arrow='amber')
    s.path('M 1060 910 H 1020 V 430 H 960', AMBER, arrow='amber')
    s.text(1175, 1090, 'Runner 协调交付', 20, AMBER, 650, anchor='middle')

    for x in (600, 1810):
        s.path(f'M {x} 1200 V 1270', PURPLE, arrow='purple')
        s.label(x, 1244, '系统反馈', PURPLE, 20)
    s.rect(100, 1270, 2200, 200, '#F2EBFB', '#B89CDC', rx=24)
    s.text(140, 1318, 'Experience / 共享系统反馈与改进证据', 30, PURPLE, 700)
    for ident, x, title, line in [
        ('runtime-feedback', 140, '运行反馈', '耗时 / token / errors'),
        ('learning-feedback', 860, '训练与评测', '效果 / 效率 / 模型版本'),
        ('improvement-evidence', 1580, '改进结论', '策略结果 / 总结 / 证据引用'),
    ]:
        s.node(ident, x, 1345, 680, 95, title, (line,), PURPLE,
               stroke='#D7C5EC', title_size=25)
    s.path('M 100 1370 H 55 V 430 H 170', PURPLE, arrow='purple', dash='9 7')
    s.label(55, 1230, '检索', PURPLE, 18)

    s.node('runtime', 100, 1520, 1050, 145, 'Runtime / 共享服务与协调', (
        'Model · Sandbox · Artifacts · Jobs · Runner',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=28)
    s.node('snapshot', 1190, 1520, 1110, 145, 'Snapshot / 保存与恢复', (
        '内置实现支持空闲边界快照；自定义组件需显式集成',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=28)
    s.text(100, 1712, '当前训练与评测后端为 Mock。', 21, MUTED)
    s.save('system-flow.svg')


def file_architecture():
    s = SVG(2400, 1470, 'RSISys 文件架构图',
            'DecisionPolicy 位于 decision_making/harness/policy/；Exploration Policy 的代码接口 ActorPolicy 位于 exploration/harness/policy/。'
            '每个 Loop 的 Harness 与 Action Space 平级，共享 Experience、跨 Loop 协议、Runtime 和 Snapshot。')
    s.text(100, 80, 'RSISys / 文件架构', 42, weight=750)
    s.text(100, 126, 'src/rsisys/ · Policy 位于默认 Harness 内部', 26, MUTED)

    for ident, x, title, policy, interface, color, fill, stroke in [
        ('decision-dir', 100, 'loops/decision_making/', 'DecisionPolicy',
         'decide(context) → Decision', BLUE, '#EFF5FF', '#BCD1F2'),
        ('exploration-dir', 870, 'loops/exploration/', 'ActorPolicy',
         'Exploration Policy · act(context)', GREEN, '#EFF8F3', '#B9DDCF'),
    ]:
        s.parts.append(f'<g id="{ident}">')
        s.rect(x, 170, 730, 860, fill, stroke, rx=24)
        s.text(x+35, 220, title, 29, color, 700, mono=True, max_width=660)
        s.text(x+35, 267, 'loop.py  ·  Harness 与 Action Space 的组合端点', 22, MUTED)
        s.rect(x+30, 310, 670, 490, '#FFFFFF', stroke)
        s.text(x+60, 355, 'harness/', 29, color, 650, mono=True)
        s.text(x+60, 395, '__init__.py：Harness 协议', 22, MUTED)
        s.node(ident+'-policy', x+60, 430, 610, 120, 'policy/  ·  '+policy,
               (interface,), color, fill, stroke, title_size=28)
        if ident == 'decision-dir':
            lines = (
                'engine.py：决策执行编排',
                'tools/ · learner/ · memory/',
                'decision.py · state.py',
            )
            space = 'DecisionActionSpace · 准备操作与系统指令'
        else:
            lines = (
                'engine.py · commands.py：执行与指令分派',
                'environment/ · learner/ · evaluation/ · memory/',
                'rl.py · state.py',
            )
            space = 'ActionSpace · Actor 环境动作'
        for i, line in enumerate(lines):
            s.text(x+60, 610+i*51, line, 22, MUTED, max_width=610)
        s.node(ident+'-action-space', x+30, 850, 670, 120, 'action_space/',
               (space,), color, stroke=stroke, title_size=28)
        s.parts.append('</g>')

    s.rect(1640, 170, 660, 860, '#F2EBFB', '#B89CDC', rx=24)
    s.text(1680, 220, 'experience/', 31, PURPLE, 700, mono=True)
    s.text(1680, 267, '跨轮系统反馈与改进证据', 23, MUTED)
    for ident, y, title, line in [
        ('experience-records', 310, 'records.py · provenance.py', '记录类型与来源关联'),
        ('experience-retrieval', 530, 'retrieval.py', '写入与检索接口'),
        ('experience-storage', 750, 'storage.py · sqlite.py', '内存 / SQLite 存储'),
    ]:
        s.node(ident, 1680, y, 580, 160, title, (line,), PURPLE,
               stroke='#D7C5EC', title_size=26)

    s.node('contracts-dir', 100, 1080, 2200, 120, 'loops/contracts/  ·  跨 Loop 消息与公开端口', (
        'Directive · Feedback · coordination · requests · jobs · model_service · artifacts · candidate · training',
    ), AMBER, '#FFF8E8', '#E4C47F', title_size=28)
    s.node('runtime-dir', 100, 1250, 1050, 170, 'runtime/  ·  共享服务与协调', (
        'model.py · sandbox.py · local_process.py',
        'artifacts.py · files.py · jobs.py · runner.py',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=28)
    s.node('snapshot-dir', 1190, 1250, 1110, 170, 'snapshot/ + 系统装配', (
        'snapshot/：捕获 / 编解码 / 保存 / 恢复',
        'application.py · config.py：系统装配',
    ), INK, '#EAF0F6', '#CBD6E3', title_size=28)
    s.save('file-architecture.svg')


if __name__ == '__main__':
    system_flow()
    file_architecture()
