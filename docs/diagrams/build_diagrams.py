#!/usr/bin/env python3
"""Generate the canonical meta-agent / Lab architecture as editable native SVGs."""
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
        left = x if anchor == 'start' else x-width if anchor == 'end' else x-width/2
        assert 0 <= left and left+width <= self.width, (text, left, width)
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
                assert title_y+34+i*29 <= y+h-12, (ident, line, h)
                self.text(x+24,title_y+34+i*29,line,20,MUTED,max_width=w-48)
        self.parts.append('</g>')

    def save(self, name):
        self.parts.append('</svg>')
        s='\n'.join(self.parts)+'\n'
        ET.fromstring(s)
        (HERE/name).write_text(s, encoding='utf-8')
        print(f'{name}: {self.width}x{self.height}, {self.text_count} editable text elements')




def section(s, x, y, w, h, ident, title, lines, color, fill):
    s.node(ident, x, y, w, h, title, lines, color=color,
           fill=fill, stroke=color, title_size=28)


def heading(s, title, subtitle):
    s.text(72, 78, title, 40, weight=750)
    s.text(72, 121, subtitle, 23, MUTED)
    s.chip(1538, 50, 190, '2026-11-05', '#E8EEF8', BLUE, 18)


def system_flow():
    s = SVG(1800, 1370, 'RSISys：meta-agent 两侧职责与 Lab',
            'meta-agent（decision侧）选择任务，Lab 执行并返回原生结果。'
            'meta-agent（整理侧）的接入窗口按用途选择即时 context 或 memory 正文路径。'
            '即时路径不要求模型整理；memory 按需检索成上下文。'
            '两侧标准装配共享捕获的模型权重身份，使用独立上下文。'
            '原始归档与提案为独立可选能力；训练和固定正式评测在 Lab；看板只面向人。'
            '本图表示职责和信息流，不强制串行模型调用；兼容实现边界见 docs/architecture.md。')
    heading(s, 'RSISys / 统一系统架构',
            '两侧职责，一个 meta-agent；Lab 是执行环境，context 是下一次模型输入。')
    section(s, 72, 150, 1656, 92, 'host', 'Host / 宿主', (
        '统一装配 · 权限与资源准入 · 全部模型调用计费 · 截止时间 · 调度、恢复与 owned 资源清理',
    ), INK, '#EAF0F6')

    # Optional raw-event storage travels outside the working-context loop.
    s.path('M 900 300 V 274 H 36 V 1135 H 72', PURPLE,
           arrow='purple', dash='7 7')
    s.label(36, 986, '原文', PURPLE, 16)

    section(s, 72, 300, 480, 270, 'meta-agent-decision', 'meta-agent（decision侧）', (
        '根据目标、真实状态与 context 决策',
        '选择动作、写代码、发起 Lab 任务',
        '选择构建、训练、评测与应用',
        '实际执行受已安装能力和权限约束',
        'meta-agent harness 装配输入',
    ), BLUE, '#EFF5FF')
    section(s, 660, 300, 480, 270, 'lab', 'Lab / 执行环境', (
        '任务 · 数据 · 模型或 agent · harness',
        '工具 / 代码 / 训练 / 推理 / 评测',
        '内部组织自由；Actor 是可选实现',
        '输出原生文本、JSON、日志或引用',
        '执行设施提供任务身份与真实状态',
    ), GREEN, '#EFF8F3')
    section(s, 1248, 300, 480, 270, 'meta-agent-organization', 'meta-agent（整理侧）', (
        '反馈接入与路由 · 解释与记忆组织',
        '模型整理时：同权重、独立上下文',
        '即时路径无需额外整理模型调用',
        'memory provider 与算法可替换',
        '可选提出建议；执行仍由 decision侧',
    ), PURPLE, '#F4EFFB')
    s.path('M 552 435 H 660', BLUE, arrow='blue')
    s.label(606, 417, '动作 / 任务', BLUE, 16)
    s.path('M 1140 435 H 1248', GREEN, arrow='green')
    s.label(1194, 417, '原生结果', GREEN, 16)

    s.path('M 1488 570 V 620 H 900 V 680', PURPLE, arrow='purple')
    s.path('M 1488 620 V 680', PURPLE, arrow='purple')
    s.label(1194, 635, '正文主消费路径二选一', PURPLE, 20)
    section(s, 72, 680, 480, 200, 'current-context', '本次 context', (
        '当前目标、状态与必要事实',
        '选中的即时结果 + 检索记忆',
        '必要摘要 / 可选建议 / 原文入口',
        'harness 选择、计量与 compact',
    ), BLUE, '#EFF5FF')
    section(s, 660, 680, 480, 200, 'instant-context', '即时 context 路径', (
        '真实结果支持下一步快速决策',
        '不触发 LangMem / 长期记忆写入',
        '近期正文与活跃依赖有界保留',
        '旧正文退出视图，原文按引用读取',
    ), GREEN, '#EFF8F3')
    section(s, 1248, 680, 480, 200, 'retrievable-memory', 'memory 路径', (
        '整理可复用发现、经验与限制',
        'LangMem 或其他可替换实现',
        '保存可检索记忆与来源 / 修订',
        '模块可用，不代表每条结果都写入',
    ), PURPLE, '#F4EFFB')
    s.path('M 312 680 V 570', BLUE, arrow='blue')
    s.label(312, 630, '下一次输入', BLUE, 19)
    s.path('M 660 780 H 552', GREEN, arrow='green')
    s.label(606, 762, '真实结果', GREEN, 16)
    s.path('M 1488 880 V 940 H 312 V 880', PURPLE, arrow='purple')
    s.label(900, 933, '按需检索 → context 文本', PURPLE, 20)

    s.rect(72, 966, 1656, 44, '#FFF7E8', 'none', rx=10)
    s.text(900, 995,
           '可选改进提案：整理侧提出 → 经 context 交回 decision侧 → 以正式任务进入 Lab。',
           20, AMBER, 500, 'middle', max_width=1608)
    section(s, 72, 1050, 480, 200, 'raw-archive', '原始轨迹归档（独立可选）', (
        '公开原始事件、来源与覆盖范围',
        '默认开启，可关闭或替换',
        '原始轨迹不等于可训练数据集',
        '执行 journal 仍承担事实与恢复',
    ), PURPLE, '#F4EFFB')
    section(s, 660, 1050, 480, 200, 'lab-improvement', 'Lab 内改进任务（可选）', (
        '数据构建、SFT / RL、harness 构建',
        '由 decision侧正式调用与验证',
        '提案 ≠ 训练 ≠ 应用 / 采纳',
        '正式评测协议由领域装配固定',
    ), AMBER, '#FFF7E8')
    section(s, 1248, 1050, 480, 200, 'human-monitor', '看板 / Monitor（面向人）', (
        'Host 写公开运行投影 → 人查看',
        '不进入模型输入或反馈链路',
        '模型用 Lab / 状态工具读信息',
        '投影不等于完整历史或执行账本',
    ), INK, '#EAF0F6')
    s.path('M 552 1150 H 660', AMBER, arrow='amber', dash='7 7')
    s.label(606, 1132, '固定引用', AMBER, 16)
    s.text(72, 1300, '实线：主要数据流；虚线：独立可选能力。任务状态与整理状态分别核对；正文路由不要求每条反馈都调用模型。',
           20, MUTED, max_width=1656)
    s.text(72, 1336, '标准医疗 Open Lab 已接通二选一路由；通用兼容装配须显式配置。术语、协议映射及实现范围见 docs/architecture.md。',
           18, MUTED, max_width=1656)
    s.save('system-flow.svg')


def file_architecture():
    s = SVG(1800, 1530, 'RSISys：统一角色与代码映射',
            '以 meta-agent（decision侧）、Lab、meta-agent（整理侧）映射当前文件。'
            '最小公开窗口只规定模块基本职能，不规定内部算法、存储或 Actor 结构。'
            '统一装配与 Host 管理生命周期；状态适配器提供原目录恢复；看板只面向人。'
            'decision_maker、exploration、feedback_learning 保留为兼容实现标识。')
    heading(s, 'RSISys / 模块与代码映射',
            '概念角色保持统一；目录、类型和持久化字段通过兼容映射接入。')
    s.text(72, 166, '核心路径相对 src/rsisys/；领域装配相对 examples/；子卡中的简写属于所在模块。',
           19, MUTED, max_width=1656)
    columns = [
        (72, BLUE, '#EFF5FF', 'meta-agent（decision侧）',
         'loops/decision_making/', [
             ('decision-input', '决策与上下文（harness/）',
              ('policy/ · engine.py', 'context.py · research_state.py')),
             ('decision-actions', '工具与研究状态',
              ('action_space/ · harness/tools/', 'harness/state.py', 'harness/research.py')),
         ]),
        (660, GREEN, '#EFF8F3', 'Lab / 执行环境',
         'loops/contracts/execution.py', [
             ('open-lab-files', '开放 Lab',
              ('adapters/local_lab.py', 'examples/medical/', 'fixed_evaluation.py')),
             ('legacy-lab-files', '兼容 Lab 适配',
              ('adapters/execution.py', 'loops/exploration/（内部可替换）')),
         ]),
        (1248, PURPLE, '#F4EFFB', 'meta-agent（整理侧）',
         'loops/decision_making/feedback/', [
             ('organization-products', '反馈接入与产物',
              ('service.py · learning.py', 'pipeline.py · proposals.py')),
             ('organization-memory', '记忆与路由',
              ('langmem.py · context_memory.py', 'adapters/local_lab_feedback.py')),
         ]),
    ]
    for x, color, fill, title, path, cards in columns:
        s.rect(x, 215, 480, 490, fill, color, rx=20)
        s.text(x+24, 264, title, 28, color, 650, max_width=432)
        s.text(x+24, 301, path, 21, MUTED, mono=True, max_width=432)
        for y, (ident, label, lines) in zip((330, 510), cards):
            s.node(ident, x+24, y, 432, 145, label, lines,
                   color=color, stroke=color, title_size=25)

    s.text(72, 772, '最小公开窗口', 30, weight=650)
    s.text(72, 810, '只约束基本职能；内部算法、格式、存储与流程由可替换模块自行实现。', 23, MUTED)
    for x, ident, title, lines, color, fill in [
        (72, 'lab-port', 'Lab 执行窗口',
         ('describe() / execute(call)', '声明真实能力，执行授权调用', '不规定 SFT / RL 算法菜单'), GREEN, '#EFF8F3'),
        (496, 'context-port', 'context / memory 窗口',
         ('context(query) → str', '按需提供文本，替换当前视图', '可选 observe 接收反馈'), PURPLE, '#F4EFFB'),
        (920, 'archive-port', '归档窗口（可选）',
         ('write(event) / read(ref)', '保存原生事件与稳定来源', '数据筛选与构建交给 Lab'), PURPLE, '#F4EFFB'),
        (1344, 'optional-ports', '提案 / 调度（可选）',
         ('propose(...) / select(...)', '分别返回建议或选择待办', '不授予执行、重放或预算'), AMBER, '#FFF7E8'),
    ]:
        s.node(ident, x, 850, 384, 180, title, lines,
               color=color, fill=fill, stroke=color, title_size=24)

    s.node('host-runtime-files', 72, 1080, 800, 190, 'Host / 统一装配与生命周期', (
        'application.py · assembly.py · research.py',
        'runtime/runner.py · run_policy.py · recovery.py',
        'loops/contracts/：身份、权限、回执、引用与状态协议',
    ), INK, '#EAF0F6', LINE, title_size=28)
    s.node('recovery-files', 912, 1080, 816, 190, '状态恢复 / 显式组件适配', (
        'runtime/component_state.py · snapshot/',
        'adapters/local_lab_state.py · medical/open_lab_recovery.py',
        '原目录续跑保留预算；未知效果先核对，不自动重放',
    ), BLUE, '#EFF5FF', BLUE, title_size=28)
    s.node('domain-and-monitor', 72, 1320, 1656, 135, '领域装配与面向人的看板', (
        'examples/medical/open_lab_experiment.py：固定正式评测、资源配置、路由与恢复接线。',
        'src/rsimonitor/：只读 monitor.json / HTTP API；不把看板页面作为模型输入。',
    ), INK, '#FFFFFF', LINE, title_size=28)
    s.text(72, 1500,
           'decision_maker / exploration / feedback_learning 仅为兼容实现标识；统一定义与实现边界见 docs/architecture.md。',
           19, MUTED, max_width=1656)
    s.save('file-architecture.svg')


if __name__ == '__main__':
    system_flow()
    file_architecture()
