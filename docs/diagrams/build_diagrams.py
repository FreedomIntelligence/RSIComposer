#!/usr/bin/env python3
"""Generate the RSI controller / Labs architecture as editable native SVGs."""
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
    s.chip(1538, 50, 190, '2026-11-08', '#E8EEF8', BLUE, 18)


def system_flow():
    s = SVG(1800, 1510, 'RSISys：RSI 决策器与 Labs',
        'RSI 决策器选择行动，Labs 执行并返回反馈；决策器可以是规则、搜索策略或 meta-agent。'
        'Host 提供基础设施；memory 和研究任务窗口属于 meta-agent 的可选实现。')
    heading(s, 'RSISys / RSI 决策器与 Labs',
        '决策器 → Lab 执行 → 反馈 → 决策器；通过库外注册和独立配置选择实现。')
    section(s, 72, 158, 1656, 100, 'host', 'Host / 宿主', (
        '固定目标与正式评测 · 权限和资源准入 · 调用 / token 账本 · 原截止时间 · 恢复与 owned 资源清理',
    ), INK, '#EAF0F6')
    section(s, 72, 320, 760, 350, 'controller', 'RSI 决策器 / Controller', (
        'decide(input) → 下一步行动',
        '固定算法 · 搜索策略 · meta-agent · 混合实现',
        '读取真实反馈，维护自己的算法状态',
        'single_loop：按回执推进配置的迭代',
        'meta_agent：模型自主研究与调试',
        '规则实现不要求模型、Memory 或 Reflect',
        '公开接入：rsisys.api / Registry',
    ), BLUE, '#EFF5FF')
    section(s, 968, 320, 760, 350, 'lab', 'Labs / 命名执行环境', (
        '任务 · 数据 · 模型或 agent · Lab harness',
        '读写代码、运行程序、训练、推理与评测',
        'describe / execute；别名.操作名统一路由',
        '原生结果：文本、JSON、日志或文件引用',
        '固定正式评测保留候选 / 数据 / 协议身份',
        '退出成功 ≠ 训练完成 ≠ 质量改善 ≠ 采纳',
    ), GREEN, '#EFF8F3')
    s.path('M 832 405 H 968', BLUE, arrow='blue')
    s.label(900, 388, '动作 / 任务', BLUE, 18)
    s.path('M 968 580 H 832', GREEN, arrow='green')
    s.label(900, 558, '原始反馈', GREEN, 18)
    section(s, 72, 810, 760, 245, 'memory', 'meta-agent 可选子模块 / memory', (
        'memory_search / read / write / delete',
        '存储、文字检索、来源与修订记录',
        '不运行模型抽取、embedding 或后台总结',
        '记忆是模型解释；引用存在不证明散文为真',
        '大记录以原文页读取，不伪装为完整摘要',
    ), PURPLE, '#F4EFFB')
    section(s, 968, 810, 760, 245, 'archive', '原始归档 / 执行账本', (
        '保存真实回执、身份与证据 SHA',
        'research_archive 按稳定引用分页读取',
        '关闭任务或删除记忆，不改写实验原文',
        '可选轨迹产物与恢复账本分别负责',
        '恢复不重放未知副作用，不延长原截止',
    ), INK, '#EAF0F6')
    s.path('M 280 670 V 810', PURPLE, arrow='purple')
    s.label(280, 750, '显式写入 / 更新 / 删除', PURPLE, 19)
    s.path('M 630 810 V 670', PURPLE, arrow='purple')
    s.label(630, 782, '按需检索', PURPLE, 19)
    s.path('M 1348 670 V 810', INK, arrow='gray', dash='7 7')
    s.label(1348, 750, '持久原文与来源', INK, 19)
    section(s, 72, 1135, 1656, 190, 'task-context', 'meta-agent 实现：当前任务的原始反馈窗口', (
        'research_task：open 固定错误 → pin 补充证据 → 读代码 / 日志、编辑和复测 → close 释放驻留',
        '中间工具不会挤掉固定的错误；当前任务关闭不等于自动写入长期 memory。',
        '完整性与分页显式标记；原始证据、模型记忆、诊断 canary 和正式评测分别识别。',
    ), BLUE, '#EFF5FF')
    s.node('monitor', 72, 1380, 1656, 90, '看板 / Monitor：Host 写公开投影 → 人查看；不进入模型反馈链路。',
        (), INK, '#FFFFFF', LINE, title_size=25)
    s.save('system-flow.svg')


def file_architecture():
    s = SVG(1800, 1240, 'RSISys：决策器与 Labs 代码映射',
        'Registry 注册决策器与 Labs；meta-agent 的 agent_memory.py 提供可选显式记忆和任务窗口。'
        '反馈设施保留归档与显式兼容处理器；默认没有独立整理 agent。')
    heading(s, 'RSISys / 模块与代码映射',
        '库外接入使用 rsisys.api；默认执行适配复用既有端点、权限与回执。')
    cards = [
        (72, BLUE, '#EFF5FF', 'RSI 决策器', 'loops/decision_making/', [
            ('decision', '决策与上下文', ('harness/policy/ · engine.py', 'context.py · research_state.py')),
            ('memory', 'meta-agent 可选记忆', ('harness/agent_memory.py', 'memory_* · research_task', 'AgentMemoryState')),
        ]),
        (660, GREEN, '#EFF8F3', 'Lab / 执行环境', 'loops/contracts/execution.py', [
            ('lab', '开放执行与原始结果', ('adapters/local_lab.py', 'adapters/local_lab_state.py')),
            ('evaluation', '固定领域评测', ('experiments/medical/', 'fixed_evaluation.py', 'candidate_adapter.py')),
        ]),
        (1248, PURPLE, '#F4EFFB', '归档与兼容设施', 'loops/decision_making/feedback/', [
            ('feedback', '原始回执与交付', ('service.py · learning.py', 'archive.py · context_memory.py')),
            ('legacy', '显式兼容处理器', ('langmem.py · pipeline.py', '默认不运行独立抽取模型', '旧冻结运行不跨版本恢复')),
        ]),
    ]
    for x, color, fill, title, path, children in cards:
        s.rect(x, 220, 480, 520, fill, color)
        s.text(x+24, 270, title, 28, color, 650, max_width=432)
        s.text(x+24, 307, path, 21, MUTED, mono=True, max_width=432)
        for y, (ident, label, lines) in zip((345, 530), children):
            s.node(ident, x+24, y, 432, 160, label, lines, color, stroke=color, title_size=24)
    s.node('assembly', 72, 810, 800, 190, 'Host / 装配、准入与生命周期', (
        'api.py · registry.py · assembly.py · methods.py',
        'components.py：内部组合可选；整体方法也可接入',
        'runtime/runner.py · run_policy.py · recovery.py',
        'component_state.py：恢复须显式注册组件状态',
    ), INK, '#EAF0F6', LINE, title_size=28)
    s.node('recovery', 912, 810, 816, 190, '医疗接线与恢复', (
        'experiments/medical/open_lab_experiment.py',
        'open_lab_recovery.py · AgentMemoryState',
        '保留记忆、任务、调用账本与绝对截止时间',
    ), BLUE, '#EFF5FF', BLUE, title_size=28)
    s.node('monitor', 72, 1070, 1656, 100, '面向人的看板：src/rsimonitor/ 只读运行投影，不承担模型反馈。',
        (), INK, '#FFFFFF', LINE, title_size=27)
    s.text(72, 1210, '顶层模块是 RSI 决策器与 Labs；meta-agent 是决策器的一种实现。', 20, MUTED)
    s.save('file-architecture.svg')


if __name__ == '__main__':
    system_flow()
    file_architecture()
