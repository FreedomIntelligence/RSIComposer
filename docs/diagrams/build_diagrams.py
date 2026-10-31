#!/usr/bin/env python3
"""Generate editable, dependency-free SVG diagrams for the RSISys design."""
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
            self.text(x+24,title_y+34+i*29,line,20,MUTED,max_width=w-48)
        self.parts.append('</g>')

    def save(self, name):
        self.parts.append('</svg>')
        s='\n'.join(self.parts)+'\n'
        ET.fromstring(s)
        (HERE/name).write_text(s)
        print(f'{name}: {self.width}x{self.height}, {self.text_count} editable text elements')


def system_flow():
    s=SVG(2000,1580,'RSISys 系统流程图','两个 loop 拥有各自流程与可替换实现，共享 Experience；Runtime 协调，Snapshot 独立。')
    s.text(64,75,'RSISys / 系统流程',42,weight=750)
    s.text(64,116,'两个 Loop 定义执行流程 · 模型、沙箱与产物后端通过依赖注入接入',23,MUTED)
    s.chip(1530,48,180,'流程与职责','#E3F3EC',GREEN)
    s.chip(1730,48,206,'实现见 README','#FFF5DD',AMBER)
    s.rect(64,155,1872,92)
    s.text(94,192,'ModelBackend / 模型调用接口',26,BLUE,700)
    s.text(94,225,'backend_id + model_id + revision；角色可共享或独立绑定，具体调用方式由后端实现。',21,MUTED)

    s.rect(64,290,776,740,fill='#EFF5FF',stroke='#BCD1F2',rx=24)
    s.rect(1160,290,776,740,fill='#EFF8F3',stroke='#B9DDCF',rx=24)
    s.text(100,338,'01  Decision-making Loop',30,BLUE,700)
    s.text(1200,338,'02  Exploration Loop',30,GREEN,700)
    s.path('M 760 247 V 385',BLUE,arrow='blue')
    s.path('M 1180 247 V 590 H 1200',GREEN,arrow='green')
    s.node('decision-policy',104,385,692,100,'Decision Policy',('结构化 Decision：invoke / explore / wait / finish',),BLUE,stroke='#9FBEEB')
    s.node('command-lifecycle',1200,385,692,100,'指令生命周期',('accept / advance / inspect / cancel / acknowledge',),GREEN,stroke='#9ACBB6')
    s.node('decision-space',104,540,692,115,'Decision Space',('describe / execute：独立执行决策操作', '文献 / 数据 / 策略操作 → 结果与 ArtifactRef'),PURPLE,'#F2EBFB','#B89CDC',title_size=25)
    s.path('M 450 485 V 540',BLUE,arrow='blue')
    s.label(450,520,'invoke',BLUE,17)
    s.node('directive',880,540,240,110,'Directive',('目标 / 输入 / 策略', '执行 / 学习 / 评测'),AMBER,'#FFF5DD','#E4C47F',title_size=25)
    s.path('M 796 435 H 855 V 592 H 880',BLUE,arrow='blue')
    s.label(925,435,'explore',BLUE,17)
    s.path('M 796 592 H 818 V 465 H 796',BLUE,arrow='blue')
    s.path('M 1120 592 H 1140 V 435 H 1200',BLUE,arrow='blue')
    s.node('action-space',1200,540,310,115,'Policy + Space',('Actor 选择行动', 'Action Space 校验'),PURPLE,'#F2EBFB','#B89CDC',title_size=24)
    s.node('environment',1582,540,310,115,'Environment',('reset / step / close', '执行行动并返回结果'),GREEN,stroke='#9ACBB6',title_size=24)
    s.path('M 1355 485 V 540',GREEN,arrow='green')
    s.path('M 1510 594 H 1582',GREEN,arrow='green')
    s.label(1546,580,'行动',GREEN,17)
    s.node('feedback',880,720,240,110,'Feedback',('结果 / 经验引用', '模拟标记 / 指标'),AMBER,'#FFF5DD','#E4C47F',title_size=25)
    s.path('M 1737 655 V 775 H 1120',AMBER,arrow='amber')
    s.label(1480,761,'汇总结果与证据',AMBER,18,190)
    s.node('observe',104,720,692,110,'Observe / 接收反馈',('关联当前指令，记录 Decision Episode', '反馈进入下一轮决策上下文'),BLUE,stroke='#9FBEEB',title_size=25)
    s.path('M 880 775 H 796',AMBER,arrow='amber')
    s.path('M 104 775 H 84 V 435 H 104',BLUE,arrow='blue')

    for ident,x,w,title,detail,color in [
        ('decision-learner',104,692,'Decision Learner','提交 / 查询 / 取消 / 模型绑定',BLUE),
        ('actor-learner',1200,310,'Actor Learner','SFT / RL 作业控制',GREEN),
        ('evaluation',1582,310,'Evaluation','候选与评测协议',GREEN)]:
        s.rect(x,880,w,100,fill='#FFFFFF',stroke=color)
        s.text(x+20,919,title,24,color,650,max_width=w-40)
        s.text(x+20,953,detail,19,MUTED,max_width=w-40)
    s.path('M 1855 485 H 1915 V 850 H 1355 V 880',GREEN,arrow='green')
    s.path('M 1737 850 V 880',GREEN,arrow='green')
    s.label(1550,850,'学习 / 评测指令',GREEN,17,190)
    s.path('M 1355 980 V 995 H 1145 V 800 H 1120',AMBER,arrow='amber')
    s.path('M 1737 980 V 995 H 1355',AMBER,arrow=None)
    s.text(1200,1020,'RL 在一条指令内部推进多次采样与版本更新',19,MUTED)
    s.text(104,1010,'各自拥有 Policy / Space / Learner',19,MUTED)

    s.rect(64,1080,1872,230,fill='#F2EBFB',stroke='#B89CDC',rx=24)
    s.text(100,1121,'Experience System / 两个 Loop 共用',29,PURPLE,700)
    s.path('M 450 1030 V 1080',PURPLE,arrow='purple')
    s.path('M 1546 1030 V 1080',PURPLE,arrow='purple')
    for x,w,title,lines in [
        (100,570,'Records / 经验记录',['Trajectory · Decision Episode','Training Run · Evaluation Record']),
        (710,570,'Evidence & Provenance',['证据 / 产物引用 / 版本','来源与关联关系']),
        (1320,576,'Memory / 可复用记忆',['Episodic · Semantic · Procedural','适用条件 / 来源 / 验证状态'])]:
        s.node(f'experience-{x}',x,1145,w,110,title,lines,PURPLE,stroke='#D7C5EC',title_size=22)
    s.text(100,1290,'记录、来源引用与显式 Memory 共用；存储和检索实现可替换。',20,MUTED)
    s.label(450,1064,'记录 / 查询',PURPLE,17)
    s.label(1546,1064,'记录 / 查询',PURPLE,17)

    s.node('runtime',64,1350,906,145,'Runtime / 运行协调与公共后端',('Runner / Jobs + Model / Sandbox / ArtifactStore', '公开端口协调、作业记账与公共后端；策略由 Loop 决定'),INK,'#EAF0F6','#CBD6E3')
    s.node('snapshot',1010,1350,926,145,'Snapshot / 系统快照',('独立组织组件状态、模型绑定、经验与产物', '保存内容依具体实现；内置实现支持空闲边界恢复'),INK,'#EAF0F6','#CBD6E3')
    s.text(64,1534,'箭头表示流程或依赖关系；policy、space、learner 等实现可替换，生命周期由所属 Loop 管理。',20,MUTED)
    s.text(64,1565,'图示展示结构；实现与限制见组件 README，跨组件协同见 ARCHITECTURE.md。',18,MUTED)
    s.save('system-flow.svg')


def file_architecture():
    s=SVG(2000,1630,'RSISys 文件架构图','按两个 loop 组织框架流程与可替换实现，Experience、Snapshot、Runtime 独立。')
    s.text(64,75,'RSISys / 文件架构',42,weight=750)
    s.text(64,115,'框架流程与具体实现各有归属 · 共享能力独立维护',24,MUTED)
    s.chip(1510,48,190,'模块归属','#E3F3EC',GREEN)
    s.chip(1720,48,216,'实现见 README','#FFF5DD',AMBER)
    s.rect(64,155,1872,80)
    s.text(94,190,'src/rsisys/',30,BLUE,700,mono=True)
    s.text(440,190,'lion:/home/jovyan/projects/RSISys',23,MUTED,mono=True)
    s.text(94,220,'Policy、Space、Learner 属于各自 Loop；只将真正共享的契约和系统能力提取出来。',21,MUTED)

    s.rect(64,270,1180,810,fill='#EFF4FC',stroke='#B9CBE6',rx=24)
    s.text(100,312,'loops/',29,BLUE,700,mono=True)
    def panel(ident,x,y,w,h,title,lines,color=INK,fill='#FFFFFF',dash=None):
        s.parts.append(f'<g id="{ident}">')
        s.rect(x,y,w,h,fill=fill,stroke=color,dash=dash)
        s.text(x+24,y+40,title,25,color,700,mono=True,max_width=w-48)
        for i,line in enumerate(lines):
            s.text(x+24,y+81+i*33,line,20,MUTED,mono=True,max_width=w-48)
        s.parts.append('</g>')
    panel('decision-dir',100,345,530,450,'decision_making/',[
        'loop.py / state.py',
        'decision.py       结构化决策',
        'policy/           决策策略',
        'decision_space/',
        '  local.py        本地操作',
        '  literature.py / data.py',
        '  strategy.py     操作协议',
        'learner/',
        '  backends/       训练后端',
        '准备 → 指令 → 接收反馈'],BLUE,'#EFF5FF')
    panel('exploration-dir',665,345,543,450,'exploration/',[
        'loop.py / state.py',
        'policy/           Actor 策略',
        'action_space/     环境行动',
        'environment/      环境接入',
        'learner/',
        '  backends/       训练后端',
        'evaluation/       评测实现',
        '',
        'commands.py / rl.py'],GREEN,'#EFF8F3')
    panel('contracts-dir',100,835,1108,205,'contracts/',[
        'directive.py / feedback.py / coordination.py',
        'model_service.py  仅类型与接口，后端实现位于 Runtime',
        'requests.py / training.py / 引用与状态类型',
        '共享纯类型与公开端口；实现通过构造函数注入'],AMBER,'#FFF8E8')

    panel('experience-dir',1290,270,646,280,'experience/',[
        'records.py       四类原始经验',
        'memory.py        显式记忆',
        'provenance.py    来源关系类型',
        'retrieval.py     角色视图接口',
        'storage.py / sqlite.py',
        '记录、检索与存储实现分开'],PURPLE,'#F2EBFB')
    panel('snapshot-dir',1290,585,646,245,'snapshot/',[
        'manifest.py      系统状态清单',
        'capture.py / restore.py',
        'storage.py       存储接口',
        'local.py / codec.py',
        '内置组件的空闲边界恢复'],AMBER,'#FFF8E8')
    panel('runtime-dir',1290,850,646,230,'runtime/',[
        'runner.py / jobs.py',
        'model.py         模型调用',
        'sandbox.py / local_process.py',
        'artifacts.py     内存 / 文件',
        '流程和策略仍由 Loop 负责'],INK,'#EAF0F6')

    s.rect(64,1120,1872,395,fill='#FFFFFF',rx=24)
    s.text(100,1163,'装配、环境与工程文件',27,INK,700)
    panel('assembly',100,1190,545,280,'装配与入口',[
        'application.py / config.py',
        'cli.py           命令逻辑',
        '__main__.py      薄入口',
        'configs/         运行配置',
        '依赖通过构造函数传入',
        '模型引用支持共享或独立'],BLUE)
    panel('engineering',685,1190,565,280,'环境与验证',[
        'pyproject.toml',
        'pixi.toml / pixi.lock',
        'tests/           行为验证',
        '',
        'pixi run demo',
        'pixi run test'],GREEN)
    panel('docs',1290,1190,606,280,'文档与图示',[
        'README.md',
        'docs/ARCHITECTURE.md',
        'docs/GUIDE.md',
        'docs/diagrams/*.svg',
        'docs/diagrams/build_diagrams.py',
        'src/ 各组件维护 README.md'],MUTED)
    s.text(64,1560,'Policy 决定模型请求；Space 定义操作；Environment 执行行动；Runtime 提供公共后端与生命周期协调。',20,MUTED)
    s.text(64,1601,'图示说明文件归属；组件职责、接口与实现范围在对应目录 README 中维护。',19,MUTED)
    s.save('file-architecture.svg')



if __name__ == '__main__':
    system_flow()
    file_architecture()
