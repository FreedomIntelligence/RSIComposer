"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  demo: false, runId: "", realRunId: "", snapshot: null, controller: null,
  epoch: 0, timer: null, auto: true, eventLimit: 30, decisionLimit: 30,
  openEvents: new Set(), openDecisions: new Set(), openJobs: new Set(), queryApplied: false,
  projects: [], projectId: "", realProjectId: "", scopes: new Map(), activeScope: null, demoSnapshots: new Map(),
};
const ACTIVE = new Set(["running", "active", "training", "evaluating", "queued", "pending", "starting", "submitted", "in_progress"]);
const DONE = new Set(["completed", "complete", "succeeded", "success", "done", "finished"]);
const FAILED = new Set(["failed", "failure", "error", "cancelled", "canceled"]);
const ACCEPTED = new Set(["accepted", "applied", "adopted", "promoted", "bound"]);
const REJECTED = new Set(["rejected", "discarded", "failed"]);
const PENDING = new Set(["pending", "proposed", "evaluating", "ready", "validated", "awaiting_evaluation", "awaiting_review", "training"]);
const labels = {
  running: "运行中", active: "进行中", training: "训练中", evaluating: "评测中", queued: "排队中", pending: "等待中", starting: "启动中", submitted: "已提交", in_progress: "进行中",
  completed: "已完成", complete: "已完成", succeeded: "已完成", success: "已完成", done: "已完成", finished: "已完成",
  failed: "失败", failure: "失败", error: "错误", cancelled: "已取消", canceled: "已取消", idle: "空闲", waiting: "等待中", paused: "已暂停", stopped: "已停止",
  accepted: "已采纳", applied: "已采纳", adopted: "已采纳", promoted: "已采纳", bound: "已绑定", rejected: "未采纳", discarded: "已放弃", proposed: "已提出", ready: "待审阅", validated: "已验证", awaiting_evaluation: "待评测", awaiting_review: "待审阅", unavailable: "快照不可用", reviewed: "已复查", cancel_requested: "已请求取消", continue: "继续执行", cancel: "请求取消",
};
const actionLabels = {
  train: "训练模型", training: "训练模型", actor_training: "训练执行模型", decision_maker_training: "训练决策模型", sft: "监督微调", rl: "强化学习",
  evaluate: "评测候选", evaluation: "评测候选", candidate_evaluation: "评测候选", select: "选择候选", apply_candidate: "应用候选", execute: "执行任务",
  finish: "结束本轮研究", finished: "结束本轮研究", continue: "继续执行", continue_training: "继续训练", cancel: "请求取消", cancel_training: "请求取消训练", stop: "停止训练", wait: "等待", review: "复查执行", prepare: "准备", queued: "排队", starting: "准备运行", collecting_result: "收集结果", accepted: "指令已接收", completed: "已完成",
  research_search: "检索研究资料", research_read: "阅读研究资料", research_archive: "查阅研究档案", research_patch: "更新研究笔记", research_write: "记录研究结论", research_memory: "查阅研究记忆", research_experience: "检索历史经验", research_history: "查阅决策历史",
  search: "搜索资料", read: "读取资料", web_search: "搜索网页", fetch: "获取资料", query_jobs: "查询任务", query_candidates: "查询候选", query_metrics: "查询指标", inspect_job: "查看任务", inspect_candidate: "查看候选", list_jobs: "查看任务列表", list_candidates: "查看候选列表",
};
const metricLabels = {
  loss: "损失", train_loss: "训练损失", "train.loss": "训练损失", "train/loss": "训练损失", eval_loss: "评测损失", learning_rate: "学习率", lr: "学习率", "train/learning_rate": "学习率",
  nll: "负对数似然", base_nll: "基线 NLL", candidate_nll: "候选 NLL", delta_nll: "上报 NLL 差值", heldout_nll_delta: "验证 NLL 差值", eval_samples: "评测样本数", num_samples: "样本数", sample_count: "样本数", accuracy: "准确率", score: "得分", grad_norm: "梯度范数", epoch: "训练轮数", runtime: "耗时", train_runtime: "训练耗时", total_flos: "浮点运算量",
};
const asText = (v, fallback = "—") => ["string", "number", "boolean"].includes(typeof v) ? String(v) : fallback;
const list = (v) => Array.isArray(v) ? v.filter(x => x && typeof x === "object" && !Array.isArray(x)) : [];
const object = (v) => v && typeof v === "object" && !Array.isArray(v) ? v : {};
const number = (v) => typeof v === "number" && Number.isFinite(v);
const present = (v) => v !== undefined && v !== null && v !== "";
const statusKey = (v) => asText(v, "unknown").toLowerCase();
const statusText = (v) => labels[statusKey(v)] || "状态未报告或未归类";
const actionText = (v) => actionLabels[asText(v, "")] || "其他操作";
const metricText = (v) => metricLabels[v] || "其他指标";
function el(tag, className = "", text) { const e = document.createElement(tag); if (className) e.className = className; if (text !== undefined) e.textContent = asText(text); return e; }
function badge(v) { const k = statusKey(v); return el("span", `badge ${FAILED.has(k) || REJECTED.has(k) ? "failure" : DONE.has(k) || ACCEPTED.has(k) ? "success" : ACTIVE.has(k) ? "active" : ""}`, statusText(v)); }
function empty(text) { return el("p", "subtle-empty", text); }
function dateText(v, withDate = true) { if (!v) return "时间未报告"; const d = new Date(v); if (Number.isNaN(d.getTime())) return "时间未报告"; return new Intl.DateTimeFormat("zh-CN", { ...(withDate ? { month: "2-digit", day: "2-digit" } : {}), hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(d); }
function secondsText(v) { if (!number(v) || v < 0) return "未报告"; if (v > 0 && v < 1) return "不足 1 秒"; const s = Math.floor(v); return [s >= 3600 ? `${Math.floor(s / 3600)} 小时` : "", s >= 60 ? `${Math.floor(s % 3600 / 60)} 分` : "", `${s % 60} 秒`].filter(Boolean).join(" "); }
function fmt(v) { if (!number(v)) return "—"; if (v !== 0 && (Math.abs(v) < .0001 || Math.abs(v) >= 1e9)) return v.toExponential(4); return new Intl.NumberFormat("en-US", { maximumFractionDigits: 6 }).format(v); }
function connection(text, tone = "") { $("connection").textContent = text; $("connection").className = `connection ${tone}`; }
function notice(text, error = false) { $("notice").textContent = text; $("notice").hidden = !text; $("notice").className = `notice${error ? " error" : ""}`; }
function showEmpty(title, text) { $("dashboard").hidden = true; $("empty").hidden = false; $("empty-title").textContent = title; $("empty-description").textContent = text; $("export-report").disabled = !state.snapshot; }
// Keep reported prose verbatim. Identifiers and long text live in disclosures.
const publicText = (value, fallback = "未报告") => asText(value, fallback);
function proseView(value, title = "决策原文") {
  const details=el("details","record-details");details.append(el("summary","",title),el("p","decision-reason",asText(value)));return details;
}
function fieldList(entries) { const dl = el("dl", "detail-grid"); for (const [k, v] of entries.filter(([, v]) => present(v))) dl.append(el("dt", "", k), el("dd", "", asText(v))); return dl; }
function rawFields(row) { return fieldList(Object.entries(object(row)).filter(([, v]) => ["string", "number", "boolean"].includes(typeof v))); }
function retainedDetails(key, store, title, content) {
  const d = el("details", "record-details"); d.open = store.has(key); d.append(el("summary", "", title), content);
  d.addEventListener("toggle", () => { if (d.isConnected) { if (d.open) store.add(key); else store.delete(key); } }); return d;
}
function flattenMetrics(source, prefix = "", output = {}, depth = 0) {
  if (depth > 5) return output;
  for (const [k, v] of Object.entries(object(source))) { const key = prefix ? `${prefix}.${k}` : k; if (number(v)) output[key] = v; else if (!Array.isArray(v)) flattenMetrics(v, key, output, depth + 1); } return output;
}
function metricStrip(metrics, raw = false) {
  const dl = el("dl", "metric-strip");
  for (const [key, value] of Object.entries(flattenMetrics(metrics)).filter(([k]) => raw || metricLabels[k])) { const item = el("div", "metric-chip"); item.append(el("dt", "", raw ? `${metricText(key)} · ${key}` : metricText(key)), el("dd", "", fmt(value))); dl.append(item); } return dl;
}
function metricsDetail(metrics) { const m = metricStrip(metrics, true); return m.childElementCount ? m : empty("尚无指标。"); }
function progressView(step, total) {
  const wrap = el("div", "step-progress");
  if (Number.isInteger(step) && step >= 0 && Number.isInteger(total) && total > 0 && step <= total) {
    const label = el("div", "progress-label"); label.append(el("span", "", `${step} / ${total} 步`), el("span", "", `${Number((step / total * 100).toFixed(1))}%`));
    const bar = el("progress"); bar.max = total; bar.value = step; bar.setAttribute("aria-label", "任务步骤进度"); wrap.append(label, bar);
  } else wrap.append(el("p", "progress-unknown", Number.isInteger(step) && step >= 0 ? `已报告 ${step} 步；总步数未报告或待核对` : "步骤进度未报告"));
  return wrap;
}
function decisionAction(row) { if (row.kind === "review") return row.action || row.mode; if (row.kind === "operation") return row.operation; return row.mode || row.action || (row.kind === "finish" ? "finish" : undefined); }
function decisionKey(row) { return asText(row.decision_id, JSON.stringify([row.timestamp, row.kind, row.round, row.directive_id, decisionAction(row), row.job_id])); }
function sortedDecisions(s) { return list(s.decisions).map((row, i) => ({ row, i })).sort((a,b) => (Date.parse(b.row.timestamp) || 0) - (Date.parse(a.row.timestamp) || 0) || a.i - b.i).map(x => x.row); }
function decisionGroups(s) {
  const groups = [];
  for (const row of sortedDecisions(s)) {
    const last = groups[groups.length - 1];
    const signature = JSON.stringify([row.kind, decisionAction(row), row.round, row.reason, row.strategy, row.status, row.actor_revision, row.decision_revision]);
    if (decisionAction(row) === "finish" && last?.signature === signature) last.rows.push(row);
    else groups.push({ row, rows: [row], signature });
  } return groups;
}
const identifier = v => typeof v === "string" && v.length > 0;
function parentIds(row) {
  return [...new Set([row.parent_decision_id, ...(Array.isArray(row.source_decision_ids) ? row.source_decision_ids : [])].filter(identifier))];
}
// Bounded spatial relaxation; only manually moved nodes stay pinned.
const LAYOUT = { iterations: 64, budget: 60000, cell: 76, bucket: 12, cache: 6000 };
function hashUnit(key) {
  let hash = 2166136261;
  for (let i = 0; i < key.length; i++) hash = Math.imul(hash ^ key.charCodeAt(i), 16777619);
  return (hash >>> 0) / 4294967296;
}
function readPositions(key) {
  try {
    const rows = JSON.parse(localStorage.getItem(`rsisys-runs-v2:${key}`) || "[]");
    return new Map((Array.isArray(rows) ? rows : []).slice(-LAYOUT.cache).filter(r => Array.isArray(r) && typeof r[0] === "string" && number(r[1]?.x) && number(r[1]?.y) && r[1].x >= 36 && r[1].y >= 36 && r[1].x <= 100000 && r[1].y <= 100000));
  } catch { return new Map(); }
}
function savePositions(scope) {
  try { localStorage.setItem(`rsisys-runs-v2:${scope.key}`, JSON.stringify([...scope.positions].slice(-LAYOUT.cache))); } catch { /* Storage can be disabled or full. In-memory positions still work. */ }
}
function networkLayout(model, positions = new Map(), { relax = false, pinned = new Set() } = {}) {
  const ordered = model.nodes.slice().sort((a,b) => a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
  const byKey = new Map(ordered.map(n => [n.key,n])), anchors = ordered.filter(n => n.type === "run"), fresh = new Set();
  const center = { x: 560, y: 245 };
  const initialize = (n, origin, radius, angle) => {
    const cached = positions.get(n.key);
    if (cached && number(cached.x) && number(cached.y)) { n.x = cached.x; n.y = cached.y; if (relax && !pinned.has(n.key)) fresh.add(n.key); }
    else { n.x = Math.max(44, origin.x + radius * Math.cos(angle)); n.y = Math.max(44, origin.y + radius * Math.sin(angle)); fresh.add(n.key); }
  };
  for (const n of ordered.filter(n => n.type === "project")) initialize(n, center, 0, 0);
  anchors.forEach((n,i) => initialize(n, center, 125 + 32 * Math.sqrt(i), i * 2.399963 + hashUnit(n.key) * .6));
  const counts = new Map();
  for (const n of ordered.filter(n => !["project","run"].includes(n.type))) {
    const origin = byKey.get(JSON.stringify(["run",n.runId])) || center;
    const i = counts.get(n.runId) || 0; counts.set(n.runId,i+1);
    initialize(n, origin, 62 + 23 * Math.sqrt(i), i * 2.399963 + hashUnit(n.key) * 1.8);
  }
  const edgeOrder = e => JSON.stringify([e.from,e.to,e.kind]);
  const edges = model.edges.slice().sort((a,b) => edgeOrder(a) < edgeOrder(b) ? -1 : edgeOrder(a) > edgeOrder(b) ? 1 : 0).map(e => ({ ...e, a: byKey.get(e.from), b: byKey.get(e.to) })).filter(e => e.a && e.b);
  const iterations = fresh.size ? Math.min(LAYOUT.iterations, Math.max(1, Math.floor(LAYOUT.budget / Math.max(1, ordered.length + edges.length)))) : 0;
  let comparisons = 0;
  for (let step = 0; step < iterations; step++) {
    const grid = new Map(), forces = new Map();
    for (const n of ordered) {
      const k = `${Math.floor(n.x / LAYOUT.cell)},${Math.floor(n.y / LAYOUT.cell)}`;
      const bucket = grid.get(k) || []; if (bucket.length < LAYOUT.bucket) bucket.push(n); grid.set(k,bucket);
      if (fresh.has(n.key)) forces.set(n.key,{ x: 0, y: 0 });
    }
    for (const n of ordered) {
      const f = forces.get(n.key); if (!f) continue;
      const gx = Math.floor(n.x / LAYOUT.cell), gy = Math.floor(n.y / LAYOUT.cell);
      for (let x = gx-1; x <= gx+1; x++) for (let y = gy-1; y <= gy+1; y++) for (const other of grid.get(`${x},${y}`) || []) {
        if (n === other) continue; comparisons++;
        let dx = n.x-other.x, dy = n.y-other.y;
        if (Math.abs(dx)+Math.abs(dy) < .001) { const angle = hashUnit(n.key+other.key)*Math.PI*2; dx = Math.cos(angle); dy = Math.sin(angle); }
        const d = Math.max(1,Math.hypot(dx,dy)), strength = Math.min(9,1500/(d*d));
        f.x += dx/d*strength; f.y += dy/d*strength;
      }
      const origin = n.type === "run" || n.type === "project" ? center : byKey.get(JSON.stringify(["run",n.runId])) || center;
      f.x += (origin.x-n.x)*.006; f.y += (origin.y-n.y)*.006;
    }
    for (const e of edges) {
      const dx = e.b.x-e.a.x, dy = e.b.y-e.a.y, distance = Math.max(1,Math.hypot(dx,dy));
      const length = e.a.type === "project" ? 155 : e.kind === "time" ? 95 : 76;
      const strength = (distance-length) * (e.kind === "time" ? .003 : .025);
      const a = forces.get(e.a.key), b = forces.get(e.b.key);
      if (a) { a.x += dx/distance*strength; a.y += dy/distance*strength; }
      if (b) { b.x -= dx/distance*strength; b.y -= dy/distance*strength; }
    }
    const temperature = 5 * (1-step/iterations) + .4;
    for (const n of ordered) {
      const f = forces.get(n.key); if (!f) continue;
      n.x = Math.max(44,n.x+Math.max(-temperature,Math.min(temperature,f.x)));
      n.y = Math.max(44,n.y+Math.max(-temperature,Math.min(temperature,f.y)));
    }
  }
  for (const n of ordered) positions.set(n.key,{ x:n.x, y:n.y });
  // Retain live nodes first; bound dormant positions from old snapshots.
  if (positions.size > LAYOUT.cache) for (const k of positions.keys()) {
    if (!byKey.has(k)) positions.delete(k); if (positions.size <= LAYOUT.cache) break;
  }
  model.layoutStats = { iterations, comparisons, nodes:ordered.length, edges:edges.length };
  return model;
}
function graphNeighborhood(model, key) {
  const keys = new Set([key]);
  for (const e of model.edges) { if (e.from === key) keys.add(e.to); if (e.to === key) keys.add(e.from); }
  return keys;
}
// The graph is a single full-width plane; selected content follows it in normal flow.
function renderGraph(host, model, key, onSelect, detailFor) {
  const previous = host.graphView, same = previous?.key === key;
  previous?.observer?.disconnect();
  const view = same ? previous : { key, autoFit: true, scale: 1, selected: model.nodes.find(n => n.selected)?.key || "project-root", left: 0, top: 0 };
  if (same) { view.left = previous.viewport.scrollLeft; view.top = previous.viewport.scrollTop; }
  const focusedKey = host.contains(document.activeElement) ? document.activeElement?.dataset.nodeKey : null;
  const controls = el("div","graph-controls"), zoomLabel = el("span","quiet-note"), viewport = el("div","graph-viewport"), extent = el("div","graph-extent"), plane = el("div","graph-plane");
  viewport.tabIndex = 0; viewport.setAttribute("role","region"); viewport.setAttribute("aria-label","关系图谱，拖拽节点或空白平移，Ctrl 或 Command 加滚轮缩放");
  const bounds = () => ({ left: Math.min(...model.nodes.map(n=>n.x))-65, top: Math.min(...model.nodes.map(n=>n.y))-45,
    right: Math.max(...model.nodes.map(n=>n.x))+65, bottom: Math.max(...model.nodes.map(n=>n.y))+65 });
  let frame = same && !view.autoFit && view.frame ? {...view.frame} : bounds();
  let width = frame.right-frame.left, height = frame.bottom-frame.top, offsetX=0, offsetY=0;
  const lines = svgEl("svg",{ width,height,"aria-hidden":"true" });
  const marker = svgEl("marker",{ id:`${host.id}-arrow`,viewBox:"0 0 8 8",refX:18,refY:4,markerWidth:5,markerHeight:5,orient:"auto" });
  marker.append(svgEl("path",{ d:"M0,0 L8,4 L0,8",fill:"context-stroke" })); const defs = svgEl("defs"); defs.append(marker); lines.append(defs);
  const byKey = new Map(model.nodes.map(n => [n.key,n])), paths = [], incident = new Map(), buttons = new Map();
  const draw = ({ edge,path }) => { const a=byKey.get(edge.from),b=byKey.get(edge.to); path.setAttribute("d",`M${a.x},${a.y} L${b.x},${b.y}`); };
  for (const edge of model.edges) {
    if (!byKey.has(edge.from) || !byKey.has(edge.to)) continue;
    const path = svgEl("path",{ class:`graph-edge ${edge.kind === "time" ? "temporal" : edge.kind === "membership" ? "membership" : "linked"}` });
    if (["explicit","run-parent","directive"].includes(edge.kind)) path.setAttribute("marker-end",`url(#${host.id}-arrow)`);
    path.append(svgEl("title",{},edge.evidence.join("、"))); const item={edge,path}; paths.push(item); draw(item); lines.append(path);
    for (const k of [edge.from,edge.to]) { const entries=incident.get(k)||[]; entries.push(item); incident.set(k,entries); }
  }
  plane.append(lines);
  const details = detailFor ? el("section","graph-detail") : null;
  if (details) details.setAttribute("aria-label","所选节点的执行流程与结果");
  const highlight = key => {
    const neighbors=graphNeighborhood(model,key);
    for (const [id,button] of buttons) button.setAttribute("data-related",neighbors.has(id)?"true":"false");
    for (const {edge,path} of paths) path.setAttribute("data-related",edge.from===key||edge.to===key?"true":"false");
  };
  const select = (node, notify=true) => {
    view.selected=node.key;
    for (const [id,button] of buttons) button.setAttribute("aria-pressed",String(id===node.key));
    if (details) details.replaceChildren(detailFor(node)); highlight(node.key);
    if (notify) onSelect?.(node);
  };
  const place = node => { const button=buttons.get(node.key); button.style.left=`${node.x}px`; button.style.top=`${node.y}px`; };
  let nodeDrag=null, suppressClick="";
  for (const node of model.nodes) {
    const button=el("button",`graph-node ${node.type}`); button.type="button"; button.dataset.nodeKey=node.key;
    button.setAttribute("aria-pressed",String(node.key===view.selected));
    button.append(el("span","node-dot"),el("span","node-label",node.title));
    button.setAttribute("aria-label", node.title);
    if (node.type==="run") {
      button.append(el("span","node-state",node.loadStatus==="pending"?"…":node.loadStatus==="error"?"!":""));
    }
    button.title=`${node.title} · ${asText(node.summary,"")}`;
    button.addEventListener("mouseenter",()=>highlight(node.key)); button.addEventListener("mouseleave",()=>highlight(view.selected));
    button.addEventListener("focus",()=>highlight(node.key)); button.addEventListener("blur",()=>highlight(view.selected));
    button.addEventListener("click",()=>{ if(suppressClick===node.key){suppressClick="";return;} select(node); });
    button.addEventListener("pointerdown",e=>{
      if(e.button!==0)return; e.stopPropagation(); suppressClick="";
      nodeDrag={node,x:e.clientX,y:e.clientY,left:node.x,top:node.y,moved:false}; view.dragging=true; button.setPointerCapture(e.pointerId);
    });
    button.addEventListener("pointermove",e=>{
      if(nodeDrag?.node!==node)return;
      const dx=(e.clientX-nodeDrag.x)/view.scale,dy=(e.clientY-nodeDrag.y)/view.scale;
      if(Math.hypot(e.clientX-nodeDrag.x,e.clientY-nodeDrag.y)>4)nodeDrag.moved=true;
      if(!nodeDrag.moved)return;
      node.x=Math.max(36,nodeDrag.left+dx); node.y=Math.max(36,nodeDrag.top+dy); place(node);
      for(const item of incident.get(node.key)||[])draw(item);
      frame.right=Math.max(frame.right,node.x+65);frame.bottom=Math.max(frame.bottom,node.y+65);
      width=frame.right-frame.left;height=frame.bottom-frame.top;resize();
    });
    const finish = () => {
      if(nodeDrag?.node!==node)return;
      if(nodeDrag.moved){view.autoFit=false;view.frame={...frame};suppressClick=node.key;const scope=state.activeScope;if(scope?.key===key){scope.positions.set(node.key,{x:node.x,y:node.y});scope.pinned ||= new Set();scope.pinned.add(node.key);scope.needsRelax=true;savePositions(scope);view.pendingRender=true;}}
      nodeDrag=null;view.dragging=false;
      if(view.pendingRender){view.pendingRender=false;renderProject();}
    };
    button.addEventListener("pointerup",finish);button.addEventListener("pointercancel",finish);button.addEventListener("lostpointercapture",finish);
    buttons.set(node.key,button);place(node);plane.append(button);
  }
  const resize=()=>{
    offsetX=Math.max(0,(viewport.clientWidth-width*view.scale)/2)-frame.left*view.scale;
    offsetY=Math.max(0,(viewport.clientHeight-height*view.scale)/2)-frame.top*view.scale;
    plane.style.width=`${frame.right}px`;plane.style.height=`${frame.bottom}px`;
    plane.style.transform=`translate(${offsetX}px,${offsetY}px) scale(${view.scale})`;
    lines.setAttribute("width",frame.right);lines.setAttribute("height",frame.bottom);
    extent.style.width=`${Math.max(viewport.clientWidth,width*view.scale)}px`;
    extent.style.height=`${Math.max(viewport.clientHeight,height*view.scale)}px`;
    zoomLabel.textContent=`${Math.round(view.scale*100)}%`;view.frame={...frame};
  };
  const scaleTo=(scale,centered=true)=>{
    const cx=(viewport.scrollLeft+viewport.clientWidth/2-offsetX)/view.scale,cy=(viewport.scrollTop+viewport.clientHeight/2-offsetY)/view.scale;
    view.autoFit=false;view.scale=Math.max(.02,Math.min(3,scale));resize();
    if(centered){viewport.scrollLeft=cx*view.scale+offsetX-viewport.clientWidth/2;viewport.scrollTop=cy*view.scale+offsetY-viewport.clientHeight/2;}
  };
  const fit=()=>{
    frame=bounds();width=frame.right-frame.left;height=frame.bottom-frame.top;
    scaleTo(Math.min((viewport.clientWidth-20)/width,(viewport.clientHeight-20)/height,1.25),false);
    view.autoFit=true;viewport.scrollLeft=0;viewport.scrollTop=0;
  };
  for(const [label,title,action] of [["−","缩小",()=>scaleTo(view.scale/1.25)],["+","放大",()=>scaleTo(view.scale*1.25)],["适应","适应全部节点",fit],["100%","恢复原始大小",()=>scaleTo(1)],["重排","自动调整节点布局",()=>{ const scope=state.activeScope;if(scope?.key!==key)return;scope.pinned.clear();scope.needsRelax=true;view.autoFit=true;renderProject(); }]]){
    const button=el("button","",label);button.type="button";button.setAttribute("aria-label",title);button.addEventListener("click",action);controls.append(button);
  }
  controls.append(zoomLabel,el("span","graph-hint quiet-note","拖拽节点 · 空白平移 · Ctrl/⌘ + 滚轮缩放"));
  viewport.addEventListener("wheel",e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();scaleTo(view.scale*(e.deltaY<0?1.1:1/1.1));}},{passive:false});
  let pan=null;
  viewport.addEventListener("pointerdown",e=>{
    if(e.button!==0||e.target.closest("button"))return;
    view.autoFit=false;pan={x:e.clientX,y:e.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};view.dragging=true;viewport.setPointerCapture(e.pointerId);viewport.classList.add("panning");
  });
  viewport.addEventListener("pointermove",e=>{if(pan){viewport.scrollLeft=pan.left-e.clientX+pan.x;viewport.scrollTop=pan.top-e.clientY+pan.y;}});
  const endPan=()=>{if(!pan)return;pan=null;view.dragging=false;viewport.classList.remove("panning");if(view.pendingRender){view.pendingRender=false;renderProject();}};
  viewport.addEventListener("pointerup",endPan);viewport.addEventListener("pointercancel",endPan);viewport.addEventListener("lostpointercapture",endPan);
  viewport.addEventListener("keydown",e=>{
    if(!e.target.dataset.nodeKey||!["ArrowRight","ArrowLeft","Home","End"].includes(e.key))return;
    e.preventDefault();const keys=[...buttons.keys()],i=keys.indexOf(e.target.dataset.nodeKey),next=e.key==="Home"?0:e.key==="End"?keys.length-1:Math.max(0,Math.min(keys.length-1,i+(e.key==="ArrowRight"?1:-1)));
    buttons.get(keys[next]).focus();
  });
  extent.append(plane);viewport.append(extent);const layout=el("div","graph-layout");layout.append(viewport);if(details)layout.append(details);
  host.replaceChildren(controls,layout);view.viewport=viewport;host.graphView=view;resize();
  if(!same || view.autoFit || model.topologyChanged)fit();else{viewport.scrollLeft=view.left;viewport.scrollTop=view.top;}
  if(typeof ResizeObserver !== "undefined") {
    let lastWidth=viewport.clientWidth,lastHeight=viewport.clientHeight;
    view.observer=new ResizeObserver(()=>{
      if(!viewport.isConnected || view.dragging)return;
      if(lastWidth===viewport.clientWidth && lastHeight===viewport.clientHeight)return;
      lastWidth=viewport.clientWidth;lastHeight=viewport.clientHeight;
      if(view.autoFit)fit();else resize();
    });view.observer.observe(viewport);
  }
  if(byKey.has(view.selected))select(byKey.get(view.selected),false);
  if(focusedKey)buttons.get(focusedKey)?.focus({preventScroll:true});
}
const sameId = (a, b) => present(a) && present(b) && a === b;
function linkedJob(row, s) {
  const jobs = list(s.jobs);
  // A declared but missing job must not silently fall back to another task.
  if (present(row.job_id)) { const matches=jobs.filter(j => sameId(j.job_id,row.job_id)); return matches.length===1?matches[0]:undefined; }
  const matches = jobs.filter(j => sameId(j.directive_id, row.directive_id));
  return matches.length === 1 ? matches[0] : undefined;
}
function relatedResults(row,s) {
  const results=[],job=linkedJob(row,s);
  if(job)results.push({type:"job",row:job,evidence:present(row.job_id)?"明确任务 ID 关联；结果为当前快照。":"唯一指令 ID 关联；结果为当前快照。"});
  const candidates=list(s.candidates),matches=present(row.candidate_id)?candidates.filter(c=>sameId(c.candidate_id,row.candidate_id)):[];
  if(matches.length===1)results.push({type:"candidate",row:matches[0],evidence:"明确候选 ID 关联；结果为当前快照。"});
  // Original directives can report multiple resulting candidates; reviews never
  // become origins and repeated directive IDs cannot claim each other's results.
  if(!present(row.candidate_id)&&row.kind==="directive"&&identifier(row.directive_id)&&list(s.decisions).filter(d=>d.kind==="directive"&&sameId(d.directive_id,row.directive_id)).length===1){
    for(const c of candidates.filter(c=>sameId(c.directive_id,row.directive_id)))results.push({type:"candidate",row:c,evidence:"唯一原始指令关联；结果为当前快照。"});
  }
  return results;
}
function feedbackFor(row, s) {
  const feedback = object(row.feedback), current = object(s.current), job = linkedJob(row, s);
  const matchesCurrent = row.kind !== "review" && (sameId(row.decision_id, current.decision_id) || sameId(row.directive_id, current.directive_id));
  // Snapshot job/candidate values are current results, not historical values at review time.
  const metrics = row.kind === "review" ? { ...object(row.metrics), ...object(feedback.metrics) } : {
    ...object(job?.metrics), ...(matchesCurrent ? object(current.metrics) : {}), ...object(row.metrics), ...object(feedback.metrics),
    ...object(feedback.training_metrics), ...object(feedback.quality_metrics), ...object(feedback.execution_metrics),
  };
  const candidate = list(s.candidates).find(c => sameId(c.candidate_id, row.candidate_id));
  if (row.kind !== "review" && ["evaluate", "evaluation", "apply_candidate", "select"].includes(decisionAction(row)) && !Object.keys(flattenMetrics(metrics)).length && candidate) Object.assign(metrics, object(candidate.metrics));
  let status = feedback.status || row.status;
  if (!status && job) status = job.status;
  let text;
  if (row.kind === "review") text = row.action === "cancel" ? "已请求取消；任务是否停止以任务回报为准。" : row.action === "continue" ? "复查决定继续执行。" : "复查结果未报告。";
  else if (decisionAction(row) === "finish") text = "模型提出结束本轮研究。";
  else if (FAILED.has(statusKey(status))) text = "执行失败或已取消。";
  else if (DONE.has(statusKey(status))) text = "执行已完成。";
  else if (ACTIVE.has(statusKey(status))) text = "执行中，等待最终反馈。";
  else if (status === "proposed") text = "已提出，尚未报告执行结果。";
  else text = "尚未报告执行反馈。";
  if (candidate && row.kind !== "review" && decisionAction(row) !== "train") text += ` 候选${ACCEPTED.has(statusKey(candidate.status)) ? "已采纳" : REJECTED.has(statusKey(candidate.status)) ? "未采纳" : "状态尚未确定"}（当前快照）。`;
  const explicit = feedback.summary || row.feedback_summary;
  if (explicit) text = "已报告执行反馈，见详情。";
  return { text, metrics, job, status, source: row.kind === "review" ? "复查记录" : "当前公开快照" };
}
function renderExecution(s) {
  const current = object(s.current), workflow = object(s.workflow);
  const ended = DONE.has(statusKey(current.status)) || FAILED.has(statusKey(current.status)) || current.status === "stopped";
  $("execution-title").textContent = ended ? "最近执行" : "当前执行";
  $("workflow-summary").textContent = [number(workflow.round) ? `第 ${workflow.round} 轮` : "", number(workflow.elapsed_seconds) ? `耗时 ${secondsText(workflow.elapsed_seconds)}` : ""].filter(Boolean).join(" · ");
  if (!Object.keys(current).length) { $("current-execution").replaceChildren(empty("尚未报告当前执行。")); return; }
  const card = el("div", "execution-card"), head = el("div", "execution-head"), body = el("div", "execution-content");
  head.append(el("h3", "", actionText(current.operation || current.mode || current.phase)), badge(current.status));
  if (present(current.strategy)) body.append(proseView(current.strategy, "执行原文"));
  const job = linkedJob(current, s), telemetry = el("div");
  if (present(current.step) || present(job?.step)) telemetry.append(progressView(current.step ?? job?.step, current.total_steps ?? job?.total_steps));
  telemetry.append(metricStrip({ ...object(job?.metrics), ...object(current.metrics) }));
  if (number(current.next_review_seconds) && current.next_review_seconds >= 0) telemetry.append(el("p", "execution-timing", `下次复查：${secondsText(current.next_review_seconds)}后（快照）`));
  body.append(telemetry); card.append(head, body, retainedDetails("current", state.openJobs, "执行详情", rawFields(current))); $("current-execution").replaceChildren(card);
}
function renderDecisions() {
  if (!state.snapshot) return;
  const s = state.snapshot, filter = $("decision-filter").value, query = $("decision-search").value.trim().toLowerCase();
  const groups = decisionGroups(s).filter(g => (filter === "all" || g.row.kind === filter) && g.rows.some(row => `${JSON.stringify(row)} ${actionText(decisionAction(row))} ${JSON.stringify(feedbackFor(row,s).metrics)} ${feedbackFor(row,s).text}`.toLowerCase().includes(query)));
  const records = groups.reduce((n, g) => n + g.rows.length, 0);
  $("decision-count").textContent = `${groups.length} 项${records !== groups.length ? ` · ${records} 条记录` : ""}`;
  $("more-decisions").hidden = groups.length <= state.decisionLimit;
  $("decision-list").replaceChildren(...(groups.length ? groups.slice(0, state.decisionLimit).map(g => {
    const row = g.row, card = el("article", "decision-card"), meta = el("div", "decision-meta"), body = el("div", "decision-body"), action = el("div"), feedback = el("div", "feedback"), f = feedbackFor(row,s);
    const time = el("time", "", dateText(row.timestamp)); if (Number.isFinite(Date.parse(row.timestamp))) time.dateTime = row.timestamp;
    meta.append(time); if (number(row.round)) meta.append(el("span", "", `第 ${row.round} 轮`)); if (row.kind === "review") meta.append(el("span", "", "复查"));
    if (g.rows.length > 1) meta.append(el("span", "repeat-count", `重复报告 ${g.rows.length} 次`));
    action.append(el("h3", "", actionText(decisionAction(row))));
    if (present(row.reason) || present(row.strategy)) action.append(proseView(row.reason || row.strategy));
    feedback.append(el("h4", "", "反馈"), el("p", "", f.text));
    const metrics = metricStrip(f.metrics); if (metrics.childElementCount) feedback.append(metrics);
    if (f.job && row.kind !== "review" && present(f.job.step)) feedback.append(progressView(f.job.step, f.job.total_steps));
    if (f.job && row.kind === "review") feedback.append(el("p", "quiet-note", `关联任务当前${statusText(f.job.status)}；不代表复查当时状态。`));
    if (decisionAction(row) === "finish") feedback.append(el("p", "quiet-note", `运行当前${statusText(s.status)}。`));
    body.append(action, feedback); card.append(meta, body);
    const detail = el("div");
    for (const record of g.rows) { const raw = el("div", "raw-record"); raw.append(rawFields(record)); const m = feedbackFor(record,s).metrics; if (Object.keys(flattenMetrics(m)).length) raw.append(metricsDetail(m)); detail.append(raw); }
    card.append(retainedDetails(`decision:${decisionKey(g.rows[g.rows.length-1])}`, state.openDecisions, g.rows.length > 1 ? `展开 ${g.rows.length} 条记录与详情` : "决策与反馈详情", detail));
    return card;
  }) : [empty(list(s.decisions).length ? "没有符合筛选条件的决策。" : "尚无决策记录。未报告的决策与反馈不会补全。") ]));
}
function jobGroup(j) { const k = statusKey(j.status); return ACTIVE.has(k) ? "active" : DONE.has(k) ? "completed" : FAILED.has(k) ? "failed" : "other"; }
function historyRows(j) {
  return list(j.metric_history).slice(-1000).map(p => ({
    step: Number.isInteger(p.step) && p.step >= 0 ? p.step : undefined,
    elapsed_seconds: number(p.elapsed_seconds) && p.elapsed_seconds >= 0 ? p.elapsed_seconds : undefined,
    metrics: Object.fromEntries(Object.entries(object(p.metrics)).filter(([,v]) => number(v))),
  }));
}
const lossKeys = ["loss", "train_loss", "train/loss", "train.loss"];
const lrKeys = ["learning_rate", "lr", "train/learning_rate", "train.learning_rate"];
function metricValue(metrics, keys) { for (const k of keys) if (number(metrics[k])) return metrics[k]; return undefined; }
function svgEl(tag, attrs = {}, text) { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, String(v)); if (text !== undefined) e.textContent = text; return e; }
function seriesFor(j, keys) {
  const history = historyRows(j);
  const usable = history.filter(p => number(metricValue(p.metrics,keys)));
  const axis = usable.length && usable.every(p => number(p.step)) ? "step" : "elapsed_seconds";
  const byX = new Map();
  for (const p of history) if (number(p[axis])) byX.set(p[axis], { x: p[axis], y: metricValue(p.metrics, keys), step: p.step, elapsed: p.elapsed_seconds });
  return { axis, rows: [...byX.values()].sort((a,b) => a.x-b.x), history };
}
function chartView(j, keys, title) {
  const figure = el("figure", "chart"), heading = el("div", "chart-heading"), series = seriesFor(j,keys), points = series.rows.filter(p => number(p.y));
  const latest = metricValue(flattenMetrics(j.metrics), keys) ?? [...series.history].reverse().map(p => metricValue(p.metrics,keys)).find(number);
  heading.append(el("h4", "", title), el("span", "chart-value", number(latest) ? fmt(latest) : "未报告")); figure.append(heading);
  if (!points.length) { figure.append(el("div", "chart-empty", number(latest) ? "仅有当前指标，未记录可绘图的历史。" : "尚未报告该指标的历史。")); return figure; }
  const width = 480, height = 205, left = 82, right = 18, top = 18, bottom = 34;
  const ys = points.map(p=>p.y), xs = points.map(p=>p.x), ymin = Math.min(...ys), ymax = Math.max(...ys), xmin = Math.min(...xs), xmax = Math.max(...xs);
  // Scale before subtraction to keep even extreme finite inputs inside the SVG.
  const yunit = Math.max(Math.abs(ymin),Math.abs(ymax)) || 1, xunit = Math.max(Math.abs(xmin),Math.abs(xmax)) || 1;
  const ylo = ymin / yunit, yhi = ymax / yunit, yspan = yhi - ylo, ypad = yspan ? yspan * .12 : .1;
  const py = v => top + (yhi + ypad - v / yunit) / (yspan + 2 * ypad) * (height - top - bottom);
  const px = v => xmax === xmin ? (left + width - right) / 2 : left + ((v / xunit - xmin / xunit) / (xmax / xunit - xmin / xunit)) * (width - left - right);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", tabindex: "0", "aria-label": `${title}，${points.length} 个历史点；左右方向键逐点查看，Home 和 End 查看首尾。` });
  svg.append(svgEl("title", {}, `${title} · ${series.axis === "step" ? "步数" : "耗时（秒）"}`));
  const ticks = ymin === ymax ? [ymin] : [ymax, ymin / 2 + ymax / 2, ymin];
  for (const v of ticks) { svg.append(svgEl("line", { x1: left, x2: width-right, y1: py(v), y2: py(v), class: "chart-grid" }), svgEl("text", { x: left - 8, y: py(v)+4, "text-anchor": "end", class: "chart-axis" }, fmt(v))); }
  const xLabel = series.axis === "step" ? "步" : "秒";
  svg.append(svgEl("text", { x: left, y: height-9, class: "chart-axis" }, `${fmt(xmin)} ${xLabel}`));
  if (xmin !== xmax) svg.append(svgEl("text", { x: width-right, y: height-9, "text-anchor": "end", class: "chart-axis" }, `${fmt(xmax)} ${xLabel}`));
  let path = "", connected = false;
  for (const p of series.rows) { if (!number(p.y)) { connected = false; continue; } path += `${connected ? "L" : "M"}${px(p.x).toFixed(2)},${py(p.y).toFixed(2)} `; connected = true; }
  svg.append(svgEl("path", { d: path, class: "chart-line" }));
  // Keep an honest single dot for one-point history. The inspection marker follows real samples.
  const marker = svgEl("circle", { cx: px(points[points.length-1].x), cy: py(points[points.length-1].y), r: 4, class: "chart-point" }); svg.append(marker);
  const readout = el("p", "chart-readout"); let selected = points.length-1;
  const inspect = index => { selected = Math.max(0,Math.min(points.length-1,index)); const p = points[selected]; marker.setAttribute("cx",px(p.x)); marker.setAttribute("cy",py(p.y)); readout.textContent = `${number(p.step) ? `第 ${p.step} 步` : `${fmt(p.x)} 秒`} · ${title} ${fmt(p.y)}${number(p.elapsed) ? ` · 耗时 ${secondsText(p.elapsed)}` : ""}${points.length === 1 ? " · 仅 1 个历史点" : ""}`; };
  inspect(selected);
  svg.addEventListener("pointermove", event => { const bounds = svg.getBoundingClientRect(); const x = (event.clientX - bounds.left) / bounds.width * width; let index = 0; for (let i=1; i<points.length; i++) if (Math.abs(px(points[i].x)-x) < Math.abs(px(points[index].x)-x)) index=i; inspect(index); });
  svg.addEventListener("keydown", event => { const actions = { ArrowLeft: selected-1, ArrowRight: selected+1, Home: 0, End: points.length-1 }; if (Object.hasOwn(actions,event.key)) { event.preventDefault(); inspect(actions[event.key]); } });
  figure.append(svg,readout); return figure;
}
function historyDetail(j) {
  const rows = historyRows(j), wrap = el("div");
  if (!rows.length) { wrap.append(empty("未记录历史；当前指标不能还原训练曲线。")); return wrap; }
  const table = el("table", "metric-table"), head = el("thead"), hr = el("tr"), body = el("tbody");
  for (const name of ["步数", "耗时（秒）", "已报告指标"]) { const th = el("th", "", name); th.scope = "col"; hr.append(th); } head.append(hr);
  for (const p of rows) { const row = el("tr"); row.append(el("td", "", fmt(p.step)),el("td", "", fmt(p.elapsed_seconds)), el("td", "", Object.entries(p.metrics).map(([k,v])=>`${k}: ${fmt(v)}`).join("；") || "未报告")); body.append(row); }
  table.append(head,body); wrap.append(table); return wrap;
}
function renderJobs() {
  if (!state.snapshot) return;
  const rows = list(state.snapshot.jobs), shown = rows.filter(j => $("job-filter").value === "all" || jobGroup(j) === $("job-filter").value);
  $("jobs-count").textContent = `${shown.length} 项`;
  $("job-list").replaceChildren(...(shown.length ? shown.map(j => {
    const card = el("article", "job-record"), heading = el("div", "job-heading"), meta = el("p", "job-sub");
    heading.append(el("h3", "", actionText(j.kind)), badge(j.status));
    meta.textContent = [number(j.round) ? `第 ${j.round} 轮` : "", number(j.elapsed_seconds) ? `耗时 ${secondsText(j.elapsed_seconds)}` : ""].filter(Boolean).join(" · "); card.append(heading,meta);
    const telemetry = el("div", "job-telemetry"); if (present(j.step) || present(j.total_steps)) telemetry.append(progressView(j.step,j.total_steps)); telemetry.append(metricStrip(j.metrics)); card.append(telemetry);
    const history = historyRows(j), isTraining = /train|sft|finetun|fine_tun/.test(asText(j.kind,"")) || history.some(p => number(metricValue(p.metrics,lossKeys)) || number(metricValue(p.metrics,lrKeys))) || number(metricValue(object(j.metrics),lossKeys));
    if (isTraining) { const charts = el("div", "charts"); charts.append(chartView(j,lossKeys,"损失"), chartView(j,lrKeys,"学习率")); card.append(charts); }
    const detail = el("div"); detail.append(rawFields(j), metricsDetail(j.metrics));
    const historyDisclosure = el("details", "record-details"); historyDisclosure.append(el("summary", "", `历史指标 · ${history.length} 点`));
    let loaded = false; historyDisclosure.addEventListener("toggle", () => { if (historyDisclosure.open && !loaded) { historyDisclosure.append(historyDetail(j)); loaded = true; } }); detail.append(historyDisclosure);
    card.append(retainedDetails(`job:${asText(j.job_id,JSON.stringify([j.kind,j.round,j.directive_id]))}`,state.openJobs,"指标与任务详情",detail)); return card;
  }) : [empty(rows.length ? "没有符合筛选条件的任务。" : "尚无任务记录；无法确认是否执行了训练或评测。") ]));
}
const baseAliases = ["baseline", "base", "before", "old"], newAliases = ["candidate", "new", "after", "current"];
function metricRows(source) {
  const metrics = object(source), pairs = [], used = new Set();
  const baseGroup = baseAliases.find(k => metrics[k] && typeof metrics[k] === "object"), newGroup = newAliases.find(k => metrics[k] && typeof metrics[k] === "object");
  if (baseGroup || newGroup) { const a = flattenMetrics(metrics[baseGroup]), b = flattenMetrics(metrics[newGroup]); for (const key of new Set([...Object.keys(a),...Object.keys(b)])) pairs.push({ name: key, base: a[key], next: b[key] }); if (baseGroup) used.add(baseGroup); if (newGroup) used.add(newGroup); }
  for (const [key,v] of Object.entries(metrics)) { if (used.has(key) || !v || typeof v !== "object") continue; const a = baseAliases.find(k=>number(v[k])), b = newAliases.find(k=>number(v[k])); if (a || b) { pairs.push({ name:key,base:v[a],next:v[b] }); used.add(key); } }
  const flat = flattenMetrics(Object.fromEntries(Object.entries(metrics).filter(([k])=>!used.has(k)))), grouped = new Map(), paired = new Set();
  for (const [key,v] of Object.entries(flat)) { for (const [aliases,side] of [[baseAliases,"base"],[newAliases,"next"]]) { const alias = aliases.find(a=>key.startsWith(a+"_") || key.endsWith("_"+a)); if (!alias) continue; const name = key.startsWith(alias+"_") ? key.slice(alias.length+1) : key.slice(0,-alias.length-1); const row = grouped.get(name) || { name }; row[side]=v; grouped.set(name,row); paired.add(key); break; } }
  return { pairs:[...pairs,...grouped.values()],extra:Object.fromEntries(Object.entries(flat).filter(([k])=>!paired.has(k))) };
}
function comparisonView(metrics) {
  const wrap = el("div"), { pairs } = metricRows(metrics);
  if (!pairs.length) { const strip=metricStrip(metrics); wrap.append(strip.childElementCount ? strip : empty("评测指标未报告。")); return wrap; }
  const table=el("table","metric-table"), head=el("thead"), hr=el("tr"), body=el("tbody");
  for (const name of ["指标","基线","候选","候选 − 基线"]) {const th=el("th","",name);th.scope="col";hr.append(th);}head.append(hr);
  for (const m of pairs) { const row=el("tr"), delta=number(m.base)&&number(m.next)?m.next-m.base:undefined;row.append(el("td","",metricText(m.name)),el("td","",fmt(m.base)),el("td","",fmt(m.next)),el("td","",number(delta)?`${delta>0?"+":""}${fmt(delta)}`:"—"));body.append(row); }table.append(head,body);wrap.append(table);return wrap;
}
function candidateGroup(c) { const k=statusKey(c.status);return ACCEPTED.has(k)?"accepted":REJECTED.has(k)?"rejected":PENDING.has(k)?"pending":"other"; }
function renderCandidates() {
  if(!state.snapshot)return;const rows=list(state.snapshot.candidates),shown=rows.filter(c=>$("candidate-filter").value==="all"||candidateGroup(c)===$("candidate-filter").value);$("candidate-count").textContent=`${shown.length} 项`;
  $("candidate-list").replaceChildren(...(shown.length?shown.map(c=>{const card=el("article","candidate-card"),head=el("div","candidate-top");head.append(el("h3","",`候选 ${rows.indexOf(c)+1}`),badge(c.status));card.append(head,comparisonView(c.metrics));const content=el("div");content.append(rawFields(c),metricsDetail(c.metrics));card.append(retainedDetails(`candidate:${asText(c.candidate_id,rows.indexOf(c))}`,state.openJobs,"候选与指标详情",content));return card;}):[empty(rows.length?"没有符合筛选条件的候选。":"尚无候选记录。") ]));
}
function renderRounds(s) {
  const rows=list(s.round_results).slice().sort((a,b)=>(b.round||0)-(a.round||0));$("round-count").textContent=`${rows.length} 轮`;
  $("round-list").replaceChildren(...(rows.length?rows.map(row=>{const card=el("article","round-card"),head=el("div","round-heading");head.append(el("h3","",number(row.round)?`第 ${row.round} 轮`:"轮次未报告"),el("span","badge",row.applied===true?"已应用":row.applied===false?"未应用":"应用状态未报告"));card.append(head,el("p","",row.answer_correct===true?"答案验证正确":row.answer_correct===false?"答案验证不正确":"答案验证未报告"),comparisonView(row.metrics));const detail=el("div");detail.append(rawFields(row),metricsDetail(row.metrics));card.append(retainedDetails(`round:${row.round}`,state.openJobs,"轮次详情",detail));return card;}):[empty("尚无轮次结果；不能据此推断是否完成过执行。") ]));
}
const eventLabels={job:"任务状态更新",progress:"执行进度更新",phase:"运行阶段更新",inference:"模型推理",feedback:"执行反馈",model_decision_requested:"请求模型决策",model_decision_validated:"决策通过校验",model_decision_rejected:"决策未通过校验",training_review_requested:"请求训练复查",training_review_completed:"训练复查完成",activation_verified:"候选验证通过","training.started":"训练开始","evaluation.completed":"评测完成","candidate.accepted":"候选已采纳","candidate.rejected":"候选未采纳","model.bound":"模型绑定更新","run.started":"运行开始"};
function updateEventTypes(){const select=$("event-filter"),previous=select.value,types=[...new Set(list(state.snapshot?.events).map(e=>asText(e.kind,"未分类")))].sort();select.replaceChildren(...[["all","全部类型"],...types.map(t=>[t,eventLabels[t]||"其他事件"])].map(([v,t])=>{const o=el("option","",t);o.value=v;return o;}));select.value=types.includes(previous)?previous:"all";}
function renderEvents(){if(!state.snapshot)return;const rows=list(state.snapshot.events),query=$("event-search").value.trim().toLowerCase(),filter=$("event-filter").value,shown=rows.filter(e=>(filter==="all"||e.kind===filter)&&JSON.stringify(e).toLowerCase().includes(query)).slice().sort((a,b)=>(Date.parse(b.timestamp)||0)-(Date.parse(a.timestamp)||0));$("event-count").textContent=`${shown.length} 条`;$("more-events").hidden=shown.length<=state.eventLimit;$("event-list").replaceChildren(...(shown.length?shown.slice(0,state.eventLimit).map(e=>{const row=el("article","event"),content=el("div","event-content");content.append(el("p","",eventLabels[e.kind]||"其他事件"),retainedDetails(JSON.stringify(e),state.openEvents,"事件详情",rawFields({...e,...object(e.details)})));row.append(el("time","event-time",dateText(e.timestamp)),content);return row;}):[empty(rows.length?"没有符合筛选条件的事件。":"尚无事件。") ]));}
function renderSnapshot(s){state.snapshot=s;$("empty").hidden=true;$("dashboard").hidden=false;$("export-report").disabled=false;$("objective").textContent=publicText(s.objective,"运行目标未报告");$("run-status").replaceChildren(badge(s.status));$("updated").textContent=`${state.demo?"示例时间":"更新"} ${dateText(s.updated_at)}`;const detail=el("div");detail.append(rawFields({run_id:s.run_id,status:s.status,updated_at:s.updated_at}));for(const [role,model] of Object.entries(object(s.models))){detail.append(el("h4","",role==="actor"?"执行模型":"决策模型"),rawFields(model));}for(const [name,loop] of Object.entries(object(s.loops))){detail.append(el("h4","",name==="exploration"?"探索循环":"决策循环"),rawFields(loop));}$("run-info").replaceChildren(detail);renderExecution(s);renderDecisions();renderJobs();renderCandidates();renderRounds(s);updateEventTypes();renderEvents();$("run-label").textContent=`${projectMeta(s.project).name} / 当前运行`;}
// Pure snapshot-to-Markdown conversion: no model call, network request or upload.
const md = value => asText(value, "未报告").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replace(/([\\`*_{}\[\]()#+.!|~-])/g, "\\$1");
function reportMetrics(metrics) {
  const entries = Object.entries(flattenMetrics(metrics));
  return entries.length ? entries.map(([key,v]) => `- ${md(metricText(key))}（${md(key)}）：${fmt(v)}`).join("\n") : "未报告指标。";
}
function reportDetails(row) {
  const lines = Object.entries(object(row)).filter(([,v])=>["string","number","boolean"].includes(typeof v)).map(([k,v])=>`- ${md(k)}：${md(v)}`);
  if (parentIds(row).length) lines.push(`- 显式父决策：${parentIds(row).map(md).join("、")}`);
  return `<details>\n<summary>技术详情与原始记录</summary>\n\n${lines.join("\n")}\n\n</details>`;
}
function reportMarkdown(s) {
  const jobs=list(s.jobs), decisions=list(s.decisions), candidates=list(s.candidates), rounds=list(s.round_results);
  const lines=["# RSISys 运行报告", "", state.demo ? "> 示例数据 · 静态演示，不代表真实运行。" : "> 来自当前公开快照；不调用模型，不上传数据。", "", `项目：${md(projectMeta(s.project).name)}`, `系统：${md(projectMeta(s.project).system_id || "未报告")}`, `目标：${md(s.objective)}`, "", `运行状态：${md(statusText(s.status))}`, `快照时间：${md(s.updated_at)}`, `导出时间：${md(new Date().toISOString())}`, "", "本报告反映已读取的快照。缺失字段不代表零、成功或未执行；结束本轮研究不等于停止整个运行。"];
  if($("connection").classList.contains("offline")) lines.push("","> 连接中断：本报告使用上次成功读取的快照，数据未更新。");
  if(!state.demo && ACTIVE.has(statusKey(s.status)) && Date.now()-Date.parse(s.updated_at)>120000) lines.push("","> 快照超过 2 分钟未更新。");
  lines.push("", "## 决策与反馈", "");
  if(!decisions.length)lines.push("未报告决策记录，无法重建决策或反馈。");
  for(const g of decisionGroups(s)) {
    const r=g.row,f=feedbackFor(r,s);
    lines.push(`### ${md(actionText(decisionAction(r)))}${g.rows.length>1?` · 重复报告 ${g.rows.length} 次`:""}`, "", `${md(dateText(r.timestamp))}${number(r.round)?` · 第 ${r.round} 轮`:""}`, "", `反馈：${md(f.text)}`, "",reportMetrics(f.metrics),"");
    if(r.kind==="review" && f.job)lines.push(`关联任务当前${md(statusText(f.job.status))}；不代表复查当时状态。`,"");
    if(decisionAction(r)==="finish")lines.push(`运行当前${md(statusText(s.status))}。`,"");
    if(present(r.reason)||present(r.strategy))lines.push("决策原文：", "", md(r.reason||r.strategy), "");
    lines.push(`<details>\n<summary>全部 ${g.rows.length} 条原始记录</summary>\n`,...g.rows.map(record=>reportDetails(record)),"\n</details>","");
  }
  lines.push("## 训练与评测", "");
  if(!jobs.length)lines.push("未报告任务，无法确认是否执行过训练或评测。未补造训练曲线。", "");
  for(const [i,j] of jobs.entries()) {
    lines.push(`### 任务 ${i+1} · ${md(actionText(j.kind))}`,"",`状态：${md(statusText(j.status))}`,`耗时：${secondsText(j.elapsed_seconds)}`,`步骤：${number(j.step)?j.step:"未报告"} / ${number(j.total_steps)?j.total_steps:"未报告"}`,"",reportMetrics(j.metrics),"");
    if(["queued","pending","proposed","submitted"].includes(statusKey(j.status)))lines.push("尚未报告实际开始执行。","");
    const history=historyRows(j);lines.push(`快照历史：${history.length} 点（最多 1000 点）。`);
    if(!history.length)lines.push("无历史数据；当前指标不能还原训练过程。");
    for(const [keys,title] of [[lossKeys,"损失"],[lrKeys,"学习率"]]) {
      const values=history.map(p=>metricValue(p.metrics,keys)).filter(number);
      if(!values.length)lines.push(`${title}：未报告历史。`);
      else lines.push(`${title}：${values.length} 个数值点；首点 ${fmt(values[0])}，末点 ${fmt(values[values.length-1])}，范围 ${fmt(Math.min(...values))} 至 ${fmt(Math.max(...values))}${values.length===1?"；仅单点，无法判断趋势":""}。`);
    }
    lines.push("",reportDetails(j),"");
  }
  lines.push("## 候选与轮次", "");
  if(!candidates.length)lines.push("未报告候选记录。", "");
  for(const [i,c] of candidates.entries()) lines.push(`### 候选 ${i+1}`,"",`状态：${md(statusText(c.status))}`,"",reportMetrics(c.metrics),"",reportDetails(c),"");
  if(!rounds.length)lines.push("未报告轮次结果，无法确认是否执行或应用候选。", "");
  for(const r of rounds)lines.push(`### 第 ${number(r.round)?r.round:"未知"} 轮`,"",r.applied===true?"已应用候选。":r.applied===false?"未应用候选。":"是否应用：未报告。",r.answer_correct===true?"答案验证正确。":r.answer_correct===false?"答案验证不正确。":"答案验证：未报告。","",reportMetrics(r.metrics),"",reportDetails(r),"");
  lines.push("## 数据范围", "", "仅包含当前公开快照（API 最多返回 300 条决策），不代表完整实验历史。数值为上报指标；差值的定义以生产端为准，不自动判定改善。缺失指标不补零。旧运行没有历史时不生成趋势。", "",reportDetails({run_id:s.run_id,status:s.status,updated_at:s.updated_at}),"");
  return lines.join("\n");
}
function downloadReport() {
  if(!state.snapshot)return;
  const blob=new Blob([reportMarkdown(state.snapshot)],{type:"text/markdown;charset=utf-8"}),url=URL.createObjectURL(blob),a=el("a");
  a.href=url;a.download=`RSISys-${state.demo?"示例-":""}报告-${new Date().toISOString().replace(/[:.]/g,"-")}.md`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$("export-report").addEventListener("click",downloadReport);

function clearRun() {
  state.snapshot = null;
  $("export-report").disabled = true;
  $("dashboard").hidden = true;
  state.openEvents.clear(); state.openDecisions.clear(); state.openJobs.clear();
  state.decisionLimit = 30; state.eventLimit = 30;

}
function projectMeta(value) {
  const p = object(value), id = identifier(p.project_id) ? p.project_id : "unassigned";
  return { project_id: id, name: id === "unassigned" ? "未归属" : asText(p.name, id), description: asText(p.description, ""), system_id: asText(p.system_id, "") };
}
function runRows(project) {
  return list(project?.runs).filter(r => identifier(r.run_id)).slice().sort((a, b) => {
    const at = Date.parse(a.updated_at), bt = Date.parse(b.updated_at);
    return (Number.isFinite(at) ? at : Infinity) - (Number.isFinite(bt) ? bt : Infinity) || a.run_id.localeCompare(b.run_id);
  });
}
function selectedProject() { return state.projects.find(p => p.project_id === state.projectId); }
function runParentIds(row) {
  return [...new Set([row.parent_run_id, ...(Array.isArray(row.source_run_ids) ? row.source_run_ids : [])].filter(identifier))];
}
function abortEntry(entry) {
  entry.controller?.abort(); entry.controller = null; entry.token = (entry.token || 0) + 1;
  if (entry.status === "pending") entry.status = entry.snapshot ? "ready" : "idle";
}
function activateProject() {
  const key = JSON.stringify([state.demo, state.projectId]);
  if (state.activeScope?.key === key) return state.activeScope;
  if (state.activeScope) {
    state.activeScope.view = $("project-graph").graphView;
    state.activeScope.view?.observer?.disconnect();
    for (const entry of state.activeScope.cache.values()) abortEntry(entry);
  }
  let scope = state.scopes.get(key);
  if (!scope) { scope = { key, pinned: new Set(), cache: new Map(), selected: "project-root", runId: "", view: null, positions: readPositions(key) }; state.scopes.set(key, scope); }
  state.activeScope = scope; $("project-graph").graphView = scope.view;
  state.runId = scope.runId; clearRun();
  return scope;
}
// Build only evidenced relationships; geometry is handled separately by networkLayout.
function projectModel(project, scope) {
  const runs = runRows(project), nodes = [{ key: "project-root", type: "project", row: project, title: project.name, label: "项目", x: 20, y: 90 }], edges = [];
  const ids = new Map();
  runs.forEach((row, i) => {
    const key = JSON.stringify(["run", row.run_id]), entry = scope.cache.get(row.run_id);
    const n = { key, type: "run", row, runId: row.run_id, title: `运行 ${i + 1}`, label: "运行", summary: statusText(row.status),
       loadStatus: entry?.status,
      missing: [], ambiguous: [] };
    nodes.push(n); const matches = ids.get(row.run_id) || []; matches.push(n); ids.set(row.run_id, matches);
    edges.push({ from: "project-root", to: key, kind: "membership", evidence: ["项目归属"] });
  });
  const runNodes = nodes.slice(1);
  for (const node of runNodes) {
    const snapshot = scope.cache.get(node.runId)?.snapshot;
    // A loaded snapshot is authoritative, including absence of optional parents.
    for (const id of runParentIds(snapshot || node.row)) {
      const matches = ids.get(id);
      if (!matches) node.missing.push(id);
      else if (matches.length !== 1) node.ambiguous.push(id);
      else if (matches[0] !== node) edges.push({ from: matches[0].key, to: node.key, kind: "run-parent", evidence: [`显式运行引用 ${id}`] });
    }
  }
  for (let i = 1; i < runNodes.length; i++) {
    const a = runNodes[i - 1], b = runNodes[i], at = Date.parse(a.row.updated_at), bt = Date.parse(b.row.updated_at);
    if (Number.isFinite(at) && Number.isFinite(bt) && at < bt)
      edges.push({ from: a.key, to: b.key, kind: "time", evidence: ["记录更新时间顺序，不表示继承"] });
  }
  const signature = JSON.stringify([nodes.map(n => n.key).sort(), edges.map(e => [e.from,e.to,e.kind]).sort()]);
  const changed = scope.layoutSignature !== undefined && scope.layoutSignature !== signature;
  const model = networkLayout({ nodes, edges }, scope.positions, { relax: changed || !!scope.needsRelax, pinned: scope.pinned });
  model.topologyChanged = changed;
  scope.layoutSignature = signature; scope.needsRelax = false;
  if (model.layoutStats.iterations) savePositions(scope);
  return model;
}
function executionGroups(s) {
  // Known timestamps establish order. Unknown timestamps retain snapshot order at
  // the end, with an explicit caveat; list position is never treated as lineage.
  return decisionGroups(s).sort((a,b) => {
    const at=Date.parse(a.rows[a.rows.length-1].timestamp),bt=Date.parse(b.rows[b.rows.length-1].timestamp);
    return (Number.isFinite(at)?at:Infinity)-(Number.isFinite(bt)?bt:Infinity) || 0;
  });
}
function executionSequence(s, scope) {
  scope.flowLimits ||= new Map(); scope.flowOpen ||= new Set(); scope.flowSelected ||= new Map();
  const section=el("section","execution-sequence"),heading=el("div","section-heading"),groups=executionGroups(s);
  heading.append(el("h4","", "执行流程"),el("span","quiet-note",`${groups.length} 步 · ${list(s.decisions).length} 条记录`));section.append(heading);
  const ordering=el("details","record-details");ordering.append(el("summary","","流程顺序说明"),el("p","quiet-note","按已报告时间从早到晚展示，不表示因果。未报告时间的记录按快照顺序列于末尾；当前结果不代表各决策时点的状态。"));section.append(ordering);
  const navigation=el("ol","flow-map");navigation.setAttribute("aria-label","运行内执行步骤");section.append(navigation);
  const entries=el("ol","flow-steps");section.append(entries);
  const fill=(restoreFocus=false)=>{
    let selectedButton;
    entries.replaceChildren();navigation.replaceChildren();
    const limit=scope.flowLimits.get(s.run_id)||30;
    const keyOf=(g,i)=>JSON.stringify([decisionKey(g.rows[g.rows.length-1]),i]);
    let selected=scope.flowSelected.get(s.run_id);
    if(!groups.slice(0,limit).some((g,i)=>keyOf(g,i)===selected))selected=groups.length?keyOf(groups[0],0):null;
    scope.flowSelected.set(s.run_id,selected);
    for(const [i,g] of groups.slice(0,limit).entries()){
      const row=g.row,f=feedbackFor(row,s),key=keyOf(g,i),item=el("li","flow-map-item"),button=el("button","flow-choice");
      button.type="button";button.setAttribute("aria-pressed",String(key===selected));
      button.setAttribute("aria-label",`步骤 ${i+1} · ${actionText(decisionAction(row))}`);
      button.append(el("span","flow-number",String(i+1).padStart(2,"0")),el("span","",actionText(decisionAction(row))));
      if(g.rows.length>1)button.append(el("span","quiet-note",`×${g.rows.length}`));
      button.addEventListener("click",()=>{scope.flowSelected.set(s.run_id,key);fill(true);});item.append(button);navigation.append(item);
      if(key!==selected)continue;
      selectedButton=button;
      const step=el("li","flow-step"),index=el("span","flow-index",String(i+1).padStart(2,"0"));
      const decision=el("div","flow-decision"),feedback=el("div","flow-feedback"),metrics=el("div","flow-metrics");
      const meta=el("p","quiet-note",`${dateText(row.timestamp)}${number(row.round)?` · 第 ${row.round} 轮`:""}`);
      decision.append(el("p","flow-label","决策"),el("h4","",actionText(decisionAction(row))),meta);
      if(g.rows.length>1)decision.append(el("p","repeat-count",`重复报告 ${g.rows.length} 次`));
      if(present(row.reason)||present(row.strategy))decision.append(retainedDetails(JSON.stringify([s.run_id,"reason",decisionKey(row)]),scope.flowOpen,"决策原文",el("p","decision-reason",row.reason||row.strategy)));
      feedback.append(el("p","flow-label","执行反馈"),el("p","",object(row.feedback).summary||row.feedback_summary||f.text));
      if(row.kind==="review"&&f.job)feedback.append(el("p","quiet-note",`关联任务当前${statusText(f.job.status)}；不代表复查当时状态。`));
      if(row.kind!=="review"&&f.job&&present(f.job.step))feedback.append(progressView(f.job.step,f.job.total_steps));
      metrics.append(el("p","flow-label",row.kind==="review"?"复查时已报告指标":"已报告指标 · 当前快照"),metricsDetail(f.metrics));
      const records=el("div");for(const record of g.rows)records.append(rawFields(record));
      feedback.append(retainedDetails(JSON.stringify([s.run_id,"step",decisionKey(row)]),scope.flowOpen,`记录详情（${g.rows.length} 条）`,records));
      step.append(index,decision,feedback,metrics);entries.append(step);
    }
    more.hidden=groups.length<=limit;
    if(restoreFocus)selectedButton?.focus({preventScroll:true});
  };
  const more=el("button","more-button","显示更多执行步骤");more.addEventListener("click",()=>{scope.flowLimits.set(s.run_id,(scope.flowLimits.get(s.run_id)||30)+30);fill();});
  section.append(more);fill();if(!groups.length)section.append(empty("尚无决策记录，不能重建执行流程。"));
  return section;
}
function runDetail(node, scope) {
  const entry=scope.cache.get(node.runId),s=entry?.snapshot,box=el("div","run-detail");
  const heading=el("div","detail-heading"),identity=el("div");
  identity.append(el("p","eyebrow","所选运行 · 执行流程与结果"),el("h3","",node.title));
  heading.append(identity,badge(s?.status||node.row.status));box.append(heading);
  box.append(el("p","detail-objective",s?.objective||node.row.objective||"目标未报告"));
  if(entry?.status==="pending")box.append(el("p","load-message",s?"正在刷新，保留上次快照。":"正在读取决策与反馈…"));
  if(entry?.status==="error"){
    box.append(el("p","load-message error",`${s?"保留上次快照。":""}读取失败：${entry.error}`));
    const retry=el("button","","重试读取");retry.addEventListener("click",()=>loadRun(node.runId,scope,true));box.append(retry);
  }
  if(s){
    box.append(executionSequence(s,scope));
    if(list(s.candidates).length){
      const candidates=el("details","record-details");candidates.append(el("summary","",`候选结果 · ${list(s.candidates).length}`));
      list(s.candidates).forEach((c,i)=>{const card=el("article","candidate-card");card.append(el("h4","",`候选 ${i+1}`),badge(c.status),comparisonView(c.metrics),rawFields(c));candidates.append(card);});box.append(candidates);
    }
    const report=el("button","","导出此运行报告");report.addEventListener("click",()=>{setSelectedRun(node.runId,scope);downloadReport();});box.append(report);
  }
  const technical=el("details","record-details");technical.append(el("summary","","运行详情与关联依据"),fieldList([["运行",node.runId],["记录更新",dateText(s?.updated_at||node.row.updated_at)]]));
  for(const [role,model] of Object.entries(object(s?.models))){technical.append(el("h4","",role==="actor"?"执行模型":role==="decision_maker"?"决策模型":role),rawFields(model));}
  if(s&&!Object.keys(object(s.models)).length)technical.append(empty("模型 revision 未报告。"));
  const parents=runParentIds(s||node.row);
  technical.append(el("h4","","运行关联依据"),el("p","quiet-note",parents.length?`显式引用：${parents.join("、")}`:"未报告显式父运行。"));
  if(node.missing.length)technical.append(el("p","quiet-note",`父引用不在本项目：${node.missing.join("、")}`));
  if(node.ambiguous.length)technical.append(el("p","quiet-note",`父 ID 歧义，未连线：${node.ambiguous.join("、")}`));
  box.append(technical);return box;
}
function projectDetail(node, model, scope) {
  if (node.type === "project") {
    const box = el("div"); box.append(el("p", "eyebrow", "项目"), el("h3", "", node.title), el("p", "", node.row.description || "项目说明未报告。"), el("p", "quiet-note", "选择运行，查看下方执行流程与结果。")); return box;
  }
  if (node.type === "run") return runDetail(node, scope);
  return el("div");
}

function setSelectedRun(id, scope) {
  scope.runId = id; state.runId = id;
  const s = scope.cache.get(id)?.snapshot;
  if (s) renderSnapshot(s); else clearRun();
  $("run-select").value = id;
  if (!state.demo) {
    const url = new URL(location.href); url.searchParams.set("project", state.projectId);
    if (id) url.searchParams.set("run", id); else url.searchParams.delete("run");
    history.replaceState(null, "", url);
  }
}
function renderProject() {
  if ($("project-graph").graphView?.dragging) { $("project-graph").graphView.pendingRender = true; return; }
  const project = selectedProject(); $("projects").hidden = !project;
  if (!project) return;
  const scope = state.activeScope || activateProject(), model = projectModel(project, scope);
  $("project-name").textContent = project.name; $("project-description").textContent = project.description || "";
  $("project-system").textContent = "";
  $("project-run-count").textContent = `${runRows(project).length} 次运行`;
  $("project-overview").hidden = false;
  if (!model.nodes.some(n => n.key === scope.selected)) scope.selected = scope.runId ? JSON.stringify(["run", scope.runId]) : "project-root";
  if ($("project-graph").graphView) $("project-graph").graphView.selected = scope.selected;
  for (const n of model.nodes) n.selected = n.key === scope.selected;
  renderGraph($("project-graph"), model, scope.key, node => {
    scope.selected = node.key;
    if (node.type === "run") selectRun(node.runId);
    else if (node.runId) setSelectedRun(node.runId, scope);
    else { selectRun(""); }
  }, node => projectDetail(node, model, scope));
  scope.view = $("project-graph").graphView;
  $("empty").hidden = true;
}
function chooseProjects(input) {
  state.projects = list(input).map(p => ({ ...projectMeta(p), runs: list(p.runs) }));
  let deepRun = "";
  if (!state.demo && !state.queryApplied) {
    const params = new URLSearchParams(location.search), run = params.get("run"), project = params.get("project");
    const owner = state.projects.find(p => list(p.runs).some(r => r.run_id === run));
    if (owner) { state.projectId = owner.project_id; deepRun = run; }
    else if (state.projects.some(p => p.project_id === project)) state.projectId = project;
    state.queryApplied = true;
  }
  if (!selectedProject()) state.projectId = state.projects[0]?.project_id || "";
  const scope = activateProject(), runs = runRows(selectedProject()), valid = new Set(runs.map(r => r.run_id));
  for (const [id, entry] of scope.cache) if (!valid.has(id)) { abortEntry(entry); scope.cache.delete(id); }
  if (scope.runId && !valid.has(scope.runId)) { scope.runId = ""; state.runId = ""; clearRun(); }
  if (deepRun) { scope.runId = deepRun; state.runId = deepRun; scope.selected = JSON.stringify(["run", deepRun]); }
  const projects = state.projects.map(p => { const option = el("option", "", p.name); option.value = p.project_id; return option; });
  if (!projects.length) { const option = el("option", "", "暂无项目"); option.value = ""; projects.push(option); }
  $("project-select").replaceChildren(...projects); $("project-select").value = state.projectId; $("project-select").disabled = !state.projects.length;
  const all = el("option", "", "选择运行"); all.value = "";
  $("run-select").replaceChildren(all, ...runs.map((r, i) => { const option = el("option", "", `运行 ${i + 1} · ${statusText(r.status)}`); option.value = r.run_id; return option; }));
  $("run-select").value = state.runId; $("run-select").disabled = !runs.length;
  if (scope.runId) setSelectedRun(scope.runId, scope);
  renderProject();
}
function selectRun(id) {
  const scope = activateProject();
  if (id && !runRows(selectedProject()).some(r => r.run_id === id)) return;
  if (scope.runId && scope.runId !== id) {
    const previous = scope.cache.get(scope.runId); if (previous) abortEntry(previous);
  }
  scope.selected = id ? JSON.stringify(["run", id]) : "project-root";
  setSelectedRun(id, scope); renderProject();
  if (id) loadRun(id, scope);
}
async function loadRun(id, scope, force = false) {
  if (scope !== state.activeScope) return;
  let entry = scope.cache.get(id);
  if (!entry) { entry = { status: "idle", snapshot: null, token: 0 }; scope.cache.set(id, entry); }
  if (entry.status === "pending" || (!force && entry.snapshot && entry.status !== "error")) return;
  const controller = new AbortController(), token = ++entry.token;
  entry.controller = controller; entry.status = "pending"; entry.error = ""; renderProject();
  const timeout = setTimeout(() => controller.abort(), 8000);
  const current = () => scope === state.activeScope && entry.token === token;
  try {
    const s = state.demo ? state.demoSnapshots.get(id) : await fetchJSON(`/api/runs/${encodeURIComponent(id)}`, controller.signal);
    if (!current() || controller.signal.aborted) return;
    if (s?.schema_version !== 1 || s.run_id !== id) throw new Error("运行快照协议或 ID 不匹配。");
    entry.snapshot = s; entry.status = "ready";
  } catch (error) {
    if (!current()) return;
    entry.status = "error"; entry.error = controller.signal.aborted ? "请求超时。" : asText(error.message, "无法连接服务。");
  } finally {
    clearTimeout(timeout);
    if (current()) {
      entry.controller = null;
      if (scope.runId === id && entry.snapshot) renderSnapshot(entry.snapshot);
      renderProject();
    }
  }
}
async function fetchJSON(url, signal) {
  const response = await fetch(url, {
    signal,
    cache: "no-store",
    credentials: "same-origin",
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("监控服务返回了无效响应。");
  }
  if (!response.ok)
    throw new Error(asText(data.error, `请求失败 (${response.status})`));
  return data;
}
function schedule() {
  clearTimeout(state.timer);
  if (state.auto && !state.demo && !document.hidden)
    state.timer = setTimeout(refresh, 5000);
}
async function refresh() {
  clearTimeout(state.timer); state.controller?.abort();
  const epoch = ++state.epoch, controller = new AbortController(); state.controller = controller;
  const timeout = setTimeout(() => controller.abort(), 8000); $("refresh").disabled = true;
  try {
    if (state.demo) {
      const s = await fetchJSON("/demo.json", controller.signal);
      if (epoch !== state.epoch) return;
      const snapshots = [s, ...list(s.demo_runs)]; state.demoSnapshots = new Map(snapshots.map(r => [r.run_id, r]));
      chooseProjects([{ ...projectMeta(s.project), runs: snapshots }]);
      connection("示例数据", "demo"); notice("示例数据 · 静态演示，不代表真实运行。"); $("footer-source").textContent = "示例数据 · 静态快照";
    } else {
      const data = await fetchJSON("/api/projects", controller.signal);
      if (epoch !== state.epoch) return;
      if (!Array.isArray(data.projects)) throw new Error("项目清单格式无效。");
      chooseProjects(data.projects); connection("已连接", "online"); notice(""); $("footer-source").textContent = "本地监控 · 只读";
    }
    clearTimeout(timeout);
    const scope = state.activeScope;
    if (scope.runId) setSelectedRun(scope.runId, scope);
    if (scope.runId) await loadRun(scope.runId, scope, true);
    if (!state.projects.length) showEmpty("等待第一个项目", "尚无项目或运行快照；有数据后将自动显示。");
  } catch (error) {
    if (epoch !== state.epoch) return;
    connection("连接中断", "offline"); notice(`保留已读取的项目图与快照。${controller.signal.aborted ? "请求超时。" : asText(error.message, "无法连接监控服务。")}`, true);
    if (!selectedProject()) showEmpty("暂时无法读取项目", "请检查本地监控服务；恢复后将自动重新读取。");
  } finally {
    clearTimeout(timeout);
    if (epoch === state.epoch) { $("refresh").disabled = false; state.controller = null; schedule(); }
  }
}
function switchDemo() {
  if (state.activeScope) {
    state.activeScope.view = $("project-graph").graphView;
    state.activeScope.view?.observer?.disconnect();
    for (const entry of state.activeScope.cache.values()) abortEntry(entry);
    state.activeScope = null;
  }
  $("project-graph").graphView = null;

  if (!state.demo) state.realProjectId = state.projectId;
  state.realRunId = state.demo ? state.realRunId : state.runId;
  state.demo = !state.demo;
  state.runId = state.demo ? "" : state.realRunId;
  state.projectId = state.demo ? "" : state.realProjectId;
  state.projects = [];
  $("projects").hidden = true;
  clearRun();
  state.snapshot = null;
    $("export-report").disabled = true;
  state.openEvents.clear();
  state.openDecisions.clear();
  state.openJobs.clear();
  state.decisionLimit = 30;
  state.eventLimit = 30;
  $("demo").textContent = state.demo ? "返回真实运行" : "查看示例";
  $("demo").setAttribute("aria-pressed", String(state.demo));
  $("auto-refresh").checked = state.demo ? false : state.auto;
  $("auto-refresh").disabled = state.demo;
  $("job-filter").value = "all";
  $("candidate-filter").value = "all";
  $("event-filter").value = "all";
  $("event-search").value = "";
  $("decision-filter").value = "all";
  $("decision-search").value = "";
  $("dashboard").hidden = true;
  $("empty").hidden = false;
  refresh();
}
$("demo").addEventListener("click", switchDemo);
$("empty-demo").addEventListener("click", () => {
  if (!state.demo) switchDemo();
  else refresh();
});
$("refresh").addEventListener("click", refresh);
$("auto-refresh").addEventListener("change", () => {
  state.auto = $("auto-refresh").checked;
  schedule();
});
$("run-select").addEventListener("change", () => selectRun($("run-select").value));
$("project-select").addEventListener("change", () => { state.controller?.abort(); ++state.epoch; state.projectId = $("project-select").value; activateProject(); chooseProjects(state.projects); refresh(); });
$("project-overview").addEventListener("click", () => selectRun(""));
$("job-filter").addEventListener("change", renderJobs);
$("candidate-filter").addEventListener("change", renderCandidates);
for (const id of ["decision-search", "decision-filter"])
  $(id).addEventListener(id === "decision-search" ? "input" : "change", () => {
    state.decisionLimit = 30;
    renderDecisions();
  });
$("more-decisions").addEventListener("click", () => {
  state.decisionLimit += 30;
  renderDecisions();
});
for (const id of ["event-search", "event-filter"])
  $(id).addEventListener(id === "event-search" ? "input" : "change", () => {
    state.eventLimit = 30;
    renderEvents();
  });
$("more-events").addEventListener("click", () => {
  state.eventLimit += 30;
  renderEvents();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearTimeout(state.timer);
  else if (state.auto && !state.demo) refresh();
});
refresh();
