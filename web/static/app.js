"use strict";
const { t } = RSII18n;
const $ = (id) => document.getElementById(id);
const state = {
  demo: false, runId: "", realRunId: "", snapshot: null, controller: null,
  epoch: 0, timer: null, auto: true,
  queryApplied: false,
  projects: [], projectId: "", realProjectId: "", scopes: new Map(), activeScope: null, demoSnapshots: new Map(),
};
const ACTIVE = new Set(["running", "active", "training", "evaluating", "queued", "pending", "starting", "submitted", "in_progress"]);
const DONE = new Set(["completed", "complete", "succeeded", "success", "done", "finished"]);
const FAILED = new Set(["failed", "failure", "error", "cancelled", "canceled"]);
const ACCEPTED = new Set(["accepted", "applied", "adopted", "promoted", "bound"]);
const REJECTED = new Set(["rejected", "discarded", "failed"]);
const labels = {
  running: "运行中", active: "进行中", training: "训练中", evaluating: "评测中", queued: "排队中", pending: "等待中", starting: "启动中", submitted: "已提交", in_progress: "进行中",
  completed: "已完成", complete: "已完成", succeeded: "已完成", success: "已完成", done: "已完成", finished: "已完成",
  failed: "失败", failure: "失败", error: "错误", cancelled: "已取消", canceled: "已取消", idle: "空闲", waiting: "等待中", paused: "已暂停", stopped: "已停止",
  accepted: "已采纳", applied: "已采纳", adopted: "已采纳", promoted: "已采纳", bound: "已绑定", rejected: "未采纳", discarded: "已放弃", proposed: "已提出", ready: "待审阅", validated: "已验证", awaiting_evaluation: "待评测", awaiting_review: "待审阅", unavailable: "快照不可用", reviewed: "已复查", cancel_requested: "已请求取消", continue: "继续执行", cancel: "请求取消",
};
const actionLabels = {
  configure_prompt: "调整提示词", define_recipe: "定义训练方案", prepare_data: "准备训练数据", define_protocol: "定义评测方案",
  train: "训练模型", training: "训练模型", actor_training: "训练执行模型", decision_maker_training: "训练决策模型", sft: "监督微调", rl: "强化学习",
  evaluate: "验证改动效果", evaluation: "验证改动效果", candidate_evaluation: "验证改动效果", select: "选择版本", apply_candidate: "采用改动", execute: "执行任务",
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
const statusText = (v) => t(labels[statusKey(v)]) || t("状态未报告或未归类");
const actionText = (v) => t(actionLabels[asText(v, "")]) || asText(v, "").trim() || t("未命名决策");
const metricText = (v) => t(metricLabels[v]) || asText(v, t("其他指标"));
function el(tag, className = "", text) { const e = document.createElement(tag); if (className) e.className = className; if (text !== undefined) e.textContent = asText(text); return e; }
function badge(v) { const k = statusKey(v); return el("span", `badge ${FAILED.has(k) || REJECTED.has(k) ? "failure" : DONE.has(k) || ACCEPTED.has(k) ? "success" : ACTIVE.has(k) ? "active" : ""}`, statusText(v)); }
function empty(text) { return el("p", "subtle-empty", text); }
function dateText(v, withDate = true) { if (!v) return t("时间未报告"); const d = new Date(v); if (Number.isNaN(d.getTime())) return t("时间未报告"); return new Intl.DateTimeFormat(RSII18n.getLanguage() === "zh" ? "zh-CN" : "en-US", { ...(withDate ? { month: "2-digit", day: "2-digit" } : {}), hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(d); }
function secondsText(v) { if (!number(v) || v < 0) return t("未报告"); if (v > 0 && v < 1) return t("不足 1 秒"); const s = Math.floor(v); return [s >= 3600 ? t("{p0} 小时", {p0: Math.floor(s / 3600)}) : "", s >= 60 ? t("{p0} 分", {p0: Math.floor(s % 3600 / 60)}) : "", t("{p0} 秒", {p0: s % 60})].filter(Boolean).join(" "); }
function fmt(v) { if (!number(v)) return "—"; if (v !== 0 && (Math.abs(v) < .0001 || Math.abs(v) >= 1e9)) return v.toExponential(4); return new Intl.NumberFormat("en-US", { maximumFractionDigits: 6 }).format(v); }
const chrome = { notice: "", noticeError: false,
  emptyTitle: () => t("等待第一次运行"), emptyDescription: () => t("尚无运行快照。") };
const copy = value => typeof value === "function" ? value() : value;
function uiError(key, params = {}) { return Object.assign(new Error(t(key, params)), { uiKey: key, uiParams: params }); }
function errorText(error, fallback = "无法连接服务。") { return error?.uiKey ? t(error.uiKey, error.uiParams) : asText(error?.message || error?.error, t(fallback)); }
function notice(text, error = false) { chrome.notice=text;chrome.noticeError=error;const value=copy(text);$("notice").textContent = value; $("notice").hidden = !value; $("notice").className = `notice${error ? " error" : ""}`; }
function showEmpty(title, text) { chrome.emptyTitle=title;chrome.emptyDescription=text; $("empty").hidden = false; $("empty-title").textContent = copy(title); $("empty-description").textContent = copy(text); }
// Keep reported prose verbatim. Identifiers and long text live in disclosures.
function retainedDetails(key, store, title, content) {
  const d = el("details", "record-details"); d.dataset.retainKey=key; d.open = store.has(key); d.append(el("summary", "", title), content);
  d.addEventListener("toggle", () => { if (d.isConnected) { if (d.open) store.add(key); else store.delete(key); } }); return d;
}
function flattenMetrics(source, prefix = "", output = {}, depth = 0) {
  if (depth > 5) return output;
  for (const [k, v] of Object.entries(object(source))) { const key = prefix ? `${prefix}.${k}` : k; if (number(v)) output[key] = v; else if (!Array.isArray(v)) flattenMetrics(v, key, output, depth + 1); } return output;
}
function metricStrip(metrics, raw = false) {
  const dl = el("dl", "metric-strip");
  for (const [key, value] of Object.entries(flattenMetrics(metrics))) { const item = el("div", "metric-chip"); const label=el("dt", "", metricText(key));label.title=key;item.append(label, el("dd", "", fmt(value))); dl.append(item); } return dl;
}
function metricsDetail(metrics) { const m = metricStrip(metrics, true); return m.childElementCount ? m : empty(t("尚无指标。")); }
function progressView(step, total) {
  const wrap = el("div", "step-progress");
  if (Number.isInteger(step) && step >= 0 && Number.isInteger(total) && total > 0 && step <= total) {
    const label = el("div", "progress-label"); label.append(el("span", "", t("{p0} / {p1} 步", {p0: step, p1: total})), el("span", "", `${Number((step / total * 100).toFixed(1))}%`));
    const bar = el("progress"); bar.max = total; bar.value = step; bar.setAttribute("aria-label", t("任务步骤进度")); wrap.append(label, bar);
  } else wrap.append(el("p", "progress-unknown", Number.isInteger(step) && step >= 0 ? t("已报告 {p0} 步；总步数未报告或待核对", {p0: step}) : t("步骤进度未报告")));
  return wrap;
}
function decisionAction(row) { if (row.kind === "review") return row.action || row.mode; if (row.kind === "operation") return row.operation || row.action || row.mode; return row.mode || row.action || (row.kind === "finish" ? "finish" : undefined); }
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
    const rows = JSON.parse(localStorage.getItem(`rsisys-runs-v5:${key}`) || "[]");
    return new Map((Array.isArray(rows) ? rows : []).slice(-LAYOUT.cache).filter(r => Array.isArray(r) && typeof r[0] === "string" && number(r[1]?.x) && number(r[1]?.y) && r[1].x >= 36 && r[1].y >= 36 && r[1].x <= 100000 && r[1].y <= 100000));
  } catch { return new Map(); }
}
function savePositions(scope) {
  try { localStorage.setItem(`rsisys-runs-v5:${scope.key}`, JSON.stringify([...scope.positions].slice(-LAYOUT.cache))); } catch { /* Storage can be disabled or full. In-memory positions still work. */ }
}
function networkLayout(model, positions = new Map(), { relax = false, pinned = new Set(), steps = LAYOUT.iterations } = {}) {
  const ordered = model.nodes.slice().sort((a,b) => a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
  const byKey = new Map(ordered.map(n => [n.key,n])), anchors = ordered.filter(n => n.type === "run"), fresh = new Set();
  const center = { x: 70, y: 245 };
  anchors.sort((a,b)=>(a.layoutOrder??0)-(b.layoutOrder??0)||(a.key<b.key?-1:a.key>b.key?1:0));
  const targets = new Map(anchors.map((n,i)=>[n.key,{x:70+(n.depth??i+1)*180,y:80+(n.treeRow??i)*110}]));
  for(const n of ordered.filter(n=>n.type==="project"))targets.set(n.key,center);
  const initialize = (n, origin, radius, angle) => {
    const cached = positions.get(n.key);
    if (cached && number(cached.x) && number(cached.y)) { n.x = cached.x; n.y = cached.y; if (relax && !pinned.has(n.key)) fresh.add(n.key); }
    else { n.x = Math.max(44, origin.x + radius * Math.cos(angle)); n.y = Math.max(44, origin.y + radius * Math.sin(angle)); fresh.add(n.key); }
  };
  for (const n of ordered.filter(n => n.type === "project")) { n.x=center.x; n.y=center.y; }
  anchors.forEach(n => initialize(n, targets.get(n.key), 0, 0));
  const counts = new Map();
  for (const n of ordered.filter(n => !["project","run"].includes(n.type))) {
    const origin = byKey.get(JSON.stringify(["run",n.runId])) || center;
    const i = counts.get(n.runId) || 0; counts.set(n.runId,i+1);
    initialize(n, origin, 62 + 23 * Math.sqrt(i), i * 2.399963 + hashUnit(n.key) * 1.8);
  }
  const edgeOrder = e => JSON.stringify([e.from,e.to,e.kind]);
  const edges = model.edges.slice().sort((a,b) => edgeOrder(a) < edgeOrder(b) ? -1 : edgeOrder(a) > edgeOrder(b) ? 1 : 0).map(e => ({ ...e, a: byKey.get(e.from), b: byKey.get(e.to) })).filter(e => e.a && e.b);
  const iterations = fresh.size ? Math.min(LAYOUT.iterations, Math.max(1, steps), Math.max(1, Math.floor(LAYOUT.budget / Math.max(1, ordered.length + edges.length)))) : 0;
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
      const origin = targets.get(n.key) || center;
      f.x += (origin.x-n.x)*.12; f.y += (origin.y-n.y)*.06;
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
      const target=targets.get(n.key);if(target)n.x=Math.max(target.x-25,Math.min(target.x+25,n.x));
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
  const pagePosition=typeof window!=="undefined"?{left:window.scrollX,top:window.scrollY}:null;
  const previous = host.graphView, same = previous?.key === key;
  previous?.observer?.disconnect();
  previous?.stopSimulation?.();
  const view = same ? previous : { key, autoFit: true, scale: 1, selected: model.nodes.find(n => n.selected)?.key || "project-root", left: 0, top: 0 };
  if (same) { view.left = previous.viewport.scrollLeft; view.top = previous.viewport.scrollTop; }
  const focusedKey = host.contains(document.activeElement) ? document.activeElement?.dataset.nodeKey : null;
  const controls = el("div","graph-controls"), zoomLabel = el("button","zoom-reset"), viewport = el("div","graph-viewport"), extent = el("div","graph-extent"), plane = el("div","graph-plane");
  viewport.tabIndex = 0; viewport.setAttribute("role","region"); viewport.setAttribute("aria-label",t("关系图谱，拖拽节点或空白平移，Ctrl 或 Command 加滚轮缩放"));
  const bounds = () => ({ left: Math.min(...model.nodes.map(n=>n.x))-65, top: Math.min(...model.nodes.map(n=>n.y))-45,
    right: Math.max(...model.nodes.map(n=>n.x))+65, bottom: Math.max(...model.nodes.map(n=>n.y))+65 });
  let frame = same && !view.autoFit && view.frame ? {...view.frame} : bounds();
  let width = frame.right-frame.left, height = frame.bottom-frame.top, offsetX=0, offsetY=0, pan=null;
  view.panX ||= 0;view.panY ||= 0;
  const lines = svgEl("svg",{ width,height,"aria-hidden":"true" });
  const marker = svgEl("marker",{ id:`${host.id}-arrow`,viewBox:"0 0 8 8",refX:18,refY:4,markerWidth:5,markerHeight:5,orient:"auto" });
  marker.append(svgEl("path",{ d:"M0,0 L8,4 L0,8",fill:"context-stroke" })); const defs = svgEl("defs"); defs.append(marker); lines.append(defs);
  const byKey = new Map(model.nodes.map(n => [n.key,n])), paths = [], incident = new Map(), buttons = new Map();
  const draw = ({ edge,path }) => { const a=byKey.get(edge.from),b=byKey.get(edge.to); path.setAttribute("d",`M${a.x},${a.y} L${b.x},${b.y}`); };
  for (const edge of model.edges) {
    if (!byKey.has(edge.from) || !byKey.has(edge.to)) continue;
    const path = svgEl("path",{ class:`graph-edge ${edge.kind === "time" || edge.kind === "run-source" ? "temporal" : edge.kind === "membership" ? "membership" : "linked"}` });
    if (["explicit","run-parent","directive"].includes(edge.kind)) path.setAttribute("marker-end",`url(#${host.id}-arrow)`);
    path.append(svgEl("title",{},(edge.evidence || [edge.kind==="run-parent"?t("运行续接"):edge.kind==="run-source"?t("资料引用"):t("项目归属")]).join(t("、")))); const item={edge,path}; paths.push(item); draw(item); lines.append(path);
    for (const k of [edge.from,edge.to]) { const entries=incident.get(k)||[]; entries.push(item); incident.set(k,entries); }
  }
  plane.append(lines);
  const details = detailFor ? (same && previous.details || el("section","graph-detail")) : null;
  view.details=details;
  if (details) details.setAttribute("aria-label",t("所选节点的执行流程与结果"));
  const highlight = key => {
    const neighbors=graphNeighborhood(model,key);
    for (const [id,button] of buttons) button.setAttribute("data-related",neighbors.has(id)?"true":"false");
    for (const {edge,path} of paths) path.setAttribute("data-related",edge.from===key||edge.to===key?"true":"false");
  };
  const select = (node, notify=true) => {
    view.selected=node.key;
    for (const [id,button] of buttons) button.setAttribute("aria-pressed",String(id===node.key));
    if (details) {
      const entry=state.activeScope?.cache.get(node.runId);
      const fingerprint=JSON.stringify([node.key,RSII18n.getLanguage(),node.row,node.missing,node.ambiguous,node.ancestryStatus,entry?.snapshot,entry?.status==="pending"&&entry?.snapshot?"ready":entry?.status,entry?.error||""]);
      if(view.detailFingerprint!==fingerprint) {
        // Store native disclosure state synchronously before replacing the DOM.
        for(const d of details.querySelectorAll?.("details[data-retain-key]")||[]) {
          const store=state.activeScope?.flowOpen;if(store){if(d.open)store.add(d.dataset.retainKey);else store.delete(d.dataset.retainKey);}
        }
        const scroll=view.detailNodeKey===node.key?(details.querySelector?.(".flow-map")?.scrollLeft||0):0;
        details.replaceChildren(detailFor(node));
        const map=details.querySelector?.(".flow-map");if(map)map.scrollLeft=scroll;
        view.detailFingerprint=fingerprint;view.detailNodeKey=node.key;
      }
    } highlight(node.key);
    if (notify) onSelect?.(node);
  };
  const place = node => { const button=buttons.get(node.key); button.style.left=`${node.x}px`; button.style.top=`${node.y}px`; };
  let nodeDrag=null, suppressClick="", animation=null, heat=0;
  const positions = () => new Map(model.nodes.map(n=>[n.key,{x:n.x,y:n.y}]));
  const cachePositions = () => {
    const scope=state.activeScope;if(scope?.key!==key)return;
    for(const n of model.nodes)scope.positions.set(n.key,{x:n.x,y:n.y});
    savePositions(scope);
  };
  view.stopSimulation=()=>{
    const active=view.simulating || animation!==null;
    if(animation!==null && typeof cancelAnimationFrame!=="undefined")cancelAnimationFrame(animation);
    animation=null;heat=0;view.simulating=false;view.dragging=false;nodeDrag=null;if(active)cachePositions();
  };
  const simulate = () => {
    const scope=state.activeScope;
    const pinned=new Set(nodeDrag?.moved?[nodeDrag.node.key]:(scope?.key===key?scope.pinned:[]));
    networkLayout(model,positions(),{relax:true,pinned,steps:2});
    for(const n of model.nodes)place(n);
    for(const item of paths)draw(item);
    // Expand the canvas without changing its camera during dragging/settling.
    frame.left=Math.min(frame.left,...model.nodes.map(n=>n.x-65));
    frame.top=Math.min(frame.top,...model.nodes.map(n=>n.y-45));
    frame.right=Math.max(frame.right,...model.nodes.map(n=>n.x+65));
    frame.bottom=Math.max(frame.bottom,...model.nodes.map(n=>n.y+65));
    width=frame.right-frame.left;height=frame.bottom-frame.top;resize();
    if(scope?.key===key)for(const n of model.nodes)scope.positions.set(n.key,{x:n.x,y:n.y});
  };
  const tick = () => {
    animation=null;
    if(!viewport.isConnected || document.hidden){view.stopSimulation();return;}
    simulate();heat--;
    if(heat>0)animation=requestAnimationFrame(tick);
    else {view.simulating=false;cachePositions();if(view.pendingRender&&!view.dragging){view.pendingRender=false;renderProject();}}
  };
  const reheat = (frames=40) => {
    if(typeof requestAnimationFrame==="undefined"){simulate();cachePositions();return;}
    heat=frames;view.simulating=true;
    if(animation===null)animation=requestAnimationFrame(tick);
  };
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
      if(e.button!==0)return; e.stopPropagation(); if(node.type==="project")return; suppressClick="";
      nodeDrag={node,x:e.clientX,y:e.clientY,left:node.x,top:node.y,moved:false}; view.dragging=true; button.setPointerCapture(e.pointerId);
    });
    button.addEventListener("pointermove",e=>{
      if(nodeDrag?.node!==node)return;
      const dx=(e.clientX-nodeDrag.x)/view.scale,dy=(e.clientY-nodeDrag.y)/view.scale;
      if(Math.hypot(e.clientX-nodeDrag.x,e.clientY-nodeDrag.y)>4)nodeDrag.moved=true;
      if(!nodeDrag.moved)return;
      const scope=state.activeScope;if(scope?.key===key)scope.pinned.clear();
      node.x=Math.max(36,nodeDrag.left+dx); node.y=Math.max(36,nodeDrag.top+dy);
      const root=model.nodes.find(n=>n.type==="project");
      if(node.type==="project")node.x=Math.min(node.x,Math.min(...model.nodes.filter(n=>n.type==="run").map(n=>n.x))-90);
      else if(root)node.x=Math.max(root.x+90,node.x);
      place(node);
      for(const item of incident.get(node.key)||[])draw(item);
      frame.left=Math.min(frame.left,node.x-65);frame.top=Math.min(frame.top,node.y-45);
      frame.right=Math.max(frame.right,node.x+65);frame.bottom=Math.max(frame.bottom,node.y+65);
      width=frame.right-frame.left;height=frame.bottom-frame.top;resize();reheat();
    });
    const finish = () => {
      if(nodeDrag?.node!==node)return;
      const moved=nodeDrag.moved;
      if(moved){
        view.autoFit=false;view.frame={...frame};suppressClick=node.key;
        const scope=state.activeScope;if(scope?.key===key){scope.pinned.clear();scope.pinned.add(node.key);cachePositions();}
      }
      nodeDrag=null;view.dragging=false;
      if(moved)reheat(18);
      if(view.pendingRender&&!view.simulating){view.pendingRender=false;renderProject();}
    };
    button.addEventListener("pointerup",finish);button.addEventListener("pointercancel",finish);button.addEventListener("lostpointercapture",finish);
    buttons.set(node.key,button);place(node);plane.append(button);
  }
  const resize=()=>{
    if((!view.dragging&&!view.simulating)||pan){
      offsetX=Math.max(0,(viewport.clientWidth-width*view.scale)/2)-frame.left*view.scale+view.panX;
      offsetY=Math.max(0,(viewport.clientHeight-height*view.scale)/2)-frame.top*view.scale+view.panY;
    }
    plane.style.width=`${frame.right}px`;plane.style.height=`${frame.bottom}px`;
    plane.style.transform=`translate(${offsetX}px,${offsetY}px) scale(${view.scale})`;
    lines.setAttribute("width",frame.right);lines.setAttribute("height",frame.bottom);
    extent.style.width=`${Math.max(viewport.clientWidth,frame.right*view.scale+offsetX)}px`;
    extent.style.height=`${Math.max(viewport.clientHeight,frame.bottom*view.scale+offsetY)}px`;
    zoomLabel.textContent=`${Math.round(view.scale*100)}%`;view.frame={...frame};
  };
  const scaleTo=(scale,centered=true)=>{
    const cx=(viewport.scrollLeft+viewport.clientWidth/2-offsetX)/view.scale,cy=(viewport.scrollTop+viewport.clientHeight/2-offsetY)/view.scale;
    view.autoFit=false;view.scale=Math.max(.02,Math.min(3,scale));resize();
    if(centered){viewport.scrollLeft=cx*view.scale+offsetX-viewport.clientWidth/2;viewport.scrollTop=cy*view.scale+offsetY-viewport.clientHeight/2;}
  };
  const fit=()=>{
    view.panX=0;view.panY=0;frame=bounds();width=frame.right-frame.left;height=frame.bottom-frame.top;
    scaleTo(Math.max(.85,Math.min((viewport.clientWidth-20)/width,(viewport.clientHeight-20)/height,1.25)),false);
    view.autoFit=true;viewport.scrollLeft=0;viewport.scrollTop=0;
  };
  const zoomGroup=el("div","graph-control-group"),layoutGroup=el("div","graph-control-group");
  zoomGroup.setAttribute("role","group");zoomGroup.setAttribute("aria-label",t("缩放"));
  layoutGroup.setAttribute("role","group");layoutGroup.setAttribute("aria-label",t("布局"));
  zoomLabel.type="button";zoomLabel.setAttribute("aria-label",t("恢复原始大小"));zoomLabel.title=t("恢复原始大小");zoomLabel.addEventListener("click",()=>scaleTo(1));
  for(const [label,title,action,group] of [["−",t("缩小"),()=>scaleTo(view.scale/1.25),zoomGroup],["+",t("放大"),()=>scaleTo(view.scale*1.25),zoomGroup],[t("适应"),t("适应全部节点"),fit,layoutGroup],[t("重排"),t("自动调整节点布局"),()=>{ const scope=state.activeScope;if(scope?.key!==key)return;scope.pinned.clear();scope.needsRelax=true;view.autoFit=true;renderProject(); },layoutGroup]]){
    const button=el("button","",label);button.type="button";button.title=title;button.setAttribute("aria-label",title);button.addEventListener("click",action);group.append(button);
    if(label==="−")group.append(zoomLabel);
  }
  controls.append(zoomGroup,layoutGroup);
  viewport.addEventListener("wheel",e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();scaleTo(view.scale*(e.deltaY<0?1.1:1/1.1));}},{passive:false});
  viewport.addEventListener("pointerdown",e=>{
    if(e.button!==0||e.target.closest("button"))return;
    view.autoFit=false;view.panX-=viewport.scrollLeft;view.panY-=viewport.scrollTop;viewport.scrollLeft=0;viewport.scrollTop=0;pan={x:e.clientX,y:e.clientY,left:view.panX,top:view.panY};view.dragging=true;viewport.setPointerCapture(e.pointerId);viewport.classList.add("panning");resize();
  });
  viewport.addEventListener("pointermove",e=>{if(pan){view.panX=pan.left+e.clientX-pan.x;view.panY=pan.top+e.clientY-pan.y;resize();}});
  const endPan=()=>{if(!pan)return;pan=null;view.dragging=false;viewport.classList.remove("panning");if(view.pendingRender){view.pendingRender=false;renderProject();}};
  viewport.addEventListener("pointerup",endPan);viewport.addEventListener("pointercancel",endPan);viewport.addEventListener("lostpointercapture",endPan);
  viewport.addEventListener("keydown",e=>{
    if(!e.target.dataset.nodeKey||!["ArrowRight","ArrowLeft","Home","End"].includes(e.key))return;
    e.preventDefault();const keys=[...buttons.keys()],i=keys.indexOf(e.target.dataset.nodeKey),next=e.key==="Home"?0:e.key==="End"?keys.length-1:Math.max(0,Math.min(keys.length-1,i+(e.key==="ArrowRight"?1:-1)));
    buttons.get(keys[next]).focus();
  });
  extent.append(plane);viewport.append(extent);
  const layout=same&&previous.layout?previous.layout:el("div","graph-layout");
  if(same&&previous.layout)layout.replaceChild(viewport,previous.viewport);
  else {layout.append(viewport);if(details)layout.append(details);host.replaceChildren(layout);}
  view.layout=layout;
  $("graph-tools").replaceChildren(controls);view.viewport=viewport;host.graphView=view;resize();
  if(!same || (view.autoFit&&model.topologyChanged))fit();else{viewport.scrollLeft=view.left;viewport.scrollTop=view.top;}
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
  if(same&&pagePosition&&(window.scrollX!==pagePosition.left||window.scrollY!==pagePosition.top))window.scrollTo({...pagePosition,behavior:"instant"});
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
  if(job)results.push({type:"job",row:job,evidence:present(row.job_id)?t("明确任务 ID 关联；结果为当前快照。"):t("唯一指令 ID 关联；结果为当前快照。")});
  const candidates=list(s.candidates),matches=present(row.candidate_id)?candidates.filter(c=>sameId(c.candidate_id,row.candidate_id)):[];
  if(matches.length===1)results.push({type:"candidate",row:matches[0],evidence:t("明确候选 ID 关联；结果为当前快照。")});
  // Original directives can report multiple resulting candidates; reviews never
  // become origins and repeated directive IDs cannot claim each other's results.
  if(!present(row.candidate_id)&&row.kind==="directive"&&identifier(row.directive_id)&&list(s.decisions).filter(d=>d.kind==="directive"&&sameId(d.directive_id,row.directive_id)).length===1){
    for(const c of candidates.filter(c=>sameId(c.directive_id,row.directive_id)))results.push({type:"candidate",row:c,evidence:t("唯一原始指令关联；结果为当前快照。")});
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
  if (row.kind === "review") text = row.action === "cancel" ? t("已请求取消；任务是否停止以任务回报为准。") : row.action === "continue" ? t("复查决定继续执行。") : t("复查结果未报告。");
  else if (decisionAction(row) === "finish") text = t("模型提出结束本轮研究。");
  else if (FAILED.has(statusKey(status))) text = t("执行失败或已取消。");
  else if (DONE.has(statusKey(status))) text = t("执行已完成。");
  else if (ACTIVE.has(statusKey(status))) text = t("执行中，等待最终反馈。");
  else if (status === "proposed") text = t("已提出，尚未报告执行结果。");
  else text = t("尚未报告执行反馈。");
  if (candidate && row.kind !== "review" && decisionAction(row) !== "train") text += t(" 候选{p0}（当前快照）。", {p0: ACCEPTED.has(statusKey(candidate.status)) ? t("已采纳") : REJECTED.has(statusKey(candidate.status)) ? t("未采纳") : t("状态尚未确定")});
  const explicit = feedback.summary || row.feedback_summary;
  if (explicit) text = t("已报告执行反馈，见详情。");
  return { text, metrics, job, status, source: row.kind === "review" ? t("复查记录") : t("当前公开快照") };
}
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
  heading.append(el("h4", "", title), el("span", "chart-value", number(latest) ? fmt(latest) : t("未报告"))); figure.append(heading);
  if (!points.length) { figure.append(el("div", "chart-empty", number(latest) ? t("仅有当前指标，未记录可绘图的历史。") : t("尚未报告该指标的历史。"))); return figure; }
  const width = 480, height = 205, left = 82, right = 18, top = 18, bottom = 34;
  const ys = points.map(p=>p.y), xs = points.map(p=>p.x), ymin = Math.min(...ys), ymax = Math.max(...ys), xmin = Math.min(...xs), xmax = Math.max(...xs);
  // Scale before subtraction to keep even extreme finite inputs inside the SVG.
  const yunit = Math.max(Math.abs(ymin),Math.abs(ymax)) || 1, xunit = Math.max(Math.abs(xmin),Math.abs(xmax)) || 1;
  const ylo = ymin / yunit, yhi = ymax / yunit, yspan = yhi - ylo, ypad = yspan ? yspan * .12 : .1;
  const py = v => top + (yhi + ypad - v / yunit) / (yspan + 2 * ypad) * (height - top - bottom);
  const px = v => xmax === xmin ? (left + width - right) / 2 : left + ((v / xunit - xmin / xunit) / (xmax / xunit - xmin / xunit)) * (width - left - right);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", tabindex: "0", "aria-label": t("{p0}，{p1} 个历史点；左右方向键逐点查看，Home 和 End 查看首尾。", {p0: title, p1: points.length}) });
  svg.append(svgEl("title", {}, `${title} · ${series.axis === "step" ? t("步数") : t("耗时（秒）")}`));
  const ticks = ymin === ymax ? [ymin] : [ymax, ymin / 2 + ymax / 2, ymin];
  for (const v of ticks) { svg.append(svgEl("line", { x1: left, x2: width-right, y1: py(v), y2: py(v), class: "chart-grid" }), svgEl("text", { x: left - 8, y: py(v)+4, "text-anchor": "end", class: "chart-axis" }, fmt(v))); }
  const xLabel = series.axis === "step" ? t("步") : t("秒");
  svg.append(svgEl("text", { x: left, y: height-9, class: "chart-axis" }, `${fmt(xmin)} ${xLabel}`));
  if (xmin !== xmax) svg.append(svgEl("text", { x: width-right, y: height-9, "text-anchor": "end", class: "chart-axis" }, `${fmt(xmax)} ${xLabel}`));
  let path = "", connected = false;
  for (const p of series.rows) { if (!number(p.y)) { connected = false; continue; } path += `${connected ? "L" : "M"}${px(p.x).toFixed(2)},${py(p.y).toFixed(2)} `; connected = true; }
  svg.append(svgEl("path", { d: path, class: "chart-line" }));
  // Keep an honest single dot for one-point history. The inspection marker follows real samples.
  const marker = svgEl("circle", { cx: px(points[points.length-1].x), cy: py(points[points.length-1].y), r: 4, class: "chart-point" }); svg.append(marker);
  const readout = el("p", "chart-readout"); let selected = points.length-1;
  const inspect = index => { selected = Math.max(0,Math.min(points.length-1,index)); const p = points[selected]; marker.setAttribute("cx",px(p.x)); marker.setAttribute("cy",py(p.y)); readout.textContent = `${number(p.step) ? t("第 {p0} 步", {p0: p.step}) : t("{p0} 秒", {p0: fmt(p.x)})} · ${title} ${fmt(p.y)}${number(p.elapsed) ? t(" · 耗时 {p0}", {p0: secondsText(p.elapsed)}) : ""}${points.length === 1 ? t(" · 仅 1 个历史点") : ""}`; };
  inspect(selected);
  svg.addEventListener("pointermove", event => { const bounds = svg.getBoundingClientRect(); const x = (event.clientX - bounds.left) / bounds.width * width; let index = 0; for (let i=1; i<points.length; i++) if (Math.abs(px(points[i].x)-x) < Math.abs(px(points[index].x)-x)) index=i; inspect(index); });
  svg.addEventListener("keydown", event => { const actions = { ArrowLeft: selected-1, ArrowRight: selected+1, Home: 0, End: points.length-1 }; if (Object.hasOwn(actions,event.key)) { event.preventDefault(); inspect(actions[event.key]); } });
  figure.append(svg,readout); return figure;
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
  const wrap = el("div"), { pairs, extra } = metricRows(metrics);
  if (!pairs.length) { const strip=metricStrip(metrics); wrap.append(strip.childElementCount ? strip : empty(t("评测指标未报告。"))); return wrap; }
  const table=el("table","metric-table"), head=el("thead"), hr=el("tr"), body=el("tbody");
  for (const name of [t("指标"),t("基线"),t("新版本"),t("新版本 − 基线")]) {const th=el("th","",name);th.scope="col";hr.append(th);}head.append(hr);
  for (const m of pairs) { const row=el("tr"), delta=number(m.base)&&number(m.next)?m.next-m.base:undefined;row.append(el("td","",metricText(m.name)),el("td","",fmt(m.base)),el("td","",fmt(m.next)),el("td","",number(delta)?`${delta>0?"+":""}${fmt(delta)}`:"—"));body.append(row); }table.append(head,body);wrap.append(table);if(Object.keys(extra).length)wrap.append(metricStrip(extra));return wrap;
}
function renderSnapshot(s) { state.snapshot=s; $("empty").hidden=true;  }
function clearRun() {
  state.snapshot = null;


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
const projectName = p => p?.project_id === "unassigned" ? t("未归属") : asText(p?.name);
function selectedProject() { return state.projects.find(p => p.project_id === state.projectId); }
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
    state.activeScope.view?.stopSimulation?.();
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
  const runs = runRows(project), nodes = [{ key: "project-root", type: "project", row: project, title: `0 · ${projectName(project)}`, label: t("项目"), x: 20, y: 90 }], edges = [];
  runs.forEach((row, i) => {
    const key=JSON.stringify(["run",row.run_id]),entry=scope.cache.get(row.run_id);
    nodes.push({key,type:"run",projectId:project.project_id,row:entry?.snapshot||row,runId:row.run_id,title:t("研究轮次 {p0}",{p0:i+1}),label:t("研究轮次"),summary:statusText(row.status),layoutOrder:i,loadStatus:entry?.status==="pending"&&entry.snapshot?"ready":entry?.status,missing:[],ambiguous:[]});
  });
  const topology=RSILineage.build(nodes,edges);
  const signature = JSON.stringify([topology.nodes.map(n => [n.key,n.depth,n.treeRow]).sort(), topology.edges.map(e => [e.from,e.to,e.kind]).sort()]);
  const changed = scope.layoutSignature !== undefined && scope.layoutSignature !== signature;
  const model = networkLayout(topology, scope.positions, { relax: changed || !!scope.needsRelax, pinned: scope.pinned });
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
const changeLabels = {instructions:"提示词内容",method:"训练方法",recipe_ref:"训练方案",protocol_ref:"评测方案",candidate_id:"待验证版本",apply_result:"训练后自动应用",steps:"训练步数",lr:"学习率",seed:"随机种子",weight_decay:"权重衰减",warmup_ratio:"预热比例",updates:"更新次数",components:"改动部分",query:"检索内容",url:"阅读地址",artifact_id:"资料引用",offset:"起始位置",limit:"数量"};
function changesView(row,s) {
  const changes=list(row.changes).filter(c=>typeof c.field==="string"&&c.field!=="base_version"&&(typeof c.after==="string"||typeof c.before==="string"));
  if(!changes.length)return null;
  const box=el("div","decision-changes");box.append(el("p","flow-label",t("提交内容")));
  for(const change of changes){
    const item=el("div","change-item");item.append(el("h5","",t(changeLabels[change.field])||change.field));
    const value=text=>{
      if(change.field==="candidate_id"){
        const candidates=list(s.candidates),matches=candidates.filter(c=>c.candidate_id===text);
        return matches.length===1?t("版本 {p0}",{p0:candidates.indexOf(matches[0])+1}):t("版本关联未记录");
      }
      return text===""?t("（清空）"):text;
    };
    if(typeof change.before==="string")item.append(el("p","change-before",t("原值：{p0}",{p0:value(change.before)})));
    if(typeof change.after==="string")item.append(el("p","change-value",value(change.after)));
    box.append(item);
  }
  return box;
}
function concreteFeedback(row) {
  const summary=object(row.feedback).summary||row.feedback_summary;
  if(typeof summary!=="string")return "";
  return /^(?:执行已完成[。.]?|执行完成[。.]?|completed[.!]?|done[.!]?|success[.!]?|configure_prompt completed)$/i.test(summary.trim())?"":summary;
}
function executionSequence(s, scope) {
  scope.flowLimits ||= new Map(); scope.flowOpen ||= new Set(); scope.flowSelected ||= new Map();
  const section=el("section","execution-sequence"),groups=executionGroups(s);
  section.id="execution-flow";

  const navigation=el("ol","flow-map");navigation.hidden=groups.length<2;navigation.setAttribute("aria-label",t("本轮研究的决策"));section.append(navigation);
  const entries=el("ol","flow-steps");section.append(entries);
  const fill=(restoreFocus=false)=>{
    let selectedButton;
    entries.replaceChildren();navigation.replaceChildren();
    const limit=scope.flowLimits.get(s.run_id)||30;
    const keyOf=g=>decisionKey(g.rows[g.rows.length-1]);
    let selected=scope.flowSelected.get(s.run_id);
    if(!groups.slice(0,limit).some((g,i)=>keyOf(g,i)===selected))selected=groups.length?keyOf(groups[0],0):null;
    scope.flowSelected.set(s.run_id,selected);
    for(const [i,g] of groups.slice(0,limit).entries()){
      const row=g.row,f=feedbackFor(row,s),key=keyOf(g,i),item=el("li","flow-map-item"),button=el("button","flow-choice");
      button.type="button";button.setAttribute("aria-pressed",String(key===selected));
      button.setAttribute("aria-label",t("步骤 {p0} · {p1}", {p0: i+1, p1: actionText(decisionAction(row))}));
      button.append(el("span","flow-number",String(i+1).padStart(2,"0")),el("span","",actionText(decisionAction(row))));
      if(g.rows.length>1)button.append(el("span","quiet-note",`×${g.rows.length}`));
      button.addEventListener("click",()=>{scope.flowSelected.set(s.run_id,key);fill(true);});item.append(button);if(groups.length>1)navigation.append(item);
      if(key!==selected)continue;
      selectedButton=button;
      const step=el("li","flow-step"),index=el("span","flow-index",String(i+1).padStart(2,"0"));
      const decision=el("div","flow-decision"),feedback=el("div","flow-feedback"),metrics=el("div","flow-metrics");
      const meta=el("p","quiet-note",`${dateText(row.timestamp)}${number(row.round)?t(" · 第 {p0} 轮", {p0: row.round}):""}`);
      decision.append(el("h4","",actionText(decisionAction(row))),meta);
      if(g.rows.length>1)decision.append(el("p","repeat-count",t("重复报告 {p0} 次", {p0: g.rows.length})));
      if(present(row.reason))decision.append(el("p","flow-label",t("决策理由")),el("p","decision-reason",row.reason));
      if(present(row.strategy)&&row.strategy!==row.reason&&row.strategy!==decisionAction(row))decision.append(el("p","decision-reason",row.strategy));
      const changes=changesView(row,s);if(changes)decision.append(changes);
      if(["evaluate","evaluation","candidate_evaluation"].includes(decisionAction(row))){
        const component=list(row.changes).find(c=>c.field==="components")?.after;
        const prompt=typeof component==="string"&&/prompt|instructions/.test(component);
        decision.append(el("p","evaluation-purpose",prompt?t("检验新提示词的效果；评测通过与实际采用是两个独立步骤。"):t("检验待验证版本的效果；评测通过与实际采用是两个独立步骤。")));
        if(!row.protocol_ref&&!list(row.changes).some(c=>c.field==="protocol_ref"))decision.append(el("p","quiet-note",t("评测方案未记录。")));
      }
      const explicit=concreteFeedback(row),error=object(row.feedback).error||row.error;
      if(explicit||error){feedback.append(el("p","flow-label",t("实际结果")));if(explicit)feedback.append(el("p","result-summary",explicit));if(error&&error!==explicit)feedback.append(el("p","result-error",error));}
      else if(!DONE.has(statusKey(f.status))||row.kind==="review"||decisionAction(row)==="finish")feedback.append(el("p","",f.text));
      if(row.kind==="review"&&f.job)feedback.append(el("p","quiet-note",t("关联任务当前{p0}；不代表复查当时状态。", {p0: statusText(f.job.status)})));
      if(row.kind!=="review"&&f.job&&present(f.job.step))feedback.append(progressView(f.job.step,f.job.total_steps));
      if(Object.keys(flattenMetrics(f.metrics)).length)metrics.append(el("p","flow-label",row.kind==="review"?t("复查时指标"):t("指标")),comparisonView(f.metrics));
      if(f.job&&number(f.job.elapsed_seconds))feedback.append(el("p","quiet-note",t("耗时 {p0}",{p0:secondsText(f.job.elapsed_seconds)})));
      step.append(index,decision);if(feedback.childElementCount)step.append(feedback);if(metrics.childElementCount)step.append(metrics);
      const results=el("div","flow-results");
      if(f.job){
        const j=f.job,h=historyRows(j),isTraining=/train|sft|finetun/.test(asText(j.kind,""))||h.some(p=>number(metricValue(p.metrics,lossKeys)));
        if(row.kind==="review")results.append(el("p","quiet-note",t("关联任务当前快照")));
        if(isTraining){const charts=el("div","charts");charts.append(chartView(j,lossKeys,t("损失")),chartView(j,lrKeys,t("学习率")));results.append(charts);}

      }
      for(const result of relatedResults(row,s).filter(r=>r.type==="candidate")){
        const reported=flattenMetrics(result.row.metrics),shown=flattenMetrics(f.metrics);
        const alreadyShown=Object.keys(reported).length>0&&Object.entries(reported).every(([key,value])=>shown[key]===value);
        if(alreadyShown){metrics.append(badge(result.row.status));continue;}
        const card=el("div","candidate-card");card.append(el("h4","",t("版本结果")),badge(result.row.status),comparisonView(result.row.metrics));results.append(card);
      }
      if(results.childElementCount)step.append(results);entries.append(step);
    }
    more.hidden=groups.length<=limit;
    if(restoreFocus)selectedButton?.focus({preventScroll:true});
  };
  const more=el("button","more-button",t("显示更多执行步骤"));more.addEventListener("click",()=>{scope.flowLimits.set(s.run_id,(scope.flowLimits.get(s.run_id)||30)+30);fill();});
  section.append(more);fill();if(!groups.length)section.append(empty(t("尚无决策记录，不能重建执行流程。")));
  return section;
}
function runDetail(node, scope) {
  const entry=scope.cache.get(node.runId),s=entry?.snapshot,box=el("div","run-detail"); scope.flowOpen ||= new Set();
  const heading=el("div","detail-heading"),identity=el("div");
  identity.append(el("h3","",node.title));
  heading.append(identity,badge(s?.status||node.row.status));box.append(heading);
  if(entry?.status==="pending"&&!s)box.append(el("p","load-message",t("正在读取决策与反馈…")));
  if(entry?.status==="error"){
    box.append(el("p","load-message error",t("{p0}读取失败：{p1}", {p0: s?t("保留上次快照。"):"", p1: errorText(entry.errorInfo || {error:entry.error})})));
    const retry=el("button","",t("重试读取"));retry.addEventListener("click",()=>loadRun(node.runId,scope,true));box.append(retry);
  }
  if(s){
    box.append(executionSequence(s,scope));
    const unlinkedCandidates=list(s.candidates).filter(c=>!list(s.decisions).some(d=>relatedResults(d,s).some(r=>r.type==="candidate"&&r.row===c)));
    if(unlinkedCandidates.length){
      const candidates=el("div");
      unlinkedCandidates.forEach((c,i)=>{const card=el("article","candidate-card");card.append(el("h4","",t("待验证版本 {p0}", {p0: i+1})),badge(c.status),comparisonView(c.metrics));candidates.append(card);});box.append(retainedDetails(JSON.stringify([node.runId,"candidates"]),scope.flowOpen,t("版本结果 · {p0}",{p0:unlinkedCandidates.length}),candidates));
    }

  }
  if(s){
    const unlinked=list(s.jobs).filter(j=>!list(s.decisions).some(d=>linkedJob(d,s)===j));
    for(const job of unlinked){
      const card=el("article","job-record"),heading=el("div","detail-heading");
      heading.append(el("h4","",actionText(job.kind)),badge(job.status));card.append(heading);
      if(present(job.step))card.append(progressView(job.step,job.total_steps));
      card.append(metricsDetail(job.metrics));box.append(card);
    }
  }
  if(node.ancestryStatus==="unknown")box.append(el("p","quiet-note",t("续接关系不完整，暂不认定完整迭代次数。")));
  return box;
}

function projectDetail(node, model, scope) {
  if (node.type === "project") {
    const box=el("div");box.append(el("h3","",projectName(node.row)));
    if(node.row.description)box.append(el("p","",node.row.description));return box;
  }
  if (node.type === "run") return runDetail(node, scope);
  return el("div");
}

function setSelectedRun(id, scope) {
  scope.runId = id; state.runId = id;
  const s = scope.cache.get(id)?.snapshot;
  if (s) renderSnapshot(s); else clearRun();
  if (!state.demo) {
    const url = new URL(location.href); url.searchParams.set("project", state.projectId);
    if (id) url.searchParams.set("run", id); else url.searchParams.delete("run");
    history.replaceState(null, "", url);
  }
}
function renderProject() {
  if ($("project-graph").graphView?.dragging || $("project-graph").graphView?.simulating) { $("project-graph").graphView.pendingRender = true; return; }
  const project = selectedProject(); $("projects").hidden = !project;
  if (!project) return;
  const scope = state.activeScope || activateProject(), model = projectModel(project, scope);
  $("project-name").textContent = projectName(project); $("project-description").textContent = project.description || "";
  $("project-system").textContent = "";
  $("project-run-count").textContent = t("{p0} 轮研究", {p0: runRows(project).length});
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
  renderSelectors();
  if (scope.runId) setSelectedRun(scope.runId, scope);
  renderProject();
}
function renderSelectors() {
  const host=$("project-list"),signature=JSON.stringify([RSII18n.getLanguage(),state.projectId,state.projects.map(p=>[p.project_id,projectName(p)])]);
  if(host.signature===signature)return;
  host.signature=signature;
  const buttons=state.projects.map(project=>{
    const button=el("button","project-choice",projectName(project));button.type="button";
    button.setAttribute("aria-current",project.project_id===state.projectId?"true":"false");
    button.addEventListener("click",()=>selectProject(project.project_id));return button;
  });
  host.replaceChildren(...(buttons.length?buttons:[empty(t("暂无项目"))]));
}
function selectProject(id) {
  if(id===state.projectId)return;
  state.controller?.abort();++state.epoch;state.projectId=id;
  activateProject();chooseProjects(state.projects);setSelectedRun(state.activeScope.runId,state.activeScope);refresh();
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
  entry.controller = controller; entry.status = "pending"; entry.error = ""; if(!entry.snapshot)renderProject();
  const timeout = setTimeout(() => controller.abort(), 8000);
  const current = () => scope === state.activeScope && entry.token === token;
  try {
    const s = state.demo ? state.demoSnapshots.get(id) : await fetchJSON(`/api/runs/${encodeURIComponent(id)}`, controller.signal);
    if (!current() || controller.signal.aborted) return;
    if (s?.schema_version !== 1 || s.run_id !== id) throw uiError("运行快照协议或 ID 不匹配。");
    if(JSON.stringify(entry.snapshot)!==JSON.stringify(s))entry.snapshot = s; entry.status = "ready"; entry.errorInfo=null;
  } catch (error) {
    if (!current()) return;
    entry.status = "error"; entry.errorInfo = controller.signal.aborted ? uiError("请求超时。") : error; entry.error = errorText(entry.errorInfo);
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
    throw uiError("监控服务返回了无效响应。");
  }
  if (!response.ok)
    throw (present(data.error) ? new Error(asText(data.error)) : uiError("请求失败 ({p0})", {p0: response.status}));
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
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    if (state.demo) {
      const s = await fetchJSON("/demo.json", controller.signal);
      if (epoch !== state.epoch) return;
      const snapshots = [s, ...list(s.demo_runs)]; state.demoSnapshots = new Map(snapshots.map(r => [r.run_id, r]));
      chooseProjects([{ ...projectMeta(s.project), runs: snapshots }]);
      notice(() => t("示例数据 · 静态演示，不代表真实运行。")); $("footer-source").textContent = t("示例数据 · 静态快照");
    } else {
      const data = await fetchJSON("/api/projects", controller.signal);
      if (epoch !== state.epoch) return;
      if (!Array.isArray(data.projects)) throw uiError("项目清单格式无效。");
      chooseProjects(data.projects); notice(""); $("footer-source").textContent = t("本地监控 · 只读");
    }
    clearTimeout(timeout);
    const scope = state.activeScope;
    if (scope.runId) setSelectedRun(scope.runId, scope);
    if (scope.runId) await loadRun(scope.runId, scope, true);
    if (!state.projects.length) showEmpty(() => t("等待第一个项目"), () => t("尚无项目或运行快照；有数据后将自动显示。"));
  } catch (error) {
    if (epoch !== state.epoch) return;
    notice(() => t("保留已读取的项目图与快照。{p0}", {p0: controller.signal.aborted ? t("请求超时。") : errorText(error,"无法连接监控服务。")}), true);
    if (!selectedProject()) showEmpty(() => t("暂时无法读取项目"), () => t("请检查本地监控服务；恢复后将自动重新读取。"));
  } finally {
    clearTimeout(timeout);
    if (epoch === state.epoch) { state.controller = null; schedule(); }
  }
}
function switchDemo() {
  if (state.activeScope) {
    state.activeScope.view = $("project-graph").graphView;
    state.activeScope.view?.observer?.disconnect();
    state.activeScope.view?.stopSimulation?.();
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
  $("demo").textContent = state.demo ? t("返回") : t("示例");
  $("demo").setAttribute("aria-pressed", String(state.demo));
  $("empty").hidden = false;
  refresh();
}
$("demo").addEventListener("click", switchDemo);
$("empty-demo").addEventListener("click", () => {
  if (!state.demo) switchDemo();
  else refresh();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { clearTimeout(state.timer); $("project-graph").graphView?.stopSimulation?.(); }
  else if (state.auto && !state.demo) refresh();
});
function renderLanguage() {
  RSII18n.applyStatic(document);
  for(const lang of ["zh","en"])$("lang-"+lang)?.setAttribute("aria-pressed",String(RSII18n.getLanguage()===lang));
  $("demo").textContent=state.demo?t("返回"):t("示例");
  notice(chrome.notice,chrome.noticeError);
  $("empty-title").textContent=copy(chrome.emptyTitle);$("empty-description").textContent=copy(chrome.emptyDescription);
  $("footer-source").textContent=state.demo?t("示例数据 · 静态快照"):t("本地监控 · 只读");
  if(state.projects.length || state.activeScope)renderSelectors();
  if(state.snapshot)renderSnapshot(state.snapshot);
  renderProject();
}
for(const lang of ["zh","en"])$("lang-"+lang)?.addEventListener("click",()=>RSII18n.setLanguage(lang));
RSII18n.subscribe(renderLanguage);
renderLanguage();
refresh();
