"use strict";
const { t } = RSII18n;
const $ = (id) => document.getElementById(id);
const state = {
  demo: false, runId: "", realRunId: "", snapshot: null, controller: null,
  epoch: 0, timer: null, auto: true,
  queryApplied: false, page: "management",
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
  system_adopt: "合并采用改动", configure_prompt: "调整提示词", define_recipe: "定义训练方案", prepare_data: "准备训练数据", define_protocol: "定义评测方案", read_artifact: "查阅资料", preview_training_data: "预览训练数据",
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
function decisionAction(row) { if (row.kind === "review") return row.action || row.mode; if (row.kind === "operation") return row.operation || row.action || row.mode; return row.mode || row.action || (["finish","wait"].includes(row.kind) ? row.kind : undefined); }
function decisionTitle(row) {
  const action=decisionAction(row);
  if(Object.hasOwn(actionLabels,action))return actionText(action);
  const strategy=typeof row.strategy==="string"?row.strategy.trim():"";
  if(strategy&&[...strategy].length<=80&&strategy!==action&&/\p{L}/u.test(strategy)&&!/^[\w.:/\\-]+$/.test(strategy)&&!/[\r\n]/.test(strategy))return strategy;
  return actionText(action);
}
function decisionKey(row) { return asText(row.decision_id, JSON.stringify([row.timestamp, row.kind, row.round, row.directive_id, decisionAction(row), row.job_id])); }
function sortedDecisions(s) { return list(s.decisions).map((row, i) => ({ row, i })).sort((a,b) => (Date.parse(b.row.timestamp) || 0) - (Date.parse(a.row.timestamp) || 0) || a.i - b.i).map(x => x.row); }
function decisionGroups(s) {
  const groups = [],lineage=["depends_on_decision_ids", "source_decision_ids", "adopted_decision_ids"],references=new Set();
  for(const row of list(s.decisions)){
    for(const key of lineage)for(const id of listStrings(row[key]))references.add(id);
    if(identifier(row.parent_decision_id))references.add(row.parent_decision_id);
  }
  for (const row of sortedDecisions(s)) {
    const last = groups[groups.length - 1];
    const signature = JSON.stringify([row.kind, decisionAction(row), row.round, row.reason, row.strategy, row.status, row.actor_revision, row.decision_revision]);
    const referenced = references.has(row.decision_id);
    if (decisionAction(row) === "finish" && !referenced && ![...lineage,"parent_decision_id"].some(key => Object.hasOwn(row,key)) && last?.signature === signature) last.rows.push(row);
    else groups.push({ row, rows: [row], signature });
  } return groups;
}
const identifier = v => typeof v === "string" && v.length > 0;
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
function clearRun() { state.snapshot = null; }
// Membership comes from the catalog's explicit project IDs, never a run name.
function normalizeProjects(input) {
  const projects=new Map(),seenRuns=new Set();
  for(const group of list(input)){
    const id=identifier(group.project_id)?group.project_id:"unassigned";
    if(!projects.has(id))projects.set(id,{project_id:id,name:asText(group.name,id),description:asText(group.description,""),runs:[]});
    const target=projects.get(id);
    for(const row of list(group.runs))if(identifier(row.run_id)&&!seenRuns.has(row.run_id)){
      seenRuns.add(row.run_id);target.runs.push(row);
    }
  }
  return [...projects.values()];
}
const listStrings = v => Array.isArray(v) ? v.filter(identifier) : [];
const runRows = project => list(project?.runs).filter(r => identifier(r.run_id));
const projectName = p => p?.project_id==="unassigned"?t("未归属"):asText(p?.name,p?.project_id||"");
const experimentName = row => asText(row?.experiment_name,row?.run_id||"");
const ownerOfRun = id => state.projects.find(p=>runRows(p).some(r=>r.run_id===id));
function selectedProject() { return state.projects.find(p => p.project_id === state.projectId); }
function abortEntry(entry) {
  entry.controller?.abort(); entry.controller = null; entry.token = (entry.token || 0) + 1;
  if (entry.status === "pending") entry.status = entry.snapshot ? "ready" : "idle";
}
function disposeGraph(scope) {
  if(!scope)return;
  scope.graph?.dispose?.();scope.graph=null;scope.detailFingerprint=null;scope.renderPending=false;
}
function activateProject() {
  const key = JSON.stringify([state.demo, state.runId]);
  if (state.activeScope?.key === key) return state.activeScope;
  if (state.activeScope) { disposeGraph(state.activeScope); for (const entry of state.activeScope.cache.values()) abortEntry(entry); }
  let scope = state.scopes.get(key);
  if (!scope) { scope = { key, cache: new Map(), runId: state.runId, details: null }; state.scopes.set(key, scope); }
  state.activeScope = scope;
  state.runId = scope.runId; clearRun();
  return scope;
}
function executionGroups(s) {
  // Known timestamps establish order. Unknown timestamps retain snapshot order at
  // the end, with an explicit caveat; list position is never treated as lineage.
  return decisionGroups(s).sort((a,b) => {
    const at=Date.parse(a.rows[a.rows.length-1].timestamp),bt=Date.parse(b.rows[b.rows.length-1].timestamp);
    return (Number.isFinite(at)?at:Infinity)-(Number.isFinite(bt)?bt:Infinity) || 0;
  });
}
// A reading projection only: the snapshot and the complete decision DAG remain
// unchanged. Candidate membership requires a unique reported identity.
function progressGroups(s, groups) {
  const auxiliary=new Set(["read_artifact","preview_training_data","define_recipe","define_protocol","query_jobs","query_candidates","query_metrics","inspect_job","inspect_candidate","list_jobs","list_candidates","research_search","research_read","research_archive","research_patch","research_write","research_memory","research_experience","research_history","wait"]);
  const refs=row=>[row.parent_decision_id,...listStrings(row.depends_on_decision_ids),...listStrings(row.adopted_decision_ids),...listStrings(row.source_decision_ids)].filter(identifier);
  const protectedIDs=new Set(groups.flatMap(g=>g.rows.flatMap(refs)));
  const folded={auxiliary:[],finish:[]},visible=[];
  for(const g of groups){
    const row=g.row,action=decisionAction(row),feedback=feedbackFor(row,s);
    const quiet=DONE.has(statusKey(feedback.status))&&!row.error&&!object(row.feedback).error&&!g.rows.some(r=>protectedIDs.has(r.decision_id));
    if(quiet&&(action==="finish"||auxiliary.has(action)))folded[action==="finish"?"finish":"auxiliary"].push(g);
    else visible.push(g);
  }
  const candidates=list(s.candidates),byCandidate=new Map(),result=[];
  const candidateFor=row=>{
    if(row.kind==="review"||FAILED.has(statusKey(feedbackFor(row,s).status))||row.error||object(row.feedback).error)return null;
    const related=relatedResults(row,s).filter(r=>r.type==="candidate").map(r=>r.row);
    if(!present(row.candidate_id)&&present(row.job_id))related.push(...candidates.filter(c=>sameId(c.candidate_id,row.job_id)));
    const matches=[...new Set(related)];
    return matches.length===1&&candidates.filter(c=>c.candidate_id===matches[0].candidate_id).length===1?matches[0]:null;
  };
  for(const g of visible){
    const candidate=candidateFor(g.row),id=candidate?.candidate_id;
    if(id&&byCandidate.has(id))byCandidate.get(id).rows.push(...g.rows);
    else {const next={...g,rows:[...g.rows],candidate};result.push(next);if(id)byCandidate.set(id,next);}
  }
  // Remap only explicit relationships between retained groups. Internal edges
  // disappear inside a candidate; no chronological or inferred adoption is added.
  const ids=new Map();
  for(const g of groups)for(const row of g.rows)if(identifier(row.decision_id))ids.set(row.decision_id,(ids.get(row.decision_id)||0)+1);
  const remap=new Map();
  for(const g of result)for(const row of g.rows)if(ids.get(row.decision_id)===1)remap.set(row.decision_id,g.row.decision_id);
  // Contracting interleaved candidate steps can introduce a cycle even when
  // the raw DAG is valid. Keep its individual steps in that case.
  const owners=new Map(result.map((g,i)=>[g.row.decision_id,i])),degree=result.map(()=>0),children=result.map(()=>new Set());
  result.forEach((g,to)=>g.rows.forEach(row=>{
    for(const id of [row.parent_decision_id,...listStrings(row.depends_on_decision_ids),...listStrings(row.adopted_decision_ids)]){
      const from=ids.get(id)===1?owners.get(remap.get(id)):undefined;
      if(from!==undefined&&from!==to&&!children[from].has(to)){children[from].add(to);degree[to]++;}
    }
  }));
  const queue=degree.map((n,i)=>n===0?i:-1).filter(i=>i>=0);
  for(let i=0;i<queue.length;i++)for(const next of children[queue[i]])if(--degree[next]===0)queue.push(next);
  if(queue.length!==result.length)return {groups:visible,folded};
  for(const g of result){
    const original=g.row,relations=["depends_on_decision_ids","adopted_decision_ids","source_decision_ids"];
    g.row={...original};
    for(const field of relations){
      const values=g.rows.flatMap(row=>[...listStrings(row[field]),...(field==="depends_on_decision_ids"&&identifier(row.parent_decision_id)?[row.parent_decision_id]:[])]);
      if(values.length||g.rows.some(row=>Object.hasOwn(row,field)))g.row[field]=[...new Set(values.map(id=>remap.get(id)||id).filter(id=>id!==original.decision_id))];
    }
    if(identifier(original.parent_decision_id))delete g.row.parent_decision_id;
    if(g.candidate)g.title=t("候选 {p0} · {p1}",{p0:candidates.indexOf(g.candidate)+1,p1:g.rows.some(row=>decisionAction(row)==="apply_candidate"&&DONE.has(statusKey(row.status)))?t("已采纳"):g.rows.some(row=>["evaluate","evaluation","candidate_evaluation"].includes(decisionAction(row))&&DONE.has(statusKey(row.status)))?t("已评测"):actionText(decisionAction(original))});
  }
  return {groups:result,folded};
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
  const normalized=summary.trim().replace(/[.!。]$/, "").toLowerCase();
  if(normalized===`${decisionAction(row)} completed`.toLowerCase())return "";
  return /^(?:执行已完成[。.]?|执行完成[。.]?|completed[.!]?|done[.!]?|success[.!]?|configure_prompt completed)$/i.test(summary.trim())?"":summary;
}
function executionSequence(s, scope) {
  scope.flowOpen ||= new Set(); scope.flowSelected ||= new Map();
  const allGroups=executionGroups(s),progress=progressGroups(s,allGroups),mode=scope.graphMode||"progress";
  const section=el("section","execution-sequence"),groups=mode==="all"?allGroups:progress.groups;
  const graphRows=groups.map(g=>g.row),graphKeys=new Map(RSIDecisions.layout(graphRows).nodes.filter(n=>n.row).map(n=>[groups[n.index],n.key]));
  section.id="execution-flow";

  scope.graphViews ||= {progress:scope.graphView||{},all:{}};
  scope.graphView=scope.graphViews[mode];scope.modeSelected ||= new Map();scope.progressSteps ||= new Map();
  const changeMode=(next,selected)=>{
    scope.graph?.dispose?.();scope.modeSelected.set(mode,scope.flowSelected.get(s.run_id));
    scope.graphMode=next;scope.graphView=scope.graphViews[next];
    scope.flowSelected.set(s.run_id,selected??scope.modeSelected.get(next));
    scope.detailFingerprint=null;renderProject();
  };
  const toolbar=el("div","trajectory-heading"),switcher=el("div","trajectory-mode");
  switcher.setAttribute("role","group");switcher.setAttribute("aria-label",t("脉络视图"));
  for(const [value,title] of [["progress","关键进展"],["all","全部记录"]]){
    const button=el("button","",t(title));button.type="button";button.setAttribute("aria-pressed",String(value===mode));button.addEventListener("click",()=>{if(value!==mode)changeMode(value);});switcher.append(button);
  }
  const foldedCount=Object.values(progress.folded).flatMap(gs=>gs.flatMap(g=>g.rows)).length;
  toolbar.append(switcher,el("p","quiet-note",mode==="all"?t("{p0} 条原始记录",{p0:list(s.decisions).length}):t("{p0} 个进展节点 · 折叠 {p1} 条辅助与收尾记录",{p0:groups.length,p1:foldedCount})));section.append(toolbar);
  const shell=el("div","trajectory-shell"),navigation=el("div","flow-map");
  navigation.setAttribute("aria-label",t("本次运行的决策"));shell.append(navigation);
  const controls=el("div","trajectory-tools");
  for(const [text,title,action] of [["−","缩小",()=>scope.graph?.zoomBy(.8)],["+","放大",()=>scope.graph?.zoomBy(1.25)],["⤢","适应全部节点",()=>scope.graph?.fit()],["↺","重排",()=>scope.graph?.reset()]]){
    const button=el("button","",text);button.type="button";button.title=t(title);button.setAttribute("aria-label",t(title));button.addEventListener("click",action);controls.append(button);
  }
  shell.append(controls);section.append(shell);
  const entries=el("ol","flow-steps");section.append(entries);
  const fill=(restoreFocus=false)=>{
    const scroll={left:navigation.scrollLeft,top:navigation.scrollTop};
    entries.replaceChildren();
    const keyOf=g=>graphKeys.get(g);
    let selected=scope.flowSelected.get(s.run_id);
    if(!groups.some(g=>keyOf(g)===selected)){
      const preferred=mode==="progress"?[...groups].reverse().find(g=>g.candidate&&g.rows.some(r=>decisionAction(r)==="apply_candidate"&&DONE.has(statusKey(r.status)))):null;
      selected=groups.length?keyOf(preferred||groups[0]):null;
    }
    scope.flowSelected.set(s.run_id,selected);
    scope.graph?.dispose?.();
    scope.graph=RSIDecisions.render({
      view:scope.graphView,
      onIdle:()=>{if(scope===state.activeScope&&state.page==="experiment"&&scope.renderPending){scope.renderPending=false;renderProject();}},
      rows:graphRows,selected,compact:mode==="progress",
      startLabel:`0 · ${t("实验起点")}`,label:(row,index)=>`${index+1} · ${groups[index].title||decisionTitle(row)}${FAILED.has(statusKey(feedbackFor(row,s).status))?` · ${statusText(feedbackFor(row,s).status)}`:""}`,
      edgeLabel:kind=>t({sequence:"记录顺序，不表示因果",parent:"明确的父决策",dependency:"依赖已完成决策",adoption:"已确认采纳结果",root:"实验起点",source:"引用的决策"}[kind]),
      onSelect:key=>{scope.flowSelected.set(s.run_id,key);fill(true);},
    });
    navigation.replaceChildren(scope.graph);
    navigation.scrollLeft=scroll.left;navigation.scrollTop=scroll.top;
    for(const [i,g] of groups.entries()){
      const key=keyOf(g);
      if(key!==selected)continue;
      let row=g.row;
      if(g.candidate){
        const chosen=scope.progressSteps.get(key);
        row=g.rows.find(r=>decisionKey(r)===chosen)||[...g.rows].reverse().find(r=>["evaluate","evaluation","candidate_evaluation"].includes(decisionAction(r)))||g.rows[g.rows.length-1];
        const stages=el("li","progress-stages");stages.append(el("h4","",g.title));
        const buttons=el("div","progress-stage-buttons");
        for(const record of g.rows){
          const button=el("button","",decisionTitle(record));button.type="button";button.setAttribute("aria-pressed",String(record===row));
          button.addEventListener("click",()=>{scope.progressSteps.set(key,decisionKey(record));fill();});buttons.append(button);
        }
        stages.append(buttons);entries.append(stages);
      }
      const f=feedbackFor(row,s);
      const step=el("li","flow-step"),index=el("span","flow-index",String(i+1).padStart(2,"0"));
      const decision=el("div","flow-decision"),feedback=el("div","flow-feedback"),metrics=el("div","flow-metrics");
      const meta=el("p","quiet-note",`${dateText(row.timestamp)}${number(row.round)?t(" · 第 {p0} 轮", {p0: row.round}):""}`);
      const title=decisionTitle(row);decision.append(el("h4","",title),meta);
      if(!g.candidate&&g.rows.length>1)decision.append(el("p","repeat-count",t("重复报告 {p0} 次", {p0: g.rows.length})));
      if(present(row.reason))decision.append(el("p","flow-label",t("决策理由")),el("p","decision-reason",row.reason));
      if(present(row.strategy)&&row.strategy!==row.reason&&row.strategy!==decisionAction(row)&&asText(row.strategy,"").trim()!==title)decision.append(el("p","decision-reason",row.strategy));
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
    if(restoreFocus)navigation.querySelector?.('[aria-pressed="true"]')?.focus({preventScroll:true});
  };
  fill();if(!groups.length)section.append(empty(t(allGroups.length?"暂无关键进展，可展开下方记录或切换全部记录。":"尚无决策记录，不能重建执行流程。")));
  if(mode==="progress"){
    const keys=new Map(RSIDecisions.layout(allGroups.map(g=>g.row)).nodes.filter(n=>n.row).map(n=>[allGroups[n.index],n.key]));
    for(const [kind,title] of [["auxiliary","辅助操作"],["finish","本轮收尾"]]){
      const folded=progress.folded[kind];if(!folded.length)continue;
      const content=el("div","folded-records");
      for(const g of folded){const button=el("button","folded-record",`${dateText(g.row.timestamp)} · ${decisionTitle(g.row)}${g.rows.length>1?` × ${g.rows.length}`:""}`);button.type="button";button.addEventListener("click",()=>changeMode("all",keys.get(g)));content.append(button);}
      section.append(retainedDetails(JSON.stringify([s.run_id,"folded",kind]),scope.flowOpen,t("{p0} · {p1} 条",{p0:t(title),p1:folded.reduce((n,g)=>n+g.rows.length,0)}),content));
    }
  }
  return section;
}
function runDetail(node, scope) {
  const entry=scope.cache.get(node.runId),s=entry?.snapshot,box=el("div","run-detail"); scope.flowOpen ||= new Set();
  const heading=el("div","detail-heading"),identity=el("div");
  identity.append(el("h3","",t("研究脉络")),el("p","quiet-note",t("{p0} 次决策",{p0:list(s?.decisions).length})));
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
  const record=s || node.row, sources=[...new Set([record.parent_run_id, ...listStrings(record.source_run_ids)].filter(identifier))];
  if(sources.length){
    const refs=el("div","run-provenance");refs.append(el("p","quiet-note",t("来源引用不表示项目归属或自动继承。")));
    for(const id of sources){
      const target=ownerOfRun(id),row=runRows(target).find(r=>r.run_id===id),link=el("a","provenance-link",row?experimentName(row):id);
      link.href=`?run=${encodeURIComponent(id)}`;
      // In demo mode navigate its local catalog without replacing the real URL.
      if(state.demo&&target)link.addEventListener("click",event=>{event.preventDefault();selectRun(id);});
      refs.append(link);
    }
    box.append(retainedDetails(JSON.stringify([node.runId,"provenance"]),scope.flowOpen,t("来源记录"),refs));
  }
  return box;
}

function setSelectedRun(id, scope) {
  scope.runId = id; state.runId = id;
  const s = scope.cache.get(id)?.snapshot;
  if (s) renderSnapshot(s); else clearRun();
  if (!state.demo) {
    const url = new URL(location.href); url.searchParams.set("project",state.projectId);
    if (id) url.searchParams.set("run", id); else url.searchParams.delete("run");
    if(state.page==="experiment")history.replaceState(null, "", url);
  }
}
function renderProject() {
  const project = selectedProject(); $("management").hidden = state.page!=="management"; $("projects").hidden = state.page!=="experiment" || !project;
  if (!project || state.page!=="experiment") return;
  const scope = state.activeScope || activateProject(), row = runRows(project).find(r=>r.run_id===state.runId);
  if(!row)return;
  if(scope.graphView?.interacting||scope.graphView?.animating){scope.renderPending=true;return;}
  $("project-name").textContent = experimentName(row); $("project-description").textContent = projectName(project); $("project-description").hidden=false;
  const host=$("project-content"),details=scope.details || (scope.details=el("section","project-detail"));
  details.setAttribute("aria-label",t("本次运行的决策"));
  const entry=scope.cache.get(row.run_id);
  const fingerprint=JSON.stringify([RSII18n.getLanguage(),scope.graphMode||"progress",row,entry?.snapshot,entry?.status==="pending"&&entry?.snapshot?"ready":entry?.status,entry?.error||"",state.projects.map(p=>[p.project_id,projectName(p)])]);
  if(scope.detailFingerprint!==fingerprint){
    const position=typeof window!=="undefined"?{left:window.scrollX,top:window.scrollY}:null;
    for(const d of details.querySelectorAll?.("details[data-retain-key]")||[]){
      scope.flowOpen ||= new Set();if(d.open)scope.flowOpen.add(d.dataset.retainKey);else scope.flowOpen.delete(d.dataset.retainKey);
    }
    const oldMap=details.querySelector?.(".flow-map"),scroll={left:oldMap?.scrollLeft||0,top:oldMap?.scrollTop||0};
    scope.graph?.dispose?.();scope.graph=null;
    details.replaceChildren(runDetail({runId:row.run_id,row},scope));
    const map=details.querySelector?.(".flow-map");if(map){map.scrollLeft=scroll.left;map.scrollTop=scroll.top;}
    scope.detailFingerprint=fingerprint;
    if(position&&host.firstElementChild===details&&(window.scrollX!==position.left||window.scrollY!==position.top))window.scrollTo({...position,behavior:"instant"});
  }
  if(host.firstElementChild!==details)host.replaceChildren(details);
  $("empty").hidden = true;
}
function chooseProjects(input) {
  state.projects=normalizeProjects(input);
  let requestedRun=state.runId,requestedProject=state.projectId;
  if(!state.demo&&!state.queryApplied){
    const params=new URLSearchParams(location.search);
    requestedRun=params.get("run")||"";requestedProject=params.get("project")||"";
    state.page=requestedRun?"experiment":"management";state.queryApplied=true;
  }
  const owner=ownerOfRun(requestedRun);
  state.projectId=owner?.project_id || state.projects.find(p=>p.project_id===requestedProject)?.project_id || "";
  state.runId=owner?requestedRun:"";
  if(state.page==="experiment"&&!owner)state.page="management";
  const scope=activateProject();setSelectedRun(state.runId,scope);
  renderSelectors();renderProject();
}
function renderSelectors() {
  const host=$("project-list"),signature=JSON.stringify([RSII18n.getLanguage(),state.projects]);
  $("experiment-count").textContent=t("{p0} 个项目 · {p1} 个实验",{p0:state.projects.length,p1:state.projects.reduce((n,p)=>n+runRows(p).length,0)});
  if(host.signature===signature)return;host.signature=signature;
  const sections=state.projects.map(project=>{
    const section=el("section","project-group"),heading=el("div","project-group-heading");
    heading.append(el("h2","",projectName(project)),el("span","quiet-note",t("{p0} 个实验",{p0:runRows(project).length})));section.append(heading);
    const grid=el("div","experiment-grid");
    for(const row of runRows(project)){
      const card=el("a","experiment-card");card.href=`?project=${encodeURIComponent(project.project_id)}&run=${encodeURIComponent(row.run_id)}`;
      const head=el("div","experiment-card-heading");head.append(el("h3","",experimentName(row)),badge(row.status));card.append(head);
      card.append(el("p","quiet-note",dateText(row.updated_at)));
      if(row.counts?.decisions!==undefined)card.append(el("p","",t("{p0} 次决策",{p0:row.counts.decisions})));
      card.addEventListener("click",event=>{if(event?.metaKey||event?.ctrlKey||event?.shiftKey||event?.altKey)return;event?.preventDefault();selectRun(row.run_id);});grid.append(card);
    }
    section.append(grid);return section;
  });
  host.replaceChildren(...(sections.length?sections:[empty(t("暂无项目"))]));
}
function navigateURL(id,replace=false) {
  if(state.demo)return;
  const url=new URL(location.href);url.searchParams.delete("project");url.searchParams.delete("run");url.hash="";
  if(id)url.searchParams.set("run",id);
  (replace?history.replaceState:history.pushState).call(history,null,"",url);
}
function showManagement(push=true) {
  clearTimeout(state.timer);state.controller?.abort();++state.epoch;
  if(state.activeScope){disposeGraph(state.activeScope);for(const entry of state.activeScope.cache.values())abortEntry(entry);}
  state.page="management";if(push)navigateURL("");renderSelectors();renderProject();schedule();
}
function selectRun(id,push=true) {
  const owner=ownerOfRun(id);if(!owner)return Promise.resolve();
  const changed=id!==state.runId||state.page!=="experiment";
  clearTimeout(state.timer);state.controller?.abort();++state.epoch;state.page="experiment";
  if(changed&&push)navigateURL(id);
  state.projectId=owner.project_id;state.runId=id;chooseProjects(state.projects);
  return loadRun(state.runId,state.activeScope).finally(schedule);
}
// Compatibility for callers that historically passed the sole run ID here.
function selectProject(id,push=true) { return selectRun(id,push); }
async function loadRun(id, scope, force = false) {
  if (scope !== state.activeScope || !runRows(selectedProject()).some(r=>r.run_id===id)) return;
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
      chooseProjects(snapshots.map(snapshot=>({...object(snapshot.project), runs:[snapshot]})));
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
    if (scope.runId && state.page==="experiment") await loadRun(scope.runId, scope, true);
    if (!state.projects.length) showEmpty(() => t("等待第一个项目"), () => t("尚无项目或运行快照；有数据后将自动显示。"));
  } catch (error) {
    if (epoch !== state.epoch) return;
    notice(() => t("保留已读取的项目与快照。{p0}", {p0: controller.signal.aborted ? t("请求超时。") : errorText(error,"无法连接监控服务。")}), true);
    if (!selectedProject()) showEmpty(() => t("暂时无法读取项目"), () => t("请检查本地监控服务；恢复后将自动重新读取。"));
  } finally {
    clearTimeout(timeout);
    if (epoch === state.epoch) { state.controller = null; schedule(); }
  }
}
function switchDemo() {
  if (state.activeScope) {
    disposeGraph(state.activeScope);
    for (const entry of state.activeScope.cache.values()) abortEntry(entry);
    state.activeScope = null;
  }

  if (!state.demo) state.realProjectId = state.projectId;
  state.realRunId = state.demo ? state.realRunId : state.runId;
  state.demo = !state.demo;
  state.runId = state.demo ? "" : state.realRunId;
  state.projectId = state.demo ? "" : state.realProjectId;
  state.projects = []; state.page="management";
  $("projects").hidden = true;
  clearRun();
  state.snapshot = null;
  $("demo").textContent = state.demo ? t("返回") : t("示例");
  $("demo").setAttribute("aria-pressed", String(state.demo));
  $("empty").hidden = false;
  refresh();
}
$("back-projects").addEventListener("click",()=>showManagement());
if(typeof window!=="undefined")window.addEventListener("popstate",()=>{
  const params=new URLSearchParams(location.search),id=params.get("run");
  if(id&&ownerOfRun(id))selectRun(id,false);
  else showManagement(false);
});
$("demo").addEventListener("click", switchDemo);
$("empty-demo").addEventListener("click", () => {
  if (!state.demo) switchDemo();
  else refresh();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { clearTimeout(state.timer); disposeGraph(state.activeScope); }
  else if(state.demo)renderProject();
  else if (state.auto) refresh();
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
