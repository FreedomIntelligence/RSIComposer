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
  failed: "失败", failure: "失败", error: "错误", cancelled: "已取消", canceled: "已取消", unknown: "状态未知", idle: "空闲", waiting: "等待中", paused: "已暂停", stopped: "已停止",
  accepted: "已采纳", applied: "已采纳", adopted: "已采纳", promoted: "已采纳", bound: "已绑定", rejected: "未采纳", discarded: "已放弃", proposed: "已提出", ready: "待审阅", validated: "已验证", awaiting_evaluation: "待评测", awaiting_review: "待审阅", unavailable: "快照不可用", reviewed: "已复查", cancel_requested: "已请求取消", continue: "继续执行", cancel: "请求取消",
};
const actionLabels = {
  lab_status: "查看实验状态", lab_start: "启动程序", lab_poll: "查看任务进度", lab_result: "读取任务结果", lab_cancel: "取消任务", lab_read_file: "读取文件", lab_read_log: "读取任务日志", lab_write_file: "写入文件", lab_export: "保存产物", lab_program: "实验程序", medical_evaluate: "提交固定评测",
  system_adopt: "合并采用改动", configure_prompt: "调整提示词", define_recipe: "定义训练方案", prepare_data: "准备训练数据", define_protocol: "定义评测方案", read_artifact: "查阅资料", preview_training_data: "预览训练数据",
  train: "训练模型", training: "训练模型", actor_training: "训练执行模型", decision_maker_training: "训练决策模型", sft: "监督微调", rl: "强化学习",
  evaluate: "验证改动效果", evaluation: "验证改动效果", candidate_evaluation: "验证改动效果", select: "选择版本", apply_candidate: "采用改动", execute: "执行任务",
  finish: "结束本轮研究", finished: "结束本轮研究", continue: "继续执行", continue_training: "继续训练", cancel: "请求取消", cancel_training: "请求取消训练", stop: "停止训练", wait: "等待", review: "复查执行", prepare: "准备", queued: "排队", starting: "准备运行", collecting_result: "收集结果", accepted: "指令已接收", completed: "已完成",
  research_search: "检索研究资料", research_read: "阅读研究资料", research_archive: "查阅研究档案", research_patch: "更新研究笔记", research_write: "记录研究结论", research_memory: "查阅研究记忆", research_experience: "检索历史经验", research_history: "查阅决策历史",
  search: "搜索资料", read: "读取资料", web_search: "搜索网页", fetch: "获取资料", query_jobs: "查询任务", query_candidates: "查询候选", query_metrics: "查询指标", inspect_job: "查看任务", inspect_candidate: "查看候选", list_jobs: "查看任务列表", list_candidates: "查看候选列表",
};
const metricLabels = {
  model_calls: "模型调用", compactions: "上下文压缩",
  loss: "损失", train_loss: "训练损失", "train.loss": "训练损失", "train/loss": "训练损失", eval_loss: "评测损失", learning_rate: "学习率", lr: "学习率", "train/learning_rate": "学习率",
  nll: "负对数似然", base_nll: "基线 NLL", candidate_nll: "候选 NLL", delta_nll: "上报 NLL 差值", heldout_nll_delta: "验证 NLL 差值", eval_samples: "评测样本数", num_samples: "样本数", sample_count: "样本数", accuracy: "准确率", score: "得分", grad_norm: "梯度范数", epoch: "训练轮数", runtime: "耗时", train_runtime: "训练耗时", total_flos: "浮点运算量",
};
const asText = (v, fallback = "—") => ["string", "number", "boolean"].includes(typeof v) ? String(v) : fallback;
const list = (v) => Array.isArray(v) ? v.filter(x => x && typeof x === "object" && !Array.isArray(x)) : [];
const object = (v) => v && typeof v === "object" && !Array.isArray(v) ? v : {};
const number = (v) => typeof v === "number" && Number.isFinite(v);
const present = (v) => v !== undefined && v !== null && v !== "";
const statusKey = (v) => asText(v, "unknown").toLowerCase();
const statusText = (v) => (present(v) ? t(labels[statusKey(v)]) : "") || t("状态未报告或未归类");
const actionText = (v) => t(actionLabels[asText(v, "")]) || asText(v, "").trim() || t("未命名决策");
const metricText = (v) => {
  const match=/^(.+)[./](accuracy|macro_f1|f1|invalid_format)$/.exec(v);
  if(match){const names={medqa:"MedQA",medxpertqa:"MedXpertQA"},labels={accuracy:"准确率",macro_f1:"Macro F1",f1:"F1",invalid_format:"格式错误数"};return `${names[match[1]]||match[1]} · ${t(labels[match[2]])||labels[match[2]]}`;}
  return t(metricLabels[v]) || asText(v, t("其他指标"));
};
const ratioMetric=(key,value)=>/(?:^|[./])(?:accuracy|macro_f1|f1)$/.test(key)&&number(value)&&value>=0&&value<=1;
const metricNumber=(key,value)=>ratioMetric(key,value)?`${Number((value*100).toFixed(2))}%`:fmt(value);
function el(tag, className = "", text) { const e = document.createElement(tag); if (className) e.className = className; if (text !== undefined) e.textContent = asText(text); return e; }
function badge(v) { const k = statusKey(v); return el("span", `badge ${FAILED.has(k) || REJECTED.has(k) ? "failure" : DONE.has(k) || ACCEPTED.has(k) ? "success" : ACTIVE.has(k) ? "active" : ""}`, statusText(v)); }
function runTerminal(s) { const k=statusKey(s?.status);return DONE.has(k)||FAILED.has(k)||["stopped","host_limit_reached","deadline_reached_before_start"].includes(k); }
function observedBadge(value,s) {
  if(runTerminal(s)&&ACTIVE.has(statusKey(value)))return el("span","badge last-observed",t("最后观测：{p0}",{p0:statusText(value)}));
  return badge(value);
}
function runBadge(value) { return DONE.has(statusKey(value))?el("span","badge",t("已结束")):badge(value); }
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
  for (const [key, value] of Object.entries(flattenMetrics(metrics))) { const item = el("div", "metric-chip"); const label=el("dt", "", metricText(key));label.title=key;item.append(label, el("dd", "", metricNumber(key,value))); dl.append(item); } return dl;
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
  if(Object.hasOwn(actionLabels,action)||["invoke","explore"].includes(row.kind))return actionText(action);
  const strategy=typeof row.strategy==="string"?row.strategy.trim():"";
  if(strategy&&[...strategy].length<=80&&strategy!==action&&/\p{L}/u.test(strategy)&&!/^[\w.:/\\-]+$/.test(strategy)&&!/[\r\n]/.test(strategy))return strategy;
  return actionText(action);
}
const identifier = v => typeof v === "string" && v.length > 0;
const sameId = (a, b) => present(a) && present(b) && a === b;
function linkedJob(row, s) {
  const jobs = list(s.jobs);
  // Recover only the known legacy evaluator alias, with both explicit identities.
  if (present(row.job_id)) {
    const matches=jobs.filter(j => sameId(j.job_id,row.job_id));
    if(matches.length)return matches.length===1?matches[0]:undefined;
    const legacy=jobs.filter(j => j.kind==="evaluate" && sameId(j.directive_id,row.directive_id)
      && identifier(row.candidate_id) && j.job_id===`evaluation:${row.candidate_id}`);
    return legacy.length===1 && jobs.filter(j=>sameId(j.directive_id,row.directive_id)).length===1 ? legacy[0] : undefined;
  }
  const matches = jobs.filter(j => sameId(j.directive_id, row.directive_id));
  return matches.length === 1 ? matches[0] : undefined;
}
function relatedResults(row,s) {
  const results=[],job=linkedJob(row,s);
  if(job)results.push({type:"job",row:job,evidence:sameId(row.job_id,job.job_id)?t("明确任务 ID 关联；结果为当前快照。"):t("唯一指令 ID 关联；结果为当前快照。")});
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
  else if (ACTIVE.has(statusKey(status))) text = runTerminal(s)?t("实验已结束，终态未报告。"):t("执行中，等待最终反馈。");
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
  for (const m of pairs) { const row=el("tr"), delta=number(m.base)&&number(m.next)?m.next-m.base:undefined;row.append(el("td","",metricText(m.name)),el("td","",metricNumber(m.name,m.base)),el("td","",metricNumber(m.name,m.next)),el("td","",number(delta)?`${delta>0?"+":""}${ratioMetric(m.name,m.base)&&ratioMetric(m.name,m.next)?`${Number((delta*100).toFixed(2))} pp`:fmt(delta)}`:"—"));body.append(row); }table.append(head,body);wrap.append(table);if(Object.keys(extra).length)wrap.append(metricStrip(extra));return wrap;
}
function formalEvaluations(s) {
  const seen=new Set(),rows=[];
  for(const row of list(s.evaluations).slice().reverse()){
    const key=identifier(row.record_sha256)?`record:${row.record_sha256.toLowerCase()}`:identifier(row.evaluation_id)?`evaluation:${row.evaluation_id}`:identifier(row.job_id)?`job:${row.job_id}`:null;
    if(key&&seen.has(key))continue;if(key)seen.add(key);rows.push(row);
  }
  return rows.reverse();
}
function dashboardData(s) {
  const definitions={...object(s.metric_definitions)},points=[];
  const evaluations=RSIDecisions.layout(list(s.decisions)).nodes.filter(n=>n.row&&n.row.kind!=="review"&&["evaluate","evaluation","candidate_evaluation"].includes(decisionAction(n.row))).map(n=>n.row);
  const explicit=Array.isArray(s.metric_points);
  if(explicit){
    for(const point of list(s.metric_points))if(identifier(point.metric)&&number(point.value))points.push({...point});
  }else{
    const measured=Array.isArray(s.evaluations)?formalEvaluations(s):evaluations;
    measured.forEach((row,index)=>{
      const metrics=flattenMetrics({...object(row.metrics),...object(object(row.feedback).metrics)});
      for(const [metric,value] of Object.entries(metrics))points.push({metric,value,evaluation:index+1,task_id:row.job_id,decision_id:row.decision_id,attempt_id:row.research_attempt_id});
    });
    for(const job of list(s.jobs).filter(job=>job.role!=="baseline"&&job.job_id!=="evaluation:initial")){
      for(const point of historyRows(job))for(const [metric,value] of Object.entries(flattenMetrics(point.metrics)))points.push({metric,value,step:point.step,elapsed_seconds:point.elapsed_seconds,task_id:job.job_id});
      for(const [metric,value] of Object.entries(flattenMetrics(job.metrics)))if(!Array.isArray(s.evaluations)&&!points.some(point=>point.metric===metric&&number(point.evaluation)))points.push({metric,value,step:job.step,elapsed_seconds:job.elapsed_seconds,task_id:job.job_id});
    }
  }
  const keys=[...new Set([...Object.keys(definitions),...points.map(p=>p.metric)])];
  for(const key of keys)if(!definitions[key])definitions[key]={label:metricText(key),unit:!explicit&&/(?:^|[./])(?:accuracy|macro_f1|f1)$/.test(key)?"percent":"number",direction:"neutral"};
  return {definitions,points,keys,evaluationCount:explicit?0:Array.isArray(s.evaluations)?formalEvaluations(s).length:evaluations.length};
}
function dashboardYConfig(value) {
  if(!value||typeof value!=="object"||Array.isArray(value))return undefined;
  const range={};for(const key of ["min","max"])if(Object.hasOwn(value,key)){if(!number(value[key]))return undefined;range[key]=value[key];}
  if(number(range.min)&&number(range.max)&&(!(range.max>range.min)||!number(range.max-range.min)))return undefined;
  return Object.keys(range).length?range:undefined;
}
function dashboardYRange(values,percent,custom={}) {
  const lo=Math.min(...values),hi=Math.max(...values),span=hi-lo;
  const rawPadding=hi===lo?(Math.abs(lo)*.05||(percent?.005:1)):(number(span)?span*.08:Math.max(Math.abs(lo),Math.abs(hi))*.08);
  const padding=Math.max(Number.MIN_VALUE,rawPadding);
  let min=Math.max(-Number.MAX_VALUE,lo-padding),max=Math.min(Number.MAX_VALUE,hi+padding);
  if(percent){min=Math.max(0,min);max=Math.min(1,max);}
  if(number(custom.min))min=custom.min;if(number(custom.max))max=custom.max;
  if(!(max>min)){const margin=Math.max(padding,Math.abs(number(custom.min)?min:max)*.05,percent?.005:1);if(number(custom.min))max=Math.min(Number.MAX_VALUE,min+margin);else min=Math.max(-Number.MAX_VALUE,max-margin);}
  return max>min?{min,max}:null;
}
function dashboardConfig(value) {
  if(value?.schema_version!==1||!Array.isArray(value.panels))return null;
  const ids=new Set(),panels=[];
  for(const panel of value.panels.slice(0,24)){
    if(!identifier(panel?.id)||ids.has(panel.id)||!["line","scalar","table"].includes(panel.type)||!Array.isArray(panel.metrics))return null;
    const layout=Number.isInteger(panel.layout?.w)&&panel.layout.w>=3&&panel.layout.w<=12&&Number.isInteger(panel.layout?.h)&&panel.layout.h>=240&&panel.layout.h<=720?{w:panel.layout.w,h:panel.layout.h}:{w:6,h:320};
    ids.add(panel.id);panels.push({id:panel.id,title:typeof panel.title==="string"?panel.title.slice(0,200):"",type:panel.type,metrics:[...new Set(panel.metrics.filter(identifier))].slice(0,12),x:["step","elapsed_seconds","evaluation"].includes(panel.x)?panel.x:"step",group_by:["task","attempt","none"].includes(panel.group_by)?panel.group_by:"none",layout});const range=dashboardYConfig(panel.y_range);if(range)panels.at(-1).y_range=range;
  }
  return {schema_version:1,panels};
}
function defaultDashboard(data) {
  const benchmark=data.keys.filter(key=>/(?:^|[./])accuracy$/.test(key));
  const training=data.keys.filter(key=>lossKeys.includes(key)||lrKeys.includes(key));
  const preferred=[...benchmark,...training].slice(0,8);
  return {schema_version:1,panels:(preferred.length?preferred:data.keys.slice(0,4)).map((key,index)=>({id:`default-${index}`,title:data.definitions[key]?.label||key,type:benchmark.includes(key)||training.includes(key)?"line":"scalar",metrics:[key],x:benchmark.includes(key)?"evaluation":"step",group_by:training.includes(key)?"task":"none"}))};
}
function dashboardFormat(value,definition) {
  if(!number(value))return "—";
  if(definition?.unit==="percent")return `${Number((value*100).toFixed(2))}%`;
  if(definition?.unit==="seconds")return secondsText(value);
  return fmt(value)+(definition?.unit==="tokens_per_second"?" tok/s":"");
}
function dashboardSeries(panel,data,s) {
  const series=new Map(),tasks=new Map(),attempts=new Map();
  for(const job of list(s.jobs))if(identifier(job.job_id)&&!tasks.has(job.job_id))tasks.set(job.job_id,tasks.size+1);
  for(const point of data.points){
    if(!panel.metrics.includes(point.metric))continue;
    const group=panel.group_by==="task"?point.task_id:panel.group_by==="attempt"?point.attempt_id:null;
    // A declared grouping never pools unassigned points into an invented task.
    const identity=JSON.stringify([point.metric,group||null]);
    if(!series.has(identity)){
      let label=data.definitions[point.metric]?.label||point.metric;
      if(group&&panel.group_by==="task"){if(!tasks.has(group))tasks.set(group,tasks.size+1);label+=` · ${t("任务 {p0}",{p0:tasks.get(group)})}`;}
      if(group&&panel.group_by==="attempt"){if(!attempts.has(group))attempts.set(group,attempts.size+1);const item=list(s.research_attempts?.items).find(item=>item.attempt_id===group);label+=` · ${item?.title||t("尝试 {p0}",{p0:attempts.get(group)})}`;}
      if(!group&&panel.group_by!=="none")label+=` · ${t("未分组")}`;
      series.set(identity,{metric:point.metric,label,points:[]});
    }
    series.get(identity).points.push(point);
  }
  return [...series.values()];
}
let dashboardTooltipCount=0;
function dashboardPanel(panel,data,s) {
  const card=el("article","benchmark-card dashboard-panel"),heading=el("div","detail-heading"),series=dashboardSeries(panel,data,s),content=el("div","dashboard-chart-body");
  heading.append(el("h5","",panel.title||t("未命名面板")));card.append(heading,content);
  const baseline=metric=>{
    const formal=formalEvaluations(s).filter(j=>j.role==="baseline"&&number(flattenMetrics(j.metrics)[metric]));
    const matches=(formal.length?formal:list(s.jobs)).filter(j=>j.role==="baseline"||(j.job_id==="evaluation:initial"&&!present(j.directive_id))).map(j=>flattenMetrics(j.metrics)[metric]).filter(number);
    return matches.length===1?matches[0]:undefined;
  };
  if(panel.type==="scalar"){
    const values=el("dl","dashboard-values");
    for(const item of series){const cell=el("div");cell.append(el("dt","",item.label),el("dd","benchmark-value",dashboardFormat(item.points.at(-1)?.value,data.definitions[item.metric])));values.append(cell);}content.append(values);
  }else if(panel.type==="table"){
    const table=el("table","metric-table"),head=el("thead"),tr=el("tr"),body=el("tbody");
    for(const label of ["指标","当前值","基线","变化"]){const th=el("th","",t(label));th.scope="col";tr.append(th);}head.append(tr);
    for(const item of series){const current=item.points.at(-1)?.value,base=baseline(item.metric),definition=data.definitions[item.metric],delta=number(base)?current-base:null,row=el("tr");
      const deltaText=number(delta)?`${delta>0?"+":""}${definition?.unit==="percent"?`${Number((delta*100).toFixed(2))} pp`:fmt(delta)}`:"—";
      const cell=el("td","",deltaText);if(number(delta)&&delta!==0&&["higher","lower"].includes(definition?.direction))cell.className=(definition.direction==="higher"?delta>0:delta<0)?"metric-better":"metric-worse";
      row.append(el("td","",item.label),el("td","",dashboardFormat(current,definition)),el("td","",dashboardFormat(base,definition)),cell);body.append(row);
    }table.append(head,body);content.append(table);
  }else{
    const units=new Set(series.map(item=>data.definitions[item.metric]?.unit||"number"));
    if(units.size>1){content.append(empty(t("请选择相同单位的指标绘制折线图。")));return card;}
    const definition=data.definitions[series[0]?.metric],percent=definition?.unit==="percent";
    const rows=series.map(item=>{
      const points=item.points.map((point,index)=>({x:panel.x==="evaluation"?(point.evaluation??index+1):point[panel.x],y:point.value})).filter(point=>number(point.x));
      const unique=new Map();for(const point of points)unique.set(point.x,point);return {...item,rows:[...unique.values()].sort((a,b)=>a.x-b.x)};
    });
    const plotted=rows.flatMap(item=>item.rows);if(!plotted.length){content.append(empty(t("当前横轴暂无数据。")));return card;}
    if(percent&&plotted.some(point=>point.y<0||point.y>1)){content.append(empty(t("百分比指标应记录为 0 到 1。")));return card;}
    const bases=rows.map(item=>baseline(item.metric)).filter(number),ys=[...plotted.map(p=>p.y),...bases],range=dashboardYRange(ys,percent,panel.y_range);
    if(!range){content.append(empty(t("纵轴范围无法绘制，请调整边界。")));return card;}
    const ymin=range.min,ymax=range.max,xmin=Math.min(...plotted.map(p=>p.x)),xmax=Math.max(...plotted.map(p=>p.x));
    const x=value=>48+(value-xmin)/(xmax-xmin||1)*440,scale=Math.max(Math.abs(ymin),Math.abs(ymax),1e-300),y=value=>142-Math.max(-1e6,Math.min(1e6,(value/scale-ymin/scale)/(ymax/scale-ymin/scale)))*116;
    const svg=svgEl("svg",{"data-y-min":ymin,"data-y-max":ymax,viewBox:"0 0 520 180",role:"img","aria-label":panel.title||t("指标趋势")}),palette=["#a85739","#4f6850","#8b754e","#74617c"];
    for(const value of [ymin,ymin/2+ymax/2,ymax])svg.append(svgEl("line",{x1:44,x2:496,y1:y(value),y2:y(value),class:"benchmark-gridline"}),svgEl("text",{x:36,y:y(value)+4,"text-anchor":"end",class:"benchmark-axis"},(percent?`${fmt(value*100)}%`:fmt(value))));
    const legend=el("div","dashboard-legend"),plot=el("div","benchmark-plot"),tooltip=el("div","benchmark-tooltip"),tooltipName=el("strong"),tooltipAxis=el("span"),tooltipValue=el("strong","benchmark-tooltip-value");
    tooltip.id=`metric-tooltip-${++dashboardTooltipCount}`;tooltip.setAttribute("role","tooltip");tooltip.hidden=true;tooltip.append(tooltipName,tooltipAxis,tooltipValue);plot.append(svg,tooltip);
    const defs=svgEl("defs"),clip=svgEl("clipPath",{id:`${tooltip.id}-clip`}),marks=svgEl("g",{"clip-path":`url(#${tooltip.id}-clip)`});clip.append(svgEl("rect",{x:44,y:20,width:452,height:126}));defs.append(clip);svg.append(defs,marks);
    const axisLabel=t({step:"训练步数",elapsed_seconds:"运行时间（秒）",evaluation:"记录顺序"}[panel.x]);
    const hidePoint=()=>{tooltip.hidden=true;};
    content.addEventListener("scroll",hidePoint);
    const showPoint=(dot,item,point)=>{
      const unit=data.definitions[item.metric]?.unit,value=unit==="percent"?`${Number((point.y*100).toPrecision(12))}%`:unit==="seconds"?t("{p0} 秒",{p0:String(point.y)}):`${String(point.y)}${unit==="tokens_per_second"?" tok/s":""}`;
      tooltipName.textContent=item.label;tooltipAxis.textContent=`${axisLabel} · ${fmt(point.x)}`;tooltipValue.textContent=value;tooltip.hidden=false;
      const bounds=plot.getBoundingClientRect(),anchor=dot.getBoundingClientRect(),box=tooltip.getBoundingClientRect(),left=anchor.left-bounds.left+anchor.width/2,top=anchor.top-bounds.top;
      tooltip.style.left=`${Math.max(4,Math.min(left-box.width/2,bounds.width-box.width-4))}px`;
      tooltip.style.top=`${Math.max(4,Math.min(top-box.height-10>=4?top-box.height-10:top+anchor.height+10,bounds.height-box.height-4))}px`;
    };
    rows.forEach((item,index)=>{
      const color=palette[index%palette.length],base=baseline(item.metric);if(number(base)){const line=svgEl("line",{x1:44,x2:496,y1:y(base),y2:y(base),class:"benchmark-baseline"});line.style.stroke=color;marks.append(line);content.append(el("p","quiet-note",t("基线 {p0}",{p0:dashboardFormat(base,data.definitions[item.metric])})));}
      let segment=[];const flush=()=>{if(segment.length>1){const line=svgEl("polyline",{points:segment.join(" "),class:"benchmark-line"});line.style.stroke=color;marks.append(line);}segment=[];};
      item.rows.forEach((point,i)=>{if(panel.x==="evaluation"&&data.evaluationCount&&i&&point.x-item.rows[i-1].x>1)flush();segment.push(`${x(point.x)},${y(point.y)}`);});flush();
      item.rows.filter((_,index)=>index%Math.max(1,Math.ceil(item.rows.length/160))===0||index===item.rows.length-1).forEach(point=>{const dot=svgEl("circle",{cx:x(point.x),cy:y(point.y),r:4,class:"benchmark-point",tabindex:0,"aria-label":item.label,"aria-describedby":tooltip.id});dot.style.fill=color;
        for(const event of ["pointerenter","focus","pointerdown"])dot.addEventListener(event,()=>showPoint(dot,item,point));for(const event of ["pointerleave","blur"])dot.addEventListener(event,hidePoint);dot.addEventListener("keydown",event=>{if(event.key==="Escape")hidePoint();});marks.append(dot);});
      const entry=el("span","",`${item.label} · ${dashboardFormat(item.points.at(-1)?.value,data.definitions[item.metric])}`);entry.style.color=color;legend.append(entry);
    });
    for(const value of [...new Set([xmin,xmax])])svg.append(svgEl("text",{x:x(value),y:163,"text-anchor":"middle",class:"benchmark-axis"},fmt(value)));
    if(series.length===1){const latest=el("div","benchmark-latest");latest.append(el("strong","benchmark-value",dashboardFormat(series[0].points.at(-1)?.value,definition)));heading.append(latest);}
    content.append(plot,el("p","benchmark-axis-label",axisLabel));if(series.length>1)content.append(legend);
  }
  if(!series.length)content.append(empty(t("尚无指标数据。")));
  return card;
}
function metricDashboard(s,scope) {
  const data=dashboardData(s),storageKey=`rsimonitor-dashboard:v1:${s.run_id}`;
  const defaults=dashboardConfig(s.dashboard)||dashboardConfig(defaultDashboard(data));
  let saved=null;try{saved=dashboardConfig(JSON.parse(localStorage.getItem(storageKey)));}catch{}
  let config=dashboardConfig(scope.dashboardOverride)||saved||defaults;
  const section=el("section","benchmark-overview metric-dashboard"),heading=el("div","detail-heading"),actions=el("div","dashboard-board-actions"),add=el("button","dashboard-add",t("添加图表"));
  add.type="button";add.setAttribute("aria-haspopup","dialog");add.disabled=config.panels.length>=24;
  heading.append(el("h4","",t("指标看板")),actions);actions.append(add);section.append(heading);
  const grid=el("div","benchmark-grid dashboard-grid");section.append(grid);
  const persist=next=>{config=dashboardConfig(next);const isDefault=JSON.stringify(config)===JSON.stringify(defaults);scope.dashboardOverride=isDefault?null:config;try{if(isDefault)localStorage.removeItem(storageKey);else localStorage.setItem(storageKey,JSON.stringify(config));}catch{}};
  const repaint=()=>{scope.detailFingerprint=null;renderProject();};
  const focusControl=(id,selector)=>[...scope.details.querySelectorAll(".dashboard-panel")].find(card=>card.dataset.panelId===id)?.querySelector(selector)?.focus({preventScroll:true});
  const cardEntries=[];
  const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
  for(const panel of config.panels){
    const card=dashboardPanel(panel,data,s),tools=el("div","dashboard-card-tools"),drag=el("button","dashboard-drag","⠿"),edit=el("button","dashboard-edit",""),resize=el("button","dashboard-resize","");
    const pencil=svgEl("svg",{viewBox:"0 0 20 20","aria-hidden":"true"});pencil.append(svgEl("path",{d:"M12.8 3.2l4 4-9.5 9.5-4.8.8.8-4.8z M11 5l4 4",fill:"none",stroke:"currentColor","stroke-width":1.4}));edit.append(pencil);
    for(const [button,label] of [[drag,"拖动排序"],[edit,"编辑图表"],[resize,"调整图表大小"]]){button.type="button";button.title=t(label);button.setAttribute("aria-label",t(label));}
    edit.setAttribute("aria-haspopup","dialog");edit.addEventListener("click",()=>openEditor(panel,false));tools.append(drag,edit);card.firstElementChild.append(tools);card.append(resize);card.dataset.panelId=panel.id;card.style.gridColumn=`span ${panel.layout.w}`;card.style.height=`${panel.layout.h}px`;grid.append(card);cardEntries.push({panel,card});
    const gesture=(handle,kind)=>{
      let active=null;
      const cleanup=()=>{if(!active)return;const pointer=active.pointer;active=null;card.style.gridColumn=`span ${panel.layout.w}`;card.style.height=`${panel.layout.h}px`;card.className=card.className.replace(/ dashboard-moving/g,"");for(const item of cardEntries){delete item.card.dataset.drop;item.card.style.order="";}handle.releasePointerCapture?.(pointer);scope.dashboardInteracting=false;scope.dashboardGesture=null;};
      handle.addEventListener("pointerdown",event=>{if(event.button!==0||event.isPrimary===false||scope.dashboardInteracting)return;event.preventDefault();active={pointer:event.pointerId,x:event.clientX,y:event.clientY,panels:[...config.panels],layout:{...panel.layout},lastTarget:null};scope.dashboardInteracting=true;scope.dashboardGesture=cleanup;card.className+=" dashboard-moving";handle.setPointerCapture(event.pointerId);});
      handle.addEventListener("pointermove",event=>{
        if(!active||active.pointer!==event.pointerId)return;event.preventDefault();
        if(kind==="resize"){
          const width=grid.getBoundingClientRect().width;active.layout={w:width<720?panel.layout.w:clamp(panel.layout.w+Math.round((event.clientX-active.x)/((width+16)/12)),3,12),h:clamp(panel.layout.h+Math.round((event.clientY-active.y)/24)*24,240,720)};card.style.gridColumn=`span ${active.layout.w}`;card.style.height=`${active.layout.h}px`;
        }else{
          const target=cardEntries.find(item=>{if(item.card===card)return false;const r=item.card.getBoundingClientRect();return event.clientX>=r.left&&event.clientX<=r.right&&event.clientY>=r.top&&event.clientY<=r.bottom;});
          if(!target){active.lastTarget=null;return;}if(active.lastTarget===target.panel.id)return;active.lastTarget=target.panel.id;
          const from=active.panels.findIndex(p=>p.id===panel.id),to=active.panels.findIndex(p=>p.id===target.panel.id);active.panels.splice(from,1);active.panels.splice(to,0,panel);
          for(const item of cardEntries){item.card.style.order=String(active.panels.findIndex(p=>p.id===item.panel.id));delete item.card.dataset.drop;}target.card.dataset.drop="true";
        }
      });
      const finish=(event,cancelled)=>{if(!active||active.pointer!==event.pointerId)return;const next=active;cleanup();if(!cancelled){persist({...config,panels:kind==="resize"?config.panels.map(p=>p.id===panel.id?{...p,layout:next.layout}:p):next.panels});}else{card.style.gridColumn=`span ${panel.layout.w}`;card.style.height=`${panel.layout.h}px`;}repaint();};
      handle.addEventListener("pointerup",event=>finish(event,false));handle.addEventListener("pointercancel",event=>finish(event,true));handle.addEventListener("lostpointercapture",event=>finish(event,true));
      handle.addEventListener("keydown",event=>{
        if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(event.key))return;event.preventDefault();const panels=[...config.panels],index=panels.findIndex(p=>p.id===panel.id);
        if(kind==="resize"){const layout={...panel.layout};if(event.key==="ArrowLeft")layout.w=clamp(layout.w-1,3,12);if(event.key==="ArrowRight")layout.w=clamp(layout.w+1,3,12);if(event.key==="ArrowUp")layout.h=clamp(layout.h-24,240,720);if(event.key==="ArrowDown")layout.h=clamp(layout.h+24,240,720);panels[index]={...panel,layout};}
        else {const to=index+(["ArrowLeft","ArrowUp"].includes(event.key)?-1:1);if(to<0||to>=panels.length)return;[panels[index],panels[to]]=[panels[to],panels[index]];}
        persist({...config,panels});repaint();focusControl(panel.id,kind==="resize"?".dashboard-resize":".dashboard-drag");
      });
    };
    gesture(drag,"drag");gesture(resize,"resize");
  }
  const dialog=el("dialog","dashboard-dialog"),dialogHeader=el("header","dashboard-dialog-heading"),dialogTitle=el("h3"),close=el("button","dashboard-close","×"),body=el("div","dashboard-chart-editor"),settings=el("div","dashboard-chart-settings"),preview=el("div","dashboard-chart-preview"),footer=el("footer","dashboard-dialog-footer");
  dialogTitle.id="dashboard-config-title";dialog.setAttribute("aria-labelledby",dialogTitle.id);close.type="button";close.autofocus=true;close.setAttribute("aria-label",t("关闭"));dialogHeader.append(dialogTitle,close);body.append(settings,preview);dialog.append(dialogHeader,body,footer);section.append(dialog);
  let draft=null,editingID=null,isNew=false;
  const field=(label,input)=>{const wrap=el("label","dashboard-field");wrap.append(el("span","",t(label)),input);return wrap;};
  function openEditor(panel,adding){
    draft=JSON.parse(JSON.stringify(panel));scope.dashboardDraft=draft;editingID=panel.id;isNew=adding;
    dialogTitle.textContent=t(adding?"添加图表":"编辑图表");settings.replaceChildren();preview.replaceChildren();footer.replaceChildren();
    const renderPreview=()=>{preview.replaceChildren(el("h4","",t("预览")),dashboardPanel(draft,data,s));};
    const title=el("input");title.value=draft.title;title.maxLength=200;title.addEventListener("input",()=>{draft.title=title.value;renderPreview();});settings.append(field("图表标题",title));
    const types=el("div","dashboard-type-options");types.setAttribute("role","group");types.setAttribute("aria-label",t("图表类型"));
    const typeButtons=[];for(const [value,label] of [["line","折线图"],["scalar","数值卡"],["table","对比表"]]){const button=el("button");button.type="button";button.setAttribute("aria-pressed",String(draft.type===value));const symbol=svgEl("svg",{viewBox:"0 0 20 20",class:"dashboard-type-icon","aria-hidden":"true"});if(value==="scalar")symbol.append(svgEl("text",{x:10,y:14,"text-anchor":"middle","font-size":11,fill:"currentColor"},"123"));else symbol.append(svgEl("path",{d:value==="line"?"M2 15l5-6 4 3 7-8":"M3 3h14v14H3z M3 8h14 M3 12h14 M8 3v14",fill:"none",stroke:"currentColor","stroke-width":1.4}));button.append(symbol,el("span","",t(label)));button.addEventListener("click",()=>{draft.type=value;for(const item of typeButtons)item.button.setAttribute("aria-pressed",String(item.value===value));axis.hidden=value!=="line";yControls.hidden=value!=="line";if(value==="line")updateRange();else{rangeError.hidden=true;save.disabled=false;renderPreview();}});typeButtons.push({value,button});types.append(button);}settings.append(types);
    const picker=el("div","dashboard-metric-picker"),search=el("input"),metricList=el("div","dashboard-metric-list"),metricHeading=el("div","detail-heading"),count=el("span","dashboard-metric-count");search.type="search";search.placeholder=t("搜索指标");search.setAttribute("aria-label",t("搜索指标"));metricHeading.append(el("span","",t("指标")),count);picker.append(metricHeading,search,metricList);settings.append(picker);
    const renderMetrics=()=>{
      count.textContent=`${draft.metrics.length} / 12`;metricList.replaceChildren();const query=(search.value||"").trim().toLocaleLowerCase(),keys=[...new Set([...data.keys,...draft.metrics])].filter(key=>`${data.definitions[key]?.label||""} ${key}`.toLocaleLowerCase().includes(query));
      for(const key of keys){const label=el("label","dashboard-metric-option"),checkbox=el("input");checkbox.type="checkbox";checkbox.checked=draft.metrics.includes(key);checkbox.disabled=!checkbox.checked&&draft.metrics.length>=12;label.title=key;checkbox.addEventListener("change",()=>{draft.metrics=checkbox.checked?[...new Set([...draft.metrics,key])].slice(0,12):draft.metrics.filter(metric=>metric!==key);renderMetrics();refreshRange();updateRange();});label.append(checkbox,el("span","",data.definitions[key]?.label||key));metricList.append(label);}if(!keys.length)metricList.append(el("p","dashboard-editor-empty",t("没有匹配的指标")));
    };search.addEventListener("input",renderMetrics);
    const select=(values,current,change)=>{const input=el("select");for(const [value,label] of values){const option=el("option","",t(label));option.value=value;input.append(option);}input.value=current;input.addEventListener("change",()=>{change(input.value);renderPreview();});return input;};
    const axis=field("横轴",select([["step","训练步数"],["elapsed_seconds","运行时间（秒）"],["evaluation","记录顺序"]],draft.x,value=>draft.x=value));axis.hidden=draft.type!=="line";settings.append(axis,field("分组",select([["none","不分组"],["task","按任务"],["attempt","按研究尝试"]],draft.group_by,value=>draft.group_by=value)));
    const yControls=el("div","dashboard-y-range"),yTitle=el("span","",t("纵轴范围")),minInput=el("input"),maxInput=el("input"),minField=field("最小值",minInput),maxField=field("最大值",maxInput),rangeError=el("p","dashboard-range-error");
    yControls.hidden=draft.type!=="line";rangeError.hidden=true;rangeError.setAttribute("role","alert");for(const input of [minInput,maxInput]){input.type="number";input.step="any";input.placeholder=t("自动");}
    const refreshRange=()=>{const percent=draft.metrics.length>0&&draft.metrics.every(key=>data.definitions[key]?.unit==="percent");minField.firstElementChild.textContent=t(percent?"最小值（%）":"最小值");maxField.firstElementChild.textContent=t(percent?"最大值（%）":"最大值");for(const [key,input] of [["min",minInput],["max",maxInput]])input.value=number(draft.y_range?.[key])?(percent?String(Number((draft.y_range[key]*100).toPrecision(15))):String(draft.y_range[key])):"";};
    const updateRange=()=>{const percent=draft.metrics.length>0&&draft.metrics.every(key=>data.definitions[key]?.unit==="percent"),range={};let error="";for(const [key,input] of [["min",minInput],["max",maxInput]]){if(input.validity?.badInput){error=t("请输入有效的数值。");break;}if(input.value.trim()!==""){const value=Number(input.value)/(percent?100:1);if(!number(value)){error=t("请输入有效的数值。");break;}range[key]=value;}}if(!error&&number(range.min)&&number(range.max)&&(!(range.max>range.min)||!number(range.max-range.min)))error=t("最大值必须大于最小值。");rangeError.textContent=error;rangeError.hidden=!error;save.disabled=!!error;if(!error){if(Object.keys(range).length)draft.y_range=range;else delete draft.y_range;renderPreview();}};
    minInput.addEventListener("input",updateRange);maxInput.addEventListener("input",updateRange);yControls.append(yTitle,minField,maxField,rangeError);settings.append(yControls);refreshRange();
    const remove=el("button","dashboard-delete",t("删除图表")),buttons=el("div","dashboard-actions"),cancel=el("button","",t("取消")),save=el("button","dashboard-save",t("保存"));for(const button of [remove,cancel,save])button.type="button";remove.hidden=adding;
    const apply=panels=>{persist({...config,panels});scope.renderPending=true;dialog.close();};remove.addEventListener("click",()=>apply(config.panels.filter(p=>p.id!==editingID)));save.addEventListener("click",()=>{if(!save.disabled)apply(adding?[...config.panels,draft]:config.panels.map(p=>p.id===editingID?draft:p));});cancel.addEventListener("click",()=>dialog.close());buttons.append(cancel,save);footer.append(remove,buttons);
    renderMetrics();renderPreview();dialog.showModal();scope.dashboardEditing=true;scope.dashboardDialog=dialog;
  }
  add.addEventListener("click",()=>{if(config.panels.length>=24)return;const ids=new Set(config.panels.map(p=>p.id));let index=1;while(ids.has(`custom-${index}`))index++;const metric=data.keys[0];openEditor({id:`custom-${index}`,title:metric?data.definitions[metric].label:t("新面板"),type:"line",metrics:metric?[metric]:[],x:"step",group_by:"none",layout:{w:6,h:320}},true);});
  close.addEventListener("click",()=>dialog.close());dialog.addEventListener("cancel",event=>{event.preventDefault();dialog.close();});dialog.addEventListener("click",event=>{if(event.target!==dialog)return;const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();});
  dialog.addEventListener("close",()=>{draft=null;scope.dashboardDraft=null;scope.dashboardEditing=false;scope.dashboardDialog=null;if(scope===state.activeScope&&state.page==="experiment"){if(scope.renderPending){scope.renderPending=false;repaint();}if(!isNew&&config.panels.some(p=>p.id===editingID))focusControl(editingID,".dashboard-edit");else scope.details.querySelector(".dashboard-add")?.focus({preventScroll:true});}});
  return section;
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
  // Use experiment activity, never catalog fetch time, as project recency.
  const updated = row => {
    const time = typeof row.updated_at === "string" ? Date.parse(row.updated_at) : NaN;
    return Number.isFinite(time) ? time : -Infinity;
  };
  const groups = [...projects.values()];
  for (const group of groups) group.runs.sort((a,b) => updated(b)-updated(a) || 0);
  return groups.sort((a,b) => updated(b.runs[0] || {})-updated(a.runs[0] || {}) || 0);
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
  scope.dashboardGesture?.();scope.dashboardGesture=null;scope.dashboardInteracting=false;
  scope.dashboardEditing=false;scope.dashboardDraft=null;scope.renderPending=false;
  if(scope.dashboardDialog?.open)scope.dashboardDialog.close();scope.dashboardDialog=null;
  scope.graph?.dispose?.();scope.graph=null;scope.attemptGraph?.dispose?.();scope.attemptGraph=null;scope.attemptView=null;scope.detailFingerprint=null;scope.renderPending=false;
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
function executionGroups(s,model=RSIDecisions.layout(list(s.decisions))) {
  return model.nodes.filter(n=>n.row).map(n=>({...n,rows:n.rows||[n.row],members:n.members||[n]}));
}
function attemptTitle(group) {
  return group.type==="attempt" ? group.title || t("未命名研究尝试") : decisionTitle(group.row);
}
function graphBusy(scope) {
  return scope?.dashboardEditing || scope?.dashboardInteracting || [scope.graphView,scope.attemptView].some(view=>view?.interacting||view?.animating);
}
function graphTools(current) {
  const controls=el("div","trajectory-tools");
  for(const [text,title,action] of [["−","缩小",()=>current()?.zoomBy(.8)],["+","放大",()=>current()?.zoomBy(1.25)],["⤢","适应全部节点",()=>current()?.fit()],["↺","重排",()=>current()?.reset()]]){
    const button=el("button","",text);button.type="button";button.title=t(title);button.setAttribute("aria-label",t(title));button.addEventListener("click",action);controls.append(button);
  }
  return controls;
}
function graphIssues(model,scope,runID) {
  const types=new Set(model.issues.map(i=>i.type)),content=el("div");
  const messages={"invalid-attempt":"部分记录的研究尝试身份无效，已保留为未分组记录。","missing-title":"部分研究尝试未记录标题。","conflicting-title":"同一研究尝试记录了不同标题；节点显示首个标题，各决策的原始标题保持不变。",unresolved:"部分关系引用缺失或不唯一，未推断关联。","raw-cycle":"无效的循环决策关系已忽略，其他关系保留。","self-reference":"无效的决策自引用已忽略。","contraction-cycle":"尝试间关系形成循环，保留研究尝试节点与已记录关系；可在尝试详情和实验日志中核对原始决策。"};
  const shown=new Set();
  for(const type of types){const message=messages[type];if(message&&!shown.has(message)){content.append(el("p","quiet-note",t(message)));shown.add(message);}}
  return content.childElementCount?retainedDetails(JSON.stringify([runID,"graph-issues"]),scope.flowOpen,t("分组与关系记录需核对"),content):null;
}
const changeLabels = {instructions:"提示词内容",method:"训练方法",recipe_ref:"训练方案",protocol_ref:"评测方案",candidate_id:"待验证版本",apply_result:"训练后自动应用",steps:"训练步数",lr:"学习率",seed:"随机种子",weight_decay:"权重衰减",warmup_ratio:"预热比例",updates:"更新次数",components:"改动部分",query:"检索内容",url:"阅读地址",artifact_id:"资料引用",offset:"起始位置",limit:"数量"};
function changesView(row,s) {
  const opaque=value=>typeof value==="string" && /(?:^|:)[a-f\d]{32,}$|^[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12}$/i.test(value.trim());
  const changes=list(row.changes).filter(c=>typeof c.field==="string"&&!['base_version','artifact_id'].includes(c.field))
    .map(c=>{
      if(c.field==="candidate_id"||["instructions","query","url"].includes(c.field))return c;
      const shown={...c};for(const key of ["before","after"])if(opaque(shown[key]))delete shown[key];return shown;
    }).filter(c=>typeof c.after==="string"||typeof c.before==="string");
  if(!changes.length)return null;
  const box=el("div","decision-changes");box.append(el("p","flow-label",t("提交内容")));
  for(const change of changes){
    const item=el("div",[change.before,change.after].some(v=>typeof v==="string"&&v.length>120)||["instructions","query","url"].includes(change.field)?"change-item is-text":"change-item");item.append(el("h5","",t(changeLabels[change.field])||change.field));
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
function feedbackRouting(row) {
  const parts=[];
  if(row.feedback_route==="context")parts.push(t("即时上下文"));
  if(row.feedback_route==="memory")parts.push(t("记忆整理请求"));
  if(row.archive_status==="archived")parts.push(t("已归档"));
  const memory={requested:"记忆整理请求",queued:"记忆整理排队中",pending:"记忆整理排队中",running:"记忆整理中",processing:"记忆整理中",completed:"记忆整理完成",failed:"记忆整理失败"}[row.memory_status];
  if(memory)parts.push(t(memory));
  return parts.length?el("p","feedback-routing",[...new Set(parts)].join(" · ")):null;
}
function runObservations(s) {
  const section=el("section","run-observations"),metrics=metricStrip(object(s.current).metrics);if(metrics.childElementCount)section.append(metrics);
  if(present(s.lifecycle?.stop_reason))section.append(el("p","run-stop",t({host_deadline_or_signal:"到达宿主截止时间或收到停止信号",host_deadline:"到达设定时间",host_signal:"收到停止信号",inference_admission_limit:"达到模型调用额度"}[s.lifecycle.stop_reason])||s.lifecycle.stop_reason));
  const state=el("dl","observation-summary");
  const entry=(label,value)=>{const pair=el("div");pair.append(el("dt","",t(label)),el("dd","",value));state.append(pair);};
  if(list(s.decisions).some(row=>["invoke","explore"].includes(row.kind))){
    entry("正式评测",Array.isArray(s.evaluations)?t("{p0} 次已报告",{p0:formalEvaluations(s).length}):t("未报告"));
    entry("训练指标",list(s.jobs).some(j=>historyRows(j).some(p=>Object.keys(flattenMetrics(p.metrics)).some(key=>lossKeys.includes(key)))||Object.keys(flattenMetrics(j.metrics)).some(key=>lossKeys.includes(key)))?t("已报告"):t("未报告"));
    const evaluated=formalEvaluations(s),adopted=evaluated.some(row=>row.adopted===true)||list(s.candidates).some(row=>ACCEPTED.has(statusKey(row.status)));
    entry("结果采纳",adopted?t("已报告采纳"):evaluated.length&&evaluated.every(row=>row.adopted===false)?t("未采纳"):t("未报告"));
    entry("权重应用",typeof s.weight_applied==="boolean"?t(s.weight_applied?"已报告应用":"未应用"):t("未报告"));
    if(runTerminal(s))entry("资源清理",typeof s.lifecycle?.cleanup?.confirmed==="boolean"?t(s.lifecycle.cleanup.confirmed?"已确认":"未确认"):t("未报告"));
  }
  if(state.childElementCount)section.append(state);
  if(s.history_window?.truncated===true)section.append(el("p","quiet-note history-window",t("显示最近 {p0} 条决策，共 {p1} 条。",{p0:list(s.decisions).length,p1:s.history_window.total??t("未报告")})));
  return section;
}
function decisionStep(row,s,index) {
  const f=feedbackFor(row,s);
  const step=el("li","flow-step"),marker=el("span","flow-index",String(index+1).padStart(2,"0"));
  const decision=el("div","flow-decision"),feedback=el("div","flow-feedback"),metrics=el("div","flow-metrics");
  const meta=el("p","quiet-note",`${dateText(row.timestamp)}${number(row.round)?t(" · 第 {p0} 轮", {p0: row.round}):""}`);
  const title=decisionTitle(row);decision.append(el("h4","",title),meta);
  if(present(row.reason))decision.append(el("p","flow-label",t(["invoke","explore"].includes(row.kind)?"模型判断":"决策理由")),el("p","decision-reason",row.reason));
  if(present(row.strategy)&&row.strategy!==row.reason&&row.strategy!==decisionAction(row)&&asText(row.strategy,"").trim()!==title)decision.append(el("p","decision-reason",row.strategy));
  const changes=changesView(row,s);if(changes)decision.append(changes);
  if(["evaluate","evaluation","candidate_evaluation"].includes(decisionAction(row))){
    const component=list(row.changes).find(c=>c.field==="components")?.after;
    const prompt=typeof component==="string"&&/prompt|instructions/.test(component);
    decision.append(el("p","evaluation-purpose",prompt?t("检验新提示词的效果。"):t("检验待验证版本的效果。")));
  }
  const explicit=concreteFeedback(row),error=object(row.feedback).error||row.error;
  if(explicit||error){feedback.append(el("p","flow-label",t(["invoke","explore"].includes(row.kind)?"即时结果":"实际结果")));if(explicit)feedback.append(el("p","result-summary",explicit));if(error&&error!==explicit)feedback.append(el("p","result-error",error));}
  else if(!DONE.has(statusKey(f.status))||row.kind==="review"||decisionAction(row)==="finish")feedback.append(el("p","",f.text));
  const routing=feedbackRouting(row);if(routing)feedback.append(routing);
  if(row.kind==="review"&&f.job)feedback.append(el("p","quiet-note",t("关联任务当前{p0}；不代表复查当时状态。", {p0: statusText(f.job.status)})));
  if(row.kind!=="review"&&f.job&&present(f.job.step))feedback.append(progressView(f.job.step,f.job.total_steps));
  if(Object.keys(flattenMetrics(f.metrics)).length)metrics.append(el("p","flow-label",row.kind==="review"?t("复查时指标"):t("指标")),comparisonView(f.metrics));
  if(f.job&&number(f.job.elapsed_seconds))feedback.append(el("p","quiet-note",t("耗时 {p0}",{p0:secondsText(f.job.elapsed_seconds)})));
  step.append(marker,decision);if(feedback.childElementCount)step.append(feedback);if(metrics.childElementCount)step.append(metrics);
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
  if(results.childElementCount)step.append(results);return step;
}
function experimentLogs(s,scope,projection) {
  const records=projection.full.nodes.filter(n=>n.row);
  if(!records.length)return null;
  const content=el("div","experiment-logs"),buttons=el("div","experiment-log-records"),items=[];
  scope.logSelected ||= new Map();
  if(!records.some(n=>n.key===scope.logSelected.get(s.run_id)))scope.logSelected.set(s.run_id,null);
  const fill=()=>{
    const position={left:window.scrollX,top:window.scrollY},height=content.getBoundingClientRect?.().height;
    if(height)content.style.minHeight=`${height}px`;
    for(const {record,button,body} of items){
      const selected=record.key===scope.logSelected.get(s.run_id);button.setAttribute("aria-pressed",String(selected));button.setAttribute("aria-expanded",String(selected));body.replaceChildren();
      if(selected){const group=projection.nodes.find(n=>n.members?.some(member=>member.key===record.key));if(group)body.append(el("li","quiet-note log-membership",attemptTitle(group)));body.append(decisionStep(record.row,s,records.indexOf(record)));}
    }
    content.style.minHeight="";
    if(window.scrollX!==position.left||window.scrollY!==position.top)window.scrollTo({...position,behavior:"instant"});
  };
  for(const [index,record] of records.entries()){
    const item=el("article","experiment-log-item"),button=el("button","experiment-log-record"),body=el("ol","flow-steps log-record-body");button.type="button";button.dataset.key=record.key;
    button.append(el("span","log-number",String(index+1).padStart(2,"0")),el("time","log-time",dateText(record.row.timestamp)),el("span","log-title",decisionTitle(record.row)));if(present(record.row.status))button.append(observedBadge(record.row.status,s));
    button.addEventListener("click",()=>{scope.logSelected.set(s.run_id,scope.logSelected.get(s.run_id)===record.key?null:record.key);fill();});item.append(button,body);buttons.append(item);items.push({record,button,body});
  }
  fill();content.append(buttons);
  return retainedDetails(JSON.stringify([s.run_id,"experiment-logs"]),scope.flowOpen,t("实验日志 · {p0}",{p0:records.length}),content);
}

function executionSequence(s, scope) {
  scope.flowOpen ||= new Set(); scope.flowSelected ||= new Map();
  const model=RSIDecisions.attempts(list(s.decisions),s.research_attempts),projection=model;
  const groups=executionGroups(s,model);
  const section=el("section","execution-sequence");section.id="execution-flow";
  scope.graphView ||= {};
  scope.attemptSteps ||= new Map();scope.attemptViews ||= new Map();
  if(!groups.some(g=>g.key===scope.flowSelected.get(s.run_id)))scope.flowSelected.set(s.run_id,null);
  for(const [key,chosen] of scope.attemptSteps){
    if(!groups.find(g=>g.key===key)?.members.some(n=>n.key===chosen))scope.attemptSteps.set(key,null);
  }
  const toolbar=el("div","trajectory-heading");toolbar.append(el("h4","",t("研究尝试")));
  section.append(toolbar);
  const issues=graphIssues(model,scope,s.run_id);if(issues)section.append(issues);
  if(!projection.attemptCount){
    disposeGraph(scope);
    section.append(el("p","attempt-empty",t("未记录研究尝试")));
    const logs=experimentLogs(s,scope,projection);if(logs)section.append(logs);
    return section;
  }
  const shell=el("div","trajectory-shell"),navigation=el("div","flow-map");
  navigation.setAttribute("aria-label",t("研究尝试"));shell.append(navigation,graphTools(()=>scope.graph));section.append(shell);
  const entries=el("ol","flow-steps");section.append(entries);
  const onIdle=()=>{if(scope===state.activeScope&&state.page==="experiment"&&scope.renderPending&&!graphBusy(scope)){scope.renderPending=false;renderProject();}};
  const edgeLabel=kind=>t({sequence:"记录顺序，不表示因果",dependency:"依赖已完成决策",adoption:"已确认采纳结果",root:"布局起点，不表示依赖",source:"引用的决策"}[kind]);
  const fill=(restoreFocus=false,childFocus=false)=>{
    const position=typeof window!=="undefined"?{left:window.scrollX,top:window.scrollY}:null;
    const minHeight=section.style.minHeight||"";
    const height=section.getBoundingClientRect?.().height;
    if(height)section.style.minHeight=`${height}px`;
    const scroll={left:navigation.scrollLeft,top:navigation.scrollTop};
    scope.attemptGraph?.dispose?.();scope.attemptGraph=null;scope.attemptView=null;
    entries.replaceChildren();
    let selected=scope.flowSelected.get(s.run_id);
    if(!groups.some(g=>g.key===selected))selected=null;
    scope.flowSelected.set(s.run_id,selected);
    scope.graph?.dispose?.();
    scope.graph=RSIDecisions.render({
      view:scope.graphView,onIdle,model,rows:list(s.decisions),selected,compact:true,
      startLabel:`0 · ${t("实验起点")}`,label:(row,index)=>{
        const group=groups.find(g=>g.index===index);
        return `${groups.indexOf(group)+1} · ${attemptTitle(group)}`;
      },edgeLabel,onSelect:(key,_row,_index,keyboard)=>{scope.flowSelected.set(s.run_id,key);fill(keyboard===true);},
    });
    navigation.replaceChildren(scope.graph);navigation.scrollLeft=scroll.left;navigation.scrollTop=scroll.top;
    for(const [i,g] of groups.entries()){
      const key=g.key;if(key!==selected)continue;
      let row=g.row,detailIndex=i;
      if(g.type==="attempt"){
        const child=RSIDecisions.subpath(projection,key),members=child.nodes.filter(n=>n.row);
        let chosen=scope.attemptSteps.has(key)?scope.attemptSteps.get(key):members[0]?.key;
        if(!members.some(n=>n.key===chosen))chosen=null;
        scope.attemptSteps.set(key,chosen);row=members.find(n=>n.key===chosen)?.row;
        detailIndex=members.findIndex(n=>n.key===chosen);
        const stages=el("li","attempt-stages");stages.append(el("h4","attempt-title",attemptTitle(g)));
        if(g.interpretation){
          const info=g.interpretation;
          if(present(info.summary))stages.append(el("p","result-summary attempt-summary",info.summary));
          if(g.missingDecisionIDs?.length)stages.append(el("p","quiet-note",t("部分详情未包含在当前快照中（{p0} 条）。",{p0:g.missingDecisionIDs.length})));

        }
        if(!members.length){entries.append(stages);continue;}
        const childHeading=el("div","attempt-heading");childHeading.append(el("p","flow-label",t("尝试内决策")),el("p","quiet-note",t("{p0} 次决策",{p0:members.length})));stages.append(childHeading);
        const childShell=el("div","trajectory-shell attempt-shell"),childMap=el("div","flow-map attempt-map");
        childMap.setAttribute("aria-label",t("尝试内决策"));
        if(!scope.attemptViews.has(key))scope.attemptViews.set(key,{});scope.attemptView=scope.attemptViews.get(key);
        scope.attemptGraph=RSIDecisions.render({model:child,selected:chosen,view:scope.attemptView,compact:true,height:280,
          startLabel:t("尝试内决策"),label:(record,index)=>`${members.findIndex(n=>n.index===index)+1} · ${decisionTitle(record)}`,
          edgeLabel,onIdle,onSelect:(childKey,_row,_index,keyboard)=>{scope.attemptSteps.set(key,childKey);fill(false,keyboard===true);}});
        childMap.append(scope.attemptGraph);childShell.append(childMap,graphTools(()=>scope.attemptGraph));stages.append(childShell);entries.append(stages);
        if(!row){stages.append(el("p","quiet-note selection-hint",t("选择节点查看决策、任务与结果。")));continue;}
      }
      entries.append(decisionStep(row,s,detailIndex));
    }
    if(groups.length&&!selected)entries.append(el("li","quiet-note selection-hint",t("选择节点查看决策、任务与结果。")));
    if(restoreFocus)navigation.querySelector?.('[aria-pressed="true"]')?.focus({preventScroll:true});
    if(childFocus)entries.querySelector?.('[aria-pressed="true"]')?.focus({preventScroll:true});
    section.style.minHeight=minHeight;
    if(position&&(window.scrollX!==position.left||window.scrollY!==position.top))window.scrollTo({...position,behavior:"instant"});
  };
  fill();
  const logs=experimentLogs(s,scope,projection);if(logs)section.append(logs);
  return section;
}
function jobRecord(job, compact=false, snapshot={}) {
  const card=el("article",compact?"job-record active-job":"job-record"),heading=el("div","detail-heading");
  heading.append(el("h4","",actionText(job.kind)),observedBadge(job.status,snapshot));card.append(heading);
  if(present(job.step))card.append(progressView(job.step,job.total_steps));
  if(number(job.elapsed_seconds))card.append(el("p","quiet-note",t("耗时 {p0}",{p0:secondsText(job.elapsed_seconds)})));
  if(!compact&&present(job.error))card.append(el("p","result-error",job.error));
  if(!compact)card.append(metricsDetail(job.metrics));
  return card;
}
function runDetail(node, scope) {
  const entry=scope.cache.get(node.runId),s=entry?.snapshot,box=el("div","run-detail"); scope.flowOpen ||= new Set();
  const heading=el("div","detail-heading"),identity=el("div");
  identity.append(el("h3","",t("研究脉络")),el("p","quiet-note",t("{p0} 次决策",{p0:list(s?.decisions).length})));
  heading.append(identity,runBadge(s?.status||node.row.status));box.append(heading);
  if(present(s?.error))box.append(el("p","result-error",s.error));
  if(entry?.status==="pending"&&!s)box.append(el("p","load-message",t("正在读取决策与反馈…")));
  if(entry?.status==="error"){
    box.append(el("p","load-message error",t("{p0}读取失败：{p1}", {p0: s?t("保留上次快照。"):"", p1: errorText(entry.errorInfo || {error:entry.error})})));
    const retry=el("button","",t("重试读取"));retry.addEventListener("click",()=>loadRun(node.runId,scope,true));box.append(retry);
  }
  if(s){
    const jobs=list(s.jobs),terminal=runTerminal(s);
    const active=terminal?[]:jobs.filter(j=>ACTIVE.has(statusKey(j.status)));
    if(active.length){
      const overview=el("section","active-jobs");overview.setAttribute("aria-label",t("当前任务"));
      overview.append(el("h4","",t("当前任务 · {p0}",{p0:active.length})));
      const cards=el("div","active-job-list");for(const job of active)cards.append(jobRecord(job,true,s));overview.append(cards);box.append(overview);
    }
    if(terminal&&jobs.some(j=>ACTIVE.has(statusKey(j.status))))box.append(el("p","quiet-note stale-job-warning",t("实验已结束；未刷新的任务仅保留最后观测状态。")));
    box.append(runObservations(s));
    const benchmarks=metricDashboard(s,scope);if(benchmarks)box.append(benchmarks);
    box.append(executionSequence(s,scope));
    const unlinkedCandidates=list(s.candidates).filter(c=>!list(s.decisions).some(d=>relatedResults(d,s).some(r=>r.type==="candidate"&&r.row===c)));
    if(unlinkedCandidates.length){
      const candidates=el("div");
      unlinkedCandidates.forEach((c,i)=>{const card=el("article","candidate-card");card.append(el("h4","",t("待验证版本 {p0}", {p0: i+1})),badge(c.status),comparisonView(c.metrics));candidates.append(card);});box.append(retainedDetails(JSON.stringify([node.runId,"candidates"]),scope.flowOpen,t("版本结果 · {p0}",{p0:unlinkedCandidates.length}),candidates));
    }
    const baselines=jobs.filter(j=>j.role==="baseline" || (j.job_id==="evaluation:initial" && !present(j.directive_id)));
    if(baselines.length){
      const records=el("div");for(const job of baselines)records.append(jobRecord(job,false,s));
      box.append(retainedDetails(JSON.stringify([node.runId,"baseline"]),scope.flowOpen,t("实验基线"),records));
    }
    const unlinked=jobs.filter(j=>!active.includes(j)&&!baselines.includes(j)&&!list(s.decisions).some(d=>linkedJob(d,s)===j));
    if(unlinked.length){
      const history=el("div","unlinked-job-history");for(const job of unlinked)history.append(jobRecord(job,false,s));
      box.append(retainedDetails(JSON.stringify([node.runId,"unlinked-jobs"]),scope.flowOpen,t("关联诊断 · {p0}",{p0:unlinked.length}),history));
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
  if(graphBusy(scope)){scope.renderPending=true;return;}
  $("project-name").textContent = experimentName(row); $("project-description").textContent = projectName(project); $("project-description").hidden=false;
  const host=$("project-content"),details=scope.details || (scope.details=el("section","project-detail"));
  details.setAttribute("aria-label",t("本次运行的决策"));
  const entry=scope.cache.get(row.run_id);
  const fingerprint=JSON.stringify([RSII18n.getLanguage(),row,entry?.snapshot,entry?.status==="pending"&&entry?.snapshot?"ready":entry?.status,entry?.error||"",state.projects.map(p=>[p.project_id,projectName(p)])]);
  if(scope.detailFingerprint!==fingerprint){
    const position=typeof window!=="undefined"?{left:window.scrollX,top:window.scrollY}:null;
    for(const d of details.querySelectorAll?.("details[data-retain-key]")||[]){
      scope.flowOpen ||= new Set();if(d.open)scope.flowOpen.add(d.dataset.retainKey);else scope.flowOpen.delete(d.dataset.retainKey);
    }
    const oldMap=details.querySelector?.(".flow-map"),scroll={left:oldMap?.scrollLeft||0,top:oldMap?.scrollTop||0};
    disposeGraph(scope);
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
      const head=el("div","experiment-card-heading");head.append(el("h3","",experimentName(row)),runBadge(row.status));card.append(head);
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
    notice(() => controller.signal.aborted ? t("监控连接超时，正在重试。") : errorText(error,"无法连接监控服务。"), true);
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
  if ($("demo")) $("demo").textContent = state.demo ? t("返回") : t("示例");
  $("demo")?.setAttribute("aria-pressed", String(state.demo));
  $("empty").hidden = false;
  refresh();
}
$("back-projects").addEventListener("click",()=>showManagement());
if(typeof window!=="undefined")window.addEventListener("popstate",()=>{
  const params=new URLSearchParams(location.search),id=params.get("run");
  if(id&&ownerOfRun(id))selectRun(id,false);
  else showManagement(false);
});
$("demo")?.addEventListener("click", switchDemo);
$("empty-demo")?.addEventListener("click", () => {
  if (!state.demo) switchDemo();
  else refresh();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { clearTimeout(state.timer); if(!state.activeScope?.dashboardEditing)disposeGraph(state.activeScope); }
  else if(state.demo)renderProject();
  else if (state.auto) refresh();
});
function renderLanguage() {
  RSII18n.applyStatic(document);
  for(const lang of ["zh","en"])$("lang-"+lang)?.setAttribute("aria-pressed",String(RSII18n.getLanguage()===lang));
  if ($("demo")) $("demo").textContent=state.demo?t("返回"):t("示例");
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
