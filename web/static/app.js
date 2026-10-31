"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  demo: false,
  runId: "",
  realRunId: "",
  snapshot: null,
  controller: null,
  epoch: 0,
  timer: null,
  auto: true,
  eventLimit: 30,
  openEvents: new Set(),
};
const ACTIVE = new Set([
  "running",
  "active",
  "training",
  "evaluating",
  "queued",
  "pending",
  "starting",
  "submitted",
  "in_progress",
]);
const DONE = new Set([
  "completed",
  "complete",
  "succeeded",
  "success",
  "done",
  "finished",
]);
const FAILED = new Set(["failed", "failure", "error", "cancelled", "canceled"]);
const ACCEPTED = new Set(["accepted", "adopted", "promoted", "bound"]);
const REJECTED = new Set(["rejected", "discarded", "failed"]);
const PENDING = new Set([
  "pending",
  "proposed",
  "evaluating",
  "ready",
  "validated",
  "awaiting_evaluation",
  "awaiting_review",
  "training",
]);
const labels = {
  running: "运行中",
  active: "进行中",
  training: "训练中",
  evaluating: "评测中",
  queued: "排队中",
  pending: "等待中",
  starting: "启动中",
  submitted: "已提交",
  in_progress: "进行中",
  completed: "已完成",
  complete: "已完成",
  succeeded: "已完成",
  success: "已完成",
  done: "已完成",
  finished: "已完成",
  failed: "失败",
  failure: "失败",
  error: "错误",
  cancelled: "已取消",
  canceled: "已取消",
  idle: "空闲",
  waiting: "等待中",
  paused: "已暂停",
  stopped: "已停止",
  accepted: "已采纳",
  adopted: "已采纳",
  promoted: "已晋升",
  bound: "已绑定",
  rejected: "未采纳",
  discarded: "已放弃",
  proposed: "已提出",
  ready: "待审阅",
  validated: "已验证",
  awaiting_evaluation: "待评测",
  awaiting_review: "待审阅",
  unavailable: "快照不可用",
};
const asText = (v, fallback = "—") =>
  ["string", "number", "boolean"].includes(typeof v) ? String(v) : fallback;
const list = (value) =>
  Array.isArray(value)
    ? value.filter((x) => x && typeof x === "object" && !Array.isArray(x))
    : [];
const object = (value) =>
  value && typeof value === "object" && !Array.isArray(value) ? value : {};
function el(tag, className = "", text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = asText(text);
  return e;
}
function statusKey(value) {
  return asText(value, "unknown").toLowerCase();
}
function badge(value) {
  const key = statusKey(value),
    tone =
      FAILED.has(key) || REJECTED.has(key)
        ? "failure"
        : DONE.has(key) || ACCEPTED.has(key)
          ? "success"
          : ACTIVE.has(key)
            ? "active"
            : "";
  return el("span", `badge ${tone}`, labels[key] || asText(value, "未报告"));
}
function empty(text) {
  return el("p", "subtle-empty", text);
}
function dateText(value, withDate = true) {
  if (!value) return "时间未报告";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return asText(value);
  return new Intl.DateTimeFormat("zh-CN", {
    ...(withDate ? { month: "2-digit", day: "2-digit" } : {}),
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(d);
}
function connection(text, tone = "") {
  $("connection").textContent = text;
  $("connection").className = `connection ${tone}`;
}
function notice(text, error = false) {
  $("notice").replaceChildren();
  $("notice").hidden = !text;
  $("notice").className = `notice${error ? " error" : ""}`;
  if (text) $("notice").append(document.createTextNode(text));
}
function showEmpty(title, description) {
  $("dashboard").hidden = true;
  $("empty").hidden = false;
  $("empty-title").textContent = title;
  $("empty-description").textContent = description;
}
function renderStats(s) {
  const jobs = list(s.jobs),
    candidates = list(s.candidates),
    events = list(s.events),
    loops = object(s.loops);
  const items = [
    [
      "活跃循环",
      Object.values(loops).filter((x) => ACTIVE.has(statusKey(x?.status)))
        .length,
      "/ 2 loops",
    ],
    [
      "进行中任务",
      jobs.filter((j) => ACTIVE.has(statusKey(j.status))).length,
      `${jobs.length} 项任务`,
    ],
    [
      "候选版本",
      candidates.length,
      `${candidates.filter((c) => ACCEPTED.has(statusKey(c.status))).length} 已采纳`,
    ],
    ["已记录事件", events.length, "当前运行"],
  ];
  $("stats").replaceChildren(
    ...items.map(([label, n, sub]) => {
      const item = el("div", "stat"),
        value = el("div", "stat-value");
      value.append(el("strong", "", n), el("span", "", sub));
      item.append(el("p", "stat-label", label), value);
      return item;
    }),
  );
}
function renderLoops(s) {
  const loops = object(s.loops);
  $("loops").replaceChildren(
    ...[
      ["decision_making", "决策循环", "Decision making", "01"],
      ["exploration", "探索循环", "Exploration", "02"],
    ].map(([key, title, en, n]) => {
      const loop = object(loops[key]),
        card = el("article", "loop-card"),
        top = el("div", "loop-top");
      top.append(el("span", "loop-number", n), badge(loop.status));
      card.append(
        top,
        el("h3", "", title),
        el("p", "loop-name", en),
        el("p", "loop-summary", asText(loop.summary, "尚无循环摘要。")),
      );
      return card;
    }),
  );
  const models = object(s.models);
  $("bindings").replaceChildren(
    ...[
      ["decision_maker", "Decision maker · 决策模型"],
      ["actor", "Actor · 执行模型"],
    ].map(([key, label]) => {
      const model = object(models[key]),
        row = el("div", "binding"),
        rev = el("div", "binding-rev");
      rev.append(
        el("span", "", "revision"),
        el("code", "", asText(model.revision, "未报告")),
      );
      row.append(
        el("p", "binding-role", label),
        el("p", "binding-model", asText(model.model_id, "尚未绑定")),
        rev,
      );
      return row;
    }),
  );
}
function jobGroup(job) {
  const key = statusKey(job.status);
  return ACTIVE.has(key)
    ? "active"
    : DONE.has(key)
      ? "completed"
      : FAILED.has(key)
        ? "failed"
        : "other";
}
function renderJobs() {
  if (!state.snapshot) return;
  const rows = list(state.snapshot.jobs),
    filter = $("job-filter").value,
    shown = rows.filter((j) => filter === "all" || jobGroup(j) === filter);
  $("jobs-count").textContent = `${shown.length} / ${rows.length}`;
  if (!shown.length) {
    $("job-list").replaceChildren(
      empty(rows.length ? "没有符合筛选条件的任务。" : "尚无任务。"),
    );
    return;
  }
  const container = el("div", "job-list");
  container.append(
    ...shown.map((j) => {
      const row = el("div", "job-row");
      row.append(
        el("code", "job-id", asText(j.job_id, "未命名任务")),
        el("span", "job-kind", asText(j.kind, "类型未报告")),
        badge(j.status),
      );
      return row;
    }),
  );
  $("job-list").replaceChildren(container);
}
const baseAliases = ["baseline", "base", "before", "old"];
const newAliases = ["candidate", "new", "after", "current"];
const number = (x) => typeof x === "number" && Number.isFinite(x);
function flattenMetrics(source, prefix = "", output = {}) {
  for (const [key, value] of Object.entries(object(source))) {
    const name = prefix ? `${prefix}.${key}` : key;
    if (number(value)) output[name] = value;
    else if (value && typeof value === "object")
      flattenMetrics(value, name, output);
  }
  return output;
}
function metricRows(source) {
  const metrics = object(source),
    pairs = [],
    used = new Set();
  const baseGroup = baseAliases.find(
    (k) => metrics[k] && typeof metrics[k] === "object",
  );
  const newGroup = newAliases.find(
    (k) => metrics[k] && typeof metrics[k] === "object",
  );
  if (baseGroup || newGroup) {
    const a = flattenMetrics(metrics[baseGroup]),
      b = flattenMetrics(metrics[newGroup]);
    for (const key of new Set([...Object.keys(a), ...Object.keys(b)]))
      pairs.push({ name: key, base: a[key], next: b[key] });
    if (baseGroup) used.add(baseGroup);
    if (newGroup) used.add(newGroup);
  }
  for (const [key, value] of Object.entries(metrics)) {
    if (used.has(key)) continue;
    if (value && typeof value === "object") {
      const a = baseAliases.find((k) => number(value[k])),
        b = newAliases.find((k) => number(value[k]));
      if (a || b) {
        pairs.push({ name: key, base: value[a], next: value[b] });
        used.add(key);
      }
    }
  }
  // Support worker metrics such as base_nll / candidate_nll without assuming
  // whether an increase or decrease is an improvement. Reported delta_* stays visible.
  const flat = flattenMetrics(
    Object.fromEntries(
      Object.entries(metrics).filter(([key]) => !used.has(key)),
    ),
  );
  const grouped = new Map(),
    pairedKeys = new Set();
  for (const [key, value] of Object.entries(flat)) {
    let side, name;
    for (const [aliases, candidateSide] of [
      [baseAliases, "base"],
      [newAliases, "next"],
    ]) {
      const alias = aliases.find(
        (a) => key === a || key.startsWith(a + "_") || key.endsWith("_" + a),
      );
      if (alias) {
        side = candidateSide;
        name =
          key === alias
            ? "score"
            : key.startsWith(alias + "_")
              ? key.slice(alias.length + 1)
              : key.slice(0, -alias.length - 1);
        break;
      }
    }
    if (side) {
      const row = grouped.get(name) || { name };
      row[side] = value;
      grouped.set(name, row);
      pairedKeys.add(key);
    }
  }
  pairs.push(...grouped.values());
  return {
    pairs,
    extra: Object.fromEntries(
      Object.entries(flat).filter(([key]) => !pairedKeys.has(key)),
    ),
  };
}
function fmt(value) {
  return number(value)
    ? new Intl.NumberFormat("en-US", { maximumFractionDigits: 5 }).format(value)
    : "—";
}
function candidateGroup(c) {
  const key = statusKey(c.status);
  return ACCEPTED.has(key)
    ? "accepted"
    : REJECTED.has(key)
      ? "rejected"
      : PENDING.has(key)
        ? "pending"
        : "other";
}
function renderCandidates() {
  if (!state.snapshot) return;
  const rows = list(state.snapshot.candidates),
    filter = $("candidate-filter").value,
    shown = rows.filter(
      (c) => filter === "all" || candidateGroup(c) === filter,
    );
  $("candidate-count").textContent = `${shown.length} / ${rows.length}`;
  if (!shown.length) {
    $("candidate-list").replaceChildren(
      empty(rows.length ? "没有符合筛选条件的候选。" : "尚无候选版本。"),
    );
    return;
  }
  $("candidate-list").replaceChildren(
    ...shown.map((c) => {
      const card = el("article", "candidate-card"),
        head = el("div", "candidate-top"),
        info = el("div"),
        revs = el("div", "revision-line");
      revs.append(
        el("span", "", "基线"),
        el("code", "", asText(c.base_revision, "未报告")),
        el("span", "arrow", "→"),
        el("span", "", "新版本"),
        el("code", "new-rev", asText(c.revision, "未报告")),
      );
      info.append(
        el("h3", "candidate-id mono", asText(c.candidate_id, "未命名候选")),
        revs,
      );
      head.append(info, badge(c.status));
      card.append(head);
      const { pairs, extra } = metricRows(c.metrics);
      if (pairs.length) {
        const table = el("table", "metric-table");
        table.setAttribute(
          "aria-label",
          `${asText(c.candidate_id, "候选")} 基线与新版本评测`,
        );
        const thead = el("thead"),
          hr = el("tr");
        for (const text of ["指标", "基线", "新版本", "变化"]) {
          const th = el("th", "", text);
          th.scope = "col";
          hr.append(th);
        }
        thead.append(hr);
        const tbody = el("tbody");
        for (const m of pairs) {
          const row = el("tr");
          let delta = "—";
          if (number(m.base) && number(m.next)) {
            const d = m.next - m.base;
            delta = `${d > 0 ? "+" : ""}${fmt(d)}`;
          }
          row.append(
            el("td", "metric-name", m.name.replaceAll("_", " ")),
            el("td", "", fmt(m.base)),
            el("td", "", fmt(m.next)),
            el("td", "delta", delta),
          );
          tbody.append(row);
        }
        table.append(thead, tbody);
        card.append(table);
      }
      const entries = Object.entries(extra);
      if (entries.length) {
        const more = el("div", "metric-note");
        more.append(
          el("span", "", "已报告指标 · "),
          document.createTextNode(
            entries
              .map(([key, value]) => `${key}: ${fmt(value)}`)
              .join("  /  "),
          ),
        );
        card.append(more);
      }
      if (!pairs.length && !entries.length)
        card.append(el("p", "metric-note", "评测结果尚未报告。"));
      return card;
    }),
  );
}
function eventKey(e, index) {
  return `${asText(e.timestamp, "")}:${asText(e.kind, "")}:${asText(e.summary, "")}:${index}`;
}
function updateEventTypes() {
  const select = $("event-filter"),
    previous = select.value,
    types = [
      ...new Set(
        list(state.snapshot?.events).map((e) => asText(e.kind, "未分类")),
      ),
    ].sort();
  const options = [["all", "全部类型"], ...types.map((t) => [t, t])];
  if (
    JSON.stringify([...select.options].map((o) => o.value)) !==
    JSON.stringify(options.map(([v]) => v))
  ) {
    select.replaceChildren(
      ...options.map(([value, text]) => {
        const option = el("option", "", text);
        option.value = value;
        return option;
      }),
    );
    select.value = types.includes(previous) ? previous : "all";
  }
}
function renderEvents() {
  if (!state.snapshot) return;
  const rows = list(state.snapshot.events),
    query = $("event-search").value.trim().toLowerCase(),
    filter = $("event-filter").value;
  const shown = rows
    .map((e, i) => ({ e, key: eventKey(e, i) }))
    .filter(
      ({ e }) =>
        (filter === "all" || asText(e.kind, "未分类") === filter) &&
        `${asText(e.summary, "")} ${asText(e.kind, "")} ${JSON.stringify(object(e.details))}`
          .toLowerCase()
          .includes(query),
    )
    .sort(
      (a, b) =>
        (Date.parse(b.e.timestamp) || 0) - (Date.parse(a.e.timestamp) || 0),
    );
  $("event-count").textContent = `${shown.length} / ${rows.length}`;
  $("more-events").hidden = shown.length <= state.eventLimit;
  if (!shown.length) {
    $("event-list").replaceChildren(
      empty(rows.length ? "没有符合筛选条件的事件。" : "尚无事件。"),
    );
    return;
  }
  const timeline = el("div", "timeline");
  for (const { e, key } of shown.slice(0, state.eventLimit)) {
    const row = el("article", "event"),
      time = el("time", "event-time", dateText(e.timestamp)),
      content = el("div", "event-content");
    if (e.timestamp) time.dateTime = asText(e.timestamp, "");
    content.append(
      el("p", "event-kind", asText(e.kind, "未分类")),
      el("p", "event-summary", asText(e.summary, "无摘要")),
    );
    const details = object(e.details);
    if (Object.keys(details).length) {
      const disclosure = el("details"),
        dl = el("dl", "event-details");
      disclosure.open = state.openEvents.has(key);
      disclosure.append(el("summary", "", "查看详情"));
      for (const [name, value] of Object.entries(details))
        dl.append(el("dt", "", name), el("dd", "", asText(value)));
      disclosure.append(dl);
      disclosure.addEventListener("toggle", () => {
        if (disclosure.isConnected) {
          if (disclosure.open) state.openEvents.add(key);
          else state.openEvents.delete(key);
        }
      });
      content.append(disclosure);
    }
    row.append(time, content);
    timeline.append(row);
  }
  $("event-list").replaceChildren(timeline);
}
function renderSnapshot(s) {
  state.snapshot = s;
  $("empty").hidden = true;
  $("dashboard").hidden = false;
  $("run-id").textContent = asText(s.run_id, "RUN");
  $("objective").textContent = asText(s.objective, "运行目标尚未报告");
  $("run-status").replaceChildren(badge(s.status));
  $("updated").textContent =
    `${state.demo ? "示例时间" : "快照更新"} · ${dateText(s.updated_at)}`;
  renderStats(s);
  renderLoops(s);
  renderJobs();
  renderCandidates();
  updateEventTypes();
  renderEvents();
}
function chooseRuns(runs) {
  const select = $("run-select"),
    available = runs.some((r) => r.run_id === state.runId);
  if (!available) {
    state.runId = asText(runs[0]?.run_id, "");
    state.snapshot = null;
    state.openEvents.clear();
    state.eventLimit = 30;
  }
  const options = runs.map((r) => {
    const option = el(
      "option",
      "",
      `${asText(r.run_id)}${r.status === "unavailable" ? " · 快照不可用" : ""}`,
    );
    option.value = asText(r.run_id, "");
    return option;
  });
  if (!options.length) {
    const option = el("option", "", "暂无运行");
    option.value = "";
    options.push(option);
  }
  select.replaceChildren(...options);
  select.value = state.runId;
  select.disabled = !runs.length;
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
  clearTimeout(state.timer);
  state.controller?.abort();
  const epoch = ++state.epoch,
    controller = new AbortController();
  state.controller = controller;
  let listConnected = false;
  const timeout = setTimeout(() => controller.abort(), 8000);
  $("refresh").disabled = true;
  try {
    if (state.demo) {
      const s = await fetchJSON("/demo.json", controller.signal);
      if (epoch !== state.epoch) return;
      chooseRuns([s]);
      renderSnapshot(s);
      connection("示例数据", "demo");
      notice("示例数据 · 静态演示，不代表真实运行。");
      $("footer-source").textContent = "示例数据 · 静态快照";
      return;
    }
    const data = await fetchJSON("/api/runs", controller.signal);
    if (epoch !== state.epoch) return;
    if (!Array.isArray(data.runs)) throw new Error("运行清单格式无效。");
    listConnected = true;
    chooseRuns(data.runs);
    connection("已连接", "online");
    $("footer-source").textContent = "本地监控 · 只读";
    if (!state.runId) {
      state.snapshot = null;
      notice("");
      showEmpty(
        "等待第一次运行",
        "运行生成 monitor.json 后，状态会自动出现在这里。",
      );
      return;
    }
    if (!state.snapshot)
      showEmpty("正在读取运行", "正在加载所选运行的监控快照。");
    const s = await fetchJSON(
      `/api/runs/${encodeURIComponent(state.runId)}`,
      controller.signal,
    );
    if (epoch !== state.epoch) return;
    if (s.schema_version !== 1) throw new Error("不支持的监控协议版本。");
    renderSnapshot(s);
    const age = Date.now() - Date.parse(s.updated_at);
    notice(
      ACTIVE.has(statusKey(s.status)) && age > 120000
        ? "快照已超过 2 分钟未更新；当前显示的是最近一次报告的状态。"
        : "",
    );
  } catch (error) {
    if (epoch !== state.epoch) return;
    const networkFailure = error instanceof TypeError || controller.signal.aborted;
    connection(listConnected && !networkFailure ? "快照不可用" : "连接中断", "offline");
    const reason = controller.signal.aborted
      ? "请求超时。"
      : error instanceof TypeError
        ? "无法连接本地监控服务。"
        : asText(error.message, "暂时无法连接。");
    notice(
      `${state.demo ? "示例数据 · " : ""}${state.snapshot ? "保留上次成功读取的快照；数据未更新。" : "暂时无法读取监控快照。"} ${reason}`,
      true,
    );
    if (!state.snapshot)
      showEmpty(
        "暂时无法读取运行",
        "请检查监控服务与 monitor.json；恢复后将自动重新读取。",
      );
  } finally {
    clearTimeout(timeout);
    if (epoch === state.epoch) {
      $("refresh").disabled = false;
      state.controller = null;
      schedule();
    }
  }
}
function switchDemo() {
  state.realRunId = state.demo ? state.realRunId : state.runId;
  state.demo = !state.demo;
  state.runId = state.demo ? "" : state.realRunId;
  state.snapshot = null;
  state.openEvents.clear();
  state.eventLimit = 30;
  $("demo").textContent = state.demo ? "返回真实运行" : "查看示例";
  $("demo").setAttribute("aria-pressed", String(state.demo));
  $("auto-refresh").checked = state.demo ? false : state.auto;
  $("auto-refresh").disabled = state.demo;
  $("job-filter").value = "all";
  $("candidate-filter").value = "all";
  $("event-filter").value = "all";
  $("event-search").value = "";
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
$("run-select").addEventListener("change", () => {
  state.runId = $("run-select").value;
  state.snapshot = null;
  state.openEvents.clear();
  state.eventLimit = 30;
  refresh();
});
$("job-filter").addEventListener("change", renderJobs);
$("candidate-filter").addEventListener("change", renderCandidates);
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
