"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(`${__dirname}/app.js`, "utf8");
class Element {
  constructor(tag = "div") { this.tagName = tag; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.scrollLeft = 0; this.scrollTop = 0; this.clientWidth = 900; this.clientHeight = 660; this.value = "all"; this.isConnected = true; this.classList = { add() {}, remove() {}, contains: () => false }; }
  append(...nodes) { this.children.push(...nodes); }
  prepend(...nodes) { this.children.unshift(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get childElementCount() { return this.children.length; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  addEventListener(k, f) { this.listeners[k] = f; }
  contains() { return false; }
  focus() {}
}
function harness(fetcher = async () => { throw new Error("unexpected fetch"); }, search = "") {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) { const e = new Element(); e.id = id; elements.set(id, e); } return elements.get(id); };
  const ctx = vm.createContext({ document: { getElementById: get, createElement: t => new Element(t), createElementNS: (_, t) => new Element(t), addEventListener() {}, hidden: false }, location: { href: `http://localhost:8769/${search}`, search }, history: { replaceState() {} }, AbortController, URL, URLSearchParams, setTimeout, clearTimeout, fetch: fetcher });
  vm.runInContext(source.slice(0, source.lastIndexOf("\nrefresh();")), ctx);
  vm.runInContext('state.auto = false; renderSnapshot = function(s) { state.snapshot = s; };', ctx);
  return { ctx, get, run: s => vm.runInContext(s, ctx), json: s => JSON.parse(JSON.stringify(vm.runInContext(s, ctx))) };
}
const run = (id, extra = {}) => ({ schema_version: 1, run_id: id, updated_at: "2026-11-01T00:00:00Z", ...extra });
const project = (id, runs) => ({ project_id: id, name: id, runs });
const response = data => ({ ok: true, json: async () => data });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
function setup(h, projects) { h.ctx.input = projects; h.run("chooseProjects(input)"); }
function model(h) { return h.json("projectModel(selectedProject(), state.activeScope)"); }
function seed(h, id, snapshot) { h.ctx.fixture = snapshot; h.ctx.id = id; h.run('state.activeScope.expanded.add(id); state.activeScope.cache.set(id, {snapshot: fixture, status: "ready", token: 0})'); }

test("project root contains every run without fetching snapshots; anchors survive multiple expansions", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b"), run("c")])]);
  const before = model(h); assert.equal(before.nodes.length, 4); assert.equal(before.edges.filter(e => e.kind === "membership").length, 3);
  seed(h, "a", run("a", { decisions: [{ decision_id: "same", mode: "train" }] })); seed(h, "b", run("b", { decisions: [{ decision_id: "same", mode: "evaluate" }] }));
  const expanded = model(h); assert.equal(expanded.nodes.filter(n => n.type === "decision").length, 2);
  assert.deepEqual(expanded.nodes.filter(n => n.type === "run").map(n => [n.key, n.x, n.y]), before.nodes.filter(n => n.type === "run").map(n => [n.key, n.x, n.y]));
  h.run('selectRun("a", true)'); assert.deepEqual(model(h).nodes.filter(n => n.type === "decision").map(n => n.runId), ["b"]);
});

test("same decision/job/directive IDs in different runs have disjoint keys and edges", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b")])]);
  for (const id of ["a", "b"]) seed(h, id, run(id, { decisions: [{ decision_id: "same", kind: "directive", directive_id: "shared" }, { decision_id: "child", parent_decision_id: "same" }], jobs: [{ job_id: "same-job", directive_id: "shared", metrics: { loss: id === "a" ? 1 : 2 } }] }));
  const graph = model(h), nodes = new Map(graph.nodes.map(n => [n.key, n]));
  assert.equal(new Set(graph.nodes.map(n => n.key)).size, graph.nodes.length);
  for (const e of graph.edges.filter(e => ["explicit", "directive"].includes(e.kind))) assert.equal(nodes.get(e.from).runId, nodes.get(e.to).runId);
  assert.deepEqual(graph.nodes.filter(n => n.type === "job").map(n => [n.runId, n.row.metrics.loss]), [["a", 1], ["b", 2]]);
});

test("run timeline uses strict updated_at order; only explicit run parents create solid cross-run edges", () => {
  const h = harness(); setup(h, [project("p", [run("c", { updated_at: "invalid", parent_run_id: "missing" }), run("b", { updated_at: "2026-11-01T01:00:00Z", parent_run_id: "a", source_run_ids: ["a", "b", "outside"] }), run("a"), run("tie")])]);
  const graph = model(h);
  assert.equal(graph.edges.filter(e => e.kind === "run-parent").length, 1);
  assert.equal(graph.edges.filter(e => e.kind === "time").length, 1);
  assert.deepEqual(graph.edges.find(e => e.kind === "time").evidence, ["记录更新时间顺序，不表示继承"]);
  assert.deepEqual(graph.nodes.find(n => n.runId === "b").missing, ["outside"]);
  assert.deepEqual(graph.nodes.find(n => n.runId === "c").missing, ["missing"]);
  const unrelated = harness(); setup(unrelated, [project("p", [run("a", { base_revision: "v1" }), run("b", { base_revision: "v2", objective: "same" })])]);
  assert.equal(model(unrelated).edges.filter(e => e.kind === "run-parent").length, 0);
});

test("ambiguous or self run references never guess a parent; multiple explicit parents work", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b"), run("c", { parent_run_id: "a", source_run_ids: ["a", "b", "c", null, 9] })])]);
  assert.equal(model(h).edges.filter(e => e.kind === "run-parent").length, 2);
  setup(h, [project("p", [run("dup"), run("dup"), run("child", { parent_run_id: "dup" })])]);
  assert.equal(model(h).edges.filter(e => e.kind === "run-parent").length, 0);
});

test("legacy ?run= locates owning project and expands inside its only graph", () => {
  const h = harness(undefined, "?project=wrong&run=target");
  setup(h, [project("first", [run("other")]), project("owner", [run("target")])]);
  assert.equal(h.run("state.projectId"), "owner"); assert.equal(h.run('state.activeScope.expanded.has("target")'), true);
  assert.equal(h.get("project-graph").graphView.selected, JSON.stringify(["run", "target"]));
});

test("parallel on-demand loads resolve out of order into their own run caches", async () => {
  const a = deferred(), b = deferred(), calls = [];
  const h = harness(url => { calls.push(url); return url.endsWith("/a") ? a.promise : b.promise; }); setup(h, [project("p", [run("a"), run("b"), run("unopened")])]);
  const pa = h.run('loadRun("a", state.activeScope)'), pb = h.run('loadRun("b", state.activeScope)');
  assert.equal(h.run('state.activeScope.cache.get("a").status'), "pending");
  b.resolve(response(run("b", { objective: "B" }))); await pb; a.resolve(response(run("a", { objective: "A" }))); await pa;
  assert.deepEqual(calls.sort(), ["/api/runs/a", "/api/runs/b"]);
  assert.equal(h.run('state.activeScope.cache.get("a").snapshot.objective'), "A"); assert.equal(h.run('state.activeScope.cache.get("b").snapshot.objective'), "B");
  await h.run('loadRun("a", state.activeScope)'); assert.equal(calls.length, 2);
});

test("project switch aborts pending request and rejects late response even when fetch ignores abort", async () => {
  const pending = deferred(), signals = [];
  const h = harness((url, options) => { signals.push(options.signal); return pending.promise; }); setup(h, [project("one", [run("same")]), project("two", [run("same")])]);
  const request = h.run('loadRun("same", state.activeScope)'); h.run('state.projectId = "two"; activateProject(); renderProject()');
  assert.equal(signals[0].aborted, true); pending.resolve(response(run("same", { objective: "OLD PROJECT" }))); await request;
  assert.equal(h.run('state.activeScope.cache.has("same")'), false); assert.equal(h.run('state.snapshot'), null);
  h.run('state.projectId = "one"; activateProject()'); assert.equal(h.run('state.activeScope.cache.get("same").snapshot'), null);
});

test("collapse aborts pending branch; reopening cannot accept older response", async () => {
  const old = deferred(), fresh = deferred(); let count = 0;
  const h = harness(() => ++count === 1 ? old.promise : fresh.promise); setup(h, [project("p", [run("a")])]);
  h.run('state.activeScope.expanded.add("a")'); const first = h.run('loadRun("a", state.activeScope)'); h.run('selectRun("a", true); state.activeScope.expanded.add("a")'); const second = h.run('loadRun("a", state.activeScope)');
  fresh.resolve(response(run("a", { objective: "fresh" }))); await second; old.resolve(response(run("a", { objective: "stale" }))); await first;
  assert.equal(h.run('state.activeScope.cache.get("a").snapshot.objective'), "fresh");
});

test("failed refresh keeps cached snapshot, exposes error, and retry recovers", async () => {
  let fail = true; const h = harness(async () => { if (fail) throw new Error("offline"); return response(run("a", { objective: "recovered" })); }); setup(h, [project("p", [run("a")])]); seed(h, "a", run("a", { objective: "cached" }));
  await h.run('loadRun("a", state.activeScope, true)'); assert.equal(h.run('state.activeScope.cache.get("a").status'), "error"); assert.equal(h.run('state.activeScope.cache.get("a").snapshot.objective'), "cached");
  assert.match(h.run('state.activeScope.cache.get("a").error'), /offline/);
  fail = false; await h.run('loadRun("a", state.activeScope, true)'); assert.equal(h.run('state.activeScope.cache.get("a").status'), "ready");
});

test("refresh preserves expanded runs, selected node, zoom and scroll; fetches only expanded runs", async () => {
  const projects = [project("p", [run("a"), run("b"), run("c")])], calls = [];
  const h = harness(async url => { calls.push(url); return response(url === "/api/projects" ? { projects } : run(url.split("/").pop(), { decisions: [{ decision_id: "same", mode: "train" }] })); }); setup(h, projects);
  for (const id of ["a", "b"]) seed(h, id, run(id, { decisions: [{ decision_id: "same", mode: "train" }] }));
  h.run('renderProject(); state.activeScope.selected = projectModel(selectedProject(), state.activeScope).nodes.find(n => n.type === "decision").key; $("project-graph").graphView.scale = .8; $("project-graph").graphView.viewport.scrollLeft = 123; $("project-graph").graphView.viewport.scrollTop = 234');
  const selected = h.run("state.activeScope.selected"); await h.run("refresh()");
  assert.deepEqual(calls.sort(), ["/api/projects", "/api/runs/a", "/api/runs/b"]);
  assert.deepEqual(h.json("[...state.activeScope.expanded]"), ["a", "b"]); assert.equal(h.get("project-graph").graphView.selected, selected);
  assert.equal(h.get("project-graph").graphView.scale, .8); assert.equal(h.get("project-graph").graphView.viewport.scrollLeft, 123); assert.equal(h.get("project-graph").graphView.viewport.scrollTop, 234);
});

test("demo has multiple expandable run snapshots and an explicit run parent", () => {
  const demo = JSON.parse(fs.readFileSync(`${__dirname}/demo.json`, "utf8")); const h = harness(); setup(h, [project("demo", [demo, ...demo.demo_runs])]);
  for (const s of [demo, ...demo.demo_runs]) seed(h, s.run_id, s);
  const graph = model(h); assert.ok(graph.nodes.filter(n => n.type === "run").length > 1); assert.ok(new Set(graph.nodes.filter(n => n.type === "decision").map(n => n.runId)).size > 1); assert.ok(graph.edges.some(e => e.kind === "run-parent"));
});

test("cached same run ID stays isolated by project and demo mode", () => {
  const h = harness(); setup(h, [project("one", [run("same")]), project("two", [run("same")])]);
  seed(h, "same", run("same", { objective: "project one" }));
  h.run('state.projectId = "two"; activateProject()'); assert.equal(h.run('state.activeScope.cache.has("same")'), false);
  seed(h, "same", run("same", { objective: "project two" }));
  h.run('state.projectId = "one"; activateProject()'); assert.equal(h.run('state.activeScope.cache.get("same").snapshot.objective'), "project one");
  h.run('state.demo = true; activateProject()'); assert.equal(h.run('state.activeScope.cache.has("same")'), false);
  h.run('state.demo = false; activateProject()'); assert.equal(h.run('state.activeScope.cache.get("same").snapshot.objective'), "project one");
});

test("switching demo invalidates real requests immediately before demo fixture loads", async () => {
  const pending = deferred(); let signal;
  const h = harness((url, options) => { signal = options.signal; return pending.promise; }); setup(h, [project("p", [run("a")])]);
  const request = h.run('loadRun("a", state.activeScope)');
  h.run('refresh = function() {}; switchDemo()'); assert.equal(signal.aborted, true); assert.equal(h.run('state.activeScope'), null);
  pending.resolve(response(run("a", { objective: "late real run" }))); await request;
  assert.equal(h.run('state.snapshot'), null); assert.equal(h.get("project-graph").graphView, null);
});
