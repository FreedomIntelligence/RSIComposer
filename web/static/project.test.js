"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const i18nSource = fs.readFileSync(`${__dirname}/i18n.js`, "utf8");
const source = fs.readFileSync(`${__dirname}/app.js`, "utf8");
class Element {
  constructor(tag = "div") { this.tagName = tag; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.scrollLeft = 0; this.scrollTop = 0; this.clientWidth = 900; this.clientHeight = 660; this.value = "all"; this.isConnected = true; this.classList = { add() {}, remove() {}, contains: () => false }; }
  append(...nodes) { this.children.push(...nodes); }
  prepend(...nodes) { this.children.unshift(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  replaceChild(next,old) { const i=this.children.indexOf(old);if(i<0)throw Error("missing child");this.children[i]=next; }
  get childElementCount() { return this.children.length; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  addEventListener(k, f) { this.listeners[k] = f; }
  contains() { return false; }
  focus() {}
}
function fakeRAF() {
  const pending = new Map(), cancelled = []; let next = 0, time = 0;
  return {
    request(callback) { const id = ++next; pending.set(id, callback); return id; },
    cancel(id) { cancelled.push(id); pending.delete(id); },
    get size() { return pending.size; },
    get ids() { return [...pending.keys()]; },
    cancelled,
    step() {
      time += 1000 / 60;
      for (const [id, callback] of [...pending]) if (pending.delete(id)) callback(time);
    },
    drain(limit = 100) {
      let frames = 0;
      while (pending.size && frames < limit) { this.step(); frames++; }
      assert.equal(pending.size, 0, `simulation must stop within ${limit} frames`);
      return frames;
    },
  };
}
function harness(fetcher = async () => { throw new Error("unexpected fetch"); }, search = "", { raf = false, fullSnapshot = false } = {}) {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) { const e = new Element(); e.id = id; elements.set(id, e); } return elements.get(id); };
  const observers = [];
  class ResizeObserver {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(target) { this.target = target; }
    disconnect() { this.disconnected = true; }
  }
  const documentListeners = {}, document = { getElementById: get, createElement: t => new Element(t), createElementNS: (_, t) => new Element(t), addEventListener(k, f) { documentListeners[k] = f; }, hidden: false };
  const frames = raf ? fakeRAF() : null;
  const ctx = vm.createContext({ ResizeObserver, document, location: { href: `http://localhost:8769/${search}`, search }, history: { replaceState() {} }, AbortController, URL, URLSearchParams, setTimeout, clearTimeout, fetch: fetcher,
    ...(frames ? { requestAnimationFrame: callback => frames.request(callback), cancelAnimationFrame: id => frames.cancel(id) } : {}) });
  vm.runInContext(i18nSource, ctx);
  vm.runInContext(fs.readFileSync(`${__dirname}/lineage.js`, "utf8"), ctx);
  vm.runInContext(source.slice(0, source.lastIndexOf("\nrefresh();")), ctx);
  vm.runInContext('state.auto = false;', ctx);
  if (!fullSnapshot) vm.runInContext('renderSnapshot = function(s) { state.snapshot = s; };', ctx);
  return { ctx, get, observers, frames, document, documentListeners, run: s => vm.runInContext(s, ctx), json: s => JSON.parse(JSON.stringify(vm.runInContext(s, ctx))) };
}
const run = (id, extra = {}) => ({ schema_version: 1, run_id: id, updated_at: "2026-11-01T00:00:00Z", ...extra });
const project = (id, runs) => ({ project_id: id, name: id, runs });
const response = data => ({ ok: true, json: async () => data });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
function setup(h, projects) { h.ctx.input = projects; h.run("chooseProjects(input)"); }
function model(h) { return h.json("projectModel(selectedProject(), state.activeScope)"); }
function seed(h, id, snapshot) { h.ctx.fixture = snapshot; h.ctx.id = id; h.run('state.activeScope.cache.set(id, {snapshot: fixture, status: "ready", token: 0})'); }

test("project graph contains only the project and all runs before and after loading internal records", () => {
  const calls = [], h = harness(async url => { calls.push(url); throw Error("unexpected fetch"); });
  setup(h, [project("p", [run("a"), run("b"), run("c")])]);
  const before = model(h);
  assert.deepEqual(before.nodes.map(n => n.type), ["project", "run", "run", "run"]);
  assert.equal(before.edges.filter(e => e.kind === "membership").length, 3);
  for (const id of ["a", "b"]) seed(h, id, run(id, {
    decisions: [{ decision_id: "same", kind: "directive", directive_id: "shared", mode: "train" }],
    jobs: [{ job_id: "same-job", directive_id: "shared", metrics: { loss: 1 } }],
    candidates: [{ candidate_id: "same-candidate", directive_id: "shared" }],
  }));
  h.run('selectRun("a"); selectRun("b"); selectRun("a")');
  const after = model(h);
  assert.deepEqual(after.nodes.map(n => [n.key, n.type, n.x, n.y]), before.nodes.map(n => [n.key, n.type, n.x, n.y]));
  assert.deepEqual(after.edges, before.edges);
  assert.equal(findClass(h.get("project-graph"), "graph-node").length, 4);
  assert.equal(findClass(h.get("project-graph"), "flow-choice").length, 1);
  assert.deepEqual(calls, []);
});

test("same decision/job/directive/candidate IDs in different runs keep their own detail results", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b")])]);
  for (const [id, loss] of [["a", 1], ["b", 2]]) seed(h, id, run(id, {
    decisions: [{ decision_id: "same", kind: "directive", directive_id: "shared", mode: "train", feedback: { summary: `feedback-${id}` } }],
    jobs: [{ job_id: "same-job", directive_id: "shared", metrics: { loss } }],
    candidates: [{ candidate_id: "same-candidate", directive_id: "shared", metrics: { accuracy: loss / 10 } }],
  }));
  for (const [id, other, loss] of [["a", "b", 1], ["b", "a", 2], ["a", "b", 1]]) {
    h.ctx.selectedId = id; h.run('selectRun(selectedId)');
    const detail = findClass(h.get("project-graph"), "run-detail")[0];
    assert.match(textOf(detail), new RegExp(`feedback-${id}`));
    assert.doesNotMatch(textOf(detail), new RegExp(`feedback-${other}`));
    assert.equal(h.run('state.snapshot.jobs[0].metrics.loss'), loss);
    assert.equal(findClass(detail, "candidate-card").length, 1);
    assert.match(textOf(findClass(detail, "flow-metrics")[0]), new RegExp(`\\b${loss}\\b`));
  }
  const graph = model(h);
  assert.equal(new Set(graph.nodes.map(n => n.key)).size, graph.nodes.length);
  assert.deepEqual(graph.nodes.map(n => n.type), ["project", "run", "run"]);
});

test("run continuation determines horizontal depth; source references do not imply parentage", () => {
  const h = harness(); setup(h, [project("p", [run("c", { updated_at: "invalid", parent_run_id: "missing" }), run("b", { updated_at: "2026-11-01T01:00:00Z", parent_run_id: "a", source_run_ids: ["a", "b", "outside"] }), run("a"), run("tie")])]);
  const graph = model(h);
  assert.equal(graph.edges.filter(e => e.kind === "run-parent").length, 1);
  assert.equal(graph.edges.filter(e => e.kind === "time").length, 0);
  assert.ok(graph.edges.every(e => ["membership", "run-parent", "run-source"].includes(e.kind)));
  const runs = graph.nodes.filter(n => n.type === "run");
  assert.deepEqual(runs.map(n => n.runId), ["a", "tie", "b", "c"]);
  assert.deepEqual(runs.map(n => n.layoutOrder), [0, 1, 2, 3]);
  assert.ok(runs.find(n=>n.runId==="b").x>runs.find(n=>n.runId==="a").x);
  assert.deepEqual(runs.map(n=>n.depth),[1,1,2,1]);
  assert.equal(graph.edges.filter(e=>e.kind==="membership").length,3);
  assert.deepEqual(graph.nodes.find(n => n.runId === "b").missing, ["outside"]);
  assert.deepEqual(graph.nodes.find(n => n.runId === "c").missing, ["missing"]);
  const unrelated = harness(); setup(unrelated, [project("p", [run("a", { base_revision: "v1" }), run("b", { base_revision: "v2", objective: "same" })])]);
  assert.equal(model(unrelated).edges.filter(e => e.kind === "run-parent").length, 0);
});

test("ambiguous or self references never guess a parent; source links remain distinct", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b"), run("c", { parent_run_id: "a", source_run_ids: ["a", "b", "c", null, 9] })])]);
  assert.equal(model(h).edges.filter(e => e.kind === "run-parent").length, 1);
  assert.equal(model(h).edges.filter(e => e.kind === "run-source").length, 2);
  setup(h, [project("p", [run("dup"), run("dup"), run("child", { parent_run_id: "dup" })])]);
  assert.equal(model(h).edges.filter(e => e.kind === "run-parent").length, 0);
});

test("legacy run-only and project/run deep links select and load the owning run", async () => {
  for (const search of ["?run=target", "?project=wrong&run=target"]) {
    const projects = [project("first", [run("other")]), project("owner", [run("target")])], calls = [];
    const h = harness(async url => { calls.push(url); return response(url === "/api/projects" ? { projects } : run("target", { objective: "deep-linked run" })); }, search);
    await h.run("refresh()");
    assert.equal(h.run("state.projectId"), "owner"); assert.equal(h.run("state.runId"), "target");
    assert.equal(h.get("project-graph").graphView.selected, JSON.stringify(["run", "target"]));
    assert.equal(h.run("state.snapshot.objective"), "deep-linked run");
    assert.deepEqual(calls, ["/api/projects", "/api/runs/target"]);
    assert.deepEqual(model(h).nodes.map(n => n.type), ["project", "run"]);
  }
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

test("repeated run clicks keep selection and the pending request, then reuse its loaded snapshot", async () => {
  const pending = deferred(), calls = [], signals = [];
  const h = harness((url, options) => { calls.push(url); signals.push(options.signal); return pending.promise; });
  setup(h, [project("p", [run("a"), run("b")])]);
  const click = () => findClass(h.get("project-graph"), "graph-node").find(n => n.dataset.nodeKey === JSON.stringify(["run", "a"])).listeners.click();
  click(); click();
  assert.equal(h.run("state.runId"), "a"); assert.equal(signals[0].aborted, false);
  assert.equal(h.run('state.activeScope.cache.get("a").status'), "pending");
  assert.deepEqual(model(h).nodes.map(n => n.type), ["project", "run", "run"]);
  pending.resolve(response(run("a", { decisions: [{ decision_id: "d", mode: "train", feedback: { summary: "loaded feedback" } }] })));
  await new Promise(resolve => setImmediate(resolve));
  click(); click();
  assert.deepEqual(calls, ["/api/runs/a"]);
  assert.equal(h.get("project-graph").graphView.selected, JSON.stringify(["run", "a"]));
  assert.equal(h.run("state.snapshot.run_id"), "a");
  assert.match(textOf(findClass(h.get("project-graph"), "run-detail")[0]), /loaded feedback/);
});

test("changing run selection rejects a late old request and reopening loads a fresh snapshot", async () => {
  const old = deferred(), fresh = deferred(), signals = []; let count = 0;
  const h = harness((url, options) => { signals.push(options.signal); return ++count === 1 ? old.promise : fresh.promise; });
  setup(h, [project("p", [run("a"), run("b")])]); seed(h, "b", run("b", { objective: "B" }));
  h.run('selectRun("a"); selectRun("b"); selectRun("a")');
  assert.equal(signals[0].aborted, true);
  fresh.resolve(response(run("a", { objective: "fresh" }))); await new Promise(resolve => setImmediate(resolve));
  old.resolve(response(run("a", { objective: "stale" }))); await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.run('state.activeScope.cache.get("a").snapshot.objective'), "fresh");
  assert.equal(h.run("state.snapshot.objective"), "fresh");
  assert.equal(h.run('state.activeScope.cache.get("b").snapshot.objective'), "B");
});

test("failed refresh keeps cached snapshot, exposes error, and retry recovers", async () => {
  let fail = true; const h = harness(async () => { if (fail) throw new Error("offline"); return response(run("a", { objective: "recovered" })); }); setup(h, [project("p", [run("a")])]); seed(h, "a", run("a", { objective: "cached" }));
  await h.run('loadRun("a", state.activeScope, true)'); assert.equal(h.run('state.activeScope.cache.get("a").status'), "error"); assert.equal(h.run('state.activeScope.cache.get("a").snapshot.objective'), "cached");
  assert.match(h.run('state.activeScope.cache.get("a").error'), /offline/);
  fail = false; await h.run('loadRun("a", state.activeScope, true)'); assert.equal(h.run('state.activeScope.cache.get("a").status'), "ready");
});

test("refresh reloads only the selected run, keeps other cached snapshots and preserves manual viewport", async () => {
  const projects = [project("p", [run("a"), run("b"), run("c")])], calls = [];
  const h = harness(async url => { calls.push(url); return response(url === "/api/projects" ? { projects } : run("a", { objective: "fresh A" })); }); setup(h, projects);
  seed(h, "a", run("a", { objective: "cached A" })); seed(h, "b", run("b", { objective: "cached B" }));
  h.run('selectRun("a"); $("project-graph").graphView.autoFit = false; $("project-graph").graphView.scale = .8; $("project-graph").graphView.viewport.scrollLeft = 123; $("project-graph").graphView.viewport.scrollTop = 234');
  const before = model(h).nodes.map(n => [n.key, n.x, n.y]);
  await h.run("refresh()");
  assert.deepEqual(calls, ["/api/projects", "/api/runs/a"]);
  assert.equal(h.run('state.activeScope.cache.get("a").snapshot.objective'), "fresh A");
  assert.equal(h.run('state.activeScope.cache.get("b").snapshot.objective'), "cached B");
  assert.equal(h.run('state.activeScope.cache.has("c")'), false);
  assert.equal(h.get("project-graph").graphView.selected, JSON.stringify(["run", "a"]));
  assert.equal(h.get("project-graph").graphView.scale, .8); assert.equal(h.get("project-graph").graphView.viewport.scrollLeft, 123); assert.equal(h.get("project-graph").graphView.viewport.scrollTop, 234);
  assert.deepEqual(model(h).nodes.map(n => [n.key, n.x, n.y]), before);
  assert.equal(model(h).layoutStats.iterations, 0);
  h.run('selectRun("b")'); assert.deepEqual(calls, ["/api/projects", "/api/runs/a"]); assert.equal(h.run("state.snapshot.objective"), "cached B");
  h.run('selectRun("")'); calls.length = 0; await h.run("refresh()");
  assert.deepEqual(calls, ["/api/projects"]); assert.equal(h.run("state.snapshot"), null);
});

test("demo keeps multiple run snapshots and explicit parents without adding internal graph nodes", () => {
  const demo = JSON.parse(fs.readFileSync(`${__dirname}/demo.json`, "utf8")); const h = harness(); setup(h, [project("demo", [demo, ...demo.demo_runs])]);
  for (const snapshot of [demo, ...demo.demo_runs]) seed(h, snapshot.run_id, snapshot);
  const graph = model(h);
  assert.ok(graph.nodes.filter(n => n.type === "run").length > 1);
  assert.ok(graph.nodes.every(n => ["project", "run"].includes(n.type)));
  assert.ok(graph.edges.some(e => e.kind === "run-parent"));
  for (const snapshot of [demo, ...demo.demo_runs]) {
    h.ctx.selectedId = snapshot.run_id; h.run('selectRun(selectedId)');
    assert.equal(h.run("state.snapshot.run_id"), snapshot.run_id);
    assert.equal(findClass(h.get("project-graph"), "execution-sequence").length, 1);
  }
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

function descendants(node) { return [node,...node.children.flatMap(descendants)]; }
function textOf(node) { return [node.textContent||"",...node.children.map(textOf)].join(" "); }
function cssClass(node,name) { return (node.className||node.attributes.class||"").split(" ").includes(name); }
function findClass(node,name) { return descendants(node).filter(n=>cssClass(n,name)); }
const pointer = (clientX, clientY) => ({ button: 0, clientX, clientY, pointerId: 1, stopPropagation() {} });
function graphButton(h, id) { return findClass(h.get("project-graph"), "graph-node").find(n => n.dataset.nodeKey === JSON.stringify(["run", id])); }
function graphPositions(h) { return findClass(h.get("project-graph"), "graph-node").map(n => [n.dataset.nodeKey, parseFloat(n.style.left), parseFloat(n.style.top)]); }
function startDrag(h, id = "a") {
  const button = graphButton(h, id); button.setPointerCapture = () => {};
  button.listeners.pointerdown(pointer(10, 20)); button.listeners.pointermove(pointer(40, 75));
  return button;
}
function assertStraightEdges(h) {
  const positions = new Map(graphPositions(h).map(([key, x, y]) => [key, `${x},${y}`]));
  const edges = model(h).edges, paths = findClass(h.get("project-graph"), "graph-edge");
  assert.equal(paths.length, edges.length);
  paths.forEach((path, i) => assert.equal(path.attributes.d, `M${positions.get(edges[i].from)} L${positions.get(edges[i].to)}`));
}

test("network layout is deterministic under input reorder and keeps cached coordinates without further simulation", () => {
  const h=harness();
  const graph={nodes:[{key:"project-root",type:"project"},...Array.from({length:8},(_,i)=>({key:JSON.stringify(["run",String(i)]),type:"run",runId:String(i),layoutOrder:7-i}))],edges:Array.from({length:8},(_,i)=>({from:"project-root",to:JSON.stringify(["run",String(i)]),kind:"membership"}))};
  h.ctx.graph=graph;
  const a=h.json("networkLayout(structuredGraph = JSON.parse(JSON.stringify(graph)))");
  h.ctx.graph={nodes:graph.nodes.slice().reverse(),edges:graph.edges.slice().reverse()};
  const b=h.json("networkLayout(JSON.parse(JSON.stringify(graph)))");
  const coords=g=>g.nodes.map(n=>[n.key,n.x,n.y]).sort((a,b)=>a[0].localeCompare(b[0]));
  assert.deepEqual(coords(a),coords(b));
  h.ctx.graph=a;
  const cached=h.json('networkLayout(graph,new Map(graph.nodes.map(n=>[n.key,{x:n.x,y:n.y}])))');
  assert.deepEqual(coords(cached),coords(a));assert.equal(cached.layoutStats.iterations,0);
  const runs = a.nodes.filter(n => n.type === "run").sort((a, b) => a.layoutOrder - b.layoutOrder);
  assert.ok(runs.every((n, i) => Math.abs(n.x - (70 + (i+1) * 180)) <= 25));
  assert.ok(runs.every((n, i) => i === 0 || n.x > runs[i - 1].x));
  const root = a.nodes.find(n => n.type === "project");
  assert.ok(root.x < runs[0].x && Math.abs(root.y - 245) < 35);
});

test("project zero starts at the left-hand center and renders its numbered name with straight edges", () => {
  const h = harness(); setup(h, [project("研究项目", [])]);
  const root = model(h).nodes[0];
  assert.deepEqual([root.x, root.y, root.title], [70, 245, "0 · 研究项目"]);
  assert.equal(findClass(h.get("project-graph"), "node-label")[0].textContent, "0 · 研究项目");
  setup(h, [project("p", [run("a"), run("b", { parent_run_id: "a" }), run("c")])]);
  const graph = model(h), runs = graph.nodes.filter(n => n.type === "run");
  assert.ok(runs.every(n => n.x > graph.nodes[0].x));
  assert.equal(runs[0].depth,1);assert.equal(runs[1].depth,2);assert.equal(runs[2].depth,1);
  assert.ok(runs[1].x>runs[0].x);assert.ok(runs[2].y>runs[0].y);
  assert.deepEqual([graph.nodes[0].x,graph.nodes[0].y],[70,245]);
  assertStraightEdges(h);
  assert.equal(findClass(h.get("project-graph"), "linked")[0].attributes["marker-end"], "url(#project-graph-arrow)");
});

test("network layout defaults to 64 steps and supports two-step relaxation with pinned coordinates", () => {
  const h = harness();
  h.ctx.graph = { nodes: [{ key: "project-root", type: "project" }, { key: "a", type: "run", layoutOrder: 0 }], edges: [{ from: "project-root", to: "a", kind: "membership" }] };
  const full = h.json("networkLayout(JSON.parse(JSON.stringify(graph)))");
  const short = h.json("networkLayout(JSON.parse(JSON.stringify(graph)), new Map(), {steps: 2})");
  assert.equal(full.layoutStats.iterations, 64); assert.equal(short.layoutStats.iterations, 2);
  assert.notDeepEqual(short.nodes.map(n => [n.x, n.y]), full.nodes.map(n => [n.x, n.y]));
  h.ctx.graph = JSON.parse(JSON.stringify(short));
  const relaxed = h.json('networkLayout(graph, new Map(graph.nodes.map(n => [n.key, {x:n.x,y:n.y}])), {relax:true, pinned:new Set(["a"]), steps:2})');
  assert.equal(relaxed.layoutStats.iterations, 0);
  assert.deepEqual(relaxed.nodes.find(n => n.key === "a"), short.nodes.find(n => n.key === "a"));
  assert.deepEqual(relaxed.nodes[0], short.nodes[0]);
});

test("large dense graph has finite budget and bounded grid probes; stationary layout does no relaxation", () => {
  const h=harness();const count=1800;
  h.ctx.graph={nodes:Array.from({length:count},(_,i)=>({key:`n${i}`,type:"run",runId:`r${i}`,layoutOrder:i})),edges:Array.from({length:count-1},(_,i)=>({from:`n${i}`,to:`n${i+1}`,kind:"run-parent"}))};
  const result=h.json('networkLayout(graph)');
  assert.ok(result.layoutStats.iterations<=64);
  assert.ok(result.layoutStats.iterations*(count+count-1)<=60000);
  assert.ok(result.layoutStats.comparisons<=result.layoutStats.iterations*count*9*12);
  assert.ok(result.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)&&n.x>=36&&n.y>=36));
  assert.equal(h.run('networkLayout(graph,new Map(graph.nodes.map(n=>[n.key,{x:n.x,y:n.y}]))).layoutStats.iterations'),0);
});

test("adding a run relaxes existing unpinned nodes while preserving the manually pinned run", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b")])]);
  h.run('state.activeScope.pinned.add(JSON.stringify(["run","a"]))');
  const before = model(h), pinned = before.nodes.find(n => n.runId === "a"), movable = before.nodes.find(n => n.runId === "b");
  h.ctx.newRun = run("c"); h.run("state.projects[0].runs.push(newRun)");
  const after = model(h);
  assert.ok(after.layoutStats.iterations > 0 && after.layoutStats.iterations <= 64);
  assert.deepEqual(after.nodes.find(n => n.runId === "a"), pinned);
  const moved = after.nodes.find(n => n.runId === "b");
  assert.notDeepEqual([moved.x, moved.y], [movable.x, movable.y]);
  assert.deepEqual(after.nodes.map(n => n.type), ["project", "run", "run", "run"]);
  const stationary = model(h);
  assert.equal(stationary.layoutStats.iterations, 0);
  assert.deepEqual(stationary.nodes.map(n => [n.key, n.x, n.y]), after.nodes.map(n => [n.key, n.x, n.y]));
});

test("updated_at order changes relax cached runs even when nodes and edges stay the same", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b", { updated_at: "2026-11-01T01:00:00Z" })])]);
  const before = model(h);
  h.run('state.projects[0].runs[0].updated_at = "2026-11-01T02:00:00Z"');
  const after = model(h), runs = after.nodes.filter(n => n.type === "run");
  const edges = g => g.edges.slice().sort((a, b) => a.to.localeCompare(b.to));
  assert.deepEqual(edges(after), edges(before));
  assert.equal(after.topologyChanged, true);
  assert.ok(after.layoutStats.iterations > 0 && after.layoutStats.iterations <= 64);
  assert.deepEqual(runs.map(n => [n.runId, n.layoutOrder]), [["b", 0], ["a", 1]]);
  assert.ok(runs[0].treeRow < runs[1].treeRow);
  assert.ok(runs.every(n => n.depth===1 && Math.abs(n.x - 250) <= 25));
  assert.equal(model(h).layoutStats.iterations, 0);
});

test("ResizeObserver preserves readable scale on dense graphs and allows scrolling", () => {
  const h = harness(); setup(h, [project("p", Array.from({ length: 15 }, (_, i) => run(String(i))))]);
  const host = h.get("project-graph"), view = host.graphView, viewport = view.viewport;
  const observer = h.observers.at(-1), initialScale = view.scale;
  assert.equal(observer.target, viewport);
  viewport.clientWidth = 320; viewport.clientHeight = 200; observer.callback();
  assert.ok(view.scale <= initialScale);assert.ok(view.scale>=.85);
  const plane = findClass(host, "graph-plane")[0];
  const [, dx, dy, scale] = plane.style.transform.match(/translate\(([-\d.e+]+)px,([-\d.e+]+)px\) scale\(([-\d.e+]+)\)/).map(Number);
  const nodes = model(h).nodes, xs = nodes.map(n => n.x * scale + dx), ys = nodes.map(n => n.y * scale + dy);
  assert.ok(Math.min(...xs) >= 0 && Math.max(...xs) <= viewport.clientWidth);
  assert.ok(Math.min(...ys)>=0);assert.ok(Math.max(...ys)>viewport.clientHeight);
  assert.ok(Math.abs((Math.min(...xs) + Math.max(...xs)) / 2 - viewport.clientWidth / 2) < 1);
  assert.ok(Number.parseFloat(findClass(host,"graph-extent")[0].style.height)>viewport.clientHeight);
  assert.equal(viewport.scrollLeft, 0); assert.equal(viewport.scrollTop, 0);
  const resizedScale = view.scale; observer.callback(); assert.equal(view.scale, resizedScale);
  h.run("renderProject()"); assert.equal(observer.disconnected, true);
  assert.equal(h.observers.at(-1).target, host.graphView.viewport);
});

test("automatic layout control clears manual pins, relaxes positions and fits the graph", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b")])]);
  h.run('state.activeScope.positions.set(JSON.stringify(["run","a"]),{x:1800,y:1200}); state.activeScope.pinned.add(JSON.stringify(["run","a"])); $("project-graph").graphView.autoFit=false; renderProject()');
  const before = model(h).nodes.find(n => n.runId === "a");
  const control = descendants(h.get("graph-tools")).find(n => n.attributes["aria-label"] === "自动调整节点布局");
  control.listeners.click();
  assert.equal(h.run("state.activeScope.pinned.size"), 0);
  const after = model(h).nodes.find(n => n.runId === "a");
  assert.notDeepEqual([after.x, after.y], [before.x, before.y]);
  assert.equal(h.get("project-graph").graphView.autoFit, true);
  assert.equal(h.get("project-graph").graphView.viewport.scrollLeft, 0);
  assert.equal(h.get("project-graph").graphView.viewport.scrollTop, 0);
  assert.equal(h.run("state.activeScope.selected"), "project-root");
  assert.equal(model(h).layoutStats.iterations, 0);
});

test("position storage survives reload, rejects malformed coordinates and handles denied storage", () => {
  const h=harness(),saved=new Map();h.ctx.localStorage={getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)};
  setup(h,[project("p",[run("a")])]);
  h.run('state.activeScope.positions.set(JSON.stringify(["run","a"]),{x:801,y:312});state.activeScope.pinned.add(JSON.stringify(["run","a"]));savePositions(state.activeScope)');
  assert.ok(saved.has('rsisys-runs-v5:' + h.run('state.activeScope.key')));
  for (const version of [2, 3]) saved.set(`rsisys-runs-v${version}:legacy`, JSON.stringify([["old", {x:900,y:900}]]));
  assert.equal(h.run('readPositions("legacy").size'), 0);
  const reloaded=harness();reloaded.ctx.localStorage=h.ctx.localStorage;setup(reloaded,[project("p",[run("a")])]);
  assert.deepEqual(model(reloaded).nodes.filter(n=>n.type==="run").map(n=>[n.x,n.y]),[[801,312]]);
  assert.equal(reloaded.run("state.activeScope.pinned.size"),0);
  saved.set('rsisys-runs-v5:bad',JSON.stringify([["nan",{x:null,y:3}],["negative",{x:-3,y:50}],["ok",{x:44,y:55}]]));
  assert.deepEqual(h.json('[...readPositions("bad")]'),[["ok",{x:44,y:55}]]);
  h.ctx.localStorage={getItem(){throw Error("denied");},setItem(){throw Error("full");}};
  assert.equal(h.run('readPositions("denied").size'),0);assert.doesNotThrow(()=>h.run('savePositions(state.activeScope)'));
});

test("hover highlights only a run's immediate neighbors and restores project selection on leave", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b")])]);
  const host = h.get("project-graph"), buttons = findClass(host, "graph-node");
  const a = buttons.find(n => n.dataset.nodeKey === JSON.stringify(["run", "a"])), b = buttons.find(n => n.dataset.nodeKey === JSON.stringify(["run", "b"]));
  const root = buttons.find(n => cssClass(n, "project"));
  a.listeners.mouseenter();
  assert.equal(a.attributes["data-related"], "true"); assert.equal(root.attributes["data-related"], "true"); assert.equal(b.attributes["data-related"], "false");
  assert.deepEqual(findClass(host, "graph-edge").map(n => n.attributes["data-related"]), ["true", "false"]);
  a.listeners.mouseleave(); assert.equal(b.attributes["data-related"], "true");
  assert.ok(findClass(host, "graph-edge").every(n => n.attributes["data-related"] === "true"));
});

test("without RAF dragging synchronously relaxes once per move and release without changing selection", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b")])]);
  h.run('$("project-graph").graphView.scale = .5');
  const host = h.get("project-graph"), key = JSON.stringify(["run", "a"]);
  const button = findClass(host, "graph-node").find(n => n.dataset.nodeKey === key);
  button.setPointerCapture = () => {};
  const x = parseFloat(button.style.left), y = parseFloat(button.style.top), before = model(h);
  const event = (cx, cy) => ({ button: 0, clientX: cx, clientY: cy, pointerId: 1, stopPropagation() {} });
  const paths = findClass(host, "graph-edge"), geometry = paths.map(p => p.attributes.d);
  h.run('simulationSteps=[]; originalNetworkLayout=networkLayout; networkLayout=function(...args){const result=originalNetworkLayout(...args);simulationSteps.push(result.layoutStats.iterations);return result;}');
  button.listeners.pointerdown(event(10, 20)); button.listeners.pointermove(event(40, 45));
  assert.equal(parseFloat(button.style.left), x + 60); assert.equal(parseFloat(button.style.top), y + 50);
  assert.notDeepEqual(paths.map(p => p.attributes.d), geometry);
  assert.deepEqual(h.json("simulationSteps"), [2]);
  assert.notDeepEqual(graphPositions(h).filter(n => n[0] !== key), before.nodes.filter(n => n.key !== key).map(n => [n.key, n.x, n.y]));
  button.listeners.pointerup(); button.listeners.click();
  assert.deepEqual(h.json("simulationSteps"), [2, 2]);
  const after = model(h);
  assert.deepEqual(h.json("[...state.activeScope.pinned]"), [key]);
  assert.deepEqual(h.json('state.activeScope.positions.get(JSON.stringify(["run","a"]))'), { x: x + 60, y: y + 50 });
  assert.equal(h.run("state.activeScope.selected"), "project-root"); assert.equal(h.run("state.runId"), "");
  assert.ok(after.nodes.some(n => n.key !== key && before.nodes.some(old => old.key === n.key && (old.x !== n.x || old.y !== n.y))));
  assert.equal(after.layoutStats.iterations, 0);
  h.run("renderProject()"); assert.deepEqual(model(h).nodes.map(n => [n.key, n.x, n.y]), after.nodes.map(n => [n.key, n.x, n.y]));
});

test("RAF moves other nodes before pointerup while the dragged node stays under the pointer", () => {
  const calls = [], h = harness(async url => { calls.push(url); throw Error("unexpected fetch"); }, "", { raf: true });
  setup(h, [project("p", [run("a"), run("b"), run("c", { parent_run_id: "b" })])]);
  seed(h, "b", run("b")); h.run('selectRun("b")');
  h.run('state.activeScope.positions.set(JSON.stringify(["run","b"]), {x:950,y:600}); state.activeScope.pinned.add(JSON.stringify(["run","b"])); renderProject(); $("project-graph").graphView.scale=.5');
  h.run('selectionCalls=0; originalSelectRun=selectRun; selectRun=function(...args){selectionCalls++;return originalSelectRun(...args);}');
  const host = h.get("project-graph"), view = host.graphView, button = graphButton(h, "a"), key = button.dataset.nodeKey;
  const before = graphPositions(h), original = before.find(n => n[0] === key);
  const unrelated = findClass(host, "linked")[0], oldLine = unrelated.attributes.d;
  const plane = findClass(host, "graph-plane")[0];
  button.setPointerCapture = () => {};
  button.listeners.pointerdown(pointer(10, 20)); button.listeners.pointermove(pointer(40, 45));
  const camera = plane.style.transform;
  const dragged = [key, original[1] + 60, original[2] + 50];
  assert.deepEqual(graphPositions(h).find(n => n[0] === key), dragged);
  assert.deepEqual(graphPositions(h).filter(n => n[0] !== key), before.filter(n => n[0] !== key), "other nodes wait for RAF");
  assert.equal(view.dragging, true); assert.equal(view.simulating, true); assert.equal(h.frames.size, 1);
  assert.equal(h.run("state.activeScope.pinned.size"), 0, "a new drag releases earlier pins");

  h.frames.step(); // Deliberately no pointerup: this is movement during the drag.
  const live = graphPositions(h);
  assert.equal(view.dragging, true);
  assert.deepEqual(live.find(n => n[0] === key), dragged, "simulation must not move the dragged node");
  assert.notDeepEqual(live.filter(n => n[0] !== key), before.filter(n => n[0] !== key));
  assert.notDeepEqual(live.find(n => n[0] === JSON.stringify(["run", "b"])), before.find(n => n[0] === JSON.stringify(["run", "b"])), "the previously pinned run moves too");
  assert.notEqual(unrelated.attributes.d, oldLine, "redraw includes edges not incident to the dragged node");
  assertStraightEdges(h);
  assert.equal(plane.style.transform, camera, "the camera stays still during simulation");
  assert.deepEqual(new Map(h.json("[...state.activeScope.positions]")), new Map(live.map(([key, x, y]) => [key, { x, y }])));
  h.frames.step();
  assert.notDeepEqual(graphPositions(h).filter(n => n[0] !== key), live.filter(n => n[0] !== key), "movement continues on the next frame");
  assert.deepEqual(graphPositions(h).find(n => n[0] === key), dragged);

  button.listeners.pointerup(); button.listeners.lostpointercapture(); button.listeners.click();
  assert.equal(view.dragging, false); assert.equal(view.simulating, true);
  assert.equal(h.frames.size, 1, "release reuses the pending RAF instead of starting another loop");
  assert.equal(h.frames.drain(), 18);
  assert.deepEqual(h.json("[...state.activeScope.pinned]"), [key]);
  assert.deepEqual(graphPositions(h).find(n => n[0] === key), dragged);
  assert.equal(h.run("selectionCalls"), 0); assert.equal(h.run("state.runId"), "b");
  assert.equal(view.selected, JSON.stringify(["run", "b"])); assert.deepEqual(calls, []);
  const settled = graphPositions(h); h.frames.step();
  assert.deepEqual(graphPositions(h), settled); assert.equal(view.simulating, false);
  assertStraightEdges(h);
});

test("RAF heat stops after 40 frames, reheats without duplicate callbacks and settles for 18 frames", () => {
  const h = harness(undefined, "", { raf: true }); setup(h, [project("p", [run("a"), run("b")])]);
  const button = startDrag(h), view = h.get("project-graph").graphView;
  for (let i = 0; i < 39; i++) { assert.equal(h.frames.size, 1); h.frames.step(); }
  assert.equal(view.simulating, true); h.frames.step();
  assert.equal(h.frames.size, 0); assert.equal(view.simulating, false); assert.equal(view.dragging, true);
  const cold = graphPositions(h); h.frames.step(); assert.deepEqual(graphPositions(h), cold);
  button.listeners.pointermove(pointer(45, 80));
  const queued = h.frames.ids;
  button.listeners.pointermove(pointer(50, 85)); button.listeners.pointermove(pointer(55, 90));
  assert.deepEqual(h.frames.ids, queued, "moves replenish heat on one existing animation");
  assert.equal(h.frames.drain(), 40); assert.equal(view.simulating, false);
  button.listeners.pointerup(); assert.equal(view.simulating, true);
  assert.equal(h.frames.drain(), 18); assert.equal(view.simulating, false);
});

test("renderProject defers redraw during dragging and settling, then renders once at rest", () => {
  const h = harness(undefined, "", { raf: true }); setup(h, [project("p", [run("a"), run("b")])]);
  const host = h.get("project-graph"), view = host.graphView, viewport = view.viewport, observer = h.observers.at(-1);
  const button = startDrag(h);
  h.run('state.projects[0].runs.push({run_id:"c", updated_at:"2026-11-01T01:00:00Z"}); renderProject(); renderProject()');
  assert.equal(view.pendingRender, true); assert.equal(view.viewport, viewport);
  assert.equal(graphPositions(h).length, 3); assert.equal(observer.disconnected, undefined);
  h.frames.drain(); // Heat can expire while the pointer is still down.
  assert.equal(view.dragging, true); assert.equal(view.simulating, false);
  assert.equal(view.viewport, viewport); assert.equal(view.pendingRender, true);
  button.listeners.pointerup(); h.run("renderProject(); renderProject()");
  assert.equal(view.viewport, viewport); assert.equal(view.simulating, true);
  assert.equal(h.frames.drain(), 18);
  assert.notEqual(view.viewport, viewport); assert.equal(observer.disconnected, true);
  assert.equal(view.pendingRender, false); assert.equal(graphPositions(h).length, 4);
  assert.equal(h.observers.length, 2, "coalesced requests create one replacement graph");
  assert.equal(h.frames.size, 0);
});

test("pointer taps below the drag threshold select once and start no RAF", () => {
  const calls = [], h = harness(async url => { calls.push(url); return response(run("a")); }, "", { raf: true });
  setup(h, [project("p", [run("a")])]); seed(h, "a", run("a"));
  h.run('selectionCalls=0; originalSelectRun=selectRun; selectRun=function(...args){selectionCalls++;return originalSelectRun(...args);}');
  const button = graphButton(h, "a"), before = graphPositions(h); button.setPointerCapture = () => {};
  button.listeners.pointerdown(pointer(10, 20)); button.listeners.pointermove(pointer(12, 21));
  button.listeners.pointerup(); button.listeners.lostpointercapture(); button.listeners.click();
  assert.equal(h.frames.size, 0); assert.equal(h.run("selectionCalls"), 1);
  assert.equal(h.run("state.runId"), "a"); assert.deepEqual(graphPositions(h), before);
  assert.equal(h.run("state.activeScope.pinned.size"), 0); assert.deepEqual(calls, []);
});

for (const phase of ["dragging", "settling"]) {
  for (const reason of ["stop", "project", "demo", "redraw", "hidden", "detached"]) {
    test(`${reason} cancels RAF while ${phase} and leaves no later node movement`, () => {
      const h = harness(undefined, "", { raf: true });
      setup(h, [project("p", [run("a"), run("b")]), project("other", [run("a")])]);
      const host = h.get("project-graph"), view = host.graphView, button = startDrag(h);
      h.frames.step(); if (phase === "settling") button.listeners.pointerup();
      const queued = h.frames.ids[0], oldViewport = view.viewport;
      if (reason === "stop") { view.stopSimulation(); view.stopSimulation(); }
      if (reason === "project") h.run('state.projectId="other"; activateProject(); renderProject()');
      if (reason === "demo") h.run('refresh=function(){}; switchDemo()');
      if (reason === "redraw") h.run('renderGraph($("project-graph"), projectModel(selectedProject(),state.activeScope), state.activeScope.key)');
      if (reason === "hidden") { h.document.hidden = true; h.documentListeners.visibilitychange(); }
      if (reason === "detached") { oldViewport.isConnected = false; h.frames.step(); }
      assert.equal(h.frames.size, 0); assert.equal(view.simulating, false); assert.equal(view.dragging, false);
      if (reason !== "detached") assert.ok(h.frames.cancelled.includes(queued), "the queued frame must be cancelled immediately");
      const stable = graphPositions(h), cached = h.json("[...state.activeScope?.positions || []]");
      // Old pointer events after cleanup must not restart the old simulation.
      button.listeners.pointermove(pointer(80, 100)); button.listeners.pointerup(); button.listeners.lostpointercapture();
      h.frames.step();
      assert.equal(h.frames.size, 0); assert.deepEqual(graphPositions(h), stable);
      assert.deepEqual(h.json("[...state.activeScope?.positions || []]"), cached);
      if (reason === "hidden") { h.document.hidden = false; h.documentListeners.visibilitychange(); assert.equal(h.frames.size, 0); }
    });
  }
}

test("redrawing an idle graph preserves newly computed positions instead of caching the old model", () => {
  const h = harness(undefined, "", { raf: true }); setup(h, [project("p", [run("a"), run("b")])]);
  const before = graphPositions(h);
  h.run('state.activeScope.positions.set(JSON.stringify(["run","a"]),{x:1000,y:800}); state.activeScope.pinned.add(JSON.stringify(["run","a"])); renderProject()');
  assert.deepEqual(graphPositions(h).find(n => n[0] === JSON.stringify(["run", "a"])), [JSON.stringify(["run", "a"]), 1000, 800]);
  assert.notDeepEqual(graphPositions(h), before);
  assert.deepEqual(h.json('state.activeScope.positions.get(JSON.stringify(["run","a"]))'), { x: 1000, y: 800 });
  assert.equal(h.frames.size, 0);
});

test("node controls have a 32px hit target and labels cannot intercept another node", () => {
  const css=fs.readFileSync(`${__dirname}/style.css`,"utf8");
  assert.match(css,/\.graph-node \{[^}]*width: 32px; height: 32px;/);
  assert.match(css,/\.graph-node \{[^}]*transform: translate\(-50%, -50%\)/);
  assert.match(css,/\.graph-node \.node-label \{[^}]*position: absolute;[^}]*pointer-events: none/);
});

test("run execution subview follows the graph viewport and never inserts internal graph nodes", () => {
  const h = harness(); setup(h, [project("p", [run("a")])]);
  seed(h, "a", run("a", { decisions: [{ decision_id: "train", mode: "train" }], jobs: [{ job_id: "j", kind: "train" }, { job_id: "e", kind: "evaluate" }], candidates: [{ candidate_id: "candidate", status: "accepted" }] }));
  h.run('selectRun("a")');
  const layout = findClass(h.get("project-graph"), "graph-layout")[0];
  assert.equal(layout.children.length, 2); assert.ok(cssClass(layout.children[0], "graph-viewport")); assert.ok(cssClass(layout.children[1], "graph-detail")); assert.equal(layout.children[1].tagName, "section");
  assert.equal(findClass(layout.children[0], "graph-node").length, 2);
  assert.equal(findClass(layout.children[0], "flow-choice").length, 0);
  assert.equal(findClass(layout.children[1], "execution-sequence").length, 1);
  assert.equal(findClass(layout.children[1], "candidate-card").length, 1);
  assert.doesNotMatch(textOf(layout.children[1]), /导出此运行报告/);
  assert.equal(h.get("export-report").listeners.click, undefined);
});

test("execution navigation pages all steps, shows one detail and retains selection across repeated run clicks", () => {
  const h = harness(); setup(h, [project("p", [run("a"), run("b")])]);
  const decisions = Array.from({ length: 35 }, (_, i) => ({ decision_id: `d${i}`, timestamp: new Date(Date.UTC(2026, 10, 1, 0, i)).toISOString(), mode: "train", reason: `reason-${i}-end`, metrics: { loss: i }, feedback: { summary: `feedback-${i}-end` } })).reverse();
  seed(h, "a", run("a", { decisions })); seed(h, "b", run("b", { decisions: [{ decision_id: "d0", mode: "evaluate", feedback: { summary: "only B" } }] }));
  h.run('selectRun("a")');
  const detail = findClass(h.get("project-graph"), "run-detail")[0];
  assert.equal(findClass(detail, "flow-choice").length, 30); assert.equal(findClass(detail, "flow-step").length, 1);
  assert.match(textOf(findClass(detail, "flow-step")[0]), /feedback-0-end/);
  const choices = findClass(detail, "flow-choice");
  assert.equal(choices[0].attributes["aria-label"], "步骤 1 · 训练模型");
  choices[29].listeners.click();
  const step = findClass(detail, "flow-step")[0];
  assert.match(textOf(step), /feedback-29-end/); assert.doesNotMatch(textOf(step), /feedback-0-end/);
  assert.match(textOf(findClass(step, "flow-decision")[0]), /reason-29-end/);
  assert.match(textOf(findClass(step, "flow-metrics")[0]), /\b29\b/);
  for (const name of ["flow-decision", "flow-feedback", "flow-metrics"]) assert.equal(findClass(step, name).length, 1);
  assert.equal(findClass(detail, "flow-choice").filter(n => n.attributes["aria-pressed"] === "true").length, 1);
  findClass(detail, "more-button")[0].listeners.click();
  assert.equal(findClass(detail, "flow-choice").length, 35); assert.equal(findClass(detail, "flow-step").length, 1);
  assert.equal(findClass(detail, "more-button")[0].hidden, true);
  findClass(detail, "flow-choice")[34].listeners.click();
  const clickA = () => findClass(h.get("project-graph"), "graph-node").find(n => n.dataset.nodeKey === JSON.stringify(["run", "a"])).listeners.click();
  clickA(); clickA();
  assert.match(textOf(findClass(h.get("project-graph"), "flow-step")[0]), /feedback-34-end/);
  assert.equal(findClass(h.get("project-graph"), "graph-node").length, 3);
  h.run('selectRun("b")'); assert.match(textOf(findClass(h.get("project-graph"), "flow-step")[0]), /only B/);
  clickA();
  assert.equal(findClass(h.get("project-graph"), "flow-choice").length, 35);
  assert.match(textOf(findClass(h.get("project-graph"), "flow-step")[0]), /feedback-34-end/);
});

test("unknown execution timestamps retain snapshot order without implying chronology; repeat finish retains originals", () => {
  const h=harness();h.ctx.fixture={decisions:[{decision_id:"u1",mode:"train"},{decision_id:"later",mode:"train",timestamp:"2026-11-01T01:00:00Z"},{decision_id:"u2",mode:"evaluate"},{decision_id:"early",mode:"train",timestamp:"2026-11-01T00:00:00Z"},{decision_id:"f1",kind:"finish",reason:"done"},{decision_id:"f2",kind:"finish",reason:"done"}]};
  const groups=h.json("executionGroups(fixture)");assert.deepEqual(groups.map(g=>g.row.decision_id),["early","later","u1","u2","f1"]);assert.equal(groups[4].rows.length,2);
});

test("execution detail keeps review metrics separate from exact current job and candidate results", () => {
  const h=harness();h.ctx.fixture={decisions:[{decision_id:"d",kind:"directive",directive_id:"dir",mode:"train"},{decision_id:"r",kind:"review",action:"continue",job_id:"job",metrics:{loss:.7}}],jobs:[{job_id:"job",directive_id:"dir",kind:"train",metrics:{loss:.1}}],candidates:[{candidate_id:"candidate",directive_id:"dir",metrics:{accuracy:.9}},{candidate_id:"other",directive_id:"outside",metrics:{accuracy:.5}}]};
  const results=h.json("relatedResults(fixture.decisions[0],fixture)");assert.deepEqual(results.map(r=>r.type),["job","candidate"]);assert.equal(results[1].row.candidate_id,"candidate");
  assert.deepEqual(h.json("feedbackFor(fixture.decisions[1],fixture).metrics"),{loss:.7});
  const detail=h.run('executionSequence({...fixture,run_id:"a"},{})');
  findClass(detail,"flow-choice")[1].listeners.click();
  assert.match(textOf(detail),/不代表复查当时状态/);
  assert.match(textOf(findClass(detail,"flow-metrics")[0]),/0\.7/);
  assert.doesNotMatch(textOf(findClass(detail,"flow-metrics")[0]),/0\.1/);
  assert.equal(h.json('relatedResults({job_id:"missing",directive_id:"dir"},fixture)').length,0);
  assert.equal(h.json('relatedResults({job_id:"dup"},{jobs:[{job_id:"dup"},{job_id:"dup"}]})').length,0);
});

test("known research actions have distinct names and custom operations retain their names in both languages",()=>{
  const h=harness();
  assert.equal(h.run('actionText(decisionAction({kind:"directive",mode:"configure_prompt"}))'),"调整提示词");
  assert.equal(h.run('actionText(decisionAction({kind:"operation",operation:"define_recipe"}))'),"定义训练方案");
  assert.equal(h.run('actionText(decisionAction({kind:"operation",action:"custom.compare"}))'),"custom.compare");
  assert.equal(h.run('actionText(undefined)'),"未命名决策");
  h.run('RSII18n.setLanguage("en")');
  assert.equal(h.run('actionText("configure_prompt")'),"Adjust prompt");
  assert.equal(h.run('actionText("define_recipe")'),"Define training recipe");
  assert.equal(h.run('actionText("custom.compare")'),"custom.compare");
});

test("root cannot be dragged and stays fixed while other nodes relax",()=>{
  const h=harness(undefined,"",{raf:true});setup(h,[project("p",[run("a"),run("b")])]);
  const root=findClass(h.get("project-graph"),"graph-node")[0];
  root.setPointerCapture=()=>{throw Error("fixed root must not capture drag");};
  root.listeners.pointerdown(pointer(0,0));root.listeners.pointermove(pointer(100,150));root.listeners.pointerup();
  assert.equal(h.frames.size,0);assert.deepEqual(model(h).nodes.filter(n=>n.type==="project").map(n=>[n.x,n.y]),[[70,245]]);
  const button=startDrag(h);h.frames.step();button.listeners.pointerup();h.frames.drain();
  assert.deepEqual(model(h).nodes.filter(n=>n.type==="project").map(n=>[n.x,n.y]),[[70,245]]);
});

test("unchanged polling reuses run detail DOM without a refreshing message",async()=>{
  const next=deferred(),snapshot=run("a",{candidates:[{candidate_id:"unlinked",status:"accepted",metrics:{score:.8}}],decisions:[{decision_id:"d",kind:"operation",operation:"define_recipe",reason:"Choose a short training run"}]});
  const h=harness(()=>next.promise);setup(h,[project("p",[run("a")])]);seed(h,"a",snapshot);h.run('selectRun("a")');
  const detail=findClass(h.get("project-graph"),"run-detail")[0],record=findClass(detail,"record-details")[0];record.open=true;
  const pending=h.run('loadRun("a",state.activeScope,true)');
  assert.equal(findClass(h.get("project-graph"),"run-detail")[0],detail);
  assert.doesNotMatch(textOf(detail),/正在刷新|正在读取/);
  next.resolve(response(JSON.parse(JSON.stringify(snapshot))));await pending;
  assert.equal(findClass(h.get("project-graph"),"run-detail")[0],detail);assert.equal(record.open,true);
  assert.equal(h.run('state.activeScope.cache.get("a").snapshot'),h.ctx.fixture);
});

test("new feedback preserves native disclosures and selected step with a stable decision identity",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{candidates:[{candidate_id:"unlinked",status:"accepted",metrics:{score:.8}}],decisions:[{decision_id:"d",kind:"operation",operation:"define_recipe",timestamp:"2026-11-01T00:00:02Z"}]}));h.run('selectRun("a")');
  const detail=findClass(h.get("project-graph"),"graph-detail")[0];
  // Real DOM selectors, represented by the existing fake-DOM tree for this test.
  detail.querySelectorAll=()=>findClass(detail,"record-details").filter(d=>d.dataset.retainKey);
  detail.querySelector=()=>findClass(detail,"flow-map")[0];
  const disclosure=findClass(detail,"record-details")[0];disclosure.open=true;
  const nav=findClass(detail,"flow-map")[0];nav.scrollLeft=123;
  h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,decisions:[{decision_id:"earlier",kind:"directive",mode:"configure_prompt",timestamp:"2026-11-01T00:00:01Z"},{...state.snapshot.decisions[0],feedback:{summary:"Recipe saved"}}]}; renderProject()');
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),"d");
  assert.match(textOf(findClass(detail,"flow-step")[0]),/Recipe saved/);
  assert.equal(findClass(detail,"record-details").find(d=>d.dataset.retainKey===disclosure.dataset.retainKey).open,true);
  assert.equal(findClass(detail,"flow-map")[0].scrollLeft,123);
});

test("execution view shows decisions and results without internal identifiers or event dumps",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{decisions:[{decision_id:"private-decision-id",directive_id:"private-directive-id",timestamp:"2026-11-01T00:00:01Z",kind:"directive",source:"model",mode:"configure_prompt",strategy:"configure_prompt",reason:"Try a shorter instruction",feedback:{summary:"Instruction updated"},decision_revision:"local-sha256:internal"}],events:[{kind:"internal-event",summary:"event-debug-text"}]}));
  h.run('selectRun("a")');
  const text=textOf(findClass(h.get("project-graph"),"run-detail")[0]);
  assert.match(text,/调整提示词/);assert.match(text,/Try a shorter instruction/);assert.match(text,/Instruction updated/);
  assert.doesNotMatch(text,/记录详情|运行详情|运行事件|private-decision-id|private-directive-id|local-sha256|configure_prompt|event-debug-text/);
  assert.equal(h.run('state.snapshot.decisions[0].decision_id'),"private-decision-id");
});

test("candidate metrics already shown for a decision are not duplicated",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{decisions:[{decision_id:"d",kind:"directive",mode:"evaluate",candidate_id:"c",metrics:{base_nll:1,candidate_nll:.8,eval_samples:10}}],candidates:[{candidate_id:"c",status:"accepted",metrics:{base_nll:1,candidate_nll:.8,eval_samples:10}}]}));h.run('selectRun("a")');
  const detail=findClass(h.get("project-graph"),"run-detail")[0];
  assert.equal(findClass(detail,"metric-table").length,1);
  assert.equal(findClass(detail,"candidate-card").length,0);
  assert.match(textOf(detail),/已采纳/);
});

test("polling retains the attached detail/layout and restores current page scroll synchronously",async()=>{
  const next=deferred(),h=harness(()=>next.promise);setup(h,[project("p",[run("a")])]);seed(h,"a",run("a",{decisions:[{decision_id:"d",kind:"finish"}]}));h.run('selectRun("a")');
  const view=h.get("project-graph").graphView,layout=view.layout,detail=view.details;
  h.ctx.window={scrollX:0,scrollY:700,scrollTo(options){this.scrollX=options.left;this.scrollY=options.top;assert.equal(options.behavior,"instant");}};
  const originalReplace=layout.replaceChild.bind(layout);
  layout.replaceChild=(...args)=>{originalReplace(...args);h.ctx.window.scrollY=0;};
  const pending=h.run('loadRun("a",state.activeScope,true)');h.ctx.window.scrollY=980;
  next.resolve(response(run("a",{decisions:[{decision_id:"d",kind:"finish",reason:"New feedback"}]})));await pending;
  assert.equal(h.ctx.window.scrollY,980);assert.equal(h.get("project-graph").graphView.layout,layout);assert.equal(h.get("project-graph").graphView.details,detail);
});

test("blank-space panning works in both directions at scroll zero and persists through polling",()=>{
  const h=harness();setup(h,[project("p",[run("a"),run("b")])]);
  const viewport=h.get("project-graph").graphView.viewport;viewport.setPointerCapture=()=>{};
  const before=findClass(h.get("project-graph"),"graph-plane")[0].style.transform;
  viewport.listeners.pointerdown({button:0,target:{closest:()=>null},clientX:100,clientY:100,pointerId:1});
  viewport.listeners.pointermove({clientX:160,clientY:220});viewport.listeners.pointerup();
  const view=h.get("project-graph").graphView;
  assert.equal(view.panX,60);assert.equal(view.panY,120);assert.equal(viewport.scrollTop,0);
  const moved=findClass(h.get("project-graph"),"graph-plane")[0].style.transform;assert.notEqual(moved,before);
  h.run('renderProject()');assert.equal(findClass(h.get("project-graph"),"graph-plane")[0].style.transform,moved);
  const current=h.get("project-graph").graphView.viewport;current.setPointerCapture=()=>{};
  current.listeners.pointerdown({button:0,target:{closest:()=>null},clientX:160,clientY:220,pointerId:2});
  current.listeners.pointermove({clientX:100,clientY:100});current.listeners.pointerup();
  assert.equal(view.panX,0);assert.equal(view.panY,0);assert.equal(findClass(h.get("project-graph"),"graph-plane")[0].style.transform,before);
});

test("ordinary refresh does not refit a scrolled automatic layout",()=>{
  const h=harness();setup(h,[project("p",[run("a"),run("b")])]);
  h.get("project-graph").graphView.viewport.scrollTop=150;
  assert.equal(h.get("project-graph").graphView.autoFit,true);
  h.run('renderProject()');assert.equal(h.get("project-graph").graphView.viewport.scrollTop,150);
});

test("sidebar selection switches project and node selection is the only run picker",()=>{
  const h=harness();setup(h,[project("p",[run("a")]),project("q",[run("b")])]);
  const buttons=h.get("project-list").children;assert.equal(buttons.length,2);assert.equal(buttons[0].attributes["aria-current"],"true");
  h.run('refresh=function(){}');buttons[1].listeners.click();assert.equal(h.run('state.projectId'),"q");
  assert.equal(h.get("project-list").children[1].attributes["aria-current"],"true");
  const stable=h.get("project-list").children[1];h.run('renderSelectors()');assert.equal(h.get("project-list").children[1],stable);
  assert.equal(h.get("run-select").listeners.change,undefined);
});

test("sidebar changes the deep link even when the new project has no selected cycle",()=>{
  const h=harness();setup(h,[project("p",[run("a")]),project("q",[run("b")])]);seed(h,"a",run("a"));h.run('selectRun("a"); refresh=function(){}; history.replaceState=function(_,__,url){savedURL=String(url)}; selectProject("q")');
  const url=new URL(h.run('savedURL'));assert.equal(url.searchParams.get('project'),'q');assert.equal(url.searchParams.has('run'),false);
});
