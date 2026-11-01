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

function descendants(node) { return [node,...node.children.flatMap(descendants)]; }
function textOf(node) { return [node.textContent||"",...node.children.map(textOf)].join(" "); }
function cssClass(node,name) { return (node.className||node.attributes.class||"").split(" ").includes(name); }
function findClass(node,name) { return descendants(node).filter(n=>cssClass(n,name)); }

test("network layout is deterministic under input reorder and keeps cached coordinates without further simulation", () => {
  const h=harness();
  const graph={nodes:[{key:"project-root",type:"project"},...Array.from({length:8},(_,i)=>({key:JSON.stringify(["run",String(i)]),type:"run",runId:String(i)}))],edges:Array.from({length:8},(_,i)=>({from:"project-root",to:JSON.stringify(["run",String(i)]),kind:"membership"}))};
  h.ctx.graph=graph;
  const a=h.json("networkLayout(structuredGraph = JSON.parse(JSON.stringify(graph)))");
  h.ctx.graph={nodes:graph.nodes.slice().reverse(),edges:graph.edges.slice().reverse()};
  const b=h.json("networkLayout(JSON.parse(JSON.stringify(graph)))");
  const coords=g=>g.nodes.map(n=>[n.key,n.x,n.y]).sort((a,b)=>a[0].localeCompare(b[0]));
  assert.deepEqual(coords(a),coords(b));
  h.ctx.graph=a;
  const cached=h.json('networkLayout(graph,new Map(graph.nodes.map(n=>[n.key,{x:n.x,y:n.y}])))');
  assert.deepEqual(coords(cached),coords(a));assert.equal(cached.layoutStats.iterations,0);
  assert.ok(new Set(a.nodes.map(n=>n.y.toFixed(1))).size>5);
});

test("large dense graph has finite budget and bounded grid probes; stationary refresh does no relaxation", () => {
  const h=harness();const count=1800;
  h.ctx.graph={nodes:Array.from({length:count},(_,i)=>({key:`n${i}`,type:"decision",runId:"r"})),edges:Array.from({length:count-1},(_,i)=>({from:`n${i}`,to:`n${i+1}`,kind:"time"}))};
  const result=h.json('networkLayout(graph)');
  assert.ok(result.layoutStats.iterations<=64);
  assert.ok(result.layoutStats.iterations*(count+count-1)<=60000);
  assert.ok(result.layoutStats.comparisons<=result.layoutStats.iterations*count*9*12);
  assert.ok(result.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)&&n.x>=36&&n.y>=36));
  assert.equal(h.run('networkLayout(graph,new Map(graph.nodes.map(n=>[n.key,{x:n.x,y:n.y}]))).layoutStats.iterations'),0);
});

test("position storage survives reload, rejects malformed coordinates and handles denied storage", () => {
  const h=harness(),saved=new Map();h.ctx.localStorage={getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)};
  setup(h,[project("p",[run("a")])]);
  h.run('state.activeScope.positions.set(JSON.stringify(["run","a"]),{x:801,y:312});savePositions(state.activeScope)');
  const reloaded=harness();reloaded.ctx.localStorage=h.ctx.localStorage;setup(reloaded,[project("p",[run("a")])]);
  assert.deepEqual(model(reloaded).nodes.filter(n=>n.type==="run").map(n=>[n.x,n.y]),[[801,312]]);
  saved.set('rsisys-network-v1:bad',JSON.stringify([["nan",{x:null,y:3}],["negative",{x:-3,y:50}],["ok",{x:44,y:55}]]));
  assert.deepEqual(h.json('[...readPositions("bad")]'),[["ok",{x:44,y:55}]]);
  h.ctx.localStorage={getItem(){throw Error("denied");},setItem(){throw Error("full");}};
  assert.equal(h.run('readPositions("denied").size'),0);assert.doesNotThrow(()=>h.run('savePositions(state.activeScope)'));
});

test("hover highlights exactly one-hop nodes and incident edges, restoring selection on leave", () => {
  const h=harness();setup(h,[project("p",[run("a"),run("b")])]);
  seed(h,"a",run("a",{decisions:[{decision_id:"d",mode:"train"}]}));h.run("renderProject()");
  const host=h.get("project-graph"),buttons=findClass(host,"graph-node"),a=buttons.find(n=>n.attributes["aria-label"]==="运行 1 · 收起"),decision=buttons.find(n=>n.attributes["aria-label"]==="训练模型 · 运行 1");
  decision.listeners.mouseenter();
  assert.equal(decision.attributes["data-related"],"true");assert.equal(a.attributes["data-related"],"true");
  assert.equal(buttons.find(n=>cssClass(n,"project")).attributes["data-related"],"false");
  decision.listeners.mouseleave();assert.equal(buttons.find(n=>cssClass(n,"project")).attributes["data-related"],"true");
  assert.equal(decision.attributes["data-related"],"false");
});

test("node dragging updates incident geometry and cache without selecting or collapsing its run", () => {
  const h=harness();setup(h,[project("p",[run("a")])]);seed(h,"a",run("a",{decisions:[{decision_id:"d",mode:"train"}]}));h.run('renderProject(); $("project-graph").graphView.scale=.5');
  const host=h.get("project-graph"),button=findClass(host,"graph-node").find(n=>cssClass(n,"run"));
  button.setPointerCapture=()=>{};const x=parseFloat(button.style.left),y=parseFloat(button.style.top);
  const event=(cx,cy)=>({button:0,clientX:cx,clientY:cy,pointerId:1,stopPropagation(){}});
  const paths=findClass(host,"graph-edge"),before=paths.map(p=>p.attributes.d);
  button.listeners.pointerdown(event(10,20));button.listeners.pointermove(event(40,45));button.listeners.pointerup();button.listeners.click();
  assert.equal(parseFloat(button.style.left),x+60);assert.equal(parseFloat(button.style.top),y+50);
  assert.equal(h.run('state.activeScope.expanded.has("a")'),true);
  assert.equal(h.run('state.activeScope.selected'),"project-root");
  assert.notDeepEqual(paths.map(p=>p.attributes.d),before);
  assert.deepEqual(h.json('state.activeScope.positions.get(JSON.stringify(["run","a"]))'),{x:x+60,y:y+50});
  h.run("renderProject()");assert.deepEqual(model(h).nodes.filter(n=>n.type==="run").map(n=>[n.x,n.y]),[[x+60,y+50]]);
});

test("node controls have a 24px hit target and labels cannot intercept another node", () => {
  const css=fs.readFileSync(`${__dirname}/style.css`,"utf8");
  assert.match(css,/\.graph-node \{[^}]*width: 24px; height: 24px;/);
  assert.match(css,/\.graph-node \{[^}]*transform: translate\(-50%, -50%\)/);
  assert.match(css,/\.graph-node \.node-label \{[^}]*position: absolute;[^}]*pointer-events: none/);
});

test("graph details follow the full-width viewport and job labels distinguish outcomes from decisions", () => {
  const h=harness();setup(h,[project("p",[run("a")])]);seed(h,"a",run("a",{decisions:[{decision_id:"train",mode:"train"}],jobs:[{job_id:"j",kind:"train"},{job_id:"e",kind:"evaluate"}]}));h.run("renderProject()");
  const graph=model(h);assert.deepEqual(graph.nodes.filter(n=>n.type==="job").map(n=>n.title),["训练结果","评测结果"]);
  const layout=findClass(h.get("project-graph"),"graph-layout")[0];
  assert.equal(layout.children.length,2);assert.ok(cssClass(layout.children[0],"graph-viewport"));assert.ok(cssClass(layout.children[1],"graph-detail"));assert.equal(layout.children[1].tagName,"section");
});

test("run detail exposes chronological decision, feedback and metrics for every step, with bounded initial rendering", () => {
  const h=harness();setup(h,[project("p",[run("a")])]);
  const decisions=Array.from({length:35},(_,i)=>({decision_id:`d${i}`,timestamp:new Date(Date.UTC(2026,10,1,0,i)).toISOString(),mode:"train",reason:`reason ${i}`,metrics:{loss:i},feedback:{summary:`feedback ${i}`}})).reverse();
  seed(h,"a",run("a",{decisions}));
  const detail=h.run('runDetail(projectModel(selectedProject(),state.activeScope).nodes.find(n=>n.type==="run"),state.activeScope)');
  let steps=findClass(detail,"flow-step");assert.equal(steps.length,30);assert.match(textOf(steps[0]),/feedback 0/);assert.match(textOf(steps[29]),/feedback 29/);
  for(const step of steps){assert.equal(findClass(step,"flow-decision").length,1);assert.equal(findClass(step,"flow-feedback").length,1);assert.equal(findClass(step,"flow-metrics").length,1);}
  const more=findClass(detail,"more-button")[0];more.listeners.click();assert.equal(findClass(detail,"flow-step").length,35);assert.equal(more.hidden,true);
  const rerender=h.run('runDetail(projectModel(selectedProject(),state.activeScope).nodes.find(n=>n.type==="run"),state.activeScope)');assert.equal(findClass(rerender,"flow-step").length,35);
});

test("unknown execution timestamps retain snapshot order without implying chronology; repeat finish retains originals", () => {
  const h=harness();h.ctx.fixture={decisions:[{decision_id:"u1",mode:"train"},{decision_id:"later",mode:"train",timestamp:"2026-11-01T01:00:00Z"},{decision_id:"u2",mode:"evaluate"},{decision_id:"early",mode:"train",timestamp:"2026-11-01T00:00:00Z"},{decision_id:"f1",kind:"finish",reason:"done"},{decision_id:"f2",kind:"finish",reason:"done"}]};
  const groups=h.json("executionGroups(fixture)");assert.deepEqual(groups.map(g=>g.row.decision_id),["early","later","u1","u2","f1"]);assert.equal(groups[4].rows.length,2);
});

test("decision detail includes only exact related results and keeps review metrics separate from current job metrics", () => {
  const h=harness();h.ctx.fixture={decisions:[{decision_id:"d",kind:"directive",directive_id:"dir",mode:"train"},{decision_id:"r",kind:"review",action:"continue",job_id:"job",metrics:{loss:.7}}],jobs:[{job_id:"job",directive_id:"dir",kind:"train",metrics:{loss:.1}}],candidates:[{candidate_id:"candidate",directive_id:"dir",metrics:{accuracy:.9}},{candidate_id:"other",directive_id:"outside",metrics:{accuracy:.5}}]};
  const results=h.json("relatedResults(fixture.decisions[0],fixture)");assert.deepEqual(results.map(r=>r.type),["job","candidate"]);assert.equal(results[1].row.candidate_id,"candidate");
  assert.deepEqual(h.json("feedbackFor(fixture.decisions[1],fixture).metrics"),{loss:.7});
  const detail=h.run('(()=>{const g=lineageModel(fixture);return graphDetail(g.nodes.find(n=>n.row.decision_id==="r"),g,fixture);})()');
  assert.match(textOf(detail),/不代表复查当时状态/);assert.match(textOf(detail),/相关结果 · 当前快照/);
  assert.equal(h.json('relatedResults({job_id:"missing",directive_id:"dir"},fixture)').length,0);
  assert.equal(h.json('relatedResults({job_id:"dup"},{jobs:[{job_id:"dup"},{job_id:"dup"}]})').length,0);
});
