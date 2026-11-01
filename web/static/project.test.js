"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(`${__dirname}/app.js`, "utf8");
const descendants = node => [node, ...node.children.flatMap(descendants)];
const textOf = node => descendants(node).map(n => n.textContent || "").join(" ");
const findClass = (node, name) => descendants(node).filter(n => (n.className || "").split(" ").includes(name));
class Element {
  constructor(tag = "div") { this.tagName = tag; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.scrollLeft = 0; this.scrollTop = 0; this.open = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get childElementCount() { return this.children.length; }
  get firstElementChild() { return this.children[0]; }
  setAttribute(k, v) { this.attributes[k] = String(v); if (k === "class") this.className = String(v); if (k.startsWith("data-")) this.dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(v); }
  getAttribute(k) { return this.attributes[k] ?? null; }
  addEventListener(k, f) { this.listeners[k] = f; }
  querySelectorAll(selector) { return descendants(this).filter(n => selector.startsWith(".") ? (n.className || "").split(" ").includes(selector.slice(1)) : selector === "details[data-retain-key]" && n.tagName === "details" && n.dataset.retainKey); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  focus() {}
}
function harness(fetcher = async () => { throw Error("unexpected fetch"); }, search = "") {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) { const e = new Element(); e.id = id; elements.set(id, e); } return elements.get(id); };
  const documentListeners = {}, document = { getElementById: get, createElement: t => new Element(t), createElementNS: (_, t) => new Element(t), addEventListener(k, f) { documentListeners[k] = f; }, hidden: false };
  const windowListeners = {}, location = { href: `http://localhost:8769/${search}`, search };
  const entries = [location.href]; let cursor = 0;
  const updateURL = url => { const next = new URL(String(url), location.href); location.href = next.href; location.search = next.search; location.hash = next.hash; };
  const history = {
    get length() { return entries.length; },
    replaceState(_, __, url) { updateURL(url); entries[cursor] = location.href; ctx.savedURL = location.href; },
    pushState(_, __, url) { updateURL(url); entries.splice(++cursor, Infinity, location.href); ctx.savedURL = location.href; },
    go(delta) { const next = cursor + delta; if (next < 0 || next >= entries.length) return; cursor = next; updateURL(entries[cursor]); ctx.savedURL = location.href; windowListeners.popstate?.({ state: null }); },
    back() { this.go(-1); }, forward() { this.go(1); },
  };
  const window = { addEventListener(k, f) { windowListeners[k] = f; }, scrollX: 0, scrollY: 0,
    scrollTo({ left = 0, top = 0 }) { this.scrollX = left; this.scrollY = top; } };
  const ctx = vm.createContext({ document, window, location, history, AbortController, URL, URLSearchParams, setTimeout, clearTimeout, fetch: fetcher });
  vm.runInContext(fs.readFileSync(`${__dirname}/i18n.js`, "utf8"), ctx);
  vm.runInContext(fs.readFileSync(`${__dirname}/decisions.js`, "utf8"), ctx);
  vm.runInContext(source.slice(0, source.lastIndexOf("\nrefresh();")), ctx);
  vm.runInContext("state.auto = false;", ctx);
  return { ctx, get, document, documentListeners, windowListeners, history, run: s => vm.runInContext(s, ctx), json: s => JSON.parse(JSON.stringify(vm.runInContext(s, ctx))) };
}
const run = (id, extra = {}) => ({ schema_version: 1, run_id: id, updated_at: "2026-11-01T00:00:00Z", ...extra });
const project = (id, runs = [run(id)], extra = {}) => ({ project_id: id, name: id, runs, ...extra });
const response = data => ({ ok: true, json: async () => data });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
function setup(h, projects) { h.ctx.input = projects; h.run("chooseProjects(input)"); }
function seed(h, id, snapshot) {
  h.ctx.fixture = snapshot; h.ctx.id = id;
  h.run('{ const key=JSON.stringify([state.demo,id]); let scope=state.scopes.get(key); if(!scope){scope={key,cache:new Map(),runId:id,details:null};state.scopes.set(key,scope);} scope.cache.set(id,{snapshot:fixture,status:"ready",token:0}); }');
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test("catalog preserves two real project groups with multiple experiments without merging equal project names", () => {
  const input = [
    project("p",[run("old",{experiment_name:"Baseline"}),run("current",{experiment_name:"Improved",parent_run_id:"old"})],{name:"Same project name"}),
    project("q",[run("third",{experiment_name:"Baseline"}),run("fourth",{experiment_name:"Comparison"})],{name:"Same project name"}),
  ];
  const before=JSON.stringify(input);
  const h = harness(); setup(h, input);
  assert.deepEqual(h.json("state.projects.map(p=>[p.project_id,p.runs.map(r=>r.run_id)])"), [["p",["old","current"]],["q",["third","fourth"]]]);
  const groups=findClass(h.get("project-list"),"project-group");assert.equal(groups.length,2);
  for(const group of groups){const heading=findClass(group,"project-group-heading")[0];assert.ok(heading);assert.match(textOf(heading),/Same project name/);assert.match(textOf(heading),/2 个实验/);}
  assert.deepEqual(groups.map(g=>findClass(g,"experiment-grid").length),[1,1]);
  assert.deepEqual(groups.map(g=>findClass(g,"experiment-card").map(c=>new URL(c.href,"http://localhost").searchParams.get("run"))),[["old","current"],["third","fourth"]]);
  assert.deepEqual(groups.map(g=>findClass(g,"experiment-card").map(c=>new URL(c.href,"http://localhost").searchParams.get("project"))),[["p","p"],["q","q"]]);
  assert.match(textOf(groups[0]),/Baseline.*Improved/);assert.match(textOf(groups[1]),/Baseline.*Comparison/);
  assert.equal(h.run("state.page"),"management");assert.equal(h.run("state.runId"),"");assert.equal(h.run("state.snapshot"),null);
  assert.equal(h.get("experiment-count").textContent,"2 个项目 · 4 个实验");
  assert.equal(JSON.stringify(input),before);
  assert.equal(findClass(h.get("project-content"), "graph-node").length, 0);
  assert.equal(findClass(h.get("project-content"), "graph-viewport").length, 0);
});

test("repeated project IDs merge their experiments while duplicate run declarations keep the first owner", () => {
  const h=harness();setup(h,[project("p",[run("a"),run("b")],{name:"Original project"}),project("p",[run("b"),run("c")]),project("q",[run("b"),run("d")])]);
  assert.deepEqual(h.json("state.projects.map(p=>[p.project_id,p.runs.map(r=>r.run_id)])"),[["p",["a","b","c"]],["q",["d"]]]);
  assert.equal(h.run("state.projects[0].name"),"Original project");
  assert.equal(findClass(h.get("project-list"),"project-group").length,2);
  assert.equal(findClass(h.get("project-list"),"experiment-card").length,4);
});

test("management home loads only the catalog until an experiment card is opened", async () => {
  const calls=[], snapshot=run("a", { objective:"Hidden research objective",decisions:[{decision_id:"d",mode:"train",reason:"Plan training",feedback:{summary:"Training result"}}] });
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("p",[run("a",{experiment_name:"Named experiment"}),run("b")],{name:"Parent project",description:"Project description is not its identity"})]}:snapshot);});
  await h.run("refresh()");
  assert.equal(h.run("state.page"),"management");assert.equal(h.get("management").hidden,false);assert.equal(h.get("projects").hidden,true);
  assert.equal(h.run("state.snapshot"),null);assert.equal(h.get("project-content").children.length,0);
  assert.deepEqual(calls,["/api/projects"]);
  assert.equal(findClass(h.get("project-list"),"experiment-card")[0].href,"?project=p&run=a");
  findClass(h.get("project-list"),"experiment-card")[0].listeners.click({preventDefault(){}});await settle();
  assert.equal(h.run("state.page"),"experiment");assert.equal(h.get("management").hidden,true);assert.equal(h.get("projects").hidden,false);
  assert.equal(h.run("state.projectId"),"p");assert.equal(h.run("state.runId"),"a");
  assert.equal(h.get("project-name").textContent,"Named experiment");assert.equal(h.get("project-description").textContent,"Parent project");assert.equal(h.get("project-description").hidden,false);
  assert.doesNotMatch(textOf(h.get("project-content")),/Hidden research objective/);
  assert.equal(h.run("state.snapshot.run_id"),"a");assert.match(textOf(h.get("project-content")),/Plan training.*Training result/);
  assert.deepEqual(calls,["/api/projects","/api/runs/a"]);
});

test("experiments within one project isolate cached histories, selected graph nodes and both scroll axes", async () => {
  const requests=[], b=run("b",{decisions:[{decision_id:"same",mode:"evaluate",reason:"Only B"}]});
  const h=harness(async url=>{requests.push(url);return response(b);});setup(h,[project("p",[run("a"),run("b")])]);
  const a=run("a",{decisions:Array.from({length:35},(_,i)=>({decision_id:`d${i}`,timestamp:new Date(Date.UTC(2026,10,1,0,i)).toISOString(),mode:"train",reason:`history-${i}-end`}))});
  seed(h,"a",a);await h.run('selectRun("a")');
  const allNodes=findClass(h.get("project-content"),"decision-node");
  assert.equal(allNodes.length,35);assert.deepEqual(allNodes.map(n=>n.dataset.key),a.decisions.map(d=>d.decision_id));
  assert.equal(findClass(h.get("project-content"),"more-button").length,0);
  const aMap=findClass(h.get("project-content"),"flow-map")[0];aMap.scrollLeft=321;aMap.scrollTop=87;
  allNodes[34].listeners.click();
  assert.equal(aMap.scrollLeft,321);assert.equal(aMap.scrollTop,87);
  assert.equal(findClass(h.get("project-content"),"decision-node")[34].getAttribute("aria-pressed"),"true");
  assert.equal(findClass(h.get("project-content"),"flow-step").length,1);
  const aScope=h.run("state.activeScope"),aCache=h.run("state.activeScope.cache");
  const detail=h.get("project-content").firstElementChild;
  h.get("back-projects").listeners.click();
  findClass(h.get("project-list"),"experiment-card")[1].listeners.click();await settle();
  assert.equal(h.run("state.projectId"),"p");assert.equal(h.run("state.runId"),"b");
  assert.equal(h.get("project-name").textContent,"b");assert.equal(h.get("project-description").textContent,"p");assert.equal(h.get("project-description").hidden,false);
  assert.match(textOf(h.get("project-content")),/Only B/);assert.doesNotMatch(textOf(h.get("project-content")),/history-34-end/);
  assert.notEqual(h.run("state.activeScope"),aScope);assert.notEqual(h.run("state.activeScope.cache"),aCache);
  assert.equal(h.run("state.activeScope.key"),JSON.stringify([false,"b"]));assert.equal(aScope.key,JSON.stringify([false,"a"]));
  assert.deepEqual(h.json('[...state.activeScope.cache.keys()]'),["b"]);
  assert.equal(h.run('state.activeScope.flowSelected.get("b")'),"same");
  const bMap=findClass(h.get("project-content"),"flow-map")[0];bMap.scrollLeft=18;bMap.scrollTop=29;
  assert.equal(new URL(h.ctx.savedURL).searchParams.get("run"),"b");
  h.get("back-projects").listeners.click();
  findClass(h.get("project-list"),"experiment-card")[0].listeners.click();await settle();
  assert.equal(h.get("project-content").firstElementChild,detail);
  assert.match(textOf(h.get("project-content")),/history-34-end/);
  assert.equal(findClass(h.get("project-content"),"decision-node").length,35);
  assert.equal(h.run("state.activeScope"),aScope);assert.equal(h.run("state.activeScope.cache"),aCache);
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),"d34");
  assert.equal(findClass(detail,"flow-map")[0].scrollLeft,321);assert.equal(findClass(detail,"flow-map")[0].scrollTop,87);
  assert.equal(h.run('state.snapshot.decisions.length'),35);
  h.run('showManagement()');await h.run('selectRun("b")');
  assert.equal(findClass(h.get("project-content"),"flow-map")[0],bMap);
  assert.equal(bMap.scrollLeft,18);assert.equal(bMap.scrollTop,29);
  assert.deepEqual(requests,["/api/runs/b"]);
});

test("run deep links open a non-first experiment and its real owner overrides a conflicting project query", async () => {
  const catalog=[project("p",[run("a"),run("b",{experiment_name:"Second P experiment"})],{name:"Project P"}),project("q",[run("c"),run("d",{experiment_name:"Second Q experiment"})],{name:"Project Q"})];
  for(const [search,expected,owner,name] of [["?run=b","b","p","Second P experiment"],["?project=wrong&run=b","b","p","Second P experiment"],["?project=p&run=b","b","p","Second P experiment"],["?project=q&run=b","b","p","Second P experiment"],["?project=p&run=d","d","q","Second Q experiment"]]){
    const calls=[],h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:catalog}:run(expected,{decisions:[{decision_id:"d",reason:`opened ${expected}`}]}));},search);
    await h.run("refresh()");
    assert.equal(h.run("state.page"),"experiment");assert.equal(h.get("management").hidden,true);assert.equal(h.get("projects").hidden,false);
    assert.equal(h.run("state.projectId"),owner);assert.equal(h.run("state.runId"),expected);
    assert.equal(h.get("project-name").textContent,name);assert.equal(h.get("project-description").textContent,owner==="p"?"Project P":"Project Q");assert.equal(h.get("project-description").hidden,false);
    assert.match(textOf(h.get("project-content")),new RegExp(`opened ${expected}`));
    assert.deepEqual(calls,["/api/projects",`/api/runs/${expected}`]);
    assert.equal(new URL(h.ctx.savedURL).searchParams.get("project"),owner);
    assert.equal(new URL(h.ctx.savedURL).searchParams.get("run"),expected);
  }
});

test("project-only deep links keep management open without reading a first experiment", async () => {
  for(const search of ["?project=p","?project=q","?project=single"]){
    const calls=[],h=harness(async url=>{calls.push(url);assert.equal(url,"/api/projects");return response({projects:[project("p",[run("a"),run("b")]),project("q",[run("c"),run("d")]),project("single")]});},search);
    await h.run("refresh()");await h.run("refresh()");
    assert.equal(h.run("state.page"),"management");assert.equal(h.get("management").hidden,false);assert.equal(h.get("projects").hidden,true);
    assert.equal(h.run("state.runId"),"");assert.equal(h.run("state.snapshot"),null);assert.equal(h.get("project-content").children.length,0);
    assert.equal(findClass(h.get("project-list"),"project-group").length,3);assert.deepEqual(calls,["/api/projects","/api/projects"]);
  }
});

test("selectProject keeps the legacy run-ID entry point without opening a multi-run project", async () => {
  const calls=[],h=harness(async url=>{calls.push(url);return response(run("single",{decisions:[{reason:"Legacy detail"}]}));});setup(h,[project("p",[run("a"),run("b")]),project("single")]);
  await h.run('selectProject("p")');assert.equal(h.run("state.page"),"management");assert.deepEqual(calls,[]);
  await h.run('selectProject("single")');assert.equal(h.run("state.projectId"),"single");assert.equal(h.run("state.runId"),"single");assert.match(textOf(h.get("project-content")),/Legacy detail/);assert.deepEqual(calls,["/api/runs/single"]);
});

test("back button returns to management, keeps cached detail and polls only the catalog", async () => {
  const calls=[],snapshot=run("a",{decisions:[{decision_id:"d",reason:"Cached A"}]});
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("a")]}:snapshot);});
  await h.run("refresh()");await h.run('selectRun("a")');
  const scope=h.run("state.activeScope"),entry=h.run('state.activeScope.cache.get("a")'),detail=h.get("project-content").firstElementChild;
  h.get("back-projects").listeners.click();
  assert.equal(h.run("state.page"),"management");assert.equal(h.get("management").hidden,false);assert.equal(h.get("projects").hidden,true);
  assert.equal(new URL(h.ctx.savedURL).searchParams.has("run"),false);assert.equal(new URL(h.ctx.savedURL).searchParams.has("project"),false);
  await h.run("refresh()");
  assert.deepEqual(calls,["/api/projects","/api/runs/a","/api/projects"]);
  assert.equal(h.run("state.activeScope"),scope);assert.equal(h.run('state.activeScope.cache.get("a")'),entry);
  await h.run('selectRun("a")');
  assert.equal(h.get("project-content").firstElementChild,detail);assert.match(textOf(detail),/Cached A/);
  assert.equal(calls.length,3);
});

test("browser back and forward restore management and experiments without adding history or fetching cached runs", async () => {
  const catalog=[project("p",[run("a"),run("b")]),project("q",[run("c"),run("d")])];
  const calls=[],h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:catalog}:run(url.endsWith("/b")?"b":"d",{decisions:[{decision_id:"shared",reason:url.endsWith("/b")?"History B":"History D"}]}));});
  await h.run("refresh()");await h.run('selectRun("b")');
  const bDetail=h.get("project-content").firstElementChild;
  await h.run('selectRun("d")');const dDetail=h.get("project-content").firstElementChild;
  const length=h.history.length;assert.equal(length,3);
  h.history.back();await settle();
  assert.equal(h.run("state.page"),"experiment");assert.equal(h.run("state.projectId"),"p");assert.equal(h.run("state.runId"),"b");assert.equal(h.get("project-content").firstElementChild,bDetail);
  h.history.back();await settle();
  assert.equal(h.run("state.page"),"management");assert.equal(h.get("projects").hidden,true);assert.equal(h.ctx.location.search,"");
  h.history.forward();await settle();
  assert.equal(h.run("state.page"),"experiment");assert.equal(h.run("state.projectId"),"p");assert.equal(h.run("state.runId"),"b");assert.equal(h.get("project-content").firstElementChild,bDetail);
  h.history.forward();await settle();
  assert.equal(h.run("state.projectId"),"q");assert.equal(h.run("state.runId"),"d");assert.equal(h.get("project-content").firstElementChild,dDetail);assert.match(textOf(dDetail),/History D/);
  assert.equal(new URL(h.ctx.savedURL).searchParams.get("project"),"q");assert.equal(new URL(h.ctx.savedURL).searchParams.get("run"),"d");
  assert.equal(h.history.length,length);assert.deepEqual(calls,["/api/projects","/api/runs/b","/api/runs/d"]);
});

test("returning to management aborts a pending detail and a late response cannot reopen it", async () => {
  const pending=deferred();let signal;
  const h=harness((url,options)=>{signal=options.signal;return pending.promise;});setup(h,[project("a")]);
  const request=h.run('selectRun("a")');h.get("back-projects").listeners.click();
  assert.equal(signal.aborted,true);
  pending.resolve(response(run("a",{decisions:[{decision_id:"late",reason:"Late detail"}]})));await request;
  assert.equal(h.run("state.page"),"management");assert.equal(h.get("projects").hidden,true);
  assert.equal(h.run("state.snapshot"),null);assert.doesNotMatch(textOf(h.get("project-content")),/Late detail/);
});

test("parent and source references are provenance only and never select, fetch or combine other runs", async () => {
  const calls=[],snapshot=run("current",{parent_run_id:"parent",source_run_ids:["parent","outside"],decisions:[{decision_id:"d",reason:"Current decision"}]});
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("current",[snapshot]),project("parent")]}:snapshot);},"?run=current");
  await h.run("refresh()");
  assert.equal(h.run("state.projectId"),"current");assert.deepEqual(h.json("state.projects.map(p=>p.runs.length)"),[1,1]);
  assert.deepEqual(h.json("[...state.activeScope.cache.keys()]"),["current"]);
  assert.deepEqual(calls,["/api/projects","/api/runs/current"]);
  assert.equal(findClass(h.get("project-content"),"provenance-link").length,2);
  assert.equal(findClass(h.get("project-content"),"provenance-link")[0].href,"?run=parent");
  assert.match(textOf(h.get("project-content")),/来源引用不表示项目归属或自动继承/);
});

test("same record IDs keep decisions, feedback, jobs and candidates isolated within and across projects", async () => {
  for(const catalog of [[project("p",[run("a"),run("b")])],[project("p",[run("a")]),project("q",[run("b")])]]){
    const h=harness();setup(h,catalog);
    for(const [id,loss] of [["a",1],["b",2]])seed(h,id,run(id,{decisions:[{decision_id:"same",kind:"directive",directive_id:"shared",mode:"train",feedback:{summary:`feedback-${id}`}}],jobs:[{job_id:"same",directive_id:"shared",metrics:{loss}}],candidates:[{candidate_id:"same",directive_id:"shared",metrics:{accuracy:loss/10}}]}));
    for(const [id,other,loss] of [["a","b",1],["b","a",2],["a","b",1]]){
      await h.run(`selectRun("${id}")`);assert.match(textOf(h.get("project-content")),new RegExp(`feedback-${id}`));
      assert.doesNotMatch(textOf(h.get("project-content")),new RegExp(`feedback-${other}`));assert.equal(h.run("state.snapshot.jobs[0].metrics.loss"),loss);
    }
  }
});

test("switching experiments within or across projects aborts pending reads and rejects late responses", async () => {
  for(const catalog of [[project("p",[run("a"),run("b")])],[project("p",[run("a")]),project("q",[run("b")])]]){
    const old=deferred(),fresh=deferred(),signals=[],h=harness((url,options)=>{signals.push(options.signal);return signals.length===1?old.promise:fresh.promise;});
    setup(h,catalog);seed(h,"b",run("b",{decisions:[{reason:"B history"}]}));
    const pending=h.run('selectRun("a")');await h.run('selectRun("b")');assert.equal(signals[0].aborted,true);
    const current=h.run('selectRun("a")');fresh.resolve(response(run("a",{decisions:[{reason:"Fresh A"}]})));await current;
    old.resolve(response(run("a",{decisions:[{reason:"Stale A"}]})));await pending;
    assert.match(textOf(h.get("project-content")),/Fresh A/);assert.doesNotMatch(textOf(h.get("project-content")),/Stale A|B history/);
  }
});

test("repeated selection reuses pending requests and cached decisions", async () => {
  const next=deferred(),calls=[],h=harness(url=>{calls.push(url);return next.promise;});setup(h,[project("a")]);
  const first=h.run('selectRun("a")');h.run('selectRun("a")');
  assert.deepEqual(calls,["/api/runs/a"]);assert.match(textOf(h.get("project-content")),/正在读取/);
  next.resolve(response(run("a",{decisions:[{reason:"Loaded"}]})));await first;await h.run('selectRun("a")');
  assert.deepEqual(calls,["/api/runs/a"]);assert.match(textOf(h.get("project-content")),/Loaded/);
});

test("refresh reloads only the selected experiment and preserves another experiment's cache in the same project", async () => {
  const calls=[],a=run("a",{decisions:Array.from({length:35},(_,i)=>({decision_id:`d${i}`,mode:"train",feedback:{summary:"New result"}}))});
  const catalog=[project("p",[run("a"),run("b")]),project("q",[run("c"),run("d")])];
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:catalog}:a);});setup(h,catalog);
  seed(h,"a",{...a,decisions:a.decisions.map(d=>({...d,feedback:{summary:"Previous result"}}))});seed(h,"b",run("b",{decisions:[{reason:"B remains"}]}));
  await h.run('selectRun("b")');const bScope=h.run("state.activeScope"),bEntry=h.run('state.activeScope.cache.get("b")'),bDetail=h.get("project-content").firstElementChild;
  await h.run('selectRun("a")');findClass(h.get("project-content"),"decision-node")[34].listeners.click();
  const map=findClass(h.get("project-content"),"flow-map")[0];map.scrollLeft=456;map.scrollTop=78;
  await h.run("refresh()");
  assert.deepEqual(calls,["/api/projects","/api/runs/a"]);assert.equal(h.run('state.activeScope.flowSelected.get("a")'),"d34");
  const nodes=findClass(h.get("project-content"),"decision-node");assert.equal(nodes.length,35);assert.equal(nodes[34].getAttribute("aria-pressed"),"true");
  assert.equal(findClass(h.get("project-content"),"flow-map")[0].scrollLeft,456);assert.equal(findClass(h.get("project-content"),"flow-map")[0].scrollTop,78);
  assert.match(textOf(h.get("project-content")),/New result/);
  assert.equal(h.run("state.projectId"),"p");assert.deepEqual(h.json('[...state.activeScope.cache.keys()]'),["a"]);
  await h.run('selectRun("b")');assert.match(textOf(h.get("project-content")),/B remains/);assert.equal(calls.length,2);
  assert.equal(h.run("state.projectId"),"p");assert.equal(h.run("state.activeScope"),bScope);assert.equal(h.run('state.activeScope.cache.get("b")'),bEntry);assert.equal(h.get("project-content").firstElementChild,bDetail);assert.deepEqual(h.json('[...state.activeScope.cache.keys()]'),["b"]);
});

test("failed reads retain decisions and a retry restores the current project", async () => {
  let fail=true;const h=harness(async()=>{if(fail)throw Error("offline");return response(run("a",{decisions:[{reason:"Recovered"}]}));});
  setup(h,[project("a")]);seed(h,"a",run("a",{decisions:[{reason:"Cached decision"}]}));await h.run('selectRun("a")');
  await h.run('loadRun("a",state.activeScope,true)');assert.match(textOf(h.get("project-content")),/offline.*Cached decision/);
  fail=false;findClass(h.get("project-content"),"run-detail")[0].children.find(n=>n.tagName==="button").listeners.click();await settle();
  assert.match(textOf(h.get("project-content")),/Recovered/);assert.equal(h.run('state.activeScope.cache.get("a").status'),"ready");
});

test("demo groups two experiments in one project, loads their decisions and keeps live caches and links", async () => {
  const demo=JSON.parse(fs.readFileSync(`${__dirname}/demo.json`,"utf8")),h=harness(async()=>response(demo));
  setup(h,[project("live")]);seed(h,"live",run("live",{decisions:[{reason:"Live history"}]}));await h.run('selectRun("live")');
  const liveURL=h.ctx.savedURL;h.run('state.demo=true;state.projectId=""');await h.run("refresh()");
  assert.equal(h.run("state.projects.length"),1);assert.equal(h.run("state.projects[0].project_id"),"example-reasoning");
  assert.equal(findClass(h.get("project-list"),"project-group").length,1);assert.equal(findClass(h.get("project-list"),"experiment-card").length,2);
  for(const snapshot of [demo,...demo.demo_runs]){
    assert.equal(snapshot.project.project_id,"example-reasoning");await h.run(`selectRun(${JSON.stringify(snapshot.run_id)})`);
    assert.equal(h.run("state.projectId"),snapshot.project.project_id);assert.equal(h.get("project-name").textContent,snapshot.experiment_name);assert.equal(h.get("project-description").textContent,snapshot.project.name);assert.equal(h.get("project-description").hidden,false);
    assert.equal(h.run("state.snapshot.run_id"),snapshot.run_id);assert.equal(findClass(h.get("project-content"),"execution-sequence").length,1);
  }
  assert.equal(h.ctx.savedURL,liveURL);
  h.run('state.demo=false;state.projectId="live"');setup(h,[project("live")]);await h.run('selectRun("live")');
  assert.match(textOf(h.get("project-content")),/Live history/);
});

test("switching to demo rejects a late live response before the demo catalog arrives", async () => {
  const pending=deferred();let signal;const h=harness((url,options)=>{signal=options.signal;return pending.promise;});setup(h,[project("a")]);
  const request=h.run('selectRun("a")');h.run('refresh=function(){};switchDemo()');assert.equal(signal.aborted,true);
  pending.resolve(response(run("a",{decisions:[{reason:"Late live history"}]})));await request;
  assert.equal(h.run("state.snapshot"),null);assert.equal(h.run("state.activeScope"),null);
});

test("empty catalog and unavailable snapshot have readable states without a project graph", async () => {
  const h=harness(async url=>response({projects:[]}));await h.run("refresh()");
  assert.equal(h.get("empty").hidden,false);assert.equal(h.get("projects").hidden,true);assert.equal(h.get("project-list").children.length,1);
  setup(h,[project("broken",[run("broken",{status:"unavailable"})])]);
  await h.run('selectRun("broken")');
  assert.equal(h.run("state.runId"),"broken");assert.match(textOf(h.get("project-content")),/快照不可用/);
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
  findClass(detail,"decision-node")[1].listeners.click();
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


test("unchanged polling reuses run detail DOM without a refreshing message",async()=>{
  const next=deferred(),snapshot=run("a",{candidates:[{candidate_id:"unlinked",status:"accepted",metrics:{score:.8}}],decisions:[{decision_id:"d",kind:"operation",operation:"define_recipe",reason:"Choose a short training run"}]});
  const h=harness(()=>next.promise);setup(h,[project("p",[run("a")])]);seed(h,"a",snapshot);h.run('selectRun("a")');
  const detail=findClass(h.get("project-content"),"run-detail")[0],record=findClass(detail,"record-details")[0];record.open=true;
  const pending=h.run('loadRun("a",state.activeScope,true)');
  assert.equal(findClass(h.get("project-content"),"run-detail")[0],detail);
  assert.doesNotMatch(textOf(detail),/正在刷新|正在读取/);
  next.resolve(response(JSON.parse(JSON.stringify(snapshot))));await pending;
  assert.equal(findClass(h.get("project-content"),"run-detail")[0],detail);assert.equal(record.open,true);
  assert.equal(h.run('state.activeScope.cache.get("a").snapshot'),h.ctx.fixture);
});


test("new feedback preserves native disclosures and selected step with a stable decision identity",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{candidates:[{candidate_id:"unlinked",status:"accepted",metrics:{score:.8}}],decisions:[{decision_id:"d",kind:"operation",operation:"define_recipe",timestamp:"2026-11-01T00:00:02Z"}]}));h.run('selectRun("a")');
  const detail=findClass(h.get("project-content"),"project-detail")[0];
  // Real DOM selectors, represented by the existing fake-DOM tree for this test.
  detail.querySelectorAll=()=>findClass(detail,"record-details").filter(d=>d.dataset.retainKey);
  detail.querySelector=()=>findClass(detail,"flow-map")[0];
  const disclosure=findClass(detail,"record-details")[0];disclosure.open=true;
  const nav=findClass(detail,"flow-map")[0];nav.scrollLeft=123;nav.scrollTop=67;
  h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,decisions:[{decision_id:"earlier",kind:"directive",mode:"configure_prompt",timestamp:"2026-11-01T00:00:01Z"},{...state.snapshot.decisions[0],feedback:{summary:"Recipe saved"}}]}; renderProject()');
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),"d");
  assert.match(textOf(findClass(detail,"flow-step")[0]),/Recipe saved/);
  assert.equal(findClass(detail,"record-details").find(d=>d.dataset.retainKey===disclosure.dataset.retainKey).open,true);
  assert.equal(findClass(detail,"flow-map")[0].scrollLeft,123);assert.equal(findClass(detail,"flow-map")[0].scrollTop,67);
});

test("duplicate decision IDs select one graph record and retain it when earlier records and feedback arrive", async()=>{
  const h=harness();setup(h,[project("a")]);
  seed(h,"a",run("a",{decisions:[
    {decision_id:"duplicate",kind:"directive",mode:"train",timestamp:"2026-11-01T00:00:01Z",reason:"First record"},
    {decision_id:"duplicate",kind:"directive",mode:"evaluate",timestamp:"2026-11-01T00:00:02Z",reason:"Second record"},
  ]}));await h.run('selectRun("a")');
  const nodes=findClass(h.get("project-content"),"decision-node");
  assert.equal(nodes.length,2);assert.notEqual(nodes[0].dataset.key,nodes[1].dataset.key);
  nodes[1].listeners.click();const key=nodes[1].dataset.key;
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),key);
  assert.equal(findClass(h.get("project-content"),"flow-step").length,1);
  assert.match(textOf(findClass(h.get("project-content"),"flow-step")[0]),/Second record/);
  assert.doesNotMatch(textOf(findClass(h.get("project-content"),"flow-step")[0]),/First record/);
  h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,decisions:[{decision_id:"duplicate",kind:"directive",mode:"configure_prompt",timestamp:"2026-11-01T00:00:00Z",reason:"Earlier record"},state.snapshot.decisions[0],{...state.snapshot.decisions[1],feedback:{summary:"Evaluation observed"}}]};renderProject()');
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),key);
  const selected=findClass(h.get("project-content"),"decision-node").filter(n=>n.getAttribute("aria-pressed")==="true");
  assert.equal(selected.length,1);assert.equal(selected[0].dataset.key,key);
  assert.equal(findClass(h.get("project-content"),"flow-step").length,1);
  assert.match(textOf(findClass(h.get("project-content"),"flow-step")[0]),/Second record.*Evaluation observed/);
});


test("execution view shows decisions and results without internal identifiers or event dumps",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{decisions:[{decision_id:"private-decision-id",directive_id:"private-directive-id",timestamp:"2026-11-01T00:00:01Z",kind:"directive",source:"model",mode:"configure_prompt",strategy:"configure_prompt",reason:"Try a shorter instruction",feedback:{summary:"Instruction updated"},decision_revision:"local-sha256:internal"}],events:[{kind:"internal-event",summary:"event-debug-text"}]}));
  h.run('selectRun("a")');
  const text=textOf(findClass(h.get("project-content"),"run-detail")[0]);
  assert.match(text,/调整提示词/);assert.match(text,/Try a shorter instruction/);assert.match(text,/Instruction updated/);
  assert.doesNotMatch(text,/记录详情|运行详情|运行事件|private-decision-id|private-directive-id|local-sha256|configure_prompt|event-debug-text/);
  assert.equal(h.run('state.snapshot.decisions[0].decision_id'),"private-decision-id");
});


test("candidate metrics already shown for a decision are not duplicated",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{decisions:[{decision_id:"d",kind:"directive",mode:"evaluate",candidate_id:"c",metrics:{base_nll:1,candidate_nll:.8,eval_samples:10}}],candidates:[{candidate_id:"c",status:"accepted",metrics:{base_nll:1,candidate_nll:.8,eval_samples:10}}]}));h.run('selectRun("a")');
  const detail=findClass(h.get("project-content"),"run-detail")[0];
  assert.equal(findClass(detail,"metric-table").length,1);
  assert.equal(findClass(detail,"candidate-card").length,0);
  assert.match(textOf(detail),/已采纳/);
});


test("decision detail shows rationale, submitted content and observed outcome without claiming application",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{objective:"Autonomous medical research",decisions:[{decision_id:"d",kind:"directive",mode:"configure_prompt",status:"completed",reason:"Responses need a consistent format",changes:[{field:"instructions",after:"Use short answers.\nDo not include a preamble."}],feedback_summary:"Prompt proposed; evaluate before application"}]}));h.run('selectRun("a")');
  const detail=findClass(h.get("project-content"),"run-detail")[0],text=textOf(detail);
  assert.match(text,/Responses need a consistent format/);assert.match(text,/Use short answers/);assert.match(text,/Prompt proposed; evaluate before application/);
  assert.doesNotMatch(text,/执行已完成|Autonomous medical research/);
  assert.equal(findClass(detail,"change-before").length,0);
});


test("generic completed feedback does not create an empty result section",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{decisions:[{decision_id:"d",kind:"directive",mode:"configure_prompt",status:"completed",feedback_summary:"执行已完成。"}]}));h.run('selectRun("a")');
  assert.equal(findClass(h.get("project-content"),"flow-feedback").length,0);
});


test("evaluation explains what is checked and keeps a failed result visible",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",run("a",{decisions:[{decision_id:"d",kind:"directive",mode:"evaluate",status:"failed",changes:[{field:"components",after:"instructions"},{field:"protocol_ref",after:"development"}],error:"No eligible examples"}]}));h.run('selectRun("a")');
  const text=textOf(findClass(h.get("project-content"),"run-detail")[0]);
  assert.match(text,/验证改动效果/);assert.match(text,/检验新提示词的效果/);assert.match(text,/development/);assert.match(text,/No eligible examples/);
});

 test("operation-name completed is only a status, not result detail",()=>{
 const h=harness();
 assert.equal(h.run('concreteFeedback({mode:"evaluate",kind:"directive",feedback_summary:"evaluate completed"})'), "");
 });
