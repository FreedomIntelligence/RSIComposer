"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(`${__dirname}/../../../src/rsimonitor/static/app.js`, "utf8");
const descendants = node => [node, ...node.children.flatMap(descendants)];
const textOf = node => descendants(node).map(n => n.textContent || "").join(" ");
const findClass = (node, name) => descendants(node).filter(n => (n.className || "").split(" ").includes(name));
function events(target,listeners={}) {
  const registered=new Map();
  target.addEventListener=(type,callback)=>{if(!registered.has(type))registered.set(type,new Set());registered.get(type).add(callback);listeners[type]=(...args)=>{for(const listener of [...registered.get(type)])if(registered.get(type).has(listener))listener(...args);};};
  target.removeEventListener=(type,callback)=>registered.get(type)?.delete(callback);
  return target;
}
class Element {
  constructor(tag = "div") { this.tagName = tag; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.scrollLeft = 0; this.scrollTop = 0; this.open = false; events(this,this.listeners); }
  append(...nodes) { for(const node of nodes)node.parentNode=this;this.children.push(...nodes); }
  replaceChildren(...nodes) { for(const node of this.children)node.parentNode=null;this.children=[];this.append(...nodes); }
  get childElementCount() { return this.children.length; }
  get firstElementChild() { return this.children[0]; }
  setAttribute(k, v) { this.attributes[k] = String(v); if (k === "class") this.className = String(v); if (k.startsWith("data-")) this.dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(v); }
  getAttribute(k) { return this.attributes[k] ?? null; }
  setPointerCapture(id) { this.capturedPointer=id; }
  releasePointerCapture(id) { if(this.capturedPointer===id)delete this.capturedPointer; }
  querySelectorAll(selector) { return descendants(this).filter(n => selector.startsWith(".") ? (n.className || "").split(" ").includes(selector.slice(1)) : selector === "details[data-retain-key]" && n.tagName === "details" && n.dataset.retainKey); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  showModal() { this.open=true; }
  close() { if(!this.open)return;this.open=false;this.listeners.close?.(); }
  focus() {}
}
function harness(fetcher = async () => { throw Error("unexpected fetch"); }, search = "") {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) { const e = new Element(); e.id = id; elements.set(id, e); } return elements.get(id); };
  const documentListeners = {}, document = events({ getElementById: get, createElement: t => new Element(t), createElementNS: (_, t) => new Element(t), hidden: false },documentListeners);
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
  const window = events({scrollX: 0, scrollY: 0,
    scrollTo({ left = 0, top = 0 }) { this.scrollX = left; this.scrollY = top; } },windowListeners);
  const frames=new Map();let frameID=0,now=0;
  const requestAnimationFrame=callback=>{const id=++frameID;frames.set(id,callback);return id;};
  const cancelAnimationFrame=id=>frames.delete(id);
  const frame=()=>{const pending=[...frames.values()];frames.clear();now+=16;for(const callback of pending)callback(now);};
  const flushFrames=(limit=600)=>{for(let i=0;frames.size&&i<limit;i++)frame();assert.equal(frames.size,0,"graph animation did not become idle");};
  Object.assign(window,{requestAnimationFrame,cancelAnimationFrame});
  const ctx = vm.createContext({ document, window, location, history, AbortController, URL, URLSearchParams, setTimeout, clearTimeout, requestAnimationFrame,cancelAnimationFrame,performance:{now:()=>now},fetch: fetcher });
  vm.runInContext(fs.readFileSync(`${__dirname}/../../../src/rsimonitor/static/i18n.js`, "utf8"), ctx);
  vm.runInContext(fs.readFileSync(`${__dirname}/../../../src/rsimonitor/static/decisions.js`, "utf8"), ctx);
  // Observe the real renderer and lifecycle rather than replacing graph behavior.
  const renders=[],decisions=ctx.RSIDecisions;
  ctx.RSIDecisions={...decisions,render(options){
    const svg=decisions.render(options),record={svg,options,disposals:0},dispose=svg.dispose;
    svg.dispose=function(...args){record.disposals++;return dispose.apply(this,args);};renders.push(record);return svg;
  }};
  vm.runInContext(source.slice(0, source.lastIndexOf("\nrefresh();")), ctx);
  vm.runInContext("state.auto=false",ctx);
  return { ctx, get, document, documentListeners, windowListeners, history, renders,frame,flushFrames,get pendingFrames(){return frames.size;},run: s => vm.runInContext(s, ctx), json: s => JSON.parse(JSON.stringify(vm.runInContext(s, ctx))) };
}
const run = (id, extra = {}) => ({ schema_version: 1, run_id: id, updated_at: "2026-11-01T00:00:00Z", ...extra });
const project = (id, runs = [run(id)], extra = {}) => ({ project_id: id, name: id, runs, ...extra });
const response = data => ({ ok: true, json: async () => data });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
function setup(h, projects) { h.ctx.input = projects; h.run("chooseProjects(input)"); }
function seed(h, id, snapshot) {
  h.ctx.fixture = snapshot; h.ctx.id = id;
  h.run('{ const key=JSON.stringify([state.demo,id]); let scope=state.scopes.get(key); if(!scope){scope={key,cache:new Map(),runId:id,details:null};state.scopes.set(key,scope);}scope.cache.set(id,{snapshot:fixture,status:"ready",token:0}); }');
}
const settle = () => new Promise(resolve => setImmediate(resolve));
const graphState = h => JSON.parse(h.run('JSON.stringify({camera:state.activeScope.graphView.camera,positions:state.activeScope.graphView.positions},(_,value)=>value instanceof Map?[...value]:value)'));
function dragGraphNode(h,key,dx=60,dy=40,release=true) {
  const nodes=mainNodes(h);const node=nodes.find(n=>n.dataset.key===key)||nodes.find(n=>n.dataset.key===attemptKey(key))||findClass(h.get("project-content"),"decision-node").find(n=>n.dataset.key===key);assert.ok(node,`missing node ${key}`);
  const event={pointerId:1,button:0,clientX:100,clientY:100,target:node,stopPropagation(){},preventDefault(){}};
  node.listeners.pointerdown(event);node.listeners.pointermove({...event,clientX:100+dx,clientY:100+dy});
  if(release)node.listeners.pointerup({...event,type:"pointerup",clientX:100+dx,clientY:100+dy});
  return node;
}

const attemptKey=id=>"attempt:"+JSON.stringify(id);
const mainNodes=h=>{const map=findClass(h.get("project-content"),"flow-map")[0];return map?findClass(map,"decision-node"):[];};
const currentGraph=h=>h.renders.findLast(record=>record.svg===h.run("state.activeScope.graph"));
function selectDecision(h,key) {
  const nodes=mainNodes(h);
  const node=key===undefined?nodes[0]:nodes.find(n=>n.dataset.key===attemptKey(key));
  if(node){
    node.listeners.click();
    const children=findClass(findClass(h.get("project-content"),"attempt-map")[0],"decision-node");
    const child=key===undefined?children[0]:children.find(n=>n.dataset.key===key);
    if(child?.getAttribute("aria-pressed")!=="true")child?.listeners.click();
    return;
  }
  const button=findClass(h.get("project-content"),"experiment-log-record").find(n=>key===undefined||n.dataset.key===key);
  assert.ok(button,`missing decision ${key ?? "first"}`);button.listeners.click();
}

// Navigation/interaction fixtures explicitly record one Attempt per decision.
// Historical fixtures use run() and keep membership absent.
function attemptRun(id,extra={}) {
  const snapshot=run(id,extra);
  return {...snapshot,...(snapshot.decisions?{decisions:snapshot.decisions.map((row,index)=>({
    research_attempt_id:row.decision_id||`fixture-${index}`,research_attempt_title:"Recorded hypothesis",...row,
  }))}:{})};
}

test("catalog preserves two real project groups with multiple experiments without merging equal project names", () => {
  const input = [
    project("p",[attemptRun("old",{experiment_name:"Baseline"}),attemptRun("current",{experiment_name:"Improved",parent_run_id:"old"})],{name:"Same project name"}),
    project("q",[attemptRun("third",{experiment_name:"Baseline"}),attemptRun("fourth",{experiment_name:"Comparison"})],{name:"Same project name"}),
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
  const calls=[], snapshot=attemptRun("a", { objective:"Hidden research objective",decisions:[{decision_id:"d",mode:"train",reason:"Plan training",feedback:{summary:"Training result"}}] });
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("p",[attemptRun("a",{experiment_name:"Named experiment"}),run("b")],{name:"Parent project",description:"Project description is not its identity"})]}:snapshot);});
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
  selectDecision(h,"d");
  assert.equal(h.run("state.snapshot.run_id"),"a");assert.match(textOf(h.get("project-content")),/Plan training.*Training result/);
  assert.deepEqual(calls,["/api/projects","/api/runs/a"]);
});

test("experiments within one project isolate cached histories, selected graph nodes and both scroll axes", async () => {
  const requests=[], b=attemptRun("b",{decisions:[{decision_id:"same",mode:"evaluate",reason:"Only B"}]});
  const h=harness(async url=>{requests.push(url);return response(b);});setup(h,[project("p",[run("a"),run("b")])]);
  const a=attemptRun("a",{decisions:Array.from({length:35},(_,i)=>({decision_id:`d${i}`,timestamp:new Date(Date.UTC(2026,10,1,0,i)).toISOString(),mode:"train",reason:`history-${i}-end`}))});
  seed(h,"a",a);await h.run('selectRun("a")');
  const allNodes=mainNodes(h);
  assert.equal(allNodes.length,35);assert.deepEqual(allNodes.map(n=>n.dataset.key),a.decisions.map(d=>attemptKey(d.research_attempt_id)));
  assert.equal(findClass(h.get("project-content"),"more-button").length,0);
  const aMap=findClass(h.get("project-content"),"flow-map")[0];aMap.scrollLeft=321;aMap.scrollTop=87;
  allNodes[34].listeners.click();
  assert.equal(aMap.scrollLeft,321);assert.equal(aMap.scrollTop,87);
  assert.equal(mainNodes(h)[34].getAttribute("aria-pressed"),"true");
  assert.equal(findClass(h.get("project-content"),"flow-step").length,1);
  const aScope=h.run("state.activeScope"),aCache=h.run("state.activeScope.cache");
  const detail=h.get("project-content").firstElementChild;
  h.get("back-projects").listeners.click();
  findClass(h.get("project-list"),"experiment-card")[1].listeners.click();await settle();
  assert.equal(h.run("state.projectId"),"p");assert.equal(h.run("state.runId"),"b");
  assert.equal(h.get("project-name").textContent,"b");assert.equal(h.get("project-description").textContent,"p");assert.equal(h.get("project-description").hidden,false);
  selectDecision(h,"same");assert.match(textOf(h.get("project-content")),/Only B/);assert.doesNotMatch(textOf(h.get("project-content")),/history-34-end/);
  assert.notEqual(h.run("state.activeScope"),aScope);assert.notEqual(h.run("state.activeScope.cache"),aCache);
  assert.equal(h.run("state.activeScope.key"),JSON.stringify([false,"b"]));assert.equal(aScope.key,JSON.stringify([false,"a"]));
  assert.deepEqual(h.json('[...state.activeScope.cache.keys()]'),["b"]);
  assert.equal(h.run('state.activeScope.flowSelected.get("b")'),attemptKey("same"));
  const bMap=findClass(h.get("project-content"),"flow-map")[0];bMap.scrollLeft=18;bMap.scrollTop=29;
  assert.equal(new URL(h.ctx.savedURL).searchParams.get("run"),"b");
  h.get("back-projects").listeners.click();
  findClass(h.get("project-list"),"experiment-card")[0].listeners.click();await settle();
  assert.equal(h.get("project-content").firstElementChild,detail);
  assert.match(textOf(h.get("project-content")),/history-34-end/);
  assert.equal(mainNodes(h).length,35);
  assert.equal(h.run("state.activeScope"),aScope);assert.equal(h.run("state.activeScope.cache"),aCache);
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),attemptKey("d34"));
  assert.equal(findClass(detail,"flow-map")[0].scrollLeft,321);assert.equal(findClass(detail,"flow-map")[0].scrollTop,87);
  assert.equal(h.run('state.snapshot.decisions.length'),35);
  h.run('showManagement()');await h.run('selectRun("b")');
  const rebuiltBMap=findClass(h.get("project-content"),"flow-map")[0];assert.notEqual(rebuiltBMap,bMap);
  assert.equal(rebuiltBMap.scrollLeft,18);assert.equal(rebuiltBMap.scrollTop,29);
  assert.deepEqual(requests,["/api/runs/b"]);
});

test("run deep links open a non-first experiment and its real owner overrides a conflicting project query", async () => {
  const catalog=[project("p",[run("a"),attemptRun("b",{experiment_name:"Second P experiment"})],{name:"Project P"}),project("q",[run("c"),attemptRun("d",{experiment_name:"Second Q experiment"})],{name:"Project Q"})];
  for(const [search,expected,owner,name] of [["?run=b","b","p","Second P experiment"],["?project=wrong&run=b","b","p","Second P experiment"],["?project=p&run=b","b","p","Second P experiment"],["?project=q&run=b","b","p","Second P experiment"],["?project=p&run=d","d","q","Second Q experiment"]]){
    const calls=[],h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:catalog}:attemptRun(expected,{decisions:[{decision_id:"d",reason:`opened ${expected}`}]}));},search);
    await h.run("refresh()");
    assert.equal(h.run("state.page"),"experiment");assert.equal(h.get("management").hidden,true);assert.equal(h.get("projects").hidden,false);
    assert.equal(h.run("state.projectId"),owner);assert.equal(h.run("state.runId"),expected);
    assert.equal(h.get("project-name").textContent,name);assert.equal(h.get("project-description").textContent,owner==="p"?"Project P":"Project Q");assert.equal(h.get("project-description").hidden,false);
    selectDecision(h,"d");assert.match(textOf(h.get("project-content")),new RegExp(`opened ${expected}`));
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
  const calls=[],h=harness(async url=>{calls.push(url);return response(attemptRun("single",{decisions:[{reason:"Legacy detail"}]}));});setup(h,[project("p",[run("a"),run("b")]),project("single")]);
  await h.run('selectProject("p")');assert.equal(h.run("state.page"),"management");assert.deepEqual(calls,[]);
  await h.run('selectProject("single")');selectDecision(h);assert.equal(h.run("state.projectId"),"single");assert.equal(h.run("state.runId"),"single");assert.match(textOf(h.get("project-content")),/Legacy detail/);assert.deepEqual(calls,["/api/runs/single"]);
});

test("back button returns to management, keeps cached detail and polls only the catalog", async () => {
  const calls=[],snapshot=attemptRun("a",{decisions:[{decision_id:"d",reason:"Cached A"}]});
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("a")]}:snapshot);});
  await h.run("refresh()");await h.run('selectRun("a")');selectDecision(h,"d");
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
  const calls=[],h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:catalog}:attemptRun(url.endsWith("/b")?"b":"d",{decisions:[{decision_id:"shared",reason:url.endsWith("/b")?"History B":"History D"}]}));});
  await h.run("refresh()");await h.run('selectRun("b")');
  const bDetail=h.get("project-content").firstElementChild;
  await h.run('selectRun("d")');selectDecision(h,"shared");const dDetail=h.get("project-content").firstElementChild;
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
  pending.resolve(response(attemptRun("a",{decisions:[{decision_id:"late",reason:"Late detail"}]})));await request;
  assert.equal(h.run("state.page"),"management");assert.equal(h.get("projects").hidden,true);
  assert.equal(h.run("state.snapshot"),null);assert.doesNotMatch(textOf(h.get("project-content")),/Late detail/);
});

test("parent and source references are provenance only and never select, fetch or combine other runs", async () => {
  const calls=[],snapshot=attemptRun("current",{parent_run_id:"parent",source_run_ids:["parent","outside"],decisions:[{decision_id:"d",reason:"Current decision"}]});
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
    for(const [id,loss] of [["a",1],["b",2]])seed(h,id,attemptRun(id,{decisions:[{decision_id:"same",kind:"directive",directive_id:"shared",mode:"train",feedback:{summary:`feedback-${id}`}}],jobs:[{job_id:"same",directive_id:"shared",metrics:{loss}}],candidates:[{candidate_id:"same",directive_id:"shared",metrics:{accuracy:loss/10}}]}));
    for(const [id,other,loss] of [["a","b",1],["b","a",2],["a","b",1]]){
      await h.run(`selectRun("${id}")`);selectDecision(h,"same");assert.match(textOf(h.get("project-content")),new RegExp(`feedback-${id}`));
      assert.doesNotMatch(textOf(h.get("project-content")),new RegExp(`feedback-${other}`));assert.equal(h.run("state.snapshot.jobs[0].metrics.loss"),loss);
    }
  }
});

test("switching experiments within or across projects aborts pending reads and rejects late responses", async () => {
  for(const catalog of [[project("p",[run("a"),run("b")])],[project("p",[run("a")]),project("q",[run("b")])]]){
    const old=deferred(),fresh=deferred(),signals=[],h=harness((url,options)=>{signals.push(options.signal);return signals.length===1?old.promise:fresh.promise;});
    setup(h,catalog);seed(h,"b",attemptRun("b",{decisions:[{reason:"B history"}]}));
    const pending=h.run('selectRun("a")');await h.run('selectRun("b")');assert.equal(signals[0].aborted,true);
    const current=h.run('selectRun("a")');fresh.resolve(response(attemptRun("a",{decisions:[{reason:"Fresh A"}]})));await current;selectDecision(h);
    old.resolve(response(attemptRun("a",{decisions:[{reason:"Stale A"}]})));await pending;
    assert.match(textOf(h.get("project-content")),/Fresh A/);assert.doesNotMatch(textOf(h.get("project-content")),/Stale A|B history/);
  }
});

test("repeated selection reuses pending requests and cached decisions", async () => {
  const next=deferred(),calls=[],h=harness(url=>{calls.push(url);return next.promise;});setup(h,[project("a")]);
  const first=h.run('selectRun("a")');h.run('selectRun("a")');
  assert.deepEqual(calls,["/api/runs/a"]);assert.match(textOf(h.get("project-content")),/正在读取/);
  next.resolve(response(attemptRun("a",{decisions:[{reason:"Loaded"}]})));await first;await h.run('selectRun("a")');selectDecision(h);
  assert.deepEqual(calls,["/api/runs/a"]);assert.match(textOf(h.get("project-content")),/Loaded/);
});

test("refresh reloads only the selected experiment and preserves another experiment's cache in the same project", async () => {
  const calls=[],a=attemptRun("a",{decisions:Array.from({length:35},(_,i)=>({decision_id:`d${i}`,mode:"train",feedback:{summary:"New result"}}))});
  const catalog=[project("p",[run("a"),run("b")]),project("q",[run("c"),run("d")])];
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:catalog}:a);});setup(h,catalog);
  seed(h,"a",{...a,decisions:a.decisions.map(d=>({...d,feedback:{summary:"Previous result"}}))});seed(h,"b",attemptRun("b",{decisions:[{reason:"B remains"}]}));
  await h.run('selectRun("b")');const bScope=h.run("state.activeScope"),bEntry=h.run('state.activeScope.cache.get("b")'),bDetail=h.get("project-content").firstElementChild;
  await h.run('selectRun("a")');mainNodes(h)[34].listeners.click();
  const map=findClass(h.get("project-content"),"flow-map")[0];map.scrollLeft=456;map.scrollTop=78;
  await h.run("refresh()");
  assert.deepEqual(calls,["/api/projects","/api/runs/a"]);assert.equal(h.run('state.activeScope.flowSelected.get("a")'),attemptKey("d34"));
  const nodes=mainNodes(h);assert.equal(nodes.length,35);assert.equal(nodes[34].getAttribute("aria-pressed"),"true");
  assert.equal(findClass(h.get("project-content"),"flow-map")[0].scrollLeft,456);assert.equal(findClass(h.get("project-content"),"flow-map")[0].scrollTop,78);
  assert.match(textOf(h.get("project-content")),/New result/);
  assert.equal(h.run("state.projectId"),"p");assert.deepEqual(h.json('[...state.activeScope.cache.keys()]'),["a"]);
  await h.run('selectRun("b")');selectDecision(h);assert.match(textOf(h.get("project-content")),/B remains/);assert.equal(calls.length,2);
  assert.equal(h.run("state.projectId"),"p");assert.equal(h.run("state.activeScope"),bScope);assert.equal(h.run('state.activeScope.cache.get("b")'),bEntry);assert.equal(h.get("project-content").firstElementChild,bDetail);assert.deepEqual(h.json('[...state.activeScope.cache.keys()]'),["b"]);
});

test("graph selection and changed polling preserve the experiment's camera and node positions", async () => {
  const initial=attemptRun("a",{decisions:[{decision_id:"d1",mode:"train",reason:"First decision"},{decision_id:"d2",mode:"evaluate",feedback:{summary:"Previous outcome"}}]});
  const updated={...initial,decisions:[initial.decisions[0],{...initial.decisions[1],feedback:{summary:"Updated outcome"}}]};
  const catalog=[project("p",[run("a"),run("b")])],calls=[];
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:catalog}:updated);});setup(h,catalog);seed(h,"a",initial);await h.run('selectRun("a")');h.flushFrames();
  const view=h.run("state.activeScope.graphView");h.run('state.activeScope.graph.zoomBy(1.25)');h.flushFrames();
  const unshifted=graphState(h);dragGraphNode(h,"d2");assert.ok(h.pendingFrames>0);h.flushFrames();assert.notDeepEqual(graphState(h).positions,unshifted.positions);
  const before=graphState(h),old=currentGraph(h);
  mainNodes(h)[1].listeners.click({detail:0});
  assert.ok(old.disposals>0);assert.equal(h.run("state.activeScope.graphView"),view);assert.deepEqual(graphState(h),before);
  h.flushFrames();const selected=currentGraph(h),beforePoll=graphState(h);
  await h.run("refresh()");
  assert.ok(selected.disposals>0);assert.notEqual(h.run("state.activeScope.graph"),selected.svg);
  assert.equal(h.run("state.activeScope.graphView"),view);assert.deepEqual(graphState(h),beforePoll);
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),attemptKey("d2"));assert.match(textOf(h.get("project-content")),/Updated outcome/);
  assert.deepEqual(calls,["/api/projects","/api/runs/a"]);h.flushFrames();
});

test("polling while graph interaction or animation is busy keeps its DOM until onIdle renders the latest snapshot", async () => {
  for(const busy of ["interacting","animating"]){
    let version=0;const calls=[],catalog=[project("p",[run("a"),run("b")])];
    const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:catalog}:attemptRun("a",{decisions:[{decision_id:"d",mode:"train",feedback:{summary:`Outcome ${++version}`}}]}));});
    setup(h,catalog);seed(h,"a",attemptRun("a",{decisions:[{decision_id:"d",mode:"train",feedback:{summary:"Initial outcome"}}]}));await h.run('selectRun("a")');selectDecision(h,"d");h.flushFrames();
    const record=currentGraph(h),detail=h.get("project-content").firstElementChild,runDOM=findClass(detail,"run-detail")[0],count=h.renders.length;
    if(busy==="interacting")record.svg.listeners.pointerdown({pointerId:2,button:0,clientX:100,clientY:100,target:record.svg});
    else dragGraphNode(h,"d");
    assert.equal(h.run(`state.activeScope.graphView.${busy}`),true);
    await h.run("refresh()");await h.run("refresh()");
    assert.equal(h.renders.length,count);assert.equal(record.disposals,0);assert.equal(h.run("state.activeScope.graph"),record.svg);
    assert.equal(h.get("project-content").firstElementChild,detail);assert.equal(findClass(detail,"run-detail")[0],runDOM);
    assert.match(textOf(detail),/Initial outcome/);assert.doesNotMatch(textOf(detail),/Outcome [12]/);
    assert.equal(h.run("state.activeScope.renderPending"),true);assert.equal(h.run('state.snapshot.decisions[0].feedback.summary'),"Outcome 2");
    if(busy==="interacting")record.svg.listeners.pointerup({pointerId:2,type:"pointerup"});else h.flushFrames();
    assert.equal(h.run("state.activeScope.renderPending"),false);assert.ok(record.disposals>0);assert.equal(h.renders.length,count+2);
    assert.equal(h.get("project-content").firstElementChild,detail);assert.notEqual(findClass(detail,"run-detail")[0],runDOM);
    assert.match(textOf(detail),/Outcome 2/);assert.doesNotMatch(textOf(detail),/Initial outcome|Outcome 1/);
    assert.deepEqual(calls,["/api/projects","/api/runs/a","/api/projects","/api/runs/a"]);h.flushFrames();
  }
});

test("leaving an experiment disposes its graph and restores isolated views without accepting stale idle callbacks", async () => {
  const h=harness();setup(h,[project("p",[run("a"),run("b")])]);
  for(const id of ["a","b"])seed(h,id,attemptRun(id,{decisions:[{decision_id:"shared",mode:"train",reason:`Only ${id}`}]}));
  await h.run('selectRun("a")');h.flushFrames();h.run('state.activeScope.graph.zoomBy(1.25)');h.flushFrames();
  dragGraphNode(h,"shared");assert.ok(h.pendingFrames>0);
  const aScope=h.run("state.activeScope"),aView=aScope.graphView,aState=graphState(h),aGraph=currentGraph(h);
  await h.run('selectRun("b")');selectDecision(h,"shared");h.flushFrames();
  assert.ok(aGraph.disposals>0);assert.equal(aScope.graph,null);assert.equal(aScope.detailFingerprint,null);assert.equal(aScope.graphView,aView);assert.equal(h.pendingFrames,0);assert.equal(aView.animating,false);assert.equal(aView.interacting,false);
  const bScope=h.run("state.activeScope"),bGraph=currentGraph(h);assert.notEqual(bScope.graphView,aView);assert.notEqual(bScope.graphView.positions,aView.positions);assert.notEqual(bScope.graphView.camera,aView.camera);
  const count=h.renders.length;aScope.renderPending=true;aGraph.options.onIdle();
  assert.equal(h.renders.length,count);assert.equal(h.run("state.activeScope.graph"),bGraph.svg);assert.match(textOf(h.get("project-content")),/Only b/);
  h.run('state.activeScope.graph.zoomBy(.8)');dragGraphNode(h,"shared",20,-70);h.flushFrames();const bState=graphState(h);assert.notDeepEqual(bState,aState);
  await h.run('selectRun("a")');
  assert.ok(bGraph.disposals>0);assert.equal(bScope.graph,null);assert.equal(h.run("state.activeScope.graphView"),aView);assert.deepEqual(graphState(h),aState);assert.notEqual(h.run("state.activeScope.graph"),aGraph.svg);h.flushFrames();
  dragGraphNode(h,"shared",0,25);assert.ok(h.pendingFrames>0);
  const returning=currentGraph(h),beforeManagement=graphState(h);h.get("back-projects").listeners.click();
  assert.ok(returning.disposals>0);assert.equal(aScope.graph,null);assert.equal(aScope.detailFingerprint,null);assert.equal(h.pendingFrames,0);assert.deepEqual(graphState(h),beforeManagement);
  aScope.renderPending=true;const hiddenCount=h.renders.length;returning.options.onIdle();assert.equal(h.renders.length,hiddenCount);assert.equal(h.run("state.page"),"management");
  await h.run('selectRun("b")');assert.equal(h.run("state.activeScope"),bScope);assert.deepEqual(graphState(h),bState);assert.notEqual(h.run("state.activeScope.graph"),bGraph.svg);h.flushFrames();
});

test("hidden recorded demo disposes its graph and showing it rebuilds the preserved view without fetching", async () => {
  const source=JSON.parse(fs.readFileSync(`${__dirname}/demo.json`,"utf8")),demo=attemptRun(source.run_id,source),calls=[];
  const h=harness(async url=>{calls.push(url);assert.equal(url,"/demo.json");return response(demo);});h.run("state.demo=true");await h.run("refresh()");await h.run(`selectRun(${JSON.stringify(demo.run_id)})`);h.flushFrames();
  h.run('state.activeScope.graph.zoomBy(1.25)');h.flushFrames();
  const nodes=mainNodes(h),dragged=dragGraphNode(h,nodes.at(-1).dataset.key,35,50,false);assert.ok(h.pendingFrames>0);assert.equal(dragged.capturedPointer,1);
  const view=h.run("state.activeScope.graphView"),before=graphState(h),old=currentGraph(h),snapshot=h.run("state.snapshot");
  h.document.hidden=true;h.documentListeners.visibilitychange();
  assert.ok(old.disposals>0);assert.equal(h.pendingFrames,0);assert.equal(dragged.capturedPointer,undefined);assert.equal(view.interacting,false);assert.equal(view.animating,false);assert.equal(h.run("state.activeScope.graph"),null);assert.equal(h.run("state.activeScope.detailFingerprint"),null);assert.deepEqual(graphState(h),before);
  h.document.hidden=false;h.documentListeners.visibilitychange();
  assert.notEqual(h.run("state.activeScope.graph"),old.svg);assert.equal(h.run("state.activeScope.graphView"),view);assert.deepEqual(graphState(h),before);assert.equal(h.run("state.snapshot"),snapshot);
  assert.equal(h.run("state.page"),"experiment");assert.equal(h.run("state.runId"),demo.run_id);assert.deepEqual(calls,["/demo.json"]);h.flushFrames();
});

test("trajectory toolbar invokes zoom, fit and rearrange on the current graph", async () => {
  const h=harness();setup(h,[project("p",[run("a")])]);seed(h,"a",attemptRun("a",{decisions:[{decision_id:"d",mode:"train"}]}));await h.run('selectRun("a")');h.flushFrames();
  const graph=h.run("state.activeScope.graph"),calls=[];
  for(const method of ["zoomBy","fit","reset"]){const original=graph[method];graph[method]=function(...args){calls.push([method,...args]);return original.apply(this,args);};}
  const toolbar=findClass(h.get("project-content"),"trajectory-tools");assert.equal(toolbar.length,1);assert.deepEqual(toolbar[0].children.map(b=>b.textContent),["−","+","⤢","↺"]);
  for(const button of toolbar[0].children){button.listeners.click();h.flushFrames();}
  assert.deepEqual(calls,[["zoomBy",.8],["zoomBy",1.25],["fit"],["reset"]]);
});

test("failed reads retain decisions and a retry restores the current project", async () => {
  let fail=true;const h=harness(async()=>{if(fail)throw Error("offline");return response(attemptRun("a",{decisions:[{reason:"Recovered"}]}));});
  setup(h,[project("a")]);seed(h,"a",attemptRun("a",{decisions:[{reason:"Cached decision"}]}));await h.run('selectRun("a")');selectDecision(h);
  await h.run('loadRun("a",state.activeScope,true)');assert.match(textOf(h.get("project-content")),/offline.*Cached decision/);
  fail=false;findClass(h.get("project-content"),"run-detail")[0].children.find(n=>n.tagName==="button").listeners.click();await settle();selectDecision(h);
  assert.match(textOf(h.get("project-content")),/Recovered/);assert.equal(h.run('state.activeScope.cache.get("a").status'),"ready");
});

test("demo groups two experiments in one project, loads their decisions and keeps live caches and links", async () => {
  const demo=JSON.parse(fs.readFileSync(`${__dirname}/demo.json`,"utf8")),h=harness(async()=>response(demo));
  setup(h,[project("live")]);seed(h,"live",attemptRun("live",{decisions:[{reason:"Live history"}]}));await h.run('selectRun("live")');selectDecision(h);
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
  pending.resolve(response(attemptRun("a",{decisions:[{reason:"Late live history"}]})));await request;
  assert.equal(h.run("state.snapshot"),null);assert.equal(h.run("state.activeScope"),null);
});

test("empty catalog and unavailable snapshot have readable states without a project graph", async () => {
  const h=harness(async url=>response({projects:[]}));await h.run("refresh()");
  assert.equal(h.get("empty").hidden,false);assert.equal(h.get("projects").hidden,true);assert.equal(h.get("project-list").children.length,1);
  setup(h,[project("broken",[attemptRun("broken",{status:"unavailable"})])]);
  await h.run('selectRun("broken")');
  assert.equal(h.run("state.runId"),"broken");assert.match(textOf(h.get("project-content")),/快照不可用/);
});
test("unknown execution timestamps retain snapshot order without implying chronology; repeat finish retains originals", () => {
  const h=harness();h.ctx.fixture={decisions:[{decision_id:"u1",mode:"train"},{decision_id:"later",mode:"train",timestamp:"2026-11-01T01:00:00Z"},{decision_id:"u2",mode:"evaluate"},{decision_id:"early",mode:"train",timestamp:"2026-11-01T00:00:00Z"},{decision_id:"f1",kind:"finish",reason:"done"},{decision_id:"f2",kind:"finish",reason:"done"}]};
  const groups=h.json("executionGroups(fixture)");assert.deepEqual(groups.map(g=>g.row.decision_id),["early","later","u1","u2","f1","f2"]);assert.equal(groups[4].rows.length,1);
});

test("referenced legacy finish nodes survive grouping and polling for every lineage relation", async () => {
  for(const relation of ["depends_on_decision_ids","adopted_decision_ids","source_decision_ids","parent_decision_id"]){
    const first={decision_id:"f1",kind:"finish",reason:"done",timestamp:"2026-11-01T00:00:01Z"},second={...first,decision_id:"f2",timestamp:"2026-11-01T00:00:02Z"};
    const next={decision_id:"next",kind:"operation",operation:"query",timestamp:"2026-11-01T00:00:03Z",[relation]:relation==="parent_decision_id"?"f1":["f1"]};
    const updated=attemptRun("a",{decisions:[first,second,next]}),h=harness(async()=>response(updated));setup(h,[project("p",[run("a")])]);seed(h,"a",attemptRun("a",{decisions:[first,second]}));await h.run('selectRun("a")');
    assert.equal(mainNodes(h).length,2);
    await h.run('loadRun("a",state.activeScope,true)');
    assert.deepEqual(mainNodes(h).map(n=>n.dataset.key),["f1","f2","next"].map(attemptKey));
    const groups=h.json("executionGroups(state.snapshot)");assert.equal(groups.length,3);
    const graph=currentGraph(h),model=h.ctx.RSIDecisions.layout(graph.options.rows);
    assert.ok(model.edges.some(edge=>edge.from==="f1"&&edge.to==="next"&&edge.kind===(relation==="adopted_decision_ids"?"adoption":relation==="source_decision_ids"?"source":"dependency")));
  }
});

test("finish records carrying their own explicit lineage are never collapsed",()=>{
  const h=harness();
  for(const relation of ["depends_on_decision_ids","adopted_decision_ids","source_decision_ids","parent_decision_id"]){
    h.ctx.rows=[{decision_id:"f1",kind:"finish",reason:"done",[relation]:relation==="parent_decision_id"?"outside":[]},{decision_id:"f2",kind:"finish",reason:"done",[relation]:relation==="parent_decision_id"?"outside":[]}];
    assert.equal(h.run("executionGroups({decisions:rows}).length"),2);
  }
});


test("execution detail keeps review metrics separate from exact current job and candidate results", () => {
  const h=harness();h.ctx.fixture={decisions:[{decision_id:"d",kind:"directive",directive_id:"dir",mode:"train"},{decision_id:"r",kind:"review",action:"continue",job_id:"job",metrics:{loss:.7}}],jobs:[{job_id:"job",directive_id:"dir",kind:"train",metrics:{loss:.1}}],candidates:[{candidate_id:"candidate",directive_id:"dir",metrics:{accuracy:.9}},{candidate_id:"other",directive_id:"outside",metrics:{accuracy:.5}}]};
  const results=h.json("relatedResults(fixture.decisions[0],fixture)");assert.deepEqual(results.map(r=>r.type),["job","candidate"]);assert.equal(results[1].row.candidate_id,"candidate");
  assert.deepEqual(h.json("feedbackFor(fixture.decisions[1],fixture).metrics"),{loss:.7});
  h.ctx.fixture=attemptRun("a",h.ctx.fixture);
  const detail=h.run("executionSequence(fixture,{})");
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

test("wait decisions render a friendly translated node and detail title",async()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);seed(h,"a",attemptRun("a",{decisions:[{decision_id:"wait",kind:"wait",reason:"Waiting for a worker"}]}));await h.run('selectRun("a")');
  selectDecision(h,"wait");
  for(const [language,title] of [["zh","等待"],["en","Wait"]]){
    h.run(`RSII18n.setLanguage("${language}")`);
    assert.equal(findClass(findClass(h.get("project-content"),"attempt-map")[0],"decision-node")[0].getAttribute("aria-label"),`1 · ${title}`);
    const heading=descendants(findClass(h.get("project-content"),"flow-decision")[0]).find(n=>n.tagName==="h4");assert.equal(heading.textContent,title);
    assert.doesNotMatch(textOf(h.get("project-content")),/未命名决策|Unnamed decision/);
  }
});

test("unknown actions use short readable strategy titles verbatim while known actions keep friendly labels",async()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  const cases=[
    {mode:"propose_component",strategy:"Propose prompt change",title:"Propose prompt change",duplicate:false},
    {mode:"propose_component",strategy:"Propose data change",title:"Propose data change",duplicate:false},
    {mode:"consume_current",strategy:"Verify adopted changes",title:"Verify adopted changes",duplicate:false},
    {mode:"custom_zh",strategy:"验证已采用的改动",title:"验证已采用的改动",duplicate:false},
    {mode:"custom_mode",strategy:"custom_mode",title:"custom_mode",duplicate:false},
    {mode:"custom_mode",strategy:"artifact://candidate-v2",title:"custom_mode",duplicate:true},
    {mode:"custom_mode",strategy:"!!!",title:"custom_mode",duplicate:true},
    {mode:"custom_mode",strategy:"A long explanation ".repeat(8),title:"custom_mode",duplicate:true},
    {mode:"custom_mode",strategy:"First line\nSecond line",title:"custom_mode",duplicate:true},
    {mode:"train",strategy:"Propose prompt change",title:{zh:"训练模型",en:"Train model"},duplicate:true},
  ];
  for(const [i,entry] of cases.entries()){
    seed(h,"a",attemptRun("a",{decisions:[{decision_id:`d${i}`,kind:"directive",mode:entry.mode,strategy:entry.strategy}]}));await h.run('selectRun("a")');
    selectDecision(h,`d${i}`);
    for(const language of ["zh","en"]){
      h.run(`RSII18n.setLanguage("${language}")`);const title=typeof entry.title==="string"?entry.title:entry.title[language];
      assert.equal(findClass(findClass(h.get("project-content"),"attempt-map")[0],"decision-node")[0].getAttribute("aria-label"),`1 · ${title}`);
      const detail=findClass(h.get("project-content"),"flow-decision")[0];assert.equal(descendants(detail).find(n=>n.tagName==="h4").textContent,title);
      assert.equal(findClass(detail,"decision-reason").some(n=>n.textContent===entry.strategy),entry.duplicate);
      assert.equal(h.run("state.snapshot.decisions[0].strategy"),entry.strategy);
    }
  }
});


test("unchanged polling reuses run detail DOM without a refreshing message",async()=>{
  const next=deferred(),snapshot=attemptRun("a",{candidates:[{candidate_id:"unlinked",status:"accepted",metrics:{score:.8}}],decisions:[{decision_id:"d",kind:"operation",operation:"define_recipe",reason:"Choose a short training run"}]});
  const h=harness(()=>next.promise);setup(h,[project("p",[run("a")])]);seed(h,"a",snapshot);h.run('selectRun("a")');
  const graph=currentGraph(h),view=h.run("state.activeScope.graphView"),beforeView=graphState(h),renderCount=h.renders.length;
  const detail=findClass(h.get("project-content"),"run-detail")[0],record=findClass(detail,"record-details")[0];record.open=true;
  const pending=h.run('loadRun("a",state.activeScope,true)');
  assert.equal(findClass(h.get("project-content"),"run-detail")[0],detail);
  assert.doesNotMatch(textOf(detail),/正在刷新|正在读取/);
  next.resolve(response(JSON.parse(JSON.stringify(snapshot))));await pending;
  assert.equal(findClass(h.get("project-content"),"run-detail")[0],detail);assert.equal(record.open,true);
  assert.equal(h.run('state.activeScope.cache.get("a").snapshot'),h.ctx.fixture);
  assert.equal(h.run("state.activeScope.graph"),graph.svg);assert.equal(graph.disposals,0);assert.equal(h.renders.length,renderCount);assert.equal(h.run("state.activeScope.graphView"),view);assert.deepEqual(graphState(h),beforeView);
});


test("new feedback preserves native disclosures and selected step with a stable decision identity",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",attemptRun("a",{candidates:[{candidate_id:"unlinked",status:"accepted",metrics:{score:.8}}],decisions:[{decision_id:"d",kind:"operation",operation:"define_recipe",timestamp:"2026-11-01T00:00:02Z"}]}));h.run('selectRun("a")');
  selectDecision(h,"d");
  const detail=findClass(h.get("project-content"),"project-detail")[0];
  // Real DOM selectors, represented by the existing fake-DOM tree for this test.
  detail.querySelectorAll=()=>findClass(detail,"record-details").filter(d=>d.dataset.retainKey);
  detail.querySelector=()=>findClass(detail,"flow-map")[0];
  const disclosure=findClass(detail,"record-details")[0];disclosure.open=true;
  const nav=findClass(detail,"flow-map")[0];nav.scrollLeft=123;nav.scrollTop=67;
  h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,decisions:[{decision_id:"earlier",kind:"directive",mode:"configure_prompt",timestamp:"2026-11-01T00:00:01Z"},{...state.snapshot.decisions[0],feedback:{summary:"Recipe saved"}}]}; renderProject()');
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),attemptKey("d"));
  assert.match(textOf(findClass(detail,"flow-step")[0]),/Recipe saved/);
  assert.equal(findClass(detail,"record-details").find(d=>d.dataset.retainKey===disclosure.dataset.retainKey).open,true);
  assert.equal(findClass(detail,"flow-map")[0].scrollLeft,123);assert.equal(findClass(detail,"flow-map")[0].scrollTop,67);
});

test("duplicate decision IDs select one log record and retain it when earlier records and feedback arrive", async()=>{
  const h=harness();setup(h,[project("a")]);
  seed(h,"a",run("a",{decisions:[
    {decision_id:"duplicate",kind:"directive",mode:"train",timestamp:"2026-11-01T00:00:01Z",reason:"First record"},
    {decision_id:"duplicate",kind:"directive",mode:"evaluate",timestamp:"2026-11-01T00:00:02Z",reason:"Second record"},
  ]}));await h.run('selectRun("a")');
  const nodes=findClass(h.get("project-content"),"experiment-log-record");
  assert.equal(nodes.length,2);assert.notEqual(nodes[0].dataset.key,nodes[1].dataset.key);
  nodes[1].listeners.click();const key=nodes[1].dataset.key;
  assert.equal(h.run('state.activeScope.logSelected.get("a")'),key);
  assert.equal(findClass(h.get("project-content"),"flow-step").length,1);
  assert.match(textOf(findClass(h.get("project-content"),"flow-step")[0]),/Second record/);
  assert.doesNotMatch(textOf(findClass(h.get("project-content"),"flow-step")[0]),/First record/);
  h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,decisions:[{decision_id:"duplicate",kind:"directive",mode:"configure_prompt",timestamp:"2026-11-01T00:00:00Z",reason:"Earlier record"},state.snapshot.decisions[0],{...state.snapshot.decisions[1],feedback:{summary:"Evaluation observed"}}]};renderProject()');
  assert.equal(h.run('state.activeScope.logSelected.get("a")'),key);
  const selected=findClass(h.get("project-content"),"experiment-log-record").filter(n=>n.getAttribute("aria-pressed")==="true");
  assert.equal(selected.length,1);assert.equal(selected[0].dataset.key,key);
  assert.equal(findClass(h.get("project-content"),"flow-step").length,1);
  assert.match(textOf(findClass(h.get("project-content"),"flow-step")[0]),/Second record.*Evaluation observed/);
});


test("execution view shows decisions and results without internal identifiers or event dumps",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",attemptRun("a",{decisions:[{decision_id:"private-decision-id",directive_id:"private-directive-id",timestamp:"2026-11-01T00:00:01Z",kind:"directive",source:"model",mode:"configure_prompt",strategy:"configure_prompt",reason:"Try a shorter instruction",feedback:{summary:"Instruction updated"},decision_revision:"local-sha256:internal"}],events:[{kind:"internal-event",summary:"event-debug-text"}]}));
  h.run('selectRun("a")');selectDecision(h,"private-decision-id");
  const text=textOf(findClass(h.get("project-content"),"run-detail")[0]);
  assert.match(text,/调整提示词/);assert.match(text,/Try a shorter instruction/);assert.match(text,/Instruction updated/);
  assert.doesNotMatch(text,/记录详情|运行详情|运行事件|private-decision-id|private-directive-id|local-sha256|configure_prompt|event-debug-text/);
  assert.equal(h.run('state.snapshot.decisions[0].decision_id'),"private-decision-id");
});


test("candidate metrics already shown for a decision are not duplicated",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",attemptRun("a",{decisions:[{decision_id:"d",kind:"directive",mode:"evaluate",candidate_id:"c",metrics:{base_nll:1,candidate_nll:.8,eval_samples:10}}],candidates:[{candidate_id:"c",status:"accepted",metrics:{base_nll:1,candidate_nll:.8,eval_samples:10}}]}));h.run('selectRun("a")');
  selectDecision(h,"d");
  const detail=findClass(h.get("project-content"),"run-detail")[0];
  assert.equal(findClass(detail,"metric-table").length,1);
  assert.equal(findClass(detail,"candidate-card").length,0);
  assert.match(textOf(detail),/已采纳/);
});


test("decision detail shows rationale, submitted content and observed outcome without claiming application",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",attemptRun("a",{objective:"Autonomous medical research",decisions:[{decision_id:"d",kind:"directive",mode:"configure_prompt",status:"completed",reason:"Responses need a consistent format",changes:[{field:"instructions",after:"Use short answers.\nDo not include a preamble."}],feedback_summary:"Prompt proposed; evaluate before application"}]}));h.run('selectRun("a")');
  selectDecision(h,"d");
  const detail=findClass(h.get("project-content"),"run-detail")[0],text=textOf(detail);
  assert.match(text,/Responses need a consistent format/);assert.match(text,/Use short answers/);assert.match(text,/Prompt proposed; evaluate before application/);
  assert.doesNotMatch(text,/执行已完成|Autonomous medical research/);
  assert.equal(findClass(detail,"change-before").length,0);
});


test("generic completed feedback does not create an empty result section",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",attemptRun("a",{decisions:[{decision_id:"d",kind:"directive",mode:"configure_prompt",status:"completed",feedback_summary:"执行已完成。"}]}));h.run('selectRun("a")');
  selectDecision(h,"d");assert.equal(findClass(h.get("project-content"),"flow-step").length,1);
  assert.equal(findClass(h.get("project-content"),"flow-feedback").length,0);
});


test("evaluation explains what is checked and keeps a failed result visible",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  seed(h,"a",attemptRun("a",{decisions:[{decision_id:"d",kind:"directive",mode:"evaluate",status:"failed",changes:[{field:"components",after:"instructions"},{field:"protocol_ref",after:"development"}],error:"No eligible examples"}]}));h.run('selectRun("a")');
  selectDecision(h,"d");
  const text=textOf(findClass(h.get("project-content"),"run-detail")[0]);
  assert.match(text,/验证改动效果/);assert.match(text,/检验新提示词的效果/);assert.match(text,/development/);assert.match(text,/No eligible examples/);
});

 test("operation-name completed is only a status, not result detail",()=>{
 const h=harness();
 assert.equal(h.run('concreteFeedback({mode:"evaluate",kind:"directive",feedback_summary:"evaluate completed"})'), "");
 });

test("management orders projects and experiments by real update time, with unknown dates last", () => {
  const h=harness();
  setup(h,[project("older",[attemptRun("old",{updated_at:"2026-11-01T00:00:00Z"})]),
    project("active",[attemptRun("invalid",{updated_at:"bad"}),attemptRun("earlier",{updated_at:"2026-11-01T01:00:00Z"}),attemptRun("latest",{updated_at:"2026-11-02T08:00:00+08:00"}),attemptRun("tie",{updated_at:"2026-11-02T00:00:00Z"})]),
    project("unknown",[attemptRun("missing",{updated_at:null})])]);
  assert.deepEqual(h.json("state.projects.map(p=>[p.project_id,p.runs.map(r=>r.run_id)])"),[
    ["active",["latest","tie","earlier","invalid"]],["older",["old"]],["unknown",["missing"]]
  ]);
});


test("dashboard hides opaque references and raw audit records while retaining useful parameters and original data",()=>{
  const hash="717a5ac34d36362ba1f610555707f3570ed238cc5b1a0c5030d03043a8892143",h=harness();
  setup(h,[project("p",[run("a")])]);
  const changes=[{field:"recipe_ref",after:"medical-native:"+hash},{field:"artifact_id",after:"internal-artifact"},{field:"steps",after:"64"},{field:"lr",after:"0.0001"},{field:"instructions",after:"Return one answer."}];
  seed(h,"a",attemptRun("a",{decisions:[{decision_id:"d",kind:"directive",mode:"train",reason:"Test a longer training schedule",changes}]}));h.run('selectRun("a")');selectDecision(h,"d");
  const text=textOf(h.get("project-content"));assert.doesNotMatch(text,/medical-native:|717a5ac|internal-artifact|整理来源与历史|原始决策记录/);assert.match(text,/64/);assert.match(text,/0.0001/);assert.match(text,/Return one answer/);
  assert.equal(h.run('state.snapshot.decisions[0].changes[0].after'),"medical-native:"+hash);
  const box=h.run('changesView({changes:[{field:"recipe_ref",after:"medical-native:'+hash+'"}]},state.snapshot)');assert.equal(box,null);
});


test("benchmark charts show measured accuracy percentages, real baselines and gaps without rewriting data",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);
  const snapshot=attemptRun("a",{jobs:[{job_id:"evaluation:initial",kind:"evaluate",metrics:{"medqa.accuracy":.65}}],decisions:[
    {decision_id:"a",kind:"directive",mode:"evaluate",metrics:{"medqa.accuracy":.7,"medxpertqa.accuracy":.2}},
    {decision_id:"b",kind:"directive",mode:"evaluate",metrics:{"medxpertqa.accuracy":.3}},
    {decision_id:"c",kind:"directive",mode:"evaluate",metrics:{"medqa.accuracy":.8,"medxpertqa.accuracy":.4}},
  ]}),before=JSON.stringify(snapshot);seed(h,"a",snapshot);h.run('selectRun("a")');
  const cards=findClass(h.get("project-content"),"benchmark-card");assert.equal(cards.length,2);
  assert.match(textOf(cards[0]),/MedQA/);assert.match(textOf(cards[0]),/80%/);assert.match(textOf(cards[0]),/基线 65%/);
  assert.equal(findClass(cards[0],"benchmark-line").length,0);assert.equal(findClass(cards[1],"benchmark-line").length,1);assert.equal(findClass(cards[1],"benchmark-baseline").length,0);
  assert.equal(JSON.stringify(snapshot),before);
  const metrics=h.run('metricStrip({"medqa.macro_f1":.721,"invalid_format":2,"score":75,"accuracy":75})');
  assert.match(textOf(metrics),/72.1%/);assert.doesNotMatch(textOf(metrics),/7500%|200%/);
  const comparison=h.run('comparisonView({before:{accuracy:.7},after:{accuracy:.75}})');assert.match(textOf(comparison),/70%/);assert.match(textOf(comparison),/75%/);assert.match(textOf(comparison),/\+5 pp/);
});
test("experiment log expands one entry in place and collapses on repeat selection",()=>{
  const h=harness();setup(h,[project("p",[run("a")])]);seed(h,"a",attemptRun("a",{decisions:[{decision_id:"a",kind:"operation",operation:"define_recipe",reason:"First reason",status:"completed"},{decision_id:"b",kind:"directive",mode:"train",reason:"Second reason",status:"running"}]}));h.run('selectRun("a")');
  const items=findClass(h.get("project-content"),"experiment-log-item"),buttons=items.map(item=>findClass(item,"experiment-log-record")[0]);assert.equal(items.length,2);
  buttons[1].listeners.click();assert.equal(findClass(items[0],"flow-step").length,0);assert.match(textOf(items[1]),/Second reason/);assert.equal(buttons[1].getAttribute("aria-expanded"),"true");
  buttons[0].listeners.click();assert.equal(findClass(items[1],"flow-step").length,0);assert.match(textOf(items[0]),/First reason/);
  buttons[0].listeners.click();assert.equal(findClass(items[0],"flow-step").length,0);assert.equal(buttons[0].getAttribute("aria-expanded"),"false");
});
