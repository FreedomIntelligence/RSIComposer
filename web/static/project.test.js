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
  constructor(tag = "div") { this.tagName = tag; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.scrollLeft = 0; this.open = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get childElementCount() { return this.children.length; }
  get firstElementChild() { return this.children[0]; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  addEventListener(k, f) { this.listeners[k] = f; }
  querySelectorAll(selector) { return descendants(this).filter(n => selector === "details[data-retain-key]" && n.tagName === "details" && n.dataset.retainKey); }
  querySelector(selector) { return selector === ".flow-map" ? findClass(this, "flow-map")[0] : null; }
  focus() {}
}
function harness(fetcher = async () => { throw Error("unexpected fetch"); }, search = "") {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) { const e = new Element(); e.id = id; elements.set(id, e); } return elements.get(id); };
  const documentListeners = {}, document = { getElementById: get, createElement: t => new Element(t), createElementNS: (_, t) => new Element(t), addEventListener(k, f) { documentListeners[k] = f; }, hidden: false };
  const ctx = vm.createContext({ document, location: { href: `http://localhost:8769/${search}`, search }, history: { replaceState(_, __, url) { ctx.savedURL = String(url); } }, AbortController, URL, URLSearchParams, setTimeout, clearTimeout, fetch: fetcher });
  vm.runInContext(fs.readFileSync(`${__dirname}/i18n.js`, "utf8"), ctx);
  vm.runInContext(source.slice(0, source.lastIndexOf("\nrefresh();")), ctx);
  vm.runInContext("state.auto = false;", ctx);
  return { ctx, get, document, documentListeners, run: s => vm.runInContext(s, ctx), json: s => JSON.parse(JSON.stringify(vm.runInContext(s, ctx))) };
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

test("catalog creates one project per startup, flattens old groups and gives repeated names distinct labels", () => {
  const input = [project("legacy", [run("old"), run("current", { parent_run_id: "old" })], { name: "Experiment" }), project("other", [run("third")], { name: "Experiment" })];
  const h = harness(); setup(h, input);
  assert.deepEqual(h.json("state.projects.map(p=>[p.project_id,p.runs.map(r=>r.run_id)])"), [["old", ["old"]], ["current", ["current"]], ["third", ["third"]]]);
  const names = h.get("project-list").children.map(textOf);
  assert.equal(new Set(names).size, 3);
  assert.ok(names.every(name => name.includes("Experiment")));
  assert.equal(h.run("state.runId"), "old");
  assert.deepEqual(input[0].runs.map(r=>r.run_id), ["old", "current"]);
  assert.equal(findClass(h.get("project-content"), "graph-node").length, 0);
  assert.equal(findClass(h.get("project-content"), "graph-viewport").length, 0);
});

test("canonical catalog metadata wins over duplicate legacy entries without losing historical projects", () => {
  const h=harness();setup(h,[project("legacy",[run("a"),run("b")]),project("a",[run("a")],{name:"Own experiment"})]);
  assert.deepEqual(h.json("state.projects.map(p=>p.project_id)"),["a","b"]);
  assert.equal(h.run("state.projects[0].name"),"Own experiment");
  setup(h,[project("a",[run("a")]),project("b",[run("b")]),project("current",[run("current",{parent_run_id:"b"})])]);
  assert.equal(h.get("project-list").children.length,3);
});

test("first project automatically loads its sole run and renders decisions without an extra click", async () => {
  const calls=[], snapshot=run("a", { decisions:[{decision_id:"d",mode:"train",reason:"Plan training",feedback:{summary:"Training result"}}] });
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("a")]}:snapshot);});
  await h.run("refresh()");
  assert.equal(h.run("state.projectId"),"a");assert.equal(h.run("state.runId"),"a");
  assert.equal(h.run("state.snapshot.run_id"),"a");assert.match(textOf(h.get("project-content")),/Plan training.*Training result/);
  assert.deepEqual(calls,["/api/projects","/api/runs/a"]);
});

test("sidebar selection loads the new project's decisions immediately and preserves each execution history", async () => {
  const requests=[], b=run("b",{decisions:[{decision_id:"same",mode:"evaluate",reason:"Only B"}]});
  const h=harness(async url=>{requests.push(url);return response(b);});setup(h,[project("a"),project("b")]);
  const a=run("a",{decisions:Array.from({length:35},(_,i)=>({decision_id:`d${i}`,timestamp:new Date(Date.UTC(2026,10,1,0,i)).toISOString(),mode:"train",reason:`history-${i}-end`}))});
  seed(h,"a",a);await h.run('selectProject("a")');
  findClass(h.get("project-content"),"more-button")[0].listeners.click();
  findClass(h.get("project-content"),"flow-choice")[34].listeners.click();
  const detail=h.get("project-content").firstElementChild;
  h.get("project-list").children[1].listeners.click();await settle();
  assert.equal(h.run("state.projectId"),"b");assert.equal(h.run("state.runId"),"b");
  assert.match(textOf(h.get("project-content")),/Only B/);assert.doesNotMatch(textOf(h.get("project-content")),/history-34-end/);
  assert.equal(new URL(h.ctx.savedURL).searchParams.get("run"),"b");
  h.get("project-list").children[0].listeners.click();await settle();
  assert.equal(h.get("project-content").firstElementChild,detail);
  assert.match(textOf(h.get("project-content")),/history-34-end/);
  assert.equal(findClass(h.get("project-content"),"flow-choice").length,35);
  assert.equal(h.run('state.snapshot.decisions.length'),35);
  assert.deepEqual(requests,["/api/runs/b"]);
});

test("run deep links and legacy project links open the owning startup directly", async () => {
  for(const [search,expected] of [["?run=b","b"],["?project=wrong&run=b","b"],["?project=legacy&run=b","b"],["?project=b","b"],["?project=legacy","a"]]){
    const calls=[],h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("legacy",[run("a"),run("b")])]}:run(expected,{decisions:[{decision_id:"d",reason:`opened ${expected}`}]}));},search);
    await h.run("refresh()");
    assert.equal(h.run("state.projectId"),expected);assert.equal(h.run("state.runId"),expected);
    assert.match(textOf(h.get("project-content")),new RegExp(`opened ${expected}`));
    assert.deepEqual(calls,["/api/projects",`/api/runs/${expected}`]);
    assert.equal(new URL(h.ctx.savedURL).searchParams.get("project"),expected);
    assert.equal(new URL(h.ctx.savedURL).searchParams.get("run"),expected);
  }
});

test("parent and source references are provenance only and never select, fetch or combine other runs", async () => {
  const calls=[],snapshot=run("current",{parent_run_id:"parent",source_run_ids:["parent","outside"],decisions:[{decision_id:"d",reason:"Current decision"}]});
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("current",[snapshot]),project("parent")]}:snapshot);});
  await h.run("refresh()");
  assert.equal(h.run("state.projectId"),"current");assert.deepEqual(h.json("state.projects.map(p=>p.runs.length)"),[1,1]);
  assert.deepEqual(h.json("[...state.activeScope.cache.keys()]"),["current"]);
  assert.deepEqual(calls,["/api/projects","/api/runs/current"]);
  assert.equal(findClass(h.get("project-content"),"provenance-link").length,2);
  assert.equal(findClass(h.get("project-content"),"provenance-link")[0].href,"?run=parent");
  assert.match(textOf(h.get("project-content")),/来源引用不表示项目归属或自动继承/);
});

test("same record IDs in different projects keep decisions, feedback, jobs and candidates isolated", async () => {
  const h=harness();setup(h,[project("a"),project("b")]);
  for(const [id,loss] of [["a",1],["b",2]])seed(h,id,run(id,{decisions:[{decision_id:"same",kind:"directive",directive_id:"shared",mode:"train",feedback:{summary:`feedback-${id}`}}],jobs:[{job_id:"same",directive_id:"shared",metrics:{loss}}],candidates:[{candidate_id:"same",directive_id:"shared",metrics:{accuracy:loss/10}}]}));
  for(const [id,other,loss] of [["a","b",1],["b","a",2],["a","b",1]]){
    await h.run(`selectProject("${id}")`);assert.match(textOf(h.get("project-content")),new RegExp(`feedback-${id}`));
    assert.doesNotMatch(textOf(h.get("project-content")),new RegExp(`feedback-${other}`));assert.equal(h.run("state.snapshot.jobs[0].metrics.loss"),loss);
  }
});

test("switch aborts pending reads and late responses cannot replace the selected project", async () => {
  const old=deferred(),fresh=deferred(),signals=[],h=harness((url,options)=>{signals.push(options.signal);return signals.length===1?old.promise:fresh.promise;});
  setup(h,[project("a"),project("b")]);seed(h,"b",run("b",{decisions:[{reason:"B history"}]}));
  const pending=h.run('selectProject("a")');await h.run('selectProject("b")');assert.equal(signals[0].aborted,true);
  const current=h.run('selectProject("a")');fresh.resolve(response(run("a",{decisions:[{reason:"Fresh A"}]})));await current;
  old.resolve(response(run("a",{decisions:[{reason:"Stale A"}]})));await pending;
  assert.match(textOf(h.get("project-content")),/Fresh A/);assert.doesNotMatch(textOf(h.get("project-content")),/Stale A|B history/);
});

test("repeated selection reuses pending requests and cached decisions", async () => {
  const next=deferred(),calls=[],h=harness(url=>{calls.push(url);return next.promise;});setup(h,[project("a")]);
  const first=h.run('selectProject("a")');h.run('selectProject("a")');
  assert.deepEqual(calls,["/api/runs/a"]);assert.match(textOf(h.get("project-content")),/正在读取/);
  next.resolve(response(run("a",{decisions:[{reason:"Loaded"}]})));await first;await h.run('selectProject("a")');
  assert.deepEqual(calls,["/api/runs/a"]);assert.match(textOf(h.get("project-content")),/Loaded/);
});

test("refresh reloads only the selected startup and retains decision selection and historical caches", async () => {
  const calls=[],a=run("a",{decisions:[{decision_id:"first",mode:"train"},{decision_id:"second",mode:"evaluate",feedback:{summary:"New result"}}]});
  const h=harness(async url=>{calls.push(url);return response(url==="/api/projects"?{projects:[project("a"),project("b")]}:a);});setup(h,[project("a"),project("b")]);
  seed(h,"a",{...a,decisions:a.decisions.map(d=>({...d,feedback:{summary:"Previous result"}}))});seed(h,"b",run("b",{decisions:[{reason:"B remains"}]}));
  await h.run('selectProject("a")');findClass(h.get("project-content"),"flow-choice")[1].listeners.click();
  await h.run("refresh()");
  assert.deepEqual(calls,["/api/projects","/api/runs/a"]);assert.equal(h.run('state.activeScope.flowSelected.get("a")'),"second");
  assert.match(textOf(h.get("project-content")),/New result/);
  await h.run('selectProject("b")');assert.match(textOf(h.get("project-content")),/B remains/);assert.equal(calls.length,2);
});

test("failed reads retain decisions and a retry restores the current project", async () => {
  let fail=true;const h=harness(async()=>{if(fail)throw Error("offline");return response(run("a",{decisions:[{reason:"Recovered"}]}));});
  setup(h,[project("a")]);seed(h,"a",run("a",{decisions:[{reason:"Cached decision"}]}));await h.run('selectProject("a")');
  await h.run('loadRun("a",state.activeScope,true)');assert.match(textOf(h.get("project-content")),/offline.*Cached decision/);
  fail=false;findClass(h.get("project-content"),"run-detail")[0].children.find(n=>n.tagName==="button").listeners.click();await settle();
  assert.match(textOf(h.get("project-content")),/Recovered/);assert.equal(h.run('state.activeScope.cache.get("a").status'),"ready");
});

test("demo contains separate projects, loads decisions directly and keeps live caches and links", async () => {
  const demo=JSON.parse(fs.readFileSync(`${__dirname}/demo.json`,"utf8")),h=harness(async()=>response(demo));
  setup(h,[project("live")]);seed(h,"live",run("live",{decisions:[{reason:"Live history"}]}));await h.run('selectProject("live")');
  const liveURL=h.ctx.savedURL;h.run('state.demo=true;state.projectId=""');await h.run("refresh()");
  assert.equal(h.run("state.projects.length"),2);
  for(const snapshot of [demo,...demo.demo_runs]){
    assert.equal(snapshot.project.project_id,snapshot.run_id);await h.run(`selectProject(${JSON.stringify(snapshot.run_id)})`);
    assert.equal(h.run("state.snapshot.run_id"),snapshot.run_id);assert.equal(findClass(h.get("project-content"),"execution-sequence").length,1);
  }
  assert.equal(h.ctx.savedURL,liveURL);
  h.run('state.demo=false;state.projectId="live"');setup(h,[project("live")]);
  assert.match(textOf(h.get("project-content")),/Live history/);
});

test("switching to demo rejects a late live response before the demo catalog arrives", async () => {
  const pending=deferred();let signal;const h=harness((url,options)=>{signal=options.signal;return pending.promise;});setup(h,[project("a")]);
  const request=h.run('loadRun("a",state.activeScope)');h.run('refresh=function(){};switchDemo()');assert.equal(signal.aborted,true);
  pending.resolve(response(run("a",{decisions:[{reason:"Late live history"}]})));await request;
  assert.equal(h.run("state.snapshot"),null);assert.equal(h.run("state.activeScope"),null);
});

test("empty catalog and unavailable snapshot have readable states without a project graph", async () => {
  const h=harness(async url=>response({projects:[]}));await h.run("refresh()");
  assert.equal(h.get("empty").hidden,false);assert.equal(h.get("projects").hidden,true);assert.equal(h.get("project-list").children.length,1);
  setup(h,[project("broken",[run("broken",{status:"unavailable"})])]);
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
