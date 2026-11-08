"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const i18nSource = fs.readFileSync(`${__dirname}/../../../src/rsimonitor/static/i18n.js`, "utf8");
const appSource = fs.readFileSync(`${__dirname}/../../../src/rsimonitor/static/app.js`, "utf8");
const html = fs.readFileSync(`${__dirname}/../../../src/rsimonitor/static/index.html`, "utf8");
// Reuse the existing graph/RAF harness without registering its regression tests.
const shared = fs.readFileSync(`${__dirname}/project.test.js`, "utf8").split('\ntest("')[0];
const { harness, setup, seed, run, project,graphState,dragGraphNode,attemptRun,currentGraph,mainNodes } = new Function("require", "__dirname", `${shared}\nreturn {harness,setup,seed,run,project,graphState,dragGraphNode,attemptRun,currentGraph,mainNodes};`)(require, __dirname);
const nodes = root => [root, ...(root.children || []).flatMap(nodes)];
const hasClass = (node, name) => (node.className || "").split(" ").includes(name);
const findClass = (root, name) => nodes(root).filter(n => hasClass(n, name));
const text = root => nodes(root).map(n => n.textContent || "").join(" ");
function api(storage) {
  const ctx = vm.createContext({ localStorage: storage }); vm.runInContext(i18nSource, ctx);
  return ctx.RSII18n;
}
function monitor(options = {}) {
  const calls = [];
  const h = harness(async url => { calls.push(url); throw Error("unexpected fetch"); }, "", { fullSnapshot: true, ...options });
  for (const id of ["decision-search", "event-search"]) h.get(id).value = "";
  return Object.assign(h, { calls });
}
function fixture(h) {
  const s = attemptRun("a", { objective: "保留目标 objective", status: "running", project: {project_id:"p",name:"研究项目"},
    decisions: [
      {decision_id:"d1",kind:"directive",mode:"train",timestamp:"2026-11-01T00:00:01Z",reason:"训练模型 原文 reason",job_id:"j"},
      {decision_id:"d2",kind:"review",action:"continue",timestamp:"2026-11-01T00:00:02Z",strategy:"继续训练 原文 strategy",job_id:"j",metrics:{loss:.8}},
    ], current:{status:"training",mode:"train",strategy:"执行原文 strategy",step:2,total_steps:4},
    jobs:[{job_id:"j",kind:"train",status:"completed",metrics:{loss:.3,learning_rate:.001,"custom.reward":.7},metric_history:[{step:2,metrics:{loss:.3,learning_rate:.001}}]}],
    events:[{kind:"custom.event",summary:"运行开始 原文 log",details:{reason:"错误 原文 log"}}], candidates:[{candidate_id:"c",status:"accepted",metrics:{loss:{baseline:.6,candidate:.3}}}],
    round_results:[{round:1,applied:true,answer_correct:true}] });
  setup(h,[project("p",[run("a",{experiment_name:"实验原文 a"}),run("b")],{name:"研究项目"})]); seed(h,"a",s);h.run('selectRun("a")');h.flushFrames();return s;
}

test("preference persists, reload restores it, and denied/invalid storage falls back safely", () => {
  const values = new Map(), storage={getItem:k=>values.get(k),setItem:(k,v)=>values.set(k,v)};
  const a=api(storage);assert.equal(a.getLanguage(),"zh");let changes=0;const off=a.subscribe(()=>changes++);
  a.setLanguage("en");assert.equal(values.get("rsisys-language"),"en");assert.equal(api(storage).getLanguage(),"en");
  a.setLanguage("en");a.setLanguage("fr");assert.equal(changes,1);off();a.setLanguage("zh");assert.equal(changes,1);
  for(const storage of [undefined,{getItem:()=>"fr",setItem(){}},{getItem(){throw Error("denied");},setItem(){throw Error("denied");}}]) {
    const b=api(storage);assert.equal(b.getLanguage(),"zh");b.setLanguage("en");assert.equal(b.t("指标"),"Metrics");
  }
});

test("dictionary covers both locales, explicit UI calls and every HTML marker", () => {
  const a=api();assert.deepEqual(Object.keys(a.messages.zh),Object.keys(a.messages.en));
  for(const [key,value] of Object.entries(a.messages.en)) {
    assert.ok(value.trim(),key);assert.doesNotMatch(value,/[\u4e00-\u9fff]/,key);
    assert.deepEqual((key.match(/\{\w+\}/g)||[]).sort(),(value.match(/\{\w+\}/g)||[]).sort(),key);
  }
  const keys=[...html.matchAll(/data-i18n(?:-aria-label|-title|-placeholder)?="([^"]+)"/g)].map(m=>m[1]);
  keys.push(...[...appSource.matchAll(/\b(?:t|uiError)\(("(?:[^"\\]|\\.)*")/g)].map(m=>JSON.parse(m[1])));
  for(const key of keys)assert.ok(Object.hasOwn(a.messages.en,key),`missing ${key}`);
  a.setLanguage("en");assert.equal(a.t("第 {p0} 轮",{p0:"<b>$&{p1}</b>"}),"Round <b>$&{p1}</b>");assert.equal(a.t("__proto__"),"__proto__");assert.equal(a.t(undefined),"");
});

test("static translation touches only explicit UI markers and preserves adjacent counters/data", () => {
  const a=api(), title={textContent:"指标"}, counter={textContent:"42"}, model={textContent:"训练模型"};
  title.getAttribute=()=>"指标";
  const input={attrs:{"data-i18n-placeholder":"搜索事件","data-i18n-aria-label":"搜索事件"},getAttribute(k){return this.attrs[k];},setAttribute(k,v){this.attrs[k]=v;}};
  const doc={documentElement:{},querySelectorAll(selector){return selector==="[data-i18n]"?[title]:selector==="[data-i18n-placeholder]"||selector==="[data-i18n-aria-label]"?[input]:[];}};
  a.setLanguage("en");a.applyStatic(doc);assert.equal(title.textContent,"Metrics");assert.equal(counter.textContent,"42");assert.equal(model.textContent,"训练模型");assert.equal(input.attrs.placeholder,"Search events");assert.equal(doc.documentElement.lang,"en");
  a.setLanguage("zh");a.applyStatic(doc);assert.equal(title.textContent,"指标");assert.equal(input.attrs["aria-label"],"搜索事件");
});

test("switching language rerenders Attempt drilldown and preserves project, selected step, cache and model prose", () => {
  const h=monitor();fixture(h);
  findClass(h.get("project-content"),"decision-node")[1].listeners.click();
  h.flushFrames();h.run('state.activeScope.graph.zoomBy(1.25)');dragGraphNode(h,"d2");h.flushFrames();
  const view=h.run("state.activeScope.graphView"),beforeView=graphState(h),old=currentGraph(h);
  assert.deepEqual(findClass(h.get("project-content"),"trajectory-tools")[0].children.map(b=>b.title),["缩小","放大","适应全部节点","重排"]);
  findClass(h.get("project-content"),"flow-map")[0].scrollLeft=123;
  findClass(h.get("project-content"),"flow-map")[0].scrollTop=67;
  const before=h.json('[state.page,state.projectId,state.runId,[...state.activeScope.flowSelected],state.activeScope.cache.size]');
  const snapshot=h.run("state.snapshot"),cache=h.run("state.activeScope.cache");
  h.get("lang-en").listeners.click();
  assert.equal(h.get("lang-en").attributes["aria-pressed"],"true");assert.equal(h.get("lang-zh").attributes["aria-pressed"],"false");
  assert.deepEqual(h.json('[state.page,state.projectId,state.runId,[...state.activeScope.flowSelected],state.activeScope.cache.size]'),before);
  assert.equal(h.run("state.snapshot"),snapshot);assert.equal(h.run("state.activeScope.cache"),cache);assert.deepEqual(h.calls,[]);
  assert.ok(old.disposals>0);assert.equal(h.run("state.activeScope.graphView"),view);assert.deepEqual(graphState(h),beforeView);
  const controls=findClass(h.get("project-content"),"trajectory-tools")[0].children;
  assert.deepEqual(controls.map(b=>b.textContent),["−","+","⤢","↺"]);
  assert.deepEqual(controls.map(b=>b.title),["Zoom out","Zoom in","Fit all nodes","Rearrange"]);
  for(const button of controls)assert.equal(button.getAttribute("aria-label"),button.title);
  assert.equal(h.run("state.projectId"),"p");assert.equal(h.run("state.runId"),"a");
  assert.equal(h.get("project-name").textContent,"实验原文 a");assert.equal(h.get("project-description").textContent,"研究项目");assert.equal(h.get("project-description").hidden,false);
  assert.equal(findClass(h.get("project-content"),"flow-map")[0].scrollLeft,123);
  assert.equal(findClass(h.get("project-content"),"flow-map")[0].scrollTop,67);
  assert.match(text(h.get("project-content")),/Metrics at review/);assert.match(text(h.get("project-content")),/继续训练 原文 strategy/);
  assert.doesNotMatch(text(h.get("project-content")),/保留目标 objective/);assert.equal(h.run("state.snapshot.objective"),"保留目标 objective");assert.doesNotMatch(text(h.get("project-content")),/错误 原文 log/);
  assert.doesNotMatch(text(h.get("project-content")),/custom.event/);
  assert.match(text(h.get("project-content")),/Research graph/);assert.match(text(h.get("project-content")),/2 decisions/);
  const graphNodes=mainNodes(h);
  assert.equal(graphNodes.length,2);assert.equal(graphNodes[1].getAttribute("aria-pressed"),"true");
  assert.equal(graphNodes[0].getAttribute("aria-label"),"1 · Recorded hypothesis");
  assert.match(text(h.get("project-content")),/Continue/);assert.match(text(h.get("project-content")),/Experiment start/);
  assert.equal(findClass(h.get("project-content"),"graph-controls").length,0);assert.match(h.run("dateText('2026-11-01T00:00:00Z')"),/,/);
  assert.match(text(h.run('metricStrip({"custom.reward":.7},true)')),/custom.reward/);
  h.flushFrames();h.get("lang-zh").listeners.click();assert.match(text(h.get("project-content")),/复查时指标/);assert.deepEqual(h.calls,[]);assert.deepEqual(graphState(h),beforeView);h.flushFrames();
});

test("language switches during graph interaction defer its DOM until idle while retaining the view and model prose", () => {
  const h=monitor();fixture(h);findClass(h.get("project-content"),"decision-node")[0].listeners.click();const record=currentGraph(h),before=graphState(h),count=h.renders.length;
  record.svg.listeners.pointerdown({pointerId:2,button:0,clientX:100,clientY:100,target:record.svg});h.get("lang-en").listeners.click();
  assert.equal(h.renders.length,count);assert.equal(record.disposals,0);assert.equal(h.run("state.activeScope.graph"),record.svg);assert.equal(h.run("state.activeScope.renderPending"),true);
  assert.match(text(h.get("project-content")),/研究脉络/);assert.doesNotMatch(text(h.get("project-content")),/Research graph/);
  record.svg.listeners.pointerup({pointerId:2,type:"pointerup"});
  assert.equal(h.run("state.activeScope.renderPending"),false);assert.ok(record.disposals>0);assert.match(text(h.get("project-content")),/Research graph/);
  assert.match(text(h.get("project-content")),/训练模型 原文 reason/);assert.deepEqual(graphState(h),before);assert.deepEqual(h.calls,[]);h.flushFrames();
});

test("unknown events and application errors/empty states switch without refetching", () => {
  const h=monitor();h.run('notice(()=>t("请求超时。"),true); showEmpty(()=>t("暂时无法读取项目"),()=>t("请检查本地监控服务；恢复后将自动重新读取。")); RSII18n.setLanguage("en")');
  assert.equal(h.get("connection").textContent,undefined);assert.equal(h.get("notice").textContent,"Request timed out.");assert.equal(h.get("empty-title").textContent,"Projects temporarily unavailable");assert.deepEqual(h.calls,[]);
  fixture(h);h.run('state.activeScope.cache.get("a").status="error"; state.activeScope.cache.get("a").errorInfo=uiError("运行快照协议或 ID 不匹配。"); renderProject(); RSII18n.setLanguage("zh"); RSII18n.setLanguage("en")');
  assert.match(text(h.get("project-content")),/Run snapshot schema or ID mismatch/);assert.match(text(h.get("project-content")),/Retaining the previous snapshot/);
  assert.doesNotMatch(text(h.get("project-content")),/custom.event/);
});

test("management language changes translate experiment cards without opening or fetching details", () => {
  const h=monitor();setup(h,[project("p",[run("a",{experiment_name:"实验名称 原文",status:"running",counts:{decisions:35}}),run("b")],{name:"研究项目 原文"}),project("q",[run("c")],{name:"研究项目 原文"})]);
  h.get("lang-en").listeners.click();
  assert.equal(h.run("state.page"),"management");assert.equal(h.get("management").hidden,false);assert.equal(h.get("projects").hidden,true);
  assert.equal(h.run("state.snapshot"),null);assert.equal(h.get("project-content").children.length,0);
  assert.equal(h.get("experiment-count").textContent,"2 projects · 3 experiments");
  const groups=findClass(h.get("project-list"),"project-group");assert.equal(groups.length,2);assert.deepEqual(groups.map(g=>findClass(g,"experiment-card").length),[2,1]);
  assert.match(text(groups[0]),/实验名称 原文/);assert.match(text(groups[0]),/2 experiments/);assert.match(text(groups[1]),/1 experiments/);
  assert.match(text(h.get("project-list")),/研究项目 原文/);assert.match(text(h.get("project-list")),/Running/);assert.match(text(h.get("project-list")),/35 decisions/);
  h.get("lang-zh").listeners.click();assert.match(text(h.get("project-list")),/35 次决策/);assert.deepEqual(h.calls,[]);
});

test("management cards replace sidebar navigation; manual refresh and export are absent",()=>{
  for(const id of ["nav-metrics","project-select","run-select","connection","refresh","auto-refresh","export-report"])
    assert.ok(!html.includes(`id="${id}"`),id);
  assert.match(html,/<section\b[^>]*id="management"[^>]*>[\s\S]*?id="project-list"[\s\S]*?<\/section>/);
  assert.doesNotMatch(html,/<aside\b/);
  assert.ok(html.includes('id="back-projects"'));
});

test("experiment detail loads the decision graph without the old multi-run project graph", () => {
  for(const name of ["/lineage.js",'id="graph-tools"','id="project-run-count"','id="project-graph"'])assert.ok(!html.includes(name),name);
  assert.ok(html.includes('id="project-content"'));
  const scripts=[...html.matchAll(/<script\b[^>]*src="([^"]+)"/g)].map(m=>m[1]);
  assert.deepEqual(scripts,["/i18n.js","/decisions.js","/app.js"]);
  assert.doesNotMatch(appSource,/RSILineage|project-root|projectModel|renderGraph|networkLayout|研究轮次/);
  const css=fs.readFileSync(`${__dirname}/../../../src/rsimonitor/static/style.css`, "utf8");
  assert.doesNotMatch(css,/graph-node|graph-viewport|graph-controls|graph-header|graph-tools/);
});
