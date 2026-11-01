"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const i18nSource = fs.readFileSync(`${__dirname}/i18n.js`, "utf8");
const appSource = fs.readFileSync(`${__dirname}/app.js`, "utf8");
const html = fs.readFileSync(`${__dirname}/index.html`, "utf8");
// Reuse the existing graph/RAF harness without registering its regression tests.
const shared = fs.readFileSync(`${__dirname}/project.test.js`, "utf8").split('\ntest("')[0];
const { harness, setup, seed, run, project } = new Function("require", "__dirname", `${shared}\nreturn {harness,setup,seed,run,project};`)(require, __dirname);
const nodes = root => [root, ...(root.children || []).flatMap(nodes)];
const hasClass = (node, name) => (node.className || "").split(" ").includes(name);
const findClass = (root, name) => nodes(root).filter(n => hasClass(n, name));
const text = root => nodes(root).map(n => n.textContent || "").join(" ");
function api(storage) {
  const ctx = vm.createContext({ localStorage: storage }); vm.runInContext(i18nSource, ctx);
  vm.runInContext(fs.readFileSync(`${__dirname}/lineage.js`, "utf8"), ctx); return ctx.RSII18n;
}
function monitor(options = {}) {
  const calls = [];
  const h = harness(async url => { calls.push(url); throw Error("unexpected fetch"); }, "", { fullSnapshot: true, ...options });
  for (const id of ["decision-search", "event-search"]) h.get(id).value = "";
  return Object.assign(h, { calls });
}
function fixture(h) {
  const s = run("a", { objective: "保留目标 objective", status: "running", project: {project_id:"p",name:"研究项目"},
    decisions: [
      {decision_id:"d1",kind:"directive",mode:"train",timestamp:"2026-11-01T00:00:01Z",reason:"训练模型 原文 reason",job_id:"j"},
      {decision_id:"d2",kind:"review",action:"continue",timestamp:"2026-11-01T00:00:02Z",strategy:"继续训练 原文 strategy",job_id:"j",metrics:{loss:.8}},
    ], current:{status:"training",mode:"train",strategy:"执行原文 strategy",step:2,total_steps:4},
    jobs:[{job_id:"j",kind:"train",status:"completed",metrics:{loss:.3,learning_rate:.001,"custom.reward":.7},metric_history:[{step:2,metrics:{loss:.3,learning_rate:.001}}]}],
    events:[{kind:"custom.event",summary:"运行开始 原文 log",details:{reason:"错误 原文 log"}}], candidates:[{candidate_id:"c",status:"accepted",metrics:{loss:{baseline:.6,candidate:.3}}}],
    round_results:[{round:1,applied:true,answer_correct:true}] });
  setup(h,[project("p",[run("a"),run("b")])]); seed(h,"a",s);h.run('selectRun("a")');return s;
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

test("switching language rerenders cached graph/run, preserves selection/step/zoom/cache/filter and model prose", () => {
  const h=monitor();fixture(h);
  findClass(h.get("project-graph"),"flow-choice")[1].listeners.click();
  h.run('state.activeScope.flowLimits.set("a",60); $("project-graph").graphView.autoFit=false; $("project-graph").graphView.scale=.75; $("project-graph").graphView.viewport.scrollLeft=123; $("project-graph").graphView.viewport.scrollTop=45;');
  const before=h.json('[state.activeScope.selected,[...state.activeScope.flowSelected],[...state.activeScope.positions],state.activeScope.cache.size]');
  const snapshot=h.run("state.snapshot"),cache=h.run("state.activeScope.cache");
  h.get("lang-en").listeners.click();
  assert.equal(h.get("lang-en").attributes["aria-pressed"],"true");assert.equal(h.get("lang-zh").attributes["aria-pressed"],"false");
  assert.deepEqual(h.json('[state.activeScope.selected,[...state.activeScope.flowSelected],[...state.activeScope.positions],state.activeScope.cache.size]'),before);
  assert.equal(h.run("state.snapshot"),snapshot);assert.equal(h.run("state.activeScope.cache"),cache);assert.deepEqual(h.calls,[]);
  assert.equal(h.get("project-graph").graphView.scale,.75);assert.equal(h.get("project-graph").graphView.viewport.scrollLeft,123);assert.equal(h.get("project-graph").graphView.viewport.scrollTop,45);
  assert.match(text(h.get("project-graph")),/Metrics at review/);assert.match(text(h.get("project-graph")),/继续训练 原文 strategy/);
  assert.doesNotMatch(text(h.get("project-graph")),/保留目标 objective/);assert.equal(h.run("state.snapshot.objective"),"保留目标 objective");assert.doesNotMatch(text(h.get("project-graph")),/错误 原文 log/);
  assert.doesNotMatch(text(h.get("project-graph")),/custom.event/);
  assert.match(text(h.get("graph-tools")),/Fit/);assert.equal(findClass(h.get("graph-tools"),"graph-control-group").length,2);assert.equal(findClass(h.get("graph-tools"),"zoom-reset")[0].textContent,"75%");
  assert.equal(findClass(h.get("project-graph"),"graph-controls").length,0);assert.match(h.run("dateText('2026-11-01T00:00:00Z')"),/,/);
  assert.match(text(h.run('metricStrip({"custom.reward":.7},true)')),/custom.reward/);
  h.get("lang-zh").listeners.click();assert.match(text(h.get("project-graph")),/复查时指标/);assert.deepEqual(h.calls,[]);
});

test("unknown events and application errors/empty states switch without refetching", () => {
  const h=monitor();h.run('notice(()=>t("请求超时。"),true); showEmpty(()=>t("暂时无法读取项目"),()=>t("请检查本地监控服务；恢复后将自动重新读取。")); RSII18n.setLanguage("en")');
  assert.equal(h.get("connection").textContent,undefined);assert.equal(h.get("notice").textContent,"Request timed out.");assert.equal(h.get("empty-title").textContent,"Projects temporarily unavailable");assert.deepEqual(h.calls,[]);
  fixture(h);h.run('state.activeScope.cache.get("a").status="error"; state.activeScope.cache.get("a").errorInfo=uiError("运行快照协议或 ID 不匹配。"); renderProject(); RSII18n.setLanguage("zh"); RSII18n.setLanguage("en")');
  assert.match(text(h.get("project-graph")),/Run snapshot schema or ID mismatch/);assert.match(text(h.get("project-graph")),/Retaining the previous snapshot/);
  assert.doesNotMatch(text(h.get("project-graph")),/custom.event/);
});

test("sidebar replaces navigation and run selectors; manual refresh and export are absent",()=>{
  for(const id of ["nav-metrics","project-select","run-select","connection","refresh","auto-refresh","export-report"])
    assert.ok(!html.includes(`id="${id}"`),id);
  assert.ok(html.includes('id="project-list"'));
});

test("changing language during graph RAF defers its redraw, retains motion and performs no request", () => {
  const h=monitor({raf:true});fixture(h);const host=h.get("project-graph"),view=host.graphView;
  const button=findClass(host,"graph-node")[1];button.setPointerCapture=()=>{};
  button.listeners.pointerdown({button:0,clientX:100,clientY:100,pointerId:1,stopPropagation(){}});
  button.listeners.pointermove({clientX:145,clientY:125,pointerId:1});assert.equal(h.frames.size,1);
  const node=button;h.get("lang-en").listeners.click();assert.equal(host.graphView,view);assert.equal(view.pendingRender,true);assert.equal(h.frames.size,1);assert.equal(findClass(host,"graph-node")[1],node);
  h.frames.step();button.listeners.pointerup();h.frames.drain();assert.equal(view.simulating,false);assert.equal(view.pendingRender,false);assert.match(text(host),/Research cycle 1/);assert.deepEqual(h.calls,[]);
});
