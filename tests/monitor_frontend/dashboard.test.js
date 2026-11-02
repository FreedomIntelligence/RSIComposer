"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs");
const shared=fs.readFileSync(`${__dirname}/project.test.js`,"utf8").split('\ntest("')[0];
const {harness,setup,seed,run,project,findClass,textOf,descendants}=new Function("require","__dirname",`${shared}\nreturn {harness,setup,seed,run,project,findClass,textOf,descendants};`)(require,__dirname);
function open(extra={}){
  const h=harness(),storage=new Map();h.ctx.localStorage={getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)};
  setup(h,[project("p",[run("a"),run("b")])]);seed(h,"a",run("a",extra));seed(h,"b",run("b",extra));h.run('selectRun("a")');return {h,storage};
}
const host=h=>h.get("project-content"),button=(node,label)=>descendants(node).find(n=>n.tagName==="button"&&n.textContent===label);
const panel=(id,type="line",extra={})=>({id,title:id,type,metrics:["loss"],x:"step",group_by:"task",...extra});
const fixture={metric_definitions:{loss:{label:"Training loss",unit:"number",direction:"lower"}},metric_points:[{metric:"loss",value:.8,step:1,task_id:"first"},{metric:"loss",value:.4,step:2,task_id:"first"},{metric:"loss",value:.6,step:1,task_id:"second"}],dashboard:{schema_version:1,panels:[panel("Loss")]}};

test("explicit metric stream is authoritative, keeps task groups separate and does not rewrite the snapshot",()=>{
  const {h}=open({...fixture,jobs:[{job_id:"legacy",metrics:{loss:999}}]}),before=h.run('JSON.stringify(state.snapshot)');
  const data=h.run('dashboardData(state.snapshot)');assert.equal(data.points.length,3);
  const series=h.run('dashboardSeries(state.snapshot.dashboard.panels[0],dashboardData(state.snapshot),state.snapshot)');assert.equal(series.length,2);assert.equal(series[0].points.length,2);assert.equal(series[1].points.length,1);
  assert.equal(findClass(host(h),"benchmark-line").length,1);assert.doesNotMatch(textOf(findClass(host(h),"metric-dashboard")[0]),/999/);assert.equal(h.run('JSON.stringify(state.snapshot)'),before);
  h.ctx.sample={...fixture,metric_points:[]};assert.equal(h.run('dashboardData(sample).points.length'),0);
});
test("declared percent units and directions drive scalar and comparison rendering without guessing a score scale",()=>{
  const sample={metric_definitions:{quality:{label:"Quality",unit:"percent",direction:"higher"},loss:{label:"Loss",unit:"number",direction:"lower"}},metric_points:[{metric:"quality",value:.8},{metric:"loss",value:2}],jobs:[{job_id:"base",role:"baseline",metrics:{quality:.7,loss:3}}],dashboard:{schema_version:1,panels:[panel("Compare","table",{metrics:["quality","loss"],group_by:"none"}),panel("Values","scalar",{metrics:["quality","loss"],group_by:"none"})]}};
  const {h}=open(sample),cards=findClass(host(h),"dashboard-panel");assert.equal(cards.length,2);assert.match(textOf(cards[0]),/80%/);assert.match(textOf(cards[0]),/\+10 pp/);assert.equal(findClass(cards[0],"metric-better").length,2);assert.match(textOf(cards[1]),/80%/);
  assert.equal(h.run('dashboardFormat(.8,{unit:"number"})'),"0.8");
});
test("line charts reject mixed units and missing x values without inventing data",()=>{
  const {h}=open(fixture);h.ctx.sample={...fixture,metric_definitions:{loss:{unit:"number"},accuracy:{unit:"percent"}},metric_points:[{metric:"loss",value:1,step:1},{metric:"accuracy",value:.7,step:1}]};
  const mixed=h.run('dashboardPanel({id:"m",type:"line",title:"Mixed",metrics:["loss","accuracy"],x:"step",group_by:"none"},dashboardData(sample),sample)');assert.match(textOf(mixed),/相同单位/);assert.equal(findClass(mixed,"benchmark-line").length,0);
  const missing=h.run('dashboardPanel({id:"m",type:"line",title:"Time",metrics:["loss"],x:"elapsed_seconds",group_by:"none"},dashboardData(sample),sample)');assert.match(textOf(missing),/当前横轴暂无数据/);
});
test("saved user layout overrides producer defaults, persists and remains isolated per experiment",()=>{
  const {h,storage}=open(fixture),editor=findClass(host(h),"dashboard-editor")[0];editor.open=true;editor.listeners.toggle();
  assert.equal(h.run('state.activeScope.dashboardEditing'),true);
  const row=findClass(editor,"dashboard-editor-panel")[0],input=descendants(row).find(n=>n.tagName==="input"&&n.type!=="checkbox"),select=descendants(row).find(n=>n.tagName==="select");input.value="My loss";input.listeners.input();select.value="scalar";select.listeners.change();
  const previous=host(h).firstElementChild;h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,updated_at:"2026-11-03"};renderProject()');assert.equal(host(h).firstElementChild,previous);assert.equal(h.run('state.activeScope.renderPending'),true);
  button(editor,"保存布局").listeners.click();assert.match(textOf(host(h)),/My loss/);assert.equal(findClass(host(h),"benchmark-line").length,0);assert.equal(JSON.parse(storage.get("rsimonitor-dashboard:v1:a")).panels[0].type,"scalar");
  h.run('selectRun("b")');assert.doesNotMatch(textOf(host(h)),/My loss/);h.run('selectRun("a")');assert.match(textOf(host(h)),/My loss/);
  const next=findClass(host(h),"dashboard-editor")[0];next.open=true;next.listeners.toggle();button(next,"恢复默认").listeners.click();assert.equal(storage.has("rsimonitor-dashboard:v1:a"),false);assert.match(textOf(host(h)),/Training loss|Loss/);
});
test("add remove and reorder panel controls preserve a deliberately empty layout",()=>{
  const {h}=open(fixture),editor=findClass(host(h),"dashboard-editor")[0];editor.open=true;editor.listeners.toggle();button(editor,"添加面板").listeners.click();assert.equal(findClass(editor,"dashboard-editor-panel").length,2);
  const rows=findClass(editor,"dashboard-editor-panel");button(rows[1],"上移").listeners.click();assert.equal(h.run('state.activeScope.dashboardDraft.panels[0].id'),"custom-1");
  button(editor,"删除面板").listeners.click();button(editor,"删除面板").listeners.click();button(editor,"保存布局").listeners.click();assert.equal(findClass(host(h),"dashboard-panel").length,0);assert.ok(findClass(host(h),"dashboard-editor").length);
});
test("line marker budget is bounded while full curve samples and scalar latest value survive",()=>{
  const sample={...fixture,metric_points:Array.from({length:1000},(_,i)=>({metric:"loss",value:i===501?100:i/1000,step:i,task_id:"first"}))},{h}=open(sample);
  assert.ok(findClass(host(h),"benchmark-point").length<170);const curve=findClass(host(h),"benchmark-line")[0];assert.equal(curve.getAttribute("points").split(" ").length,1000);assert.match(textOf(host(h)),/0.999/);
});

test("editor hides irrelevant axes and enforces the metric selection limit",()=>{
  const keys=Array.from({length:13},(_,i)=>`m${i}`),sample={metric_definitions:Object.fromEntries(keys.map(key=>[key,{label:key,unit:"number",direction:"neutral"}])),metric_points:[],dashboard:{schema_version:1,panels:[panel("Many","line",{metrics:keys.slice(0,12)})]}},{h}=open(sample),editor=findClass(host(h),"dashboard-editor")[0];editor.open=true;editor.listeners.toggle();
  const checkboxes=()=>descendants(editor).filter(n=>n.type==="checkbox");assert.equal(checkboxes().filter(n=>n.disabled).length,1);checkboxes()[0].checked=false;checkboxes()[0].listeners.change();assert.equal(checkboxes().filter(n=>n.disabled).length,0);
  const type=descendants(editor).find(n=>n.tagName==="select");type.value="scalar";type.listeners.change();assert.equal(descendants(editor).filter(n=>n.tagName==="select").length,2);
});
