"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs");
const shared=fs.readFileSync(`${__dirname}/project.test.js`,"utf8").split('\ntest("')[0];
const {harness,setup,seed,run,project,findClass,textOf,descendants}=new Function("require","__dirname",`${shared}\nreturn {harness,setup,seed,run,project,findClass,textOf,descendants};`)(require,__dirname);
function open(extra={},storage=new Map()){
  const h=harness();h.ctx.localStorage={getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)};
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
test("line marker budget is bounded while full curve samples and scalar latest value survive",()=>{
  const sample={...fixture,metric_points:Array.from({length:1000},(_,i)=>({metric:"loss",value:i===501?100:i/1000,step:i,task_id:"first"}))},{h}=open(sample);
  assert.ok(findClass(host(h),"benchmark-point").length<170);const curve=findClass(host(h),"benchmark-line")[0];assert.equal(curve.getAttribute("points").split(" ").length,1000);assert.match(textOf(host(h)),/0.999/);
});

const grid=h=>findClass(host(h),"dashboard-grid")[0],cards=h=>findClass(grid(h),"dashboard-panel");
function edit(h,index=0){findClass(cards(h)[index],"dashboard-edit")[0].listeners.click();return findClass(host(h),"dashboard-dialog")[0];}
const settings=d=>findClass(d,"dashboard-chart-settings")[0],titleInput=d=>descendants(settings(d)).find(n=>n.tagName==="input"&&n.type!=="search"&&n.type!=="checkbox");
const chooseType=(d,index)=>findClass(d,"dashboard-type-options")[0].children[index].listeners.click();
const pointer=(x=0,y=0)=>({pointerId:7,button:0,clientX:x,clientY:y,preventDefault(){}});
const stored=storage=>JSON.parse(storage.get("rsimonitor-dashboard:v1:a"));

test("per-chart editor previews only its draft, saves one chart and preserves another",()=>{
  const {h,storage}=open({...fixture,dashboard:{schema_version:1,panels:[panel("First"),panel("Second")]}}),d=edit(h),input=titleInput(d);input.value="My loss";input.listeners.input();chooseType(d,1);
  assert.match(textOf(findClass(d,"dashboard-chart-preview")[0]),/My loss/);assert.doesNotMatch(textOf(grid(h)),/My loss/);assert.equal(h.run('state.activeScope.dashboardEditing'),true);
  const previous=host(h).firstElementChild;h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,updated_at:"2026-11-03"};renderProject()');assert.equal(host(h).firstElementChild,previous);
  button(d,"保存").listeners.click();assert.equal(d.open,false);assert.match(textOf(cards(h)[0]),/My loss/);assert.match(textOf(cards(h)[1]),/Second/);assert.equal(stored(storage).panels[0].type,"scalar");
  h.run('selectRun("b")');assert.doesNotMatch(textOf(grid(h)),/My loss/);h.run('selectRun("a")');assert.match(textOf(grid(h)),/My loss/);
  const {h:reloaded}=open(fixture,storage);assert.match(textOf(grid(reloaded)),/My loss/);
  button(host(h),"恢复默认").listeners.click();assert.equal(storage.has("rsimonitor-dashboard:v1:a"),false);assert.match(textOf(grid(h)),/First/);
});
test("add and delete chart operate individually and keep a deliberately empty dashboard",()=>{
  const {h,storage}=open(fixture);button(host(h),"添加图表").listeners.click();let d=findClass(host(h),"dashboard-dialog")[0];assert.equal(findClass(d,"dashboard-delete")[0].hidden,true);button(d,"保存").listeners.click();assert.equal(cards(h).length,2);
  d=edit(h,1);button(d,"删除图表").listeners.click();assert.equal(cards(h).length,1);d=edit(h);button(d,"删除图表").listeners.click();assert.equal(cards(h).length,0);assert.equal(stored(storage).panels.length,0);
  const {h:reloaded}=open(fixture,storage);assert.equal(cards(reloaded).length,0);assert.ok(button(host(reloaded),"添加图表"));
});
test("metric search preserves selection, enforces twelve metrics and updates a live preview",()=>{
  const keys=Array.from({length:13},(_,i)=>`m${i}`),sample={metric_definitions:Object.fromEntries(keys.map(key=>[key,{label:`Metric ${key}`,unit:"number",direction:"neutral"}])),metric_points:[],dashboard:{schema_version:1,panels:[panel("Many","line",{metrics:keys.slice(0,12)})]}},{h}=open(sample),d=edit(h),checks=()=>findClass(d,"dashboard-metric-list")[0].children.flatMap(n=>descendants(n)).filter(n=>n.type==="checkbox");
  assert.equal(checks().filter(n=>n.disabled).length,1);checks()[0].checked=false;checks()[0].listeners.change();assert.equal(checks().filter(n=>n.disabled).length,0);
  const search=descendants(d).find(n=>n.type==="search");search.value="m12";search.listeners.input();assert.equal(checks().length,1);checks()[0].checked=true;checks()[0].listeners.change();assert.equal(h.run('state.activeScope.dashboardDraft.metrics.length'),12);
  search.value="not-present";search.listeners.input();assert.match(textOf(findClass(d,"dashboard-metric-list")[0]),/没有匹配/);
  chooseType(d,1);const fields=findClass(d,"dashboard-field");assert.equal(fields[1].hidden,true);assert.equal(findClass(d,"dashboard-type-options")[0].children[1].getAttribute("aria-pressed"),"true");
});
test("individual configuration is a labelled modal, cancel and Escape discard staged edits",()=>{
  const {h,storage}=open(fixture),d=edit(h);assert.equal(d.tagName,"dialog");assert.equal(d.open,true);assert.equal(d.getAttribute("aria-labelledby"),"dashboard-config-title");const input=titleInput(d);input.value="Unsaved";input.listeners.input();button(d,"取消").listeners.click();
  assert.equal(d.open,false);assert.equal(storage.size,0);assert.equal(h.run('state.activeScope.dashboardDraft'),null);assert.doesNotMatch(textOf(grid(h)),/Unsaved/);
  const next=edit(h);assert.equal(titleInput(next).value,"Loss");let prevented=false;next.listeners.cancel({preventDefault(){prevented=true;}});assert.equal(prevented,true);assert.equal(next.open,false);
});
test("backdrop closes only outside the dialog; hidden tabs preserve drafts",()=>{
  const {h}=open(fixture),d=edit(h);d.getBoundingClientRect=()=>({left:20,right:900,top:40,bottom:700});d.listeners.click({target:d,clientX:30,clientY:50});assert.equal(d.open,true);
  h.document.hidden=true;h.documentListeners.visibilitychange();assert.equal(h.run('state.activeScope.dashboardEditing'),true);d.listeners.click({target:d,clientX:0,clientY:0});assert.equal(d.open,false);
});
test("cancel consumes queued metrics without saving edits or moving page scroll",()=>{
  const {h,storage}=open(fixture);h.ctx.window.scrollY=720;const d=edit(h);titleInput(d).value="Unsaved";titleInput(d).listeners.input();
  h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,metric_points:[...state.snapshot.metric_points,{metric:"loss",value:.2,step:3,task_id:"first"}]};renderProject()');assert.equal(h.run('state.activeScope.renderPending'),true);
  button(d,"取消").listeners.click();assert.equal(storage.size,0);assert.match(textOf(grid(h)),/0.2/);assert.doesNotMatch(textOf(grid(h)),/Unsaved/);assert.equal(h.ctx.window.scrollY,720);
});
test("pointer drag changes display order and persists on release, isolated from incoming refresh",()=>{
  const sample={...fixture,dashboard:{schema_version:1,panels:[panel("First"),panel("Second"),panel("Third")]}},{h,storage}=open(sample),nodes=cards(h),handle=findClass(nodes[0],"dashboard-drag")[0];
  nodes.forEach((node,i)=>node.getBoundingClientRect=()=>({left:i*400,right:i*400+380,top:0,bottom:320,width:380}));handle.listeners.pointerdown(pointer(20,20));handle.listeners.pointermove(pointer(850,30));assert.equal(nodes[0].style.order,"2");assert.equal(storage.size,0);assert.equal(h.run('state.activeScope.dashboardInteracting'),true);
  const previous=host(h).firstElementChild;h.run('state.activeScope.cache.get("a").snapshot={...state.snapshot,updated_at:"2026-11-03"};renderProject()');assert.equal(host(h).firstElementChild,previous);handle.listeners.pointerup(pointer(850,30));assert.deepEqual(stored(storage).panels.map(p=>p.id),["Second","Third","First"]);assert.equal(cards(h)[2].dataset.panelId,"First");assert.equal(h.run('state.activeScope.dashboardInteracting'),false);
  const {h:reloaded}=open(sample,storage);assert.equal(cards(reloaded)[2].dataset.panelId,"First");
});
test("resizing snaps width to twelve-column grid and height to bounds; cancellation saves nothing",()=>{
  const {h,storage}=open(fixture);grid(h).getBoundingClientRect=()=>({width:1184});let handle=findClass(cards(h)[0],"dashboard-resize")[0];handle.listeners.pointerdown(pointer());handle.listeners.pointermove(pointer(300,120));assert.equal(cards(h)[0].style.gridColumn,"span 9");assert.equal(cards(h)[0].style.height,"440px");assert.equal(storage.size,0);handle.listeners.pointerup(pointer(300,120));assert.deepEqual(stored(storage).panels[0].layout,{w:9,h:440});
  grid(h).getBoundingClientRect=()=>({width:1184});handle=findClass(cards(h)[0],"dashboard-resize")[0];handle.listeners.pointerdown(pointer());handle.listeners.pointermove(pointer(9000,9000));assert.equal(cards(h)[0].style.height,"720px");handle.listeners.pointercancel(pointer(9000,9000));assert.deepEqual(stored(storage).panels[0].layout,{w:9,h:440});assert.equal(cards(h)[0].style.height,"440px");
});
test("small screens preserve desktop widths while changing height, and keyboard controls save order and size",()=>{
  const {h,storage}=open({...fixture,dashboard:{schema_version:1,panels:[panel("First"),panel("Second")]}});grid(h).getBoundingClientRect=()=>({width:390});let handle=findClass(cards(h)[0],"dashboard-resize")[0];handle.listeners.pointerdown(pointer());handle.listeners.pointermove(pointer(1000,48));handle.listeners.pointerup(pointer(1000,48));assert.deepEqual(stored(storage).panels[0].layout,{w:6,h:368});
  findClass(cards(h)[0],"dashboard-drag")[0].listeners.keydown({key:"ArrowRight",preventDefault(){}});assert.deepEqual(stored(storage).panels.map(p=>p.id),["Second","First"]);
  findClass(cards(h)[1],"dashboard-resize")[0].listeners.keydown({key:"ArrowRight",preventDefault(){}});assert.equal(stored(storage).panels[1].layout.w,7);
});
test("legacy layout defaults migrate without mutation and navigation closes any active chart dialog",()=>{
  const {h}=open(fixture),before=h.run('JSON.stringify(state.snapshot)');assert.equal(cards(h)[0].style.gridColumn,"span 6");assert.equal(cards(h)[0].style.height,"320px");const d=edit(h);h.run('selectRun("b")');assert.equal(d.open,false);assert.equal(h.run('JSON.stringify(state.activeScope.cache.get("b").snapshot)'),before.replace('"a"','"b"'));
});

test("lost pointer capture and tab hiding cancel unfinished layouts without saving",()=>{
  const {h,storage}=open(fixture);grid(h).getBoundingClientRect=()=>({width:1184});let resize=findClass(cards(h)[0],"dashboard-resize")[0];resize.listeners.pointerdown(pointer());resize.listeners.pointermove(pointer(300,120));resize.listeners.lostpointercapture(pointer());assert.equal(storage.size,0);assert.equal(cards(h)[0].style.height,"320px");
  grid(h).getBoundingClientRect=()=>({width:1184});resize=findClass(cards(h)[0],"dashboard-resize")[0];resize.listeners.pointerdown(pointer());resize.listeners.pointermove(pointer(300,120));h.document.hidden=true;h.documentListeners.visibilitychange();assert.equal(storage.size,0);assert.equal(h.run('state.activeScope.dashboardInteracting'),false);assert.equal(cards(h)[0].style.height,"320px");
});
test("invalid saved sizes revert to defaults and a full board cannot add beyond its limit",()=>{
  const sample={...fixture,dashboard:{schema_version:1,panels:Array.from({length:24},(_,i)=>panel(`p${i}`,"line",{layout:{w:9,h:90000}}))}},{h}=open(sample);assert.equal(cards(h)[0].style.gridColumn,"span 6");assert.equal(cards(h)[0].style.height,"320px");assert.equal(button(host(h),"添加图表").disabled,true);button(host(h),"添加图表").listeners.click();assert.equal(findClass(host(h),"dashboard-dialog")[0].open,false);
});

const tooltipGeometry=(card)=>{
  const plot=findClass(card,"benchmark-plot")[0],tip=findClass(card,"benchmark-tooltip")[0];plot.getBoundingClientRect=()=>({left:100,top:100,width:520,height:180});tip.getBoundingClientRect=()=>({width:210,height:70});
  for(const dot of findClass(card,"benchmark-point"))dot.getBoundingClientRect=()=>({left:100+Number(dot.getAttribute("cx")),top:100+Number(dot.getAttribute("cy")),width:8,height:8});return tip;
};
test("single series has no duplicate footer and hovering reports the precise recorded percentage",()=>{
  const sample={metric_definitions:{accuracy:{label:"MedQA · 准确率",unit:"percent",direction:"higher"}},metric_points:[{metric:"accuracy",value:.734375,step:15}],dashboard:{schema_version:1,panels:[panel("MedQA","line",{metrics:["accuracy"],group_by:"none"})]}},{h}=open(sample),card=cards(h)[0],before=h.run('JSON.stringify(state.snapshot)');
  assert.equal(findClass(card,"dashboard-legend").length,0);assert.match(textOf(findClass(card,"benchmark-latest")[0]),/73.44%/);
  const tip=tooltipGeometry(card),dot=findClass(card,"benchmark-point")[0];assert.equal(tip.hidden,true);dot.listeners.pointerenter();assert.equal(tip.hidden,false);assert.match(textOf(tip),/73.4375%/);assert.match(textOf(tip),/训练步数 · 15/);assert.equal(dot.getAttribute("aria-describedby"),tip.id);assert.equal(dot.getAttribute("tabindex"),"0");
  assert.ok(parseFloat(tip.style.left)>=4);assert.ok(parseFloat(tip.style.left)<=306);dot.listeners.pointerleave();assert.equal(tip.hidden,true);assert.equal(h.run('JSON.stringify(state.snapshot)'),before);
});
test("multiple curves retain their legend; hover and keyboard show the selected curve without inventing units",()=>{
  const {h}=open(fixture),card=cards(h)[0],tip=tooltipGeometry(card),dots=findClass(card,"benchmark-point");assert.equal(findClass(card,"dashboard-legend").length,1);assert.equal(findClass(card,"benchmark-latest").length,0);
  dots.at(-1).listeners.focus();assert.equal(tip.hidden,false);assert.match(textOf(tip),/任务 2/);assert.match(textOf(tip),/0.6/);assert.doesNotMatch(textOf(tip),/%/);dots.at(-1).listeners.keydown({key:"Escape"});assert.equal(tip.hidden,true);
  dots[0].listeners.pointerdown();assert.equal(tip.hidden,false);dots[0].listeners.blur();assert.equal(tip.hidden,true);
});
test("point tooltip follows the chosen axis and hides on plot scrolling",()=>{
  const {h}=open({...fixture,metric_points:[{metric:"loss",value:.123456789,elapsed_seconds:1.25}],dashboard:{schema_version:1,panels:[panel("Loss","line",{x:"elapsed_seconds",group_by:"none"})]}}),card=cards(h)[0],tip=tooltipGeometry(card),dot=findClass(card,"benchmark-point")[0];dot.listeners.pointerenter();assert.match(textOf(tip),/运行时间（秒） · 1.25/);assert.match(textOf(tip),/0.123456789/);findClass(card,"dashboard-chart-body")[0].listeners.scroll();assert.equal(tip.hidden,true);
});
