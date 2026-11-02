"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs");
const shared=fs.readFileSync(`${__dirname}/project.test.js`,"utf8").split('\ntest("')[0];
const {harness,setup,seed,run,project,findClass,textOf,selectDecision,response}=new Function("require","__dirname",`${shared}\nreturn {harness,setup,seed,run,project,findClass,textOf,selectDecision,response};`)(require,__dirname);
const host=h=>h.get("project-content");
const visibleNodes=node=>[node,...(node.tagName==="details"&&!node.open?node.children.slice(0,1):node.children).flatMap(visibleNodes)];
const visibleText=node=>visibleNodes(node).map(n=>n.textContent||"").join(" ");
const history=h=>findClass(host(h),"record-details").find(n=>n.dataset.retainKey===JSON.stringify(["a","unlinked-jobs"]));
const mainNodes=h=>findClass(findClass(host(h),"flow-map")[0],"decision-node");
const chooseAttempt=(h,title)=>{const node=mainNodes(h).find(n=>n.getAttribute("aria-label").includes(title));assert.ok(node);node.listeners.click();};
function fixture(){return run("a",{status:"running",decisions:[
  {decision_id:"past",kind:"directive",mode:"evaluate",reason:"Historical rationale",research_attempt_id:"past",research_attempt_title:"Past hypothesis",job_id:"finished",candidate_id:"result",status:"completed"},
  {decision_id:"training",kind:"directive",mode:"train",reason:"Current rationale",research_attempt_id:"current",research_attempt_title:"Current hypothesis",job_id:"live",depends_on_decision_ids:["past"]},
],jobs:[
  {job_id:"finished",kind:"evaluate",status:"succeeded",metrics:{accuracy:.81}},
  {job_id:"live",kind:"train",status:"running",step:2,total_steps:10,metrics:{loss:.3}},
  {job_id:"parallel",kind:"evaluate",status:"evaluating",step:4,total_steps:10},
  {job_id:"queued",kind:"train",status:"queued"},
  {job_id:"orphan",kind:"evaluate",status:"failed",error:"Reported orphan failure",metrics:{accuracy:.17}},
  {job_id:"uncertain",kind:"execute",status:"unknown"},
  {job_id:"cancelling",kind:"execute",status:"cancel_requested"},
],candidates:[{candidate_id:"result",status:"accepted",metrics:{accuracy:.81}}]});}
function open(snapshot=fixture(),fetcher){
  const h=harness(fetcher);setup(h,[project("p",[run("a")])]);seed(h,"a",snapshot);h.run('selectRun("a")');return h;
}

test("entry leaves both graph views unselected and shows all active linked and parallel jobs compactly",()=>{
  const h=open(),before=JSON.stringify(h.run("state.snapshot"));
  assert.equal(mainNodes(h).length,2);
  assert.equal(findClass(host(h),"attempt-map").length,0);assert.equal(findClass(host(h),"flow-step").length,0);
  assert.ok(mainNodes(h).every(n=>n.getAttribute("aria-pressed")==="false"));
  assert.equal(findClass(host(h),"selection-hint").length,1);
  assert.equal(findClass(host(h),"active-job").length,3);
  const overview=findClass(host(h),"active-jobs")[0];
  assert.match(textOf(overview),/当前任务 · 3/);assert.match(textOf(overview),/运行中/);assert.match(textOf(overview),/评测中/);assert.match(textOf(overview),/排队中/);
  assert.equal(findClass(overview,"metric-strip").length,0);
  assert.doesNotMatch(visibleText(host(h)),/Historical rationale|Current rationale|0\.81|0\.17|Reported orphan failure/);
  assert.equal(history(h).open,false);assert.equal(findClass(history(h),"job-record").length,3);
  assert.equal(findClass(host(h),"run-detail")[0].children.some(n=>n.className==="job-record"),false);
  assert.equal(h.renders[0].options.model.edges.filter(e=>e.kind==="dependency").length,1);
  assert.equal(findClass(host(h),"trajectory-mode").length,0);
  assert.equal(JSON.stringify(h.run("state.snapshot")),before);
});

test("explicit historical selection exposes exact completed job and candidate results",()=>{
  const h=open();chooseAttempt(h,"Past hypothesis");
  assert.equal(findClass(host(h),"attempt-map").length,1);assert.equal(findClass(host(h),"flow-step").length,1);
  const step=findClass(host(h),"flow-step")[0];assert.match(textOf(step),/Historical rationale/);assert.match(textOf(step),/81%/);assert.match(textOf(step),/已采纳/);
  assert.equal(findClass(host(h),"active-job").length,3);assert.equal(history(h).open,false);
});

test("terminal runs omit the active overview, preserve run failure and keep uncertain orphan reports inspectable",()=>{
  for(const status of ["completed","failed","cancelled","stopped"]){
    const snapshot=fixture();snapshot.status=status;snapshot.error="Reported run error";
    const h=open(snapshot);assert.equal(findClass(host(h),"active-jobs").length,0);
    assert.equal(findClass(host(h),"stale-job-warning").length,1);
    assert.match(visibleText(host(h)),/Reported run error/);assert.doesNotMatch(visibleText(host(h)),/当前任务 · 0|未报告活动任务/);
    assert.equal(findClass(history(h),"job-record").length,5);assert.equal(history(h).open,false);
    history(h).open=true;assert.match(visibleText(history(h)),/状态未知/);assert.match(visibleText(history(h)),/已请求取消/);assert.match(visibleText(history(h)),/Reported orphan failure/);
    h.get("lang-en").listeners.click();assert.match(visibleText(host(h)),/Unknown status/);assert.match(visibleText(host(h)),/Cancellation requested/);
    if(status==="failed")assert.match(visibleText(host(h)),/Failed/);
  }
});

test("a terminal run with only terminal tasks has no empty overview or stale warning",()=>{
  const snapshot=fixture();snapshot.status="failed";snapshot.jobs=snapshot.jobs.filter(j=>["succeeded","failed"].includes(j.status));
  const h=open(snapshot);assert.equal(findClass(host(h),"active-jobs").length,0);assert.equal(findClass(host(h),"stale-job-warning").length,0);
  assert.equal(findClass(host(h),"flow-step").length,0);assert.equal(history(h).open,false);
  assert.match(visibleText(host(h)),/失败/);assert.doesNotMatch(visibleText(host(h)),/Reported orphan failure|0\.81|0\.17/);
});

test("polling and languages preserve explicit selection and unlinked history expansion while active tasks complete",async()=>{
  const snapshot=fixture(),updated=structuredClone(snapshot);updated.updated_at="2026-11-02T01:00:00Z";
  updated.jobs.find(j=>j.job_id==="parallel").status="succeeded";
  const h=open(snapshot,async()=>response(updated));chooseAttempt(h,"Past hypothesis");history(h).open=true;
  const selected=h.run('state.activeScope.flowSelected.get("a")');
  await h.run('loadRun("a",state.activeScope,true)');
  assert.equal(findClass(host(h),"active-job").length,2);assert.equal(history(h).open,true);assert.equal(findClass(history(h),"job-record").length,4);
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),selected);assert.match(textOf(findClass(host(h),"flow-step")[0]),/Historical rationale/);
  h.get("lang-en").listeners.click();assert.equal(history(h).open,true);assert.match(visibleText(host(h)),/Association diagnostics · 4/);
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),selected);
  history(h).open=false;h.get("lang-zh").listeners.click();assert.equal(history(h).open,false);
});

test("polling and language changes never create a selection and removed main selections clear",async()=>{
  let updated=fixture();const h=open(fixture(),async()=>response(updated));
  updated.updated_at="2026-11-02T01:00:00Z";await h.run('loadRun("a",state.activeScope,true)');h.get("lang-en").listeners.click();
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),null);assert.equal(findClass(host(h),"flow-step").length,0);
  chooseAttempt(h,"Past hypothesis");
  updated={...updated,decisions:updated.decisions.filter(d=>d.decision_id!=="past")};await h.run('loadRun("a",state.activeScope,true)');
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),null);assert.equal(findClass(host(h),"attempt-map").length,0);
  assert.equal(findClass(host(h),"flow-step").length,0);
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),null);assert.equal(mainNodes(h).length,1);
});

test("removing the selected child keeps the complete surviving attempt graph without picking another decision",async()=>{
  const snapshot=fixture();snapshot.decisions.push({...snapshot.decisions[0],decision_id:"second",reason:"Second historical rationale"});
  const updated={...snapshot,decisions:snapshot.decisions.filter(d=>d.decision_id!=="past")};
  const h=open(snapshot,async()=>response(updated));chooseAttempt(h,"Past hypothesis");
  await h.run('loadRun("a",state.activeScope,true)');
  assert.equal(findClass(host(h),"attempt-map").length,1);assert.equal(findClass(host(h),"flow-step").length,0);
  const children=findClass(findClass(host(h),"attempt-map")[0],"decision-node");assert.equal(children.length,1);assert.equal(children[0].getAttribute("aria-pressed"),"false");
  children[0].listeners.click();assert.match(textOf(findClass(host(h),"flow-step")[0]),/Second historical rationale/);
});

test("legacy runs retain active tasks and experiment logs with strict job association guards",()=>{
  const snapshot=fixture();for(const d of snapshot.decisions){delete d.research_attempt_id;delete d.research_attempt_title;}
  snapshot.decisions[0].job_id="missing";snapshot.decisions[0].directive_id="shared";
  snapshot.jobs[0].directive_id="shared";
  const h=open(snapshot);assert.equal(findClass(host(h),"active-job").length,3);assert.equal(findClass(host(h),"flow-map").length,0);
  assert.match(visibleText(host(h)),/未记录研究尝试/);
  const logs=findClass(host(h),"record-details").find(n=>n.dataset.retainKey===JSON.stringify(["a","experiment-logs"]));assert.equal(logs.open,false);logs.open=true;
  const button=findClass(logs,"experiment-log-record").find(n=>n.dataset.key==="past");assert.ok(button);button.listeners.click();
  assert.equal(findClass(host(h),"flow-map").length,0);assert.match(textOf(findClass(logs,"flow-step")[0]),/Historical rationale/);
  assert.equal(h.run('linkedJob(state.snapshot.decisions[0],state.snapshot)'),undefined);
  assert.equal(findClass(history(h),"job-record").length,4);
});


test("legacy evaluator aliases recover only uniquely corroborated directives and baseline stays separate",()=>{
  const s=fixture();s.decisions[0].job_id="backend-id";s.decisions[0].directive_id="eval-directive";
  s.jobs[0]={job_id:"evaluation:result",kind:"evaluate",directive_id:"eval-directive",status:"succeeded"};
  s.jobs.push({job_id:"evaluation:initial",kind:"evaluate",status:"succeeded"});
  const h=open(s);
  assert.equal(h.run('linkedJob(state.snapshot.decisions[0],state.snapshot).job_id'),"evaluation:result");
  const baseline=findClass(host(h),"record-details").find(n=>n.dataset.retainKey===JSON.stringify(["a","baseline"]));
  assert.ok(baseline);assert.equal(findClass(baseline,"job-record").length,1);
  assert.equal(findClass(history(h),"job-record").length,3);
  h.run('state.snapshot.jobs.push({job_id:"another",directive_id:"eval-directive"})');
  assert.equal(h.run('linkedJob(state.snapshot.decisions[0],state.snapshot)'),undefined);
});


test("node selection restores page scroll after transient content collapse and only keyboard restores focus",()=>{
  const h=open(),section=findClass(host(h),"execution-sequence")[0],entries=findClass(host(h),"flow-steps")[0],nav=findClass(host(h),"flow-map")[0];
  section.getBoundingClientRect=()=>({height:1200});
  const replace=entries.replaceChildren.bind(entries);let guarded=0,focus=0;
  entries.replaceChildren=(...nodes)=>{if(section.style.minHeight==="1200px")guarded++;h.ctx.window.scrollY=0;replace(...nodes);};
  nav.querySelector=()=>({focus(options){assert.equal(options.preventScroll,true);focus++;}});
  h.ctx.window.scrollY=650;h.ctx.window.scrollX=7;
  mainNodes(h)[0].listeners.click();
  assert.equal(h.ctx.window.scrollY,650);assert.equal(h.ctx.window.scrollX,7);assert.equal(focus,0);assert.ok(guarded);assert.equal(section.style.minHeight,"");
  mainNodes(h)[0].listeners.keydown({key:"Enter",preventDefault(){}});
  assert.equal(h.ctx.window.scrollY,650);assert.equal(focus,1);
  const child=findClass(findClass(host(h),"attempt-map")[0],"decision-node")[0];
  child.listeners.click();assert.equal(h.ctx.window.scrollY,650);
});
