"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs");
const {layout,attempts,subpath}=require("../../src/rsimonitor/static/decisions.js");
// Share the real renderer, DOM, polling and RAF harness, without registering tests.
const shared=fs.readFileSync(`${__dirname}/project.test.js`,"utf8").split('\ntest("')[0];
const {harness,setup,seed,run,project,findClass,textOf,descendants,graphState,dragGraphNode,response}=new Function("require","__dirname",`${shared}\nreturn {harness,setup,seed,run,project,findClass,textOf,descendants,graphState,dragGraphNode,response};`)(require,__dirname);
const d=(id,attempt,extra={})=>({decision_id:id,kind:"operation",operation:"configure_prompt",reason:`reason-${id}`,
  timestamp:`2026-11-02T00:00:${String(extra.second||0).padStart(2,"0")}Z`,
  ...(attempt?{research_attempt_id:attempt,research_attempt_title:`Hypothesis ${attempt}`} : {}),...extra});
function fixture(){return run("a",{decisions:[
  d("prompt","prompt",{second:1,status:"completed",changes:[{field:"instructions",after:"Try concise prompts"}],feedback:{summary:"Prompt proposed"}}),
  d("data","data",{second:2,status:"running"}),
  d("wait","join",{second:3,kind:"wait",operation:undefined,source_decision_ids:["prompt","data"]}),
  d("adopt","join",{second:4,operation:"system_adopt",depends_on_decision_ids:["prompt","data"],adopted_decision_ids:["prompt","data"],source_decision_ids:["prompt","data"],feedback:{summary:"Both results adopted"}}),
  d("consume","join",{second:5,kind:"directive",mode:"evaluate",operation:undefined,depends_on_decision_ids:["adopt"],metrics:{score:.8},feedback:{summary:"Measured result"}}),
  d("finish","join",{second:6,kind:"finish",operation:undefined,depends_on_decision_ids:["consume"]}),
]});}
const nodes=model=>model.nodes.filter(n=>n.row);
const links=model=>model.edges.filter(e=>e.from!=="root");
const attempt=(model,id)=>model.nodes.find(n=>n.attemptID===id);
function assertFinite(model){for(const n of model.nodes)assert.ok(Number.isFinite(n.x)&&Number.isFinite(n.y));}
function open(snapshot=fixture(),fetcher){
  const h=harness(fetcher);setup(h,[project("p",[run("a")])]);seed(h,"a",snapshot);h.run('selectRun("a")');return h;
}
const host=h=>h.get("project-content");
const main=h=>findClass(host(h),"flow-map")[0];
const child=h=>findClass(host(h),"attempt-map")[0];
const mainNodes=h=>findClass(main(h),"decision-node");
const childNodes=h=>findClass(child(h),"decision-node");
const choose=(h,key)=>{const node=mainNodes(h).find(n=>n.dataset.key===key);assert.ok(node);node.listeners.click();};
const chooseChild=(h,key)=>{const node=childNodes(h).find(n=>n.dataset.key===key);assert.ok(node);node.listeners.click();};
const log=(h,key)=>{const button=findClass(host(h),"experiment-log-record").find(n=>n.dataset.key===key);assert.ok(button);button.listeners.click();};
const logs=h=>findClass(host(h),"record-details").find(n=>n.dataset.retainKey===JSON.stringify(["a","experiment-logs"]));

test("only explicit identities form attempts, retain every child and keep same titles and candidates independent",()=>{
  const rows=[d("a1","A",{candidate_id:"same"}),d("b1","B",{candidate_id:"same",research_attempt_title:"Hypothesis A"}),
    d("old",null,{candidate_id:"same"}),d("a2","A",{second:4,kind:"finish"}),d("a3","A",{second:5,kind:"finish"})];
  const before=JSON.stringify(rows),model=attempts(rows);
  assert.equal(model.attemptCount,2);assert.equal(model.ungroupedCount,1);assert.equal(nodes(model).length,2);
  assert.deepEqual(attempt(model,"A").rows.map(r=>r.decision_id),["a1","a2","a3"]);
  assert.deepEqual(model.ungrouped.map(n=>n.key),["old"]);assert.equal(model.full.nodes.length,6);
  assert.equal(model.edges.some(e=>e.kind==="sequence"),false);assert.equal(JSON.stringify(rows),before);assertFinite(model);
});

test("interleaved parallel attempts and multi-result adoption contract with strongest straight edges and complete provenance",()=>{
  const model=attempts(fixture().decisions),a=attempt(model,"prompt"),b=attempt(model,"data"),join=attempt(model,"join");
  assert.equal(nodes(model).length,3);assert.equal(a.x,b.x);assert.notEqual(a.y,b.y);assert.ok(join.x>a.x);
  assert.deepEqual(links(model).map(e=>[e.from,e.to,e.kind]),[[a.key,join.key,"adoption"],[b.key,join.key,"adoption"]]);
  for(const edge of links(model)){
    assert.deepEqual(new Set(edge.types),new Set(["source","dependency","adoption"]));
    assert.equal(edge.decisions.length,4);assert.equal(edge.decisions.filter(e=>e.kind==="source").length,2);
  }
  const child=subpath(model,join.key);
  assert.deepEqual(nodes(child).map(n=>n.key),["wait","adopt","consume","finish"]);
  assert.deepEqual(links(child).map(e=>[e.from,e.to,e.kind]),[["adopt","consume","dependency"],["consume","finish","dependency"]]);
  assert.equal(child.edges.some(e=>e.kind==="sequence"),false);assertFinite(child);
});

test("a valid A1 to B1 to A2 continuation stays Attempt-only and retains cyclic edges and original evidence",()=>{
  const rows=[d("a1","A"),d("b1","B",{second:1,depends_on_decision_ids:["a1"]}),d("a2","A",{second:2,adopted_decision_ids:["b1"]})];
  const model=attempts(rows),a=attempt(model,"A"),b=attempt(model,"B");
  assert.equal(model.fallback,undefined);assert.equal(model.attemptCount,2);
  assert.deepEqual(nodes(model).map(n=>n.type),["attempt","attempt"]);
  assert.deepEqual(links(model).map(e=>[e.from,e.to,e.kind]),[[b.key,a.key,"adoption"],[a.key,b.key,"dependency"]]);
  assert.ok(links(model).every(e=>e.cyclic));
  assert.ok(model.issues.some(i=>i.type==="contraction-cycle"));assertFinite(model);
  assert.deepEqual(links(model.full).map(e=>[e.from,e.to,e.kind]),[["a1","b1","dependency"],["b1","a2","adoption"]]);
  const h=open(run("a",{decisions:rows}));assert.equal(mainNodes(h).length,2);assert.equal(child(h),undefined);
  assert.match(textOf(host(h)),/保留研究尝试节点与已记录关系/);assert.doesNotMatch(textOf(host(h)),/已展开全部决策|全部记录/);
  choose(h,a.key);assert.equal(childNodes(h).length,2);chooseChild(h,"a2");assert.match(textOf(host(h)),/reason-a2/);
  log(h,"a2");assert.match(findClass(host(h),"log-evidence")[0].textContent,/"adopted_decision_ids": \[\s*"b1"/);
});

test("raw self references, raw cycles and ambiguous IDs keep existing guards and produce inspectable diagnostics",()=>{
  const rows=[d("outside","O"),d("a","A",{depends_on_decision_ids:["outside","b"]}),d("b","B",{adopted_decision_ids:["a"]}),
    d("descendant","D",{depends_on_decision_ids:["b"]}),d("self","S",{depends_on_decision_ids:["self"]}),d("dup","X"),d("dup","Y"),d("unknown","Z",{depends_on_decision_ids:["missing","dup"]})];
  const full=layout(rows),model=attempts(rows);
  assert.deepEqual(links(full).map(e=>[e.from,e.to,e.kind]),[["outside","a","dependency"],["b","descendant","dependency"]]);
  assert.ok(full.issues.some(i=>i.type==="raw-cycle"));assert.ok(full.issues.some(i=>i.type==="self-reference"));assert.ok(full.issues.some(i=>i.type==="unresolved"));
  assert.equal(model.fallback,undefined);assert.equal(model.full.nodes.length,9);assertFinite(model);
  const h=open(run("a",{decisions:rows}));assert.match(textOf(host(h)),/无效的循环决策关系已忽略/);assert.match(textOf(host(h)),/自引用已忽略/);
});

test("malformed attempt metadata never creates guessed groups or hides records",()=>{
  const rows=[d("legacy"),d("title-only",null,{research_attempt_title:"A title"}),d("numeric",null,{research_attempt_id:7}),
    d("blank",null,{research_attempt_id:"  "}),d("missing-title",null,{research_attempt_id:"A"}),
    d("named","A"),d("conflict","A",{research_attempt_title:"Different literal title"})];
  const model=attempts(rows);assert.equal(model.attemptCount,1);assert.equal(model.ungroupedCount,4);
  assert.deepEqual(attempt(model,"A").rows.map(r=>r.decision_id),["missing-title","named","conflict"]);
  assert.ok(model.issues.some(i=>i.type==="invalid-attempt"));assert.ok(model.issues.some(i=>i.type==="missing-title"));assert.ok(model.issues.some(i=>i.type==="conflicting-title"));
  assert.equal(model.full.nodes.length,8);assert.equal(model.edges.some(e=>e.kind==="sequence"),false);
});

test("all unassigned records stay in logs, including dependency, adoption and source endpoints; no bridging is inferred",()=>{
  const rows=[d("a","A"),...Array.from({length:30},(_,i)=>d(`wait-${i}`,null,{kind:"wait",operation:undefined})),
    d("dep",null,{depends_on_decision_ids:["a"]}),d("adopt",null,{adopted_decision_ids:["a"]}),d("ref",null,{source_decision_ids:["a"]}),
    d("b","B",{depends_on_decision_ids:["dep"],adopted_decision_ids:["adopt"],source_decision_ids:["ref"]})];
  const model=attempts(rows);assert.equal(nodes(model).length,2);assert.equal(model.ungrouped.length,33);
  assert.deepEqual(links(model),[]);assert.equal(links(model.full).length,6);
  const h=open(run("a",{decisions:rows}));assert.equal(mainNodes(h).length,2);assert.equal(findClass(host(h),"experiment-log-record").length,35);
  assert.equal(logs(h).open,false);
  for(const key of ["wait-20","dep","adopt","ref"]){
    log(h,key);assert.equal(mainNodes(h).length,2);assert.equal(h.run('state.activeScope.flowSelected.get("a")'),null);
    assert.equal(findClass(host(h),"log-membership")[0].textContent,"未记录研究尝试");
    assert.match(textOf(findClass(host(h),"experiment-logs")[0]),new RegExp(`reason-${key}`));
  }
  assert.equal(findClass(host(h),"trajectory-mode").length,0);
});

test("historical membership stays explicitly missing and every original decision is inspectable only in logs",()=>{
  const snapshot=run("a",{candidates:[{candidate_id:"same",status:"applied"}],decisions:Array.from({length:40},(_,i)=>d(`old-${i}`,null,{candidate_id:"same",kind:i>2?"finish":"directive",mode:i?"evaluate":"train"}))});
  const model=attempts(snapshot.decisions);assert.equal(nodes(model).length,0);assert.equal(model.ungrouped.length,40);
  const h=open(snapshot);assert.match(textOf(host(h)),/未记录研究尝试/);assert.equal(findClass(host(h),"decision-map").length,0);
  assert.equal(findClass(host(h),"trajectory-mode").length,0);assert.equal(findClass(host(h),"experiment-log-record").length,40);
  log(h,"old-39");assert.match(textOf(host(h)),/reason-old-39/);assert.equal(findClass(host(h),"decision-map").length,0);
  h.get("lang-en").listeners.click();assert.match(textOf(host(h)),/No research attempts recorded/);assert.match(textOf(host(h)),/Experiment logs · 40/);
  assert.equal(h.run('state.activeScope.logSelected.get("a")'),"old-39");
  assert.doesNotMatch(textOf(host(h)),/All records|View all decisions/);
});

test("grouped child selection exposes all existing rationale, changes, feedback and metrics without raw IDs or inferred attempt status",()=>{
  const h=open();assert.equal(mainNodes(h).length,3);const model=attempts(fixture().decisions),join=attempt(model,"join");
  choose(h,join.key);assert.equal(childNodes(h).length,4);assert.equal(h.run('state.activeScope.attemptGraph.getAttribute("height")'),"280");
  for(const key of ["wait","adopt","consume","finish"]){chooseChild(h,key);assert.match(textOf(findClass(host(h),"flow-step")[0]),new RegExp(`reason-${key}`));}
  chooseChild(h,"consume");assert.match(textOf(host(h)),/Measured result/);assert.match(textOf(host(h)),/0.8/);
  choose(h,attempt(model,"prompt").key);assert.match(textOf(host(h)),/Try concise prompts/);assert.match(textOf(host(h)),/Prompt proposed/);
  const title=mainNodes(h)[0].getAttribute("aria-label");assert.equal(title,"1 · Hypothesis prompt");assert.doesNotMatch(title,/已完成|成功|Completed|Succeeded/);
  assert.doesNotMatch(textOf(host(h)),/研究尝试记录|research_attempt_id|attempt:\"/);
  const snapshot=h.run("state.snapshot");assert.equal(snapshot.decisions[0].research_attempt_id,"prompt");
});

test("titles are literal, untrusted text and never translated or interpreted as outcomes",()=>{
  const title='<img src=x onerror=alert(1)> 已采纳';
  const h=open(run("a",{decisions:[d("d","opaque-uuid",{research_attempt_title:title,status:"failed"})]}));
  assert.equal(mainNodes(h)[0].getAttribute("aria-label"),`1 · ${title}`);assert.equal(descendants(host(h)).some(n=>n.tagName==="img"),false);
  h.get("lang-en").listeners.click();assert.equal(mainNodes(h)[0].getAttribute("aria-label"),`1 · ${title}`);assert.doesNotMatch(textOf(host(h)),/opaque-uuid/);
});

test("a hypothesis attempt contains configure prompt, benchmark, compare and confirmed adoption children",()=>{
  const title="简短提示词能提高基准准确率",rows=[
    d("optimize","hypothesis",{research_attempt_title:title,operation:"configure_prompt",changes:[{field:"instructions",after:"Explain concisely"}],feedback:{summary:"Prompt saved"}}),
    d("benchmark","hypothesis",{research_attempt_title:title,second:1,kind:"directive",operation:undefined,mode:"evaluate",depends_on_decision_ids:["optimize"],metrics:{accuracy:.82}}),
    d("compare","hypothesis",{research_attempt_title:title,second:2,operation:"query_metrics",depends_on_decision_ids:["benchmark"],feedback:{summary:"Compared with the baseline"}}),
    d("adoption","hypothesis",{research_attempt_title:title,second:3,operation:"system_adopt",depends_on_decision_ids:["compare"],adopted_decision_ids:["optimize"],feedback:{summary:"Confirmed prompt adoption"}}),
  ];
  const h=open(run("a",{decisions:rows}));assert.equal(mainNodes(h).length,1);choose(h,mainNodes(h)[0].dataset.key);assert.equal(childNodes(h).length,4);
  assert.equal(mainNodes(h)[0].getAttribute("aria-label"),`1 · ${title}`);
  for(const id of ["optimize","benchmark","compare","adoption"]){chooseChild(h,id);assert.match(textOf(findClass(host(h),"flow-step")[0]),new RegExp(`reason-${id}`));}
  assert.match(textOf(host(h)),/Confirmed prompt adoption/);chooseChild(h,"benchmark");assert.match(textOf(host(h)),/0.82/);
  chooseChild(h,"optimize");assert.match(textOf(host(h)),/Explain concisely/);h.get("lang-en").listeners.click();assert.equal(mainNodes(h)[0].getAttribute("aria-label"),`1 · ${title}`);
});

test("refresh preserves selected attempt, child, both cameras, dragged positions, disclosure and bilingual view state",async()=>{
  const snapshot=fixture();snapshot.decisions.push(d("mechanical-wait",null,{kind:"wait"}));
  const updated=structuredClone(snapshot);updated.decisions.find(d=>d.decision_id==="consume").feedback.summary="New measured result";
  updated.decisions.push(d("later","join",{second:7,depends_on_decision_ids:["finish"]}));
  const h=open(snapshot,async()=>response(updated)),join=attempt(attempts(snapshot.decisions),"join").key;
  choose(h,join);chooseChild(h,"consume");h.run('state.activeScope.graph.zoomBy(1.25);state.activeScope.attemptGraph.zoomBy(1.4)');dragGraphNode(h,"consume",15,35);h.flushFrames();
  const mainView=graphState(h),childView=h.json('state.activeScope.attemptView.camera'),childPositions=h.json('[...state.activeScope.attemptView.positions]');
  log(h,"mechanical-wait");assert.equal(h.run('state.activeScope.attemptSteps.get(state.activeScope.flowSelected.get("a"))'),"consume");
  const disclosure=logs(h);disclosure.open=true;
  assert.deepEqual(graphState(h),mainView);assert.deepEqual(h.json('state.activeScope.attemptView.camera'),childView);
  await h.run('loadRun("a",state.activeScope,true)');
  assert.equal(h.run('state.activeScope.flowSelected.get("a")'),join);assert.equal(h.run(`state.activeScope.attemptSteps.get(${JSON.stringify(join)})`),"consume");
  assert.equal(childNodes(h).length,5);assert.match(textOf(host(h)),/New measured result/);assert.deepEqual(graphState(h),mainView);
  assert.equal(logs(h).open,true);
  assert.deepEqual(h.json('state.activeScope.attemptView.camera'),childView);
  for(const [key,point] of childPositions)assert.deepEqual(h.json(`state.activeScope.attemptView.positions.get(${JSON.stringify(key)})`),point);
  h.get("lang-en").listeners.click();assert.match(textOf(host(h)),/Decisions within this attempt/);assert.match(textOf(host(h)),/Hypothesis join/);
  assert.deepEqual(h.json('state.activeScope.attemptView.camera'),childView);assert.deepEqual(graphState(h),mainView);
});

test("polling during child interaction waits for idle; leaving a run disposes both graphs",async()=>{
  const snapshot=fixture(),updated=structuredClone(snapshot);updated.decisions[0].reason="Changed while dragging";
  const h=open(snapshot,async()=>response(updated));choose(h,mainNodes(h)[0].dataset.key);
  const original=h.run('state.activeScope.attemptGraph'),graph=h.run('state.activeScope.graph'),renderCount=h.renders.length;
  original.listeners.pointerdown({pointerId:9,button:0,clientX:10,clientY:10,target:original});
  await h.run('loadRun("a",state.activeScope,true)');assert.equal(h.renders.length,renderCount);assert.equal(h.run('state.activeScope.renderPending'),true);
  original.listeners.pointerup({pointerId:9,type:"pointerup"});assert.ok(h.renders.length>renderCount);assert.match(textOf(host(h)),/Changed while dragging/);
  assert.ok(h.renders.find(r=>r.svg===original).disposals>0);assert.ok(h.renders.find(r=>r.svg===graph).disposals>0);
  const activeMain=h.run('state.activeScope.graph'),activeChild=h.run('state.activeScope.attemptGraph');
  h.get("back-projects").listeners.click();assert.ok(h.renders.find(r=>r.svg===activeMain).disposals>0);assert.ok(h.renders.find(r=>r.svg===activeChild).disposals>0);assert.equal(h.pendingFrames,0);
});

test("transition from historical empty state to new recorded attempts preserves the run and full decision access",async()=>{
  const old=d("old"),updated=run("a",{decisions:[old,...fixture().decisions]});
  const h=open(run("a",{decisions:[old]}),async()=>response(updated));assert.equal(findClass(host(h),"decision-map").length,0);
  await h.run('loadRun("a",state.activeScope,true)');assert.equal(mainNodes(h).length,3);assert.equal(h.run('state.runId'),"a");
  assert.equal(findClass(host(h),"experiment-log-record").length,7);log(h,"old");assert.equal(mainNodes(h).length,3);assert.match(textOf(host(h)),/reason-old/);
});

test("supplied projection and input remain unchanged during render and child camera controls",()=>{
  const h=open(),projection=attempts(fixture().decisions);h.ctx.projection=projection;
  const before=JSON.stringify(projection);h.run('const standalone=RSIDecisions.render({model:projection,height:280});standalone.zoomBy(1.5);standalone.reset();standalone.dispose();');
  assert.equal(JSON.stringify(projection),before);
});


const interpretedView=(items)=>({schema_version:1,authority:"model_interpretation",items,revisions:[],ungrouped_decision_ids:[]});
test("post-hoc interpretations group without mutating decisions or inventing adoption and sequence edges",()=>{
  const rows=[d("x","legacy"),d("y",null,{second:2}),d("z",null,{second:3,depends_on_decision_ids:["x","y"],adopted_decision_ids:["x","y"]})],before=JSON.stringify(rows);
  const view=interpretedView([{attempt_id:"a",title:"Prompt hypothesis",summary:"Measured a variation",member_decision_ids:["x"]},{attempt_id:"b",title:"Data hypothesis",member_decision_ids:["y"]},{attempt_id:"join",title:"Combine",member_decision_ids:["z"]}]);
  const model=attempts(rows,view);assert.equal(model.attemptCount,3);assert.equal(JSON.stringify(rows),before);
  assert.deepEqual(links(model).map(e=>e.kind),["adoption","adoption"]);
  assert.equal(attempt(model,"legacy"),undefined);assert.equal(attempt(model,"a").interpretation.summary,"Measured a variation");
  const empty=attempts(rows,interpretedView([]));assert.equal(empty.attemptCount,0);assert.equal(empty.ungroupedCount,3);
  const old=attempts(rows);assert.equal(old.attemptCount,1);
});
test("interpretations retain entirely missing members and exclude conflicting memberships",()=>{
  const rows=[d("x",null)],view=interpretedView([{attempt_id:"a",title:"A",member_decision_ids:["missing","x"]},{attempt_id:"b",title:"B",member_decision_ids:["x"]}]);
  const model=attempts(rows,view);assert.equal(model.attemptCount,2);assert.equal(model.ungroupedCount,1);assert.deepEqual(attempt(model,"a").missingDecisionIDs,["missing","x"]);assert.equal(links(model).length,0);assertFinite(model);
});
test("selected interpretation shows summary, model source, revision, and honest partial details",()=>{
  const view=interpretedView([{attempt_id:"archived",title:"Archived idea",summary:"A model summary, not success",member_decision_ids:["older-decision"],source_refs:["artifact:report"],revision:2,provenance:{job_id:"summary-job",model:{model_id:"organizer",revision:"v1"},input_digest:"digest"},simulated:true}]);
  view.revisions=[{...view.items[0],revision:1,summary:"Earlier summary"}];
  const snapshot=run("a",{decisions:[d("new",null)],research_attempts:view}),h=open(snapshot);
  const main=findClass(findClass(h.get("project-content"),"flow-map")[0],"decision-node");assert.equal(main.length,1);main[0].listeners.click();
  const content=textOf(h.get("project-content"));assert.match(content,/A model summary, not success/);assert.doesNotMatch(content,/由模型整理，分组不代表成功或采纳|条未分组记录|个研究尝试/);assert.match(content,/部分详情/);assert.match(content,/organizer/);assert.match(content,/artifact:report/);assert.match(content,/Earlier summary/);assert.equal(findClass(h.get("project-content"),"attempt-map").length,0);
});
