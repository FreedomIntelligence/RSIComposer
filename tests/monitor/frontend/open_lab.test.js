"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs");
const shared=fs.readFileSync(`${__dirname}/project.test.js`,"utf8").split('\ntest("')[0];
const {harness,setup,seed,run,project,findClass,textOf,descendants}=new Function("require","__dirname",`${shared}\nreturn {harness,setup,seed,run,project,findClass,textOf,descendants};`)(require,__dirname);
function open(extra={}){const h=harness();setup(h,[project("p",[run("a")])]);seed(h,"a",run("a",extra));h.run('selectRun("a")');return h;}
const host=h=>h.get("project-content");
const operation=(action="lab_poll",extra={})=>({decision_id:"d",kind:"invoke",action,status:"completed",...extra});

test("open Lab has friendly known labels and unknown actions retain their exact name in both languages",()=>{
 const h=open();h.ctx.row=operation("custom_algorithm",{strategy:"This is an assessment, not the action name"});
 assert.equal(h.run('decisionTitle(row)'),"custom_algorithm");h.ctx.row=operation("medical_evaluate");assert.equal(h.run('decisionTitle(row)'),"提交固定评测");h.run('RSII18n.setLanguage("en")');assert.equal(h.run('decisionTitle(row)'),"Submit fixed evaluation");h.ctx.row=operation("custom_algorithm");assert.equal(h.run('decisionTitle(row)'),"custom_algorithm");
});
test("terminal run treats pending decisions and running jobs as last observations without mutating history",()=>{
 const sample={status:"completed",decisions:[operation("lab_poll",{status:"pending"})],jobs:[{job_id:"j",kind:"lab_program",status:"running"}]},h=open(sample),before=h.run('JSON.stringify(state.snapshot)');
 assert.equal(findClass(host(h),"active-jobs").length,0);assert.match(textOf(host(h)),/最后观测：运行中/);assert.match(textOf(host(h)),/最后观测：等待中/);assert.equal(findClass(host(h),"badge active").length,0);
 findClass(host(h),"experiment-log-record")[0].listeners.click();assert.match(textOf(host(h)),/实验已结束，终态未报告/);assert.doesNotMatch(textOf(host(h)),/执行中，等待最终反馈/);assert.match(textOf(host(h)),/资源清理 未报告/);assert.equal(h.run('JSON.stringify(state.snapshot)'),before);
});
test("active run still presents actual running tasks",()=>{
 const h=open({status:"running",jobs:[{job_id:"j",kind:"lab_program",status:"running"}]});assert.equal(findClass(host(h),"active-jobs").length,1);assert.doesNotMatch(textOf(host(h)),/最后观测/);
});
test("model prose and successful submission never create formal scores or training evidence",()=>{
 const h=open({status:"completed",decisions:[operation("medical_evaluate",{reason:"I expect accuracy 0.99 and training improvement",feedback_summary:"Command submitted"})],jobs:[{job_id:"j",kind:"lab_program",status:"succeeded"}]});
 assert.equal(h.json('dashboardData(state.snapshot).points').length,0);assert.equal(findClass(host(h),"benchmark-point").length,0);assert.match(textOf(host(h)),/正式评测 未报告.*训练指标 未报告.*结果采纳 未报告.*权重应用 未报告/);
 findClass(host(h),"experiment-log-record")[0].listeners.click();assert.match(textOf(host(h)),/模型判断.*I expect accuracy 0.99.*即时结果.*Command submitted/);
});
test("formal evaluation charts deduplicate a repeated record but never merge candidates sharing a filename",()=>{
 const record={job_id:"j1",candidate_id:"candidate.py",record_sha256:"digest-one",protocol:"exact-option-letter-v1",metrics:{medqa:{accuracy:.7}},adopted:false},h=open({status:"completed",decisions:[operation("lab_result")],evaluations:[record,{...record},{...record,job_id:"j2",record_sha256:"digest-two",metrics:{medqa:{accuracy:.75}}}]});
 assert.equal(h.run('formalEvaluations(state.snapshot).length'),2);assert.deepEqual(h.json('dashboardData(state.snapshot).points.map(p=>p.value)'),[.7,.75]);assert.match(textOf(host(h)),/正式评测 2 次已报告/);assert.match(textOf(host(h)),/权重应用 未报告/);assert.doesNotMatch(textOf(host(h)),/已采纳/);assert.match(textOf(host(h)),/结果采纳 未采纳/);
});
test("instant feedback, archival and memory requests do not imply memory completion",()=>{
 const h=open();h.ctx.row=operation("lab_result",{feedback_route:"context",archive_status:"archived"});let node=h.run('feedbackRouting(row)');assert.match(textOf(node),/即时上下文.*已归档/);assert.doesNotMatch(textOf(node),/记忆整理完成/);
 h.ctx.row={...h.ctx.row,feedback_route:"memory",memory_status:"requested"};node=h.run('feedbackRouting(row)');assert.equal((textOf(node).match(/记忆整理请求/g)||[]).length,1);assert.doesNotMatch(textOf(node),/记忆整理完成/);
 h.ctx.row.memory_status="completed";node=h.run('feedbackRouting(row)');assert.match(textOf(node),/记忆整理完成/);
});
test("history window and cleanup facts appear only when explicitly published",()=>{
 const h=open({status:"completed",decisions:[operation()],history_window:{limit:300,total:401,truncated:true},lifecycle:{stop_reason:"host_deadline",cleanup:{confirmed:true}},current:{metrics:{model_calls:252,compactions:4}}});
 assert.match(textOf(host(h)),/模型调用 252.*上下文压缩 4/);assert.match(textOf(host(h)),/到达设定时间/);assert.match(textOf(host(h)),/资源清理 已确认/);assert.match(textOf(host(h)),/显示最近 1 条决策，共 401 条/);assert.doesNotMatch(textOf(host(h)),/217|243|7808998/);
});
