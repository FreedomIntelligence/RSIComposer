"use strict";

class RunSync {
  constructor() {
    this.runs = [];
    this.runId = '';
    this.snapshot = null;
    this.newest = null;
    this.revision = 0;
  }

  select(id) {
    if (this.runId === id) return;
    this.runId = id;
    this.snapshot = null;
    this.revision++;
  }

  async refresh(request) {
    const revision = ++this.revision;
    const data = await request('/api/runs');
    if (revision !== this.revision) return {};
    if (!Array.isArray(data.runs)) throw new Error('实验清单格式无效');
    this.runs = data.runs.filter(row => row && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(row.run_id));
    const key = row => Date.parse(row.started_at);
    const newest = this.runs.filter(row => Number.isFinite(key(row)))
      .sort((a,b) => key(b) - key(a) || b.run_id.localeCompare(a.run_id))[0];
    const newer = newest && (!this.newest || key(newest) > key(this.newest)
      || key(newest) === key(this.newest) && newest.run_id.localeCompare(this.newest.run_id) > 0);
    const id = newer && (this.newest || !this.runId) ? newest.run_id
      : this.runs.some(row => row.run_id === this.runId) ? this.runId : newest?.run_id || this.runs[0]?.run_id || '';
    const switched = id !== this.runId;
    if (id) {
      const snapshot = await request(`/api/runs/${encodeURIComponent(id)}`);
      if (revision !== this.revision) return {};
      if (snapshot?.schema_version !== 1 || snapshot.run_id !== id) throw new Error('快照与当前实验 ID 不匹配');
      this.snapshot = snapshot;
    } else {
      this.snapshot = null;
    }
    if (newer) this.newest = newest;
    this.runId = id;
    return { switched };
  }
}

const liveBoard = {
  sync: new RunSync(),
  selected: new Set(),
  compared: new Map(),
  decisionId: '',
  evaluationId: '',
  stageLogs: '',
  renderKey: '',
  controller: null,
  timer: null,
  error: '',
};
const rowsOf = value => Array.isArray(value) ? value.filter(row => row && typeof row === 'object') : [];
const liveStatus = {
  unavailable:'快照不可用', starting:'启动中', queued:'排队中', pending:'等待中', running:'运行中', active:'运行中', waiting:'等待中',
  progressed:'进行中', completed:'已完成', finished:'已结束', succeeded:'已完成', failed:'失败',
  cancelled:'已取消', stopped:'已停止', partial:'部分完成', step_limit:'达到决策步数上限',
};
const liveBadge = status => `<span class="badge ${['completed','finished','succeeded'].includes(status)?'completed':status==='failed'?'failed':status==='running'?'running':'pending'}">${escapeHTML(liveStatus[status] || status || '状态未上报')}</span>`;
const display = value => escapeHTML(value === null || value === undefined || value === '' ? '—' : value);
const missing = label => `<div class="empty-state">${escapeHTML(label)}未上报</div>`;
const textTable = (headers, rows) => table(headers, rows.map(row=>`<tr>${row.map(value=>`<td>${display(value)}</td>`).join('')}</tr>`));
const jsonBlock = value => `<pre class="code-block">${escapeHTML(JSON.stringify(value,null,2))}</pre>`;
const fields = object => Object.entries(object || {}).filter(([,value])=>['number','boolean','string'].includes(typeof value));
const fmtPercent = val => Number.isFinite(val)?`${format(val*100,2)}%`:'—';
const decisionKey = row => row.decision_id || row.directive_id;
const runLabel = row => row?.experiment_name || row?.run_id;
const liveButton = (label, symbol, action) => `<button class="button small" data-live-action="${action}">${icon(symbol)}${label}</button>`;

function liveHeader(run) {
  return `<div class="page-heading"><div><div class="title-row"><h1>${display(runLabel(run))}</h1>${liveBadge(run.status)}${run.simulated===true?tag('模拟运行'):''}</div><div class="subheading"><span class="mono">${display(run.run_id)}</span>${run.config?.seed!==undefined?tag(`seed ${run.config.seed}`):''}${run.project?.name?tag(run.project.name):''}</div></div><div class="heading-actions">${liveButton('导出记录','download','export-run')}${liveButton('运行配置','settings-2','config')}</div></div>`;
}

function liveTabs(run) {
  return `<nav class="tabs view-nav" aria-label="实验详情">${['overview','trajectory','metrics','samples','artifacts','resources','logs'].map(page=>`<a href="#${page}" class="tab ${state.page===page?'active':''}">${labels[page]}${page==='trajectory'?`<span class="count">${rowsOf(run.decisions).length}</span>`:''}</a>`).join('')}</nav>`;
}

function currentLiveStage(run) {
  const stages = rowsOf(run.decisions);
  const current = run.current || {};
  const chosen = stages.find(row=>decisionKey(row)===liveBoard.decisionId);
  if (chosen && liveBoard.decisionId) return chosen;
  const stage = decisionKey(current) ? stages.find(row=>decisionKey(row)===decisionKey(current)) : stages.at(-1);
  return stage ? {...stage,...current} : current;
}

function liveStageBanner(run) {
  const stage = currentLiveStage(run);
  const progressKnown = Number.isFinite(stage.step) && Number.isFinite(stage.total_steps) && stage.total_steps > 0;
  return `<div class="stage-banner ${liveBoard.decisionId?'historical':''}"><div><div class="stage-banner-label">${icon(liveBoard.decisionId?'history':'radio')}${liveBoard.decisionId?'所选阶段':'当前执行位置'}</div><h3>${display(stage.operation || stage.phase || stage.mode || '等待阶段上报')}</h3><div class="stage-banner-meta"><span class="mono">${display(decisionKey(stage) || stage.job_id)}</span>${Object.keys(stage).length?liveBadge(stage.status):''}</div></div><div>${progressKnown?`<div class="progress-label"><span>执行进度</span><strong>${stage.step} / ${stage.total_steps}</strong></div>${progress(Math.max(0,Math.min(100,100*stage.step/stage.total_steps)))}`:Number.isFinite(stage.step)?`已上报 ${stage.step} 步`:'步数未上报'}</div><div class="banner-actions">${liveBoard.decisionId?liveButton('回到当前阶段','radio','follow'):liveButton('查看轨迹','route','trajectory')}<div class="updated">${display(run.updated_at)}</div></div></div>`;
}

function liveStageList(run) {
  const stages = rowsOf(run.decisions);
  const current = decisionKey(currentLiveStage(run));
  if (!stages.length) return missing('执行阶段');
  return `<div class="stage-list">${stages.filter(decisionKey).map(row=>`<button class="stage-row ${decisionKey(row)===current?'active':''}" data-live-stage="${display(decisionKey(row))}" aria-pressed="${decisionKey(row)===current}"><span class="stage-icon">${icon('circle-dot')}</span><span class="stage-text"><strong>${display(row.operation || row.kind || row.mode)}</strong><small class="mono">${display(decisionKey(row))}</small></span><span class="stage-meta">${liveBadge(row.status)}</span></button>`).join('')}</div>`;
}

function liveMetricCharts(run, decisionId) {
  const groups = new Map();
  for (const point of rowsOf(run.metric_points)) {
    if (decisionId && point.decision_id !== decisionId && point.task_id !== decisionId) continue;
    if (typeof point.metric !== 'string' || !Number.isFinite(point.value)) continue;
    const axis = Number.isFinite(point.step) ? 'step' : Number.isFinite(point.elapsed_seconds) ? 'elapsed_seconds' : null;
    if (!axis) continue;
    const key = JSON.stringify([point.metric, axis]);
    if (!groups.has(key)) groups.set(key, new Map());
    const stages = groups.get(key), stage = point.decision_id || point.task_id || '';
    if (!stages.has(stage)) stages.set(stage, new Map());
    stages.get(stage).set(point[axis], point.value);
  }
  const colors = ['#a85739','#4f6850','#63767b','#8b754e'];
  const charts = Array.from(groups, ([key, stages])=>{
    const [metric,axis]=JSON.parse(key);
    const definition=run.metric_definitions?.[metric];
    const percent=definition?.unit==='percent'||metric==='accuracy';
    const sets=Array.from(stages, ([name, values],index)=>series(name || metric,colors[index%colors.length],
      Array.from(values,([x,y])=>({x,y:percent?y*100:y})).sort((a,b)=>a.x-b.x)));
    const values=sets.flatMap(set=>set.points.map(point=>point.y));
    const low=Math.min(...values), high=Math.max(...values), last=values.at(-1);
    const padding=(high-low)*.1||Math.max(Math.abs(high)*.1,1e-8);
    const min=percent?Math.max(0,low-padding):low-padding;
    const max=percent?Math.min(100,high+padding):high+padding;
    const digits=high!==0&&Math.abs(high)<.001?8:4;
    return chart(definition?.label||metric,metric,`${format(last,digits)}${percent?'%':''}`,sets,
      {min,max,percent,axisLabel:axis==='step'?'上报步数':'运行时间 / 秒',discrete:values.length<15});
  });
  return charts.length?`<div class="chart-grid">${charts.join('')}</div>`:missing('指标曲线');
}

function liveOverview(run) {
  const stage=currentLiveStage(run);
  const recent=rowsOf(run.logs || run.events).slice(-5).reverse();
  const evals=rowsOf(run.evaluations);
  const dev=evals.filter(row=>row.split==='dev'&&Number.isFinite(row.metrics?.accuracy));
  const best=dev.length?Math.max(...dev.map(row=>row.metrics.accuracy)):null;
  const observed=stage.metrics?.completion_loss;
  return `${liveStageBanner(run)}<div class="stats">${stat('最佳开发集正确率',fmtPercent(best),'已上报开发集评测','target')}${stat('阶段训练损失',Number.isFinite(observed)?format(observed,4):'—','当前所选阶段','chart-no-axes-combined')}${stat('执行记录',rowsOf(run.decisions).length,'已上报阶段','route')}${stat('累计运行时间',Number.isFinite(run.workflow?.elapsed_seconds)?`${format(run.workflow.elapsed_seconds)} 秒`:'—','实验运行端记录','timer')}</div><section class="section">${sectionTitle('执行阶段','route',liveButton('完整轨迹','arrow-up-right','trajectory'))}${liveStageList(run)}</section><section class="section">${sectionTitle('阶段指标','chart-no-axes-combined')}${decisionKey(stage)?liveMetricCharts(run,decisionKey(stage)):missing('阶段指标')}</section><div class="two-columns"><section>${sectionTitle('评测记录','list-checks')}${evaluationTable(evals)}</section><section>${sectionTitle('最近事件','clock-3')}${recent.length?textTable(['时间','事件'],recent.map(row=>[row.timestamp,row.message||row.summary])):missing('事件')}</section></div>`;
}

function liveTrajectory(run) {
  const stage=currentLiveStage(run);
  return `${liveStageBanner(run)}<div class="stage-workspace"><section>${sectionTitle('执行记录','list-tree')}${liveStageList(run)}</section><section class="stage-detail">${sectionTitle('阶段详情','file-text')}${Object.keys(stage).length?`<div class="detail-facts">${fields(stage).map(([key,value])=>fact(display(key),display(value))).join('')}</div>${sectionTitle('阶段指标','chart-no-axes-combined')}${fields(stage.metrics).length?textTable(['指标','数值'],fields(stage.metrics)):missing('阶段指标')}<div class="dialog-actions">${liveButton('关联日志','terminal','stage-logs')}</div>`:missing('阶段详情')}</section></div>`;
}

function evaluationTable(evals) {
  if (!evals.length) return missing('评测记录');
  return textTable(['评测','数据划分','模型','正确率','样本数'],evals.map(row=>[
    row.label||row.evaluation_id||row.protocol,row.split,row.model||row.weight_identity?.model_id,
    fmtPercent(row.metrics?.accuracy),row.metrics?.total??row.metrics?.examples]));
}

function liveSamples(run) {
  const evaluations=rowsOf(run.evaluations);
  const samples=rowsOf(run.samples);
  const selected=evaluations.find(row=>row.evaluation_id===liveBoard.evaluationId);
  const shown=selected?samples.filter(row=>row.decision_id===(selected.decision_id||selected.evaluation_id)):samples;
  const flags = value => value === true ? '是' : value === false ? '否' : '—';
  return `${sectionTitle('评测记录','list-checks')}${evaluationTable(evaluations)}<section class="section section-band">${sectionTitle('逐样本结果','list-checks',samples.length?liveButton('导出样本','download','export-samples'):'')}<select id="live-evaluation" aria-label="评测记录"><option value="">全部已上报评测</option>${evaluations.filter(row=>row.evaluation_id).map(row=>`<option value="${display(row.evaluation_id)}" ${selected===row?'selected':''}>${display(row.label||row.evaluation_id)}</option>`).join('')}</select>${shown.length?textTable(['样本 ID','关联阶段','正确','可解析','预测答案','标准答案'],shown.map(row=>[row.id,row.decision_id,flags(row.correct),flags(row.parsed),row.predicted,row.expected])):missing('逐样本结果')}</section>`;
}

function liveArtifacts(run) {
  const records=rowsOf(run.artifacts);
  const rows=records.map((row,index)=>`<tr><td><button class="text-link mono" data-live-artifact="${index}">${display(row.artifact_id)}</button></td><td>${display(row.kind)}</td><td>${display(row.decision_id)}</td><td>${display(row.model)}</td></tr>`);
  return `${sectionTitle('版本与产物','git-branch',records.length?liveButton('导出清单','download','export-artifacts'):'')}${rows.length?table(['产物 ID','类型','关联阶段','模型版本'],rows):missing('版本或产物')}`;
}

function liveResources(run) {
  const resources=run.resources||{};
  const values=fields(resources);
  const metrics=fields(resources.latest_execution_metrics);
  return `${sectionTitle('资源与成本','cpu')}${values.length||metrics.length?textTable(['字段','已上报值'],[...values,...metrics]):missing('资源用量')}`;
}

function liveLogs(run) {
  const records=rowsOf(run.logs || run.events);
  const shown=liveBoard.stageLogs?records.filter(row=>(row.decision_id||row.details?.decision_id)===liveBoard.stageLogs):records;
  return `${sectionTitle('运行日志与事件','terminal',shown.length?liveButton('导出日志','download','export-logs'):'')}${liveBoard.stageLogs?`<div class="notice-line">${display(liveBoard.stageLogs)} ${liveButton('清除阶段筛选','x','clear-logs')}</div>`:''}${shown.length?textTable(['时间','级别','来源','关联阶段','记录'],shown.map(row=>[row.timestamp,row.level,row.source||row.kind,row.decision_id||row.details?.decision_id,row.message||row.summary])):missing('日志或事件')}<section class="section section-band">${sectionTitle('运行配置','settings-2',fields(run.config).length?liveButton('下载配置','download','export-config'):'')}${fields(run.config).length?textTable(['参数','实际值'],fields(run.config)):missing('公开配置')}</section>`;
}

function liveExperiments() {
  const runs=liveBoard.sync.runs;
  const rows=runs.map(row=>`<tr><td><input type="checkbox" data-live-compare="${display(row.run_id)}" aria-label="对比 ${display(runLabel(row))}" ${liveBoard.selected.has(row.run_id)?'checked':''}></td><td><button class="text-link" data-live-run="${display(row.run_id)}">${display(runLabel(row))}</button><small class="cell-sub mono">${display(row.run_id)}</small></td><td>${liveBadge(row.status)}</td><td>${display(row.started_at)}</td><td>${display(row.updated_at)}</td></tr>`);
  return `<div class="page-heading"><div class="title-row"><h1>实验管理</h1>${tag(`${runs.length} 个实验`)}</div><div class="heading-actions">${liveButton('对比已选实验','git-compare-arrows','compare')}</div></div>${rows.length?table(['对比','实验','状态','开始时间','最后更新'],rows):missing('实验快照')}`;
}

function liveComparison() {
  const compared=Array.from(liveBoard.selected,id=>liveBoard.compared.get(id)).filter(Boolean);
  const split=state.compareMetric;
  const evaluations=run=>rowsOf(run.evaluations).filter(row=>row.split===split&&Number.isFinite(row.metrics?.accuracy));
  const data=compared.flatMap(run=>evaluations(run).map(row=>[
    runLabel(run),row.label||row.evaluation_id,row.round,fmtPercent(row.metrics.accuracy),row.metrics.total??row.metrics.examples]));
  const palette=['#a85739','#4f6850','#63767b','#8b754e'];
  const axis=state.compareAxis==='time'?'elapsed_seconds':'round';
  const sets=compared.map((run,i)=>series(runLabel(run),palette[i%palette.length],evaluations(run)
    .filter(row=>Number.isFinite(row[axis])).map(row=>({x:row[axis],y:row.metrics.accuracy*100})).sort((a,b)=>a.x-b.x)))
    .filter(set=>set.points.length);
  const curve=sets.length?chart(split==='dev'?'开发集正确率':'测试集正确率',`${split}/accuracy`,
    `${format(Math.max(...sets.flatMap(set=>set.points.map(point=>point.y))),2)}%`,sets,
    {min:0,max:100,percent:true,axisLabel:axis==='round'?'上报轮次':'运行时间 / 秒',discrete:true}):'';
  const toggles=liveBoard.sync.runs.map(row=>`<label class="compare-chip"><input type="checkbox" data-live-compare="${display(row.run_id)}" ${liveBoard.selected.has(row.run_id)?'checked':''}>${display(runLabel(row))}</label>`).join('');
  return `<div class="page-heading"><div class="title-row"><h1>实验对比</h1>${tag(`${liveBoard.selected.size} 个已选实验`)}</div><div class="heading-actions">${compared.length?liveButton('导出对比','download','export-comparison'):''}</div></div><div class="compare-selection">${toggles}</div><div class="metrics-toolbar"><div class="segmented"><button data-live-action="dev" class="${split==='dev'?'active':''}">开发集</button><button data-live-action="test" class="${split==='test'?'active':''}">测试集</button></div><label class="control-group">横轴<select id="live-compare-axis"><option value="round" ${axis==='round'?'selected':''}>上报轮次</option><option value="time" ${axis==='elapsed_seconds'?'selected':''}>运行时间</option></select></label></div>${curve}${data.length?textTable(['实验','评测','轮次','正确率','样本数'],data):missing(liveBoard.selected.size?'所选实验的该数据划分评测':'对比数据')}`;
}

function renderLivePage() {
  const sync=liveBoard.sync;
  const run=sync.snapshot;
  const index=state.page==='experiments'||state.page==='compare';
  $('sidebar-run').textContent=runLabel(run) || sync.runId || '等待实验';
  for(const node of document.querySelectorAll('[data-live-run-count]')) node.textContent=sync.runs.length;
  for(const node of document.querySelectorAll('[data-live-stage-count]')) node.textContent=rowsOf(run?.decisions).length;
  const notice=liveBoard.error?`<div class="notice-line" role="status">${display(liveBoard.error)}${run?' · 显示上次成功快照':''}</div>`:'';
  if (index) return notice+(state.page==='experiments'?liveExperiments():liveComparison());
  if (!run) return `${notice}<div class="page-heading"><h1>${labels[state.page]}</h1></div>${missing(sync.runId?'该实验快照':'实验快照')}`;
  const pages={overview:liveOverview,trajectory:liveTrajectory,metrics:liveMetricCharts,samples:liveSamples,artifacts:liveArtifacts,resources:liveResources,logs:liveLogs};
  return notice+liveHeader(run)+liveTabs(run)+pages[state.page](run);
}

async function refreshLive() {
  clearTimeout(liveBoard.timer);
  liveBoard.controller?.abort();
  const controller=new AbortController();
  liveBoard.controller=controller;
  const timeout=setTimeout(()=>controller.abort(),8000);
  const request=async path=>{
    const response=await fetch(path,{signal:controller.signal,cache:'no-store'});
    if(!response.ok) throw new Error(`同步失败（HTTP ${response.status}）`);
    return response.json();
  };
  const active=()=>liveBoard.controller===controller;
  try {
    const {switched}=await liveBoard.sync.refresh(request);
    if(!active()) return;
    liveBoard.error='';
    if(switched){
      liveBoard.decisionId=liveBoard.evaluationId=liveBoard.stageLogs='';
      liveBoard.compared.clear();
      if(location.hash!=='#overview') location.hash='overview';
      $('detail-dialog').close();
    }
    if(state.page==='compare'){
      const selected=Array.from(liveBoard.selected).filter(id=>liveBoard.sync.runs.some(row=>row.run_id===id));
      const snapshots=await Promise.allSettled(selected.map(id=>request(`/api/runs/${encodeURIComponent(id)}`)));
      if(!active()) return;
      liveBoard.compared=new Map(snapshots.flatMap((result,i)=>result.status==='fulfilled'&&result.value.schema_version===1&&result.value.run_id===selected[i]?[[selected[i],result.value]]:[]));
      if(liveBoard.compared.size<selected.length) liveBoard.error='部分对比快照暂时无法同步';
    }
  } catch(error) {
    if(active()) liveBoard.error=controller.signal.aborted?'同步超时，正在重试':error.message;
  } finally {
    clearTimeout(timeout);
    if(active()){
      liveBoard.controller=null;
      const renderKey=JSON.stringify([liveBoard.sync.runs,liveBoard.sync.snapshot,Array.from(liveBoard.compared),liveBoard.error]);
      if(liveBoard.renderKey!==renderKey){liveBoard.renderKey=renderKey;render(true);}
      if(!document.hidden) liveBoard.timer=setTimeout(refreshLive,2500);
    }
  }
}

function selectLiveRun(id) {
  if(!liveBoard.sync.runs.some(row=>row.run_id===id)) return;
  liveBoard.sync.select(id);
  liveBoard.decisionId=liveBoard.evaluationId=liveBoard.stageLogs='';
  liveBoard.error='';
  liveBoard.renderKey='';
  navigate('overview');
  render();
  refreshLive();
}

function openLiveRuns() {
  modal('切换实验',liveBoard.sync.runs.length?`<div class="switch-list">${liveBoard.sync.runs.map(row=>`<button data-live-run="${display(row.run_id)}"><span>${display(runLabel(row))}</span>${liveBadge(row.status)}</button>`).join('')}</div>`:missing('实验快照'),'真实运行');
}

function liveAction(action) {
  const run=liveBoard.sync.snapshot;
  if(action==='compare'||action==='trajectory') {navigate(action);return;}
  if(action==='dev'||action==='test'){state.compareMetric=action;render();return;}
  if(action==='follow'){liveBoard.decisionId='';render();return;}
  if(action==='clear-logs'){liveBoard.stageLogs='';render();return;}
  if(action==='stage-logs'&&run){liveBoard.stageLogs=decisionKey(currentLiveStage(run))||'';navigate('logs');return;}
  if(action==='config'&&run){modal('运行配置',fields(run.config).length?jsonBlock(run.config)+liveButton('下载配置','download','export-config'):missing('公开配置'),run.run_id);return;}
  if(action==='export-comparison'){download('monitor-comparison.json',JSON.stringify(Array.from(liveBoard.compared.values(),item=>({run_id:item.run_id,evaluations:item.evaluations})),null,2));return;}
  if(!run)return;
  const evaluation=rowsOf(run.evaluations).find(row=>row.evaluation_id===liveBoard.evaluationId);
  const values={run,config:run.config,samples:rowsOf(run.samples).filter(row=>!evaluation||row.decision_id===(evaluation.decision_id||evaluation.evaluation_id)),artifacts:run.artifacts,
    logs:rowsOf(run.logs||run.events).filter(row=>!liveBoard.stageLogs||(row.decision_id||row.details?.decision_id)===liveBoard.stageLogs)};
  const key=action.replace(/^export-/, '');
  if(Object.hasOwn(values,key)&&values[key]!==undefined) download(`${run.run_id}-${key}.json`,JSON.stringify(values[key],null,2));
}

function liveClick(event) {
  const run=event.target.closest('[data-live-run]');
  if(run){selectLiveRun(run.dataset.liveRun);return;}
  const stage=event.target.closest('[data-live-stage]');
  if(stage){liveBoard.decisionId=stage.dataset.liveStage;render();return;}
  const artifact=event.target.closest('[data-live-artifact]');
  if(artifact){const row=rowsOf(liveBoard.sync.snapshot?.artifacts)[Number(artifact.dataset.liveArtifact)];if(row)modal(row.artifact_id,jsonBlock(row),liveBoard.sync.runId);return;}
  const action=event.target.closest('[data-live-action]');
  if(action)liveAction(action.dataset.liveAction);
}

function liveChange(event) {
  const node=event.target;
  if(node.dataset.liveCompare){
    if(node.checked)liveBoard.selected.add(node.dataset.liveCompare);
    else{liveBoard.selected.delete(node.dataset.liveCompare);liveBoard.compared.delete(node.dataset.liveCompare);}
    if(state.page==='compare')refreshLive();
  }
  if(node.id==='live-compare-axis'){state.compareAxis=node.value;render();}
  if(node.id==='live-evaluation'){liveBoard.evaluationId=node.value;render();}
}

function startLive() {
  const runId = new URLSearchParams(location.search).get('run');
  if (runId) liveBoard.sync.select(runId);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)clearTimeout(liveBoard.timer);else refreshLive();});
  refreshLive();
}
