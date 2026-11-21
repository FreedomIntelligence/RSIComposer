/** RSIComposer observations -> native chart subscriptions and browser-owned panels. */
import {normalizeSmoothing} from './swanboard/charts/package/smooth.js';
export const SYSTEM_AXES = ['observation', 'step', 'elapsed_seconds', 'timestamp'];
export const AGGREGATES = ['both', 'lines', 'mean'];
export function sourceName(run) {
  return run.experiment_name || run.run_id;
}
/** Shortest legend label that still tells lines apart: run only when comparing runs, metric only when a
 *  panel mixes metrics, then the series. Two runs with the same name fall back to their IDs. */
function lineLabel(run, runs, panel, metric, series) {
  const name=sourceName(run),ambiguous=runs.some(other=>other!==run&&sourceName(other)===name);
  const parts=[runs.length>1?(ambiguous?run.run_id:name):null,panel.metrics.length>1?metric:null,series||null].filter(Boolean);
  return parts.length?parts.join(" · "):name;
}
export function availableMetrics(runs) {
  return [...new Set(runs.flatMap(run => (run.metric_points || [])
    .filter(p => typeof p.metric === 'string' && Number.isFinite(p.value)).map(p => p.metric)))].sort();
}
export function normalizePanel(value) {
  if (!value || typeof value !== 'object') throw new Error('图表配置无效');
  const text = (field, fallback='') => String(value[field] ?? fallback).trim();
  const id=text('id'), name=text('name'), namespace=text('namespace','自定义');
  if (!id || id.length>200 || !name || name.length>160 || !namespace || namespace.length>100) throw new Error('请填写图表名称和分组（名称最长 160 字）');
  if (!Array.isArray(value.metrics) || !value.metrics.length || value.metrics.length>12
      || value.metrics.some(key=>typeof key!=='string'||!key||key.length>100)) throw new Error('请选择 1–12 个 Y 轴指标');
  const xAxis=text('xAxis','observation'), xMetric=text('xMetric'), match=text('match','step');
  const aggregate=text('aggregate','both');
  if (!AGGREGATES.includes(aggregate)) throw new Error('请选择有效的多序列显示方式');
  if (![...SYSTEM_AXES,'metric'].includes(xAxis) || xAxis==='metric' && !xMetric) throw new Error('请选择有效的 X 轴');
  if (!['step','previous'].includes(match)) throw new Error('请选择自定义 X 轴的对齐方式');
  const result={id,name,namespace,metrics:[...new Set(value.metrics)],xAxis,xMetric,match,aggregate,
    smoothing:normalizeSmoothing(value.smoothing),xLabel:text('xLabel'),yLabel:text('yLabel'),xScale:text('xScale','linear'),yScale:text('yScale','linear')};
  if (![result.xScale,result.yScale].every(scale=>['linear','log'].includes(scale))) throw new Error('坐标轴刻度无效');
  for(const field of ['xMin','xMax','yMin','yMax']) {
    if(value[field]===undefined || value[field]===null || value[field]==='') continue;
    if(!['number','string'].includes(typeof value[field])) throw new Error('坐标轴范围必须是有限数值');
    const number=typeof value[field]==='number'?value[field]:Number(value[field]);
    if(!Number.isFinite(number)) throw new Error('坐标轴范围必须是有限数值');
    result[field]=number;
  }
  for(const axis of ['x','y']) {
    if(result[axis+'Min']!==undefined&&result[axis+'Max']!==undefined&&result[axis+'Min']>=result[axis+'Max']) throw new Error('坐标轴最小值必须小于最大值');
    if(result[axis+'Scale']==='log'&&['Min','Max'].some(edge=>result[axis+edge]!==undefined&&result[axis+edge]<=0)) throw new Error('对数轴范围必须大于 0');
  }
  return result;
}
/** Distinct `series` values a run reported for one metric, in first-seen order. */
export function seriesNames(run, metric) {
  const names=[];
  for(const p of run.metric_points||[]) if(p.metric===metric&&typeof p.series==='string'&&!names.includes(p.series)) names.push(p.series);
  return names.sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
}
/** Mean across series at each X; an X reported by only some series averages those that have it. */
export function meanSeries(run, metric, panel) {
  const sums=new Map();
  for(const name of seriesNames(run,metric)) for(const p of panelSeries(run,metric,panel,name)) {
    const entry=sums.get(p.index)||{total:0,count:0,create_time:p.create_time};
    entry.total+=p.data;entry.count++;sums.set(p.index,entry);
  }
  return [...sums].sort((a,b)=>a[0]-b[0]).map(([index,e],i,all)=>({index,data:e.total/e.count,create_time:e.create_time,_last:i===all.length-1}));
}
export function panelSeries(run, metric, panel, series) {
  const points=run.metric_points||[],scope=p=>p.call_id||p.decision_id||p.task_id||'';
  const xByStep=new Map(),previous=new Map(),xByObservation=new Map();
  // A log call is one simultaneous observation, regardless of dictionary order.
  if(panel.xAxis==='metric') for(const p of points) {
    if(p.metric===panel.xMetric&&Number.isFinite(p.value)&&p.observation_id) xByObservation.set(JSON.stringify([scope(p),p.observation_id]),p.value);
  }
  let ordinal=0;
  const values=[];
  for(const point of points) {
    if(panel.xAxis==='metric'&&point.metric===panel.xMetric&&Number.isFinite(point.value)) {
      previous.set(scope(point),point.value);
      if(Number.isFinite(point.step))xByStep.set(JSON.stringify([scope(point),point.step]),point.value);
    }
    if(point.metric!==metric||!Number.isFinite(point.value))continue;
    // Each series is its own line, so observation order counts within the series.
    if(series!==undefined&&point.series!==series)continue;
    let x;
    if(panel.xAxis==='observation') x=ordinal;
    else if(panel.xAxis==='timestamp') x=typeof point.timestamp==='string'?Date.parse(point.timestamp)/1000:undefined;
    else if(panel.xAxis==='metric') {
      x=point.observation_id?xByObservation.get(JSON.stringify([scope(point),point.observation_id])):undefined;
      if(x===undefined)x=panel.match==='previous'?previous.get(scope(point)):Number.isFinite(point.step)?xByStep.get(JSON.stringify([scope(point),point.step])):undefined;
    }
    else x=point[panel.xAxis];
    ordinal++;
    if(!Number.isFinite(x)||panel.xScale==='log'&&x<=0||panel.yScale==='log'&&point.value<=0)continue;
    values.push({index:x,data:point.value,create_time:point.timestamp||run.updated_at});
  }
  return values.sort((a,b)=>a.index-b.index).map((p,i,all)=>({...p,_last:i===all.length-1}));
}
export function metricSeries(run, metric, axis='observation') {
  return panelSeries(run,metric,{xAxis:axis,xScale:'linear',yScale:'linear'});
}
export function panelDefinitions(runs,preferences={},axis='observation') {
  const removed=Array.isArray(preferences.removed)?preferences.removed:[];
  const automatic=availableMetrics(runs).filter(key=>!removed.includes(key)).map(metric=>({
    // The chart sits under its group heading, so its own title is just the last path segment.
    id:metric,name:metric.slice(metric.lastIndexOf('/')+1),namespace:metric.includes('/')?metric.slice(0,metric.lastIndexOf('/')):'概览',metrics:[metric],
    // Plot against the reported step when there is one; observation order is only a fallback.
    xAxis:axis==='observation'&&runs.some(run=>(run.metric_points||[]).some(p=>p.metric===metric&&Number.isFinite(p.step)))?'step':axis,...preferences.overrides?.[metric],
  }));
  const result=[];
  for(const raw of [...automatic,...(Array.isArray(preferences.panels)?preferences.panels:[])]) {
    try {const panel=normalizePanel(raw);if(!result.some(p=>p.id===panel.id))result.push(panel);} catch { /* Ignore corrupt browser-local configuration. */ }
  }
  return result;
}
export function chartData(chart,runs) {
  const result={};
  for(const binding of chart.bindings) {
    const run=runs.find(run=>run.run_id===binding.runId);
    if(run) result[binding.source]={experiment_id:run.run_id,list:binding.mean?meanSeries(run,binding.metric,chart.panel)
      :panelSeries(run,binding.metric,chart.panel,binding.series)};
  }
  return result;
}
export function buildGroups(runs,preferences={},axis='observation') {
  const groups=new Map();
  for(const panel of panelDefinitions(runs,preferences,axis)) {
    const bindings=[];
    for(const run of runs)for(const metric of panel.metrics) {
      if(!panelSeries(run,metric,panel).length)continue;
      const names=seriesNames(run,metric),label=series=>lineLabel(run,runs,panel,metric,series);
      if(!names.length){bindings.push({runId:run.run_id,metric,source:label()});continue;}
      // One line per reported series (GPU, worker, seed...), and their mean.
      if(panel.aggregate!=='mean')for(const series of names)bindings.push({runId:run.run_id,metric,series,source:label(series)});
      if(panel.aggregate!=='lines'&&names.length>1)bindings.push({runId:run.run_id,metric,mean:true,source:label('平均')});
    }
    const status=preferences.charts?.[panel.id]||0,id=status===1?-1:status===-1?-2:panel.namespace;
    if(!groups.has(id))groups.set(id,{id,name:panel.namespace,opened:preferences.opened?.[id]??id!==-2,charts:[]});
    // Keep empty custom panels visible so users can correct their axes.
    groups.get(id).charts.push({id:panel.id,name:panel.name,type:'line',multi:bindings.length>1,
      reference:panel.xAxis,source:bindings.map(b=>b.source),bindings,panel,error:null});
  }
  return [...groups.values()].sort((a,b)=>a.id===-1?-1:b.id===-1?1:a.id===-2?1:b.id===-2?-1:String(a.id).localeCompare(String(b.id)));
}
