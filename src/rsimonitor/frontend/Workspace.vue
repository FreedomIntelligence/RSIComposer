<template>
  <div>
    <div class="rsi-metrics-toolbar">
      <div v-if="catalog.length>1" class="rsi-metrics-runs" role="group" aria-label="叠加对比的实验">
        <button v-for="run in catalog" :key="run.run_id" type="button" class="chip" :class="{active:selected.includes(run.run_id)}" :aria-pressed="selected.includes(run.run_id)" :disabled="run.run_id===props.state.run?.run_id" @click="select(run.run_id,!selected.includes(run.run_id))">{{ run.experiment_name || run.run_id }}</button>
      </div>
      <span v-if="error" class="rsi-metrics-error" role="status">{{ error }}</span>
      <span class="rsi-toolbar-space"></span>
      <button class="button" type="button" @click="manageOpen=!manageOpen">管理</button><button class="button" type="button" @click="openEditor()">添加图表</button>
    </div>
    <div v-if="manageOpen" class="rsi-chart-manager">
      <p v-if="!allPanels.length">暂无图表，可从上方添加。</p>
      <div v-for="panel in allPanels" :key="panel.id"><span>{{ panel.name }}<small>{{ panel.namespace }} · {{ panel.metrics.join(', ') }}</small></span><button class="button" type="button" @click="removePanel(panel)">移除</button></div>
      <button v-if="preferences.removed?.length" class="button" type="button" @click="restoreAutomatic">恢复已移除的自动图表</button>
    </div>
    <ChartsDashboard :key="chartKey" :edit-chart="chart=>openEditor(chart.panel)" :groups="groups" :update-chart-status="updateChartStatus"
      :update-namespace-status="updateNamespaceStatus" :media="media" :default-color="colors[0]"
      :get-color="getColor" :subscribe="subscribe" :unsubscribe="unsubscribe" experiment-prefix="?run=" />
    <p v-if="!metricCount" class="rsi-metrics-empty">还没有指标。在代码里调用 rsicomposer.log({...}) 就会出现在这里。</p>
    <dialog ref="editorDialog" class="rsi-panel-editor" aria-labelledby="panel-editor-title" @cancel.prevent="closeEditor">
      <form v-if="draft" novalidate @submit.prevent="savePanel">
        <header><div><small>图表配置</small><h2 id="panel-editor-title">{{ editingNew ? '添加图表' : '图表设置' }}</h2></div><button type="button" aria-label="关闭图表配置" @click="closeEditor">✕</button></header>
        <nav class="rsi-editor-tabs" role="tablist" aria-label="图表设置分类">
          <button v-for="tab in editorTabs" :key="tab.id" :id="'chart-tab-'+tab.id" type="button" role="tab" :aria-selected="editorTab===tab.id" :aria-controls="'chart-panel-'+tab.id" :tabindex="editorTab===tab.id?0:-1" @click="editorTab=tab.id" @keydown="switchTab($event,tab.id)">{{ tab.label }}</button>
        </nav>
        <div class="rsi-editor-body">
          <section v-show="editorTab==='data'" id="chart-panel-data" role="tabpanel" aria-labelledby="chart-tab-data">
          <label>图表名称<input v-model="draft.name" required maxlength="160" placeholder="例如：训练质量与成本"></label>
          <label>多序列显示<select v-model="draft.aggregate"><option value="both">每条序列 + 平均</option><option value="lines">仅每条序列</option><option value="mean">仅平均</option></select></label>
          <p class="rsi-editor-help">同一指标按 series（如各 GPU、各 worker）上报时生效。</p>
          <label>分组<input v-model="draft.namespace" required maxlength="100" placeholder="例如：训练分析"></label>
          <fieldset><legend>Y 轴指标</legend>
            <label>筛选指标<input v-model="metricQuery" type="search" placeholder="搜索已上报的指标"></label>
            <div class="rsi-y-options"><label v-for="metric in filteredMetricKeys" :key="metric"><input v-model="draft.metrics" type="checkbox" :value="metric">{{ metric }}</label><p v-if="!filteredMetricKeys.length" class="rsi-editor-help">尚无匹配指标</p></div>
            <p class="rsi-editor-help">已选 {{ draft.metrics.length }} 项，最多 12 项；每个指标 × 实验形成一条曲线。</p>
          </fieldset>
          </section>
          <section v-show="editorTab==='axes'" id="chart-panel-axes" role="tabpanel" aria-labelledby="chart-tab-axes">
          <fieldset><legend>X 轴</legend>
            <label>数据<select v-model="xSelection"><optgroup label="内置坐标"><option value="observation">观测序号</option><option value="step">上报步数</option><option value="elapsed_seconds">运行时间 / 秒</option><option value="timestamp">记录时间 / Unix 秒</option></optgroup><optgroup label="自定义指标"><option v-for="metric in metricKeys" :key="metric" :value="'metric:'+metric">{{ metric }}</option></optgroup></select></label>
            <label v-if="draft.xAxis==='metric'">指标对齐<select v-model="draft.match"><option value="step">同一操作、同一步数</option><option value="previous">同一操作最近已上报的 X 值</option></select></label>
            <p v-if="draft.xAxis==='metric'" class="rsi-editor-help">优先对齐同一次观测；分开记录时先报 X 再报 Y，不跨实验或操作配对，缺失 X 的点跳过。</p>
            <label>轴标题<input v-model="draft.xLabel" maxlength="120" placeholder="留空使用默认标题"></label>
            <div class="rsi-axis-grid"><label>刻度<select v-model="draft.xScale"><option value="linear">线性</option><option value="log">对数</option></select></label><label>最小值<input v-model="draft.xMin" type="number" step="any" placeholder="自动"></label><label>最大值<input v-model="draft.xMax" type="number" step="any" placeholder="自动"></label></div>
          </fieldset>
          <fieldset><legend>Y 轴</legend>
            <label>轴标题<input v-model="draft.yLabel" maxlength="120" placeholder="留空显示指标数值"></label>
            <div class="rsi-axis-grid"><label>刻度<select v-model="draft.yScale"><option value="linear">线性</option><option value="log">对数</option></select></label><label>最小值<input v-model="draft.yMin" type="number" step="any" placeholder="自动"></label><label>最大值<input v-model="draft.yMax" type="number" step="any" placeholder="自动"></label></div>
            <p v-if="draft.xScale==='log'||draft.yScale==='log'" class="rsi-editor-help">对数轴仅展示大于 0 的值。</p>
          </fieldset>
          </section>
          <section v-show="editorTab==='smoothing'" id="chart-panel-smoothing" role="tabpanel" aria-labelledby="chart-tab-smoothing">
            <label>平滑算法<select v-model.number="draft.smoothing.id" @change="changeSmoothingMethod"><option v-for="method in SMOOTHING_METHODS" :key="method.id" :value="method.id">{{ method.name }}</option></select></label>
            <p class="rsi-editor-help">{{ smoothingMethod.description }}</p>
            <template v-if="draft.smoothing.id!==0">
              <label>{{ smoothingMethod.parameter }}<div class="rsi-smoothing-inputs"><input v-model.number="draft.smoothing.value" :aria-label="smoothingMethod.parameter+'滑块'" type="range" :min="smoothingMethod.min" :max="smoothingMethod.max" :step="smoothingMethod.step"><input v-model.number="draft.smoothing.value" :aria-label="smoothingMethod.parameter" type="number" :min="smoothingMethod.min" :max="smoothingMethod.max" :step="smoothingMethod.step" required></div></label>
              <label class="rsi-original-option"><input v-model="draft.smoothing.showOriginal" type="checkbox">同时显示原始曲线</label>
            </template>
            <p class="rsi-editor-help">仅影响当前图表的展示。原始观测记录保持不变，放大和导出沿用此图的设置。</p>
          </section>
          <p v-if="editorError" class="rsi-editor-error" role="alert">{{ editorError }}</p>
        </div>
        <footer><span></span><button type="button" class="button" @click="closeEditor">取消</button><button type="submit" class="button primary">{{ editingNew?'添加图表':'应用设置' }}</button></footer>
      </form>
    </dialog>
    <SLMessages ref="messages" />
  </div>
</template>
<script setup>
import {computed,ref,watch,onMounted,onBeforeUnmount,shallowRef,nextTick} from 'vue';
import ChartsDashboard from './swanboard/charts/ChartsDashboard.vue';
import {SLMessages,installMessage} from './swanboard/components/message';
import {SMOOTHING_METHODS} from './swanboard/charts/package/smooth.js';
import {SYSTEM_AXES,availableMetrics,normalizePanel,panelDefinitions,chartData,buildGroups} from './data.js';
const props=defineProps({state:{type:Object,required:true}});
const colors=['#a85739','#4f6850','#63767b','#8b754e','#77668f','#3b8791','#b66b85','#c08a2e','#5b6fa8','#7a8f3c','#a34a6a','#2f7f6b'];
const messages=ref(null),axis=ref('observation'),selected=ref([]),catalog=ref([]),snapshots=shallowRef([]),error=ref('');
const preferences=ref({charts:{},opened:{},panels:[],overrides:{},removed:[]}),groups=ref([]),subscriptions=new Map();
const editorDialog=ref(null),draft=ref(null),editingNew=ref(false),editorError=ref(''),metricQuery=ref(''),manageOpen=ref(false);
const editorTabs=[{id:'data',label:'数据'},{id:'axes',label:'坐标轴'},{id:'smoothing',label:'平滑'}],editorTab=ref('data');
const smoothingMethod=computed(()=>SMOOTHING_METHODS.find(method=>method.id===draft.value?.smoothing.id)||SMOOTHING_METHODS[0]);
function changeSmoothingMethod(){draft.value.smoothing.value=smoothingMethod.value.defaultValue;}
function switchTab(event,id){
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  event.preventDefault();const index=editorTabs.findIndex(tab=>tab.id===id);
  const next=event.key==='Home'?0:event.key==='End'?editorTabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+editorTabs.length)%editorTabs.length;
  editorTab.value=editorTabs[next].id;nextTick(()=>editorDialog.value.querySelector('#chart-tab-'+editorTab.value)?.focus());
}
const metricKeys=computed(()=>availableMetrics(snapshots.value));
const filteredMetricKeys=computed(()=>metricKeys.value.filter(key=>key.toLowerCase().includes(metricQuery.value.toLowerCase())));
const allPanels=computed(()=>panelDefinitions(snapshots.value,preferences.value,axis.value));
const xSelection=computed({get:()=>draft.value?.xAxis==='metric'?'metric:'+draft.value.xMetric:draft.value?.xAxis,
  set:value=>{draft.value.xAxis=value.startsWith('metric:')?'metric':value;draft.value.xMetric=value.startsWith('metric:')?value.slice(7):'';}});
async function openEditor(panel){
  editingNew.value=!panel;metricQuery.value='';editorError.value='';editorTab.value='data';
  draft.value=JSON.parse(JSON.stringify(panel||{id:'custom:'+crypto.randomUUID(),name:'',namespace:'自定义',metrics:metricKeys.value.slice(0,1),xAxis:axis.value,xMetric:'',match:'step',aggregate:'both',xScale:'linear',yScale:'linear',xLabel:'',yLabel:'',smoothing:{id:0,value:0,showOriginal:true}}));
  draft.value.smoothing ||= {id:0,value:0,showOriginal:true};
  draft.value.aggregate ||= 'both';
  await nextTick();editorDialog.value.showModal();
}
function closeEditor(){editorDialog.value?.close();draft.value=null;}
function savePanel(){
  try{
    const panel=normalizePanel(draft.value);preferences.value.panels ||= [];preferences.value.overrides ||= {};
    if(panel.id.startsWith('custom:')){const index=preferences.value.panels.findIndex(p=>p.id===panel.id);if(index<0)preferences.value.panels.push(panel);else preferences.value.panels[index]=panel;}
    else preferences.value.overrides[panel.id]=panel;
    savePreferences();rebuild();closeEditor();
  }catch(error){editorError.value=error.message;}
}
function removePanel(panel){
  if(panel.id.startsWith('custom:'))preferences.value.panels=(preferences.value.panels||[]).filter(p=>p.id!==panel.id);
  else preferences.value.removed=[...new Set([...(preferences.value.removed||[]),panel.id])];
  delete preferences.value.charts?.[panel.id];savePreferences();rebuild();
}
function restoreAutomatic(){preferences.value.removed=[];savePreferences();rebuild();}
const metricCount=computed(()=>groups.value.reduce((total,group)=>total+group.charts.length,0));
const chartKey=computed(()=>JSON.stringify([axis.value,selected.value,groups.value.map(g=>[g.id,g.charts.map(c=>[c.id,c.panel,c.source])])]));
const getColor=(_,index)=>colors[Math.max(0,index)%colors.length];
const media={get:()=>Promise.reject(new Error('Only scalar probes are exposed by this data adapter'))};
const storageKey=()=>`rsisys.metrics.layout.${props.state.run?.run_id}`;
function loadPreferences(){
  let saved;try{saved=JSON.parse(localStorage.getItem(storageKey()));}catch{}
  const object=value=>value&&typeof value==='object'&&!Array.isArray(value)?value:{};
  saved=object(saved);
  preferences.value={charts:object(saved.charts),opened:object(saved.opened),overrides:object(saved.overrides),
    panels:Array.isArray(saved.panels)?saved.panels:[],removed:Array.isArray(saved.removed)?saved.removed:[],
    axis:SYSTEM_AXES.includes(saved.axis)?saved.axis:'observation'};
  axis.value=preferences.value.axis;
}
function savePreferences(){try{localStorage.setItem(storageKey(),JSON.stringify(preferences.value));}catch{error.value='布局无法保存到当前浏览器';}}
function rebuild(){groups.value=buildGroups(snapshots.value,preferences.value,axis.value);}
function updateChartStatus(chart,status){preferences.value.charts ||= {};preferences.value.charts[chart.id]=status;savePreferences();rebuild();return Promise.resolve({groups:groups.value});}
function updateNamespaceStatus(opened,namespace){preferences.value.opened ||= {};preferences.value.opened[namespace.id]=opened;savePreferences();return Promise.resolve();}
function deliver(sources,cid,callback){
  const chart=groups.value.flatMap(group=>group.charts).find(chart=>chart.id===cid);if(!chart)return;
  const data=chartData(chart,snapshots.value);
  for(const source of sources)if(data[source])Promise.resolve(callback(source,data[source])).catch(()=>{});
}
function subscribe(sources,cid,callback){subscriptions.set(cid,{sources,callback});nextTick(()=>deliver(sources,cid,callback));}
function unsubscribe(_sources,cid){subscriptions.delete(cid);}
let revision=0,disposed=false;
async function update(){
  const current=++revision,base=props.state.run;if(!base)return;
  catalog.value=props.state.runs||[];
  const results=await Promise.allSettled(selected.value.map(async id=>{
    if(id===base.run_id)return base;
    const response=await fetch(`/api/runs/${encodeURIComponent(id)}`,{cache:'no-store'});
    if(!response.ok)throw new Error('无法读取所选实验');return response.json();
  }));
  if(disposed||current!==revision)return;
  snapshots.value=results.flatMap(item=>item.status==='fulfilled'?[item.value]:[]);
  error.value=results.some(item=>item.status==='rejected')?'部分实验暂时无法读取':'';
  rebuild();await nextTick();for(const [cid,{sources,callback}] of subscriptions)deliver(sources,cid,callback);
}
function select(id,enabled){selected.value=enabled?[...selected.value,id]:selected.value.filter(v=>v!==id);update();}
watch(()=>props.state.run?.run_id,()=>{closeEditor();manageOpen.value=false;loadPreferences();selected.value=[props.state.run.run_id];update();},{immediate:true});
watch(()=>props.state.run,update);watch(axis,()=>{preferences.value.axis=axis.value;savePreferences();rebuild();});
onMounted(()=>installMessage(messages));onBeforeUnmount(()=>{disposed=true;revision++;subscriptions.clear();});
</script>
