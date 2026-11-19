<template>
  <LiveStage :run="run" />
  <DecisionGraph v-if="graphRows.length > 1" :run-id="run.run_id" :rows="graphRows" :selected="row.decision_id || ''"
    @select="row => store.decisionId = decisionKey(row)" />
  <div class="split">
    <section><StageTable :run="run" :selected="current?.id || ''" @select="id => store.decisionId = id" /></section>
    <aside v-if="current" class="detail">
      <div class="detail-head"><h2>{{ current.label }}</h2><StatusBadge :status="row.status" /></div>
      <p class="muted">{{ [dateText(row.started_at), durationText(stageSeconds(row))].filter(text => text !== '—').join(' · ') }}</p>
      <div v-if="row.reason || row.strategy" class="detail-block"><h3>{{ t('模型判断') }}</h3><p class="verbatim">{{ row.reason || row.strategy }}</p></div>
      <div v-if="row.feedback_summary" class="detail-block"><h3>{{ t('结果摘要') }}</h3><p class="verbatim">{{ row.feedback_summary }}</p></div>
      <div v-if="row.error" class="detail-block"><h3>{{ t('错误') }}</h3><p class="verbatim bad">{{ row.error }}</p></div>
      <div v-if="metrics.length" class="detail-block"><h3>{{ t('指标') }}</h3>
        <dl class="facts"><template v-for="[key, value] in metrics" :key="key"><dt>{{ key }}</dt><dd class="num">{{ value }}</dd></template></dl></div>
      <div v-if="changes.length" class="detail-block"><h3>{{ t('提交的改动') }}</h3>
        <DataTable :headers="[t('字段'), t('改动前'), t('改动后')]" :rows="changes" /></div>
      <div v-for="relation in relations" :key="relation.label" class="detail-block"><h3>{{ relation.label }}</h3>
        <button v-for="item in relation.items" :key="item.id" type="button" class="chip" @click="store.decisionId = item.id">{{ item.label }}</button></div>
      <div v-if="charts.length" class="detail-block chart-stack">
        <SeriesChart v-for="chart in charts" :key="chart.metric + chart.axis" :title="chart.definition.label || chart.metric"
          :value="formatMetric(chart.last, chart.definition)" :sets="chart.sets" :definition="chart.definition"
          :axis-label="t(chart.axis === 'step' ? '步数' : '运行时间 / 秒')" />
      </div>
      <div class="detail-actions"><ActionButton :label="t('查看该阶段日志')" @click="showLogs" /></div>
      <details class="technical"><summary>{{ t('技术标识') }}</summary>
        <dl class="facts mono"><template v-for="[key, value] in identifiers" :key="key"><dt>{{ key }}</dt><dd>{{ value }}</dd></template></dl></details>
    </aside>
  </div>
</template>
<script setup>
import {computed} from 'vue';
import LiveStage from '../components/LiveStage.vue';
import StageTable from '../components/StageTable.vue';
import StatusBadge from '../components/StatusBadge.vue';
import ActionButton from '../components/ActionButton.vue';
import DataTable from '../components/DataTable.vue';
import SeriesChart from '../components/SeriesChart.vue';
import DecisionGraph from '../components/DecisionGraph.vue';
import {store, navigate} from '../app/store.js';
import {t, dateText, durationText} from '../app/i18n.js';
import {rowsOf, stagesOf, decisionKey, stageSeconds, flattenMetrics, metricDefinition, formatMetric, metricSeries} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const RELATIONS = [['depends_on_decision_ids', '依赖'], ['adopted_decision_ids', '采纳'], ['source_decision_ids', '参考']];
const stages = computed(() => stagesOf(props.run));
const graphRows = computed(() => {
  const groups = new Map();
  for (const {row, label, id} of stages.value) {
    if (row.phase === 'host_finish') continue;
    const key = row.decision_id || id;
    if (!groups.has(key)) groups.set(key, {row, labels: []});
    groups.get(key).labels.push(label);
  }
  return [...groups.values()].map(({row, labels}) => ({...row, label: labels.join(' · ')}));
});
const current = computed(() => stages.value.find(stage => stage.id === store.decisionId) || stages.value.at(-1));
const row = computed(() => current.value?.row || {});
const metrics = computed(() => Object.entries(flattenMetrics(row.value.metrics))
  .map(([key, value]) => [metricDefinition(props.run, key).label || key, formatMetric(value, metricDefinition(props.run, key))]));
const changes = computed(() => rowsOf(row.value.changes).map(change => [change.field, change.before, change.after]));
const relations = computed(() => RELATIONS.map(([field, label]) => ({label: t(label),
  items: (Array.isArray(row.value[field]) ? row.value[field] : []).map(id => stages.value.find(stage =>
    stage.row.decision_id === id || stage.id === id)).filter(Boolean)}))
  .filter(relation => relation.items.length));
// Curves only: a stage that logged one or two values has them in its metrics, not as a chart.
const charts = computed(() => current.value ? metricSeries(props.run, current.value.id)
  .filter(chart => chart.sets.some(set => set.points.length > 2)) : []);
const identifiers = computed(() => ['call_id', 'decision_id', 'directive_id', 'job_id', 'candidate_id', 'model', 'kind']
  .filter(key => typeof row.value[key] === 'string' && row.value[key]).map(key => [key, row.value[key]]));
function showLogs() {
  store.stageLogs = current.value.id;
  navigate('logs');
}
</script>
