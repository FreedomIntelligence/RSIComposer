<template>
  <header class="page-head"><h1>{{ t('对比') }}</h1></header>
  <div class="chips">
    <button v-for="row in store.runs" :key="row.run_id" type="button" class="chip" :class="{active: store.compareIds.includes(row.run_id)}"
      :aria-pressed="store.compareIds.includes(row.run_id)" @click="toggleCompare(row.run_id, !store.compareIds.includes(row.run_id))">{{ runLabel(row) }}</button>
  </div>
  <div v-if="metricKeys.length" class="toolbar">
    <label>{{ t('指标') }}<select v-model="metric"><option v-for="key in metricKeys" :key="key" :value="key">{{ definition(key).label || key }}</option></select></label>
    <label v-if="splits.length > 1">{{ t('数据划分') }}<select v-model="split"><option :value="ALL">{{ t('全部') }}</option>
      <option v-for="value in splits" :key="value" :value="value">{{ value || t('未标注') }}</option></select></label>
    <label>{{ t('横轴') }}<select v-model="axis"><option value="index">{{ t('评测次序') }}</option><option value="round">{{ t('轮次') }}</option>
      <option value="elapsed_seconds">{{ t('运行时间') }}</option></select></label>
  </div>
  <section v-if="sets.length" class="section">
    <SeriesChart :key="metric + axis + split" :title="definition(metric).label || metric" :value="best" :sets="sets"
      :definition="definition(metric)" :axis-label="t(AXIS_LABELS[axis])" />
  </section>
  <DataTable :headers="[t('实验'), t('评测'), definition(metric).label || metric || t('指标')]" :numeric="[2]" :rows="table"
    :empty="t(store.compareIds.length ? '所选实验没有可对比的评测' : '选择上方的实验开始对比')" />
</template>
<script setup>
import {computed, ref, watchEffect} from 'vue';
import DataTable from '../components/DataTable.vue';
import SeriesChart from '../components/SeriesChart.vue';
import {store, compared, toggleCompare} from '../app/store.js';
import {t} from '../app/i18n.js';
import {runLabel, evaluationRows, evaluationMetricKeys, metricDefinition, formatMetric, primaryMetric, splitsOf, stageLabel,
  UNLABELLED} from '../app/insight.js';
const ALL = '\u0000all', AXIS_LABELS = {index: '评测次序', round: '轮次', elapsed_seconds: '运行时间 / 秒'};
const metric = ref(''), split = ref(ALL), axis = ref('index');
const runs = computed(() => store.compareIds.map(id => compared.value.get(id)).filter(Boolean));
const metricKeys = computed(() => evaluationMetricKeys(runs.value));
const splits = computed(() => splitsOf(runs.value));
// Units and labels come from the first compared run that declares the metric.
const definition = key => metricDefinition(runs.value.find(run => run.metric_definitions?.[key]) || runs.value[0], key);
watchEffect(() => {
  if (!metricKeys.value.includes(metric.value)) metric.value = runs.value.map(primaryMetric).find(Boolean) || metricKeys.value[0] || '';
  if (split.value !== ALL && !splits.value.includes(split.value)) split.value = ALL;
});
const selected = run => evaluationRows(run).filter(row => typeof row.flat[metric.value] === 'number'
  && (split.value === ALL || (typeof row.split === 'string' ? row.split : UNLABELLED) === split.value));
const sets = computed(() => runs.value.map(run => ({name: runLabel(run), points: selected(run)
  .map((row, index) => ({x: axis.value === 'index' ? index + 1 : row[axis.value], y: row.flat[metric.value]}))
  .filter(point => Number.isFinite(point.x)).sort((a, b) => a.x - b.x)})).filter(set => set.points.length));
const best = computed(() => {
  const values = sets.value.flatMap(set => set.points.map(point => point.y));
  if (!values.length) return '';
  return formatMetric(definition(metric.value).direction === 'lower' ? Math.min(...values) : Math.max(...values), definition(metric.value));
});
const table = computed(() => runs.value.flatMap(run => selected(run).map((row, index) => [runLabel(run),
  row.label || row.protocol || stageLabel(run, row.call_id || row.decision_id) || `#${index + 1}`, formatMetric(row.flat[metric.value], definition(metric.value))])));
</script>
