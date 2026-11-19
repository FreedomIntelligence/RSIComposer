<template>
  <LiveStage :run="run" />
  <div class="stats">
    <div v-for="card in cards" :key="card.label" class="stat">
      <div class="stat-label">{{ card.label }}</div>
      <div class="stat-value num">{{ card.value }}</div>
      <div v-if="card.foot" class="stat-foot" :class="card.tone">{{ card.foot }}</div>
    </div>
  </div>
  <section v-if="trend.length > 1" class="section">
    <SeriesChart :title="definition.label || metric"
      :sets="[{name: definition.label || metric, points: trend}]" :definition="definition" :axis-label="t('评测次序')" />
  </section>
  <section class="section">
    <SectionTitle :title="t('阶段')" :count="stageCount">
      <ActionButton v-if="stageCount" :label="t('查看轨迹')" @click="navigate('trajectory')" />
    </SectionTitle>
    <StageTable :run="run" @select="open" />
  </section>
  <section v-if="evaluations" class="section">
    <SectionTitle :title="t('评测')" :count="evaluations" />
    <EvaluationTable :run="run" />
  </section>
</template>
<script setup>
import {computed} from 'vue';
import LiveStage from '../components/LiveStage.vue';
import StageTable from '../components/StageTable.vue';
import SectionTitle from '../components/SectionTitle.vue';
import ActionButton from '../components/ActionButton.vue';
import SeriesChart from '../components/SeriesChart.vue';
import EvaluationTable from '../components/EvaluationTable.vue';
import {store, navigate} from '../app/store.js';
import {t, durationText} from '../app/i18n.js';
import {rowsOf, stagesOf, primaryMetric, bestEvaluation, evaluationTrend, metricDefinition, formatMetric, formatNumber} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const metric = computed(() => primaryMetric(props.run));
const definition = computed(() => metricDefinition(props.run, metric.value));
const trend = computed(() => metric.value ? evaluationTrend(props.run, metric.value) : []);
const stageCount = computed(() => stagesOf(props.run).length);
const evaluations = computed(() => rowsOf(props.run.evaluations).length);
const cards = computed(() => {
  const run = props.run, best = metric.value ? bestEvaluation(run, metric.value) : null, first = trend.value[0]?.y;
  // Change from the first evaluation, in the metric's own unit: the one number a reader wants next to "best".
  let foot = '', tone = '';
  if (best && trend.value.length > 1) {
    const ratio = definition.value.unit === 'percent' || definition.value.unit === 'ratio', delta = best.value - first;
    const amount = ratio ? `${formatNumber(Math.abs(delta) * 100, 2)} pt` : formatNumber(Math.abs(delta));
    foot = delta === 0 ? t('与首次评测持平') : t(delta > 0 ? '较首次评测 +{p0}' : '较首次评测 −{p0}', {p0: amount});
    tone = delta === 0 ? '' : (delta > 0) === (definition.value.direction !== 'lower') ? 'good' : 'bad';
  }
  return [
    {label: metric.value ? t('最佳 {p0}', {p0: definition.value.label || metric.value}) : t('主要指标'),
      value: best ? formatMetric(best.value, definition.value) : '—', foot, tone},
    {label: t('评测'), value: String(evaluations.value)},
    {label: t('阶段'), value: String(stageCount.value)},
    {label: t('运行时长'), value: durationText(run.workflow?.elapsed_seconds)},
  ];
});
function open(id) {
  store.decisionId = id;
  navigate('trajectory');
}
</script>
