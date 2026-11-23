<template>
  <LiveStage :run="run" />
  <section v-if="sealed.length && reported?.kind !== 'bin_packing'" class="section results-section">
    <SectionTitle :title="t('封存评测')" />
    <div class="benchmark-panels">
      <ResultPanel v-for="panel in sealed" :key="panel.key" :title="panelTitle(panel, '封存集')"
        :context="panel.definition.label" v-bind="comparison(panel, panel.initial, panel.selected, t('最终选中版本'))" />
    </div>
  </section>
  <section v-if="reported?.kind === 'promptpg'" class="section">
    <SectionTitle :title="t('TabMWP · 封存集策略对照')" />
    <DataTable :headers="[t('策略'), t('答对 / 总数'), t('准确率')]" :numeric="[1, 2]"
      :rows="reported.comparison.map(score => [t(armLabels[score.arm]), `${score.correct}/${score.total}`, tallyRate(score)])" />
  </section>
  <section v-else-if="reported?.kind === 'reflexion'" class="section">
    <SectionTitle :title="t('HumanEval · 隐藏测试')" />
    <dl class="benchmark-scores">
      <div><dt>{{ t('通过题数') }}</dt><dd class="num">
        {{ reported.score.correct }}/{{ reported.score.total }} <small>{{ tallyRate(reported.score) }}</small>
      </dd></div>
    </dl>
    <details v-if="reported.tasks.length" class="technical">
      <summary>{{ t('逐题结果') }} · {{ reported.tasks.length }}/{{ reported.score.total }}</summary>
      <DataTable :headers="[t('题号'), t('尝试次数'), t('结果')]" :numeric="[0, 1]"
        :rows="reported.tasks.map(([index, task]) => [index + 1, task.attempt, t(task.passed ? '通过' : '未通过')])" />
    </details>
  </section>
  <section v-else-if="reported?.kind === 'bin_packing'" class="section">
    <SectionTitle :title="t('Bin Packing · 封存集')" />
    <ResultPanel :title="t('总用箱数（越少越好）')" :change="packingDelta"
      :tone="reported.delta < 0 ? 'good' : reported.delta > 0 ? 'bad' : ''"
      :context="t('{p0} 个实例', {p0: formatNumber(reported.instances)})"
      :rows="reported.strategies.map(({arm, score}) => ({label: t(packingArms[arm]), value: formatNumber(score.bins),
        amount: score.bins, chosen: arm === 'selected'}))" :maximum="reported.max" :scale-end="formatNumber(reported.max)" />
  </section>
  <div v-if="reported?.kind !== 'bin_packing' && (performance || selected.length)" class="section benchmark-panels">
    <section v-if="performance" class="benchmark-panel">
      <SectionTitle :title="panelTitle(performance, '搜索集')" />
      <SeriesChart v-if="performance.points.length > 1" :title="t('采纳版本表现')"
        :sets="[{name: metricLabel, points: performance.points}]" :definition="performance.definition" :axis-label="t('版本次序')" />
      <dl v-else class="benchmark-scores">
        <div><dt>{{ t('当前版本') }}</dt><dd class="num">{{ formatMetric(performance.value, performance.definition) }}</dd></div>
      </dl>
    </section>
    <ResultPanel v-for="panel in selected" :key="panel.key" :title="panelTitle(panel, '评测')"
      :context="panel.definition.label || panel.metric"
      v-bind="comparison(panel, panel.initial, panel,
        [t(actionLabels[panel.action] || '已选定'), panel.version].filter(Boolean).join(' · '))" />
  </div>
  <dl class="run-stats">
    <div v-for="card in cards" :key="card.label" class="stat">
      <dt class="stat-label">{{ card.label }}</dt>
      <dd class="stat-value num">{{ card.value }}</dd>
      <dd v-if="card.foot" class="stat-foot" :class="card.tone">{{ card.foot }}</dd>
    </div>
  </dl>
  <section class="section">
    <SectionTitle :title="t('阶段')" :count="stageCount">
      <ActionButton v-if="stageCount" :label="t('查看轨迹')" @click="navigate('trajectory')" />
    </SectionTitle>
    <StageTable :run="run" @select="open" />
  </section>
  <details v-if="evaluations" class="section technical">
    <summary>{{ t('评测回执') }} · {{ evaluations }}</summary>
    <EvaluationTable :run="run" />
  </details>
</template>
<script setup>
import {computed} from 'vue';
import LiveStage from '../components/LiveStage.vue';
import StageTable from '../components/StageTable.vue';
import SectionTitle from '../components/SectionTitle.vue';
import ActionButton from '../components/ActionButton.vue';
import SeriesChart from '../components/SeriesChart.vue';
import EvaluationTable from '../components/EvaluationTable.vue';
import DataTable from '../components/DataTable.vue';
import ResultPanel from '../components/ResultPanel.vue';
import {store, navigate} from '../app/store.js';
import {t, durationText} from '../app/i18n.js';
import {rowsOf, stagesOf, incumbentPerformance, selectedPerformance, sealedPerformance, methodPerformance,
  benchmarkLabel, formatMetric, formatNumber} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const performance = computed(() => incumbentPerformance(props.run));
const selected = computed(() => performance.value ? [] : selectedPerformance(props.run));
const sealed = computed(() => sealedPerformance(props.run));
const reported = computed(() => methodPerformance(props.run));
const armLabels = {trained: '训练后的选择器', random: '随机选择', nearest: '最近邻选择'};
const packingArms = {first_fit: 'First Fit', best_fit: 'Best Fit', selected: '最终选中版本'};
const packingDelta = computed(() => {
  const delta = reported.value?.delta;
  if (delta === 0) return t('与 Best Fit 持平');
  return t(delta < 0 ? '比 Best Fit 少用 {p0} 个箱子' : '比 Best Fit 多用 {p0} 个箱子',
    {p0: formatNumber(Math.abs(delta))});
});
const tallyRate = score => formatMetric(score.correct / score.total, {unit: 'ratio'});
const actionLabels = {kept: '保留原版本', selected: '已选定', applied: '已生效'};
const metricLabel = computed(() => performance.value?.definition.label || performance.value?.metric);
const stageCount = computed(() => stagesOf(props.run).length);
const evaluations = computed(() => rowsOf(props.run.evaluations).length);
const cards = computed(() => {
  const run = props.run;
  const current = performance.value || (selected.value.length === 1 ? selected.value[0] : null);
  let foot = current ? [benchmarkLabel(current.benchmark), splitLabel(current.split)].filter(Boolean).join(' · ') : '';
  let tone = '';
  const baseline = current?.baseline ?? current?.initial?.value;
  if (performance.value && Number.isFinite(baseline)) {
    const {change, tone: direction} = changeFrom(current.definition, baseline, current.value);
    foot = [foot, change].filter(Boolean).join(' · ');
    tone = direction;
  }
  const currentCard = performance.value && current ? {
    label: `${t(actionLabels[current.action] || '当前版本')} · ${current.definition.label || current.metric}`,
    value: formatMetric(current.value, current.definition), foot, tone,
  } : null;
  return [
    ...(currentCard ? [currentCard] : []),
    {label: t('评测回执'), value: String(evaluations.value)},
    {label: t('阶段'), value: String(stageCount.value)},
    {label: t('运行时长'), value: durationText(run.workflow?.elapsed_seconds)},
  ];
});
function splitLabel(split) {
  return split === 'search' ? t('搜索集') : split === 'heldout' || split === 'test' ? t('封存集')
    : ['dev', 'development', 'validation'].includes(split) ? t('验证集') : split;
}
function panelTitle(panel, fallback) {
  return [benchmarkLabel(panel.benchmark), splitLabel(panel.split) || t(fallback)].filter(Boolean).join(' · ');
}
const isRatio = definition => ['ratio', 'percent'].includes(definition.unit);
function changeFrom(definition, baseline, value) {
  if (!Number.isFinite(baseline) || !Number.isFinite(value)) return {};
  const delta = value - baseline;
  const amount = isRatio(definition) ? `${formatNumber(Math.abs(delta) * 100, 2)} pt` : formatNumber(Math.abs(delta));
  const change = delta === 0 ? t('与初始版本持平') : t(delta > 0 ? '较初始版本 +{p0}' : '较初始版本 −{p0}', {p0: amount});
  const tone = delta && ['higher', 'lower'].includes(definition.direction)
    ? (delta > 0) === (definition.direction === 'higher') ? 'good' : 'bad' : '';
  return {change, tone};
}
function comparison({definition}, initial, chosen, chosenLabel) {
  const rows = [[initial, t('初始版本'), false], [chosen, chosenLabel, true]].filter(([score]) => score)
    .map(([score, label, selected]) => ({label, value: formatMetric(score.value, definition), amount: score.value,
      detail: score.correct != null && score.count != null ? `${score.correct}/${score.count}` : '', chosen: selected}));
  const maximum = isRatio(definition) && rows.some(({amount}) => amount >= 0 && amount <= 1) ? 1 : 0;
  return {rows, maximum, scaleStart: '0%', scaleEnd: '100%', ...changeFrom(definition, initial?.value, chosen?.value)};
}
function open(id) {
  store.decisionId = id;
  navigate('trajectory');
}
</script>
<style scoped>
.benchmark-panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr)); gap: 24px; }
.benchmark-panel { min-width: 0; }
.benchmark-scores { margin: 0; border-top: 1px solid var(--line); }
.benchmark-scores > div { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px 12px;
  padding: 14px 0; border-bottom: 1px solid var(--line); }
.benchmark-scores dt { overflow-wrap: anywhere; }
.benchmark-scores dd { display: flex; align-items: baseline; gap: 8px; margin: 0; font-weight: 500; }
.benchmark-scores small { color: var(--muted); font-weight: 400; }
</style>
