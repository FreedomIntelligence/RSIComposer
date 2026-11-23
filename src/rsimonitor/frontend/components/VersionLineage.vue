<template>
  <section class="lineage">
    <header class="lineage-head">
      <h2>{{ t(showGraph ? '版本谱系' : '候选版本') }}</h2>
      <div v-if="currentId" class="lineage-key"><span></span>{{ t('已生效') }}</div>
    </header>
    <p v-if="cyclic" class="muted lineage-warning">{{ t('版本关系有循环，已显示候选列表') }}</p>
    <DecisionGraph v-if="showGraph" :run-id="`${run.run_id}:versions`" :rows="graphRows" :lineage="graph"
      :selected="keyOf(active?.version_id)" @select="row => selected = row.version_id" />
    <div v-else class="lineage-groups">
      <div v-for="[round, rows] in groups" :key="round" class="lineage-group">
        <h3>{{ round === 0 ? t('初始版本') : round === -1 ? t('候选版本') : t('第 {p0} 轮', {p0: round}) }}</h3>
        <button v-for="row in rows" :key="row.version_id" type="button" class="version" :class="styleOf(row)"
          :aria-pressed="active?.version_id === row.version_id" @click="selected = row.version_id">
          <span>{{ labelOf(row) }}</span><strong>{{ summary(row) }}</strong>
        </button>
      </div>
    </div>
    <div v-if="active" class="lineage-detail">
      <strong>{{ labelOf(active) }}</strong>
      <span v-if="active.version_id === currentId" class="selection-label">{{ t('当前版本') }}</span>
      <span v-else-if="adopted.has(active.version_id)" class="selection-label">{{ t('已生效') }}</span>
      <span v-if="latestActions.get(active.version_id)?.action === 'kept'" class="selection-label">{{ t('保留原版本') }}</span>
      <span v-else-if="latestActions.get(active.version_id)?.action === 'selected'" class="selection-label">{{ t('已选定') }}</span>
      <span v-if="active.parent_ids === undefined" class="muted">{{ t('未记录父版本') }}</span>
      <span v-else-if="active.parent_ids.some(id => !byId.has(id))" class="muted">{{ t('部分父版本不在当前记录中') }}</span>
      <VersionContent :key="active.version_id" :run-id="run.run_id"
        :version="{...active, content_available: published.has(active.version_id)}" :labels="versionLabels" />
      <div v-if="measurements(active).length" class="lineage-measurements">
        <div v-for="item in measurements(active)" :key="item.evaluation_id" class="lineage-measurement">
          <span>{{ evaluationLabel(item) }}</span><strong>{{ score(item) }}</strong>
          <button v-if="stageId(item)" type="button" class="chip" @click="$emit('stage', stageId(item))">{{ t('查看评测阶段') }}</button>
        </div>
      </div>
      <div v-if="sources.length" class="lineage-sources">
        <span class="muted">{{ t('参考评测') }}</span>
        <button v-for="stage in sources" :key="stage.id" type="button" class="chip" @click="$emit('stage', stage.id)">{{ stage.label }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import {computed, ref} from 'vue';
import DecisionGraph from './DecisionGraph.vue';
import VersionContent from './VersionContent.vue';
import {t} from '../app/i18n.js';
import {formatMetric, metricDefinition} from '../app/insight.js';
import RSIDecisions from '../app/decisions.js';

const props = defineProps({run: Object, stages: Array});
defineEmits(['stage']);
const selected = ref('');
const versions = computed(() => props.run.versions || []);
const published = computed(() => new Set((props.run.observation_events || []).filter(row => row.content_available).map(row => row.subject)));
const byId = computed(() => new Map(versions.value.map(row => [row.version_id, row])));
const selections = computed(() => (props.run.version_selections || []).filter(row => (row.scope || 'run') === 'run'));
const applied = computed(() => selections.value.filter(item => item.action === 'applied'));
const latestActions = computed(() => new Map(selections.value.map(item => [item.version_id, {...item, action: item.action || 'selected'}])));
const adopted = computed(() => new Set(applied.value.map(item => item.version_id)));
const currentId = computed(() => applied.value.at(-1)?.version_id);
const selectedId = computed(() => selections.value.at(-1)?.version_id);
const active = computed(() => byId.value.get(selected.value) || byId.value.get(currentId.value)
  || byId.value.get(selectedId.value) || versions.value[0]);
const evaluations = computed(() => new Map((props.run.evaluations || []).map(row => [row.evaluation_id, row])));
const keyOf = id => id ? `version:${id}` : '';
const predecessor = computed(() => new Map(applied.value.slice(1).map((item, i) =>
  [item.version_id, applied.value[i].version_id])));
const graphRows = computed(() => versions.value.map(row => {
  const parentIds = (row.parent_ids || []).filter(id => byId.value.has(id));
  const previous = predecessor.value.get(row.version_id);
  return {version_id: row.version_id, decision_id: keyOf(row.version_id),
    label: [labelOf(row), summary(row)].filter(Boolean).join(' · '),
    version_adopted: adopted.value.has(row.version_id),
    depends_on_decision_ids: parentIds.map(keyOf),
    adopted_decision_ids: parentIds.includes(previous) ? [keyOf(previous)] : []};
}));
const graph = computed(() => {
  // Keep the chosen branch straight when there is a confirmed parent or just one parent.
  const previous = new Map(graphRows.value.map(row => [row.decision_id, row.adopted_decision_ids[0]
    || (row.depends_on_decision_ids.length === 1 ? row.depends_on_decision_ids[0] : undefined)]));
  const mainline = new Set();
  for (let id = keyOf(currentId.value || selectedId.value); id && !mainline.has(id); id = previous.get(id)) mainline.add(id);
  const model = RSIDecisions.layout(graphRows.value, {mainline: [...mainline]});
  for (const edge of model.edges) if (edge.from === 'root') edge.kind = 'root';
  return model;
});
const cyclic = computed(() => graph.value.issues.some(issue => issue.type === 'raw-cycle'));
const showGraph = computed(() => !cyclic.value && (
  graph.value.edges.some(edge => edge.from !== 'root') ||
  (versions.value.length === 1 && versions.value[0].parent_ids?.length === 0)
));
const groups = computed(() => {
  const rounds = new Map();
  for (const row of versions.value) {
    const round = row.round ?? -1;
    if (!rounds.has(round)) rounds.set(round, []);
    rounds.get(round).push(row);
  }
  return [...rounds].sort(([a], [b]) => (a < 0 ? Infinity : a) - (b < 0 ? Infinity : b));
});
const sources = computed(() => (active.value?.source_decision_ids || []).map(id =>
  props.stages.find(stage => stage.row.decision_id === id)).filter(Boolean));
const labelOf = row => row.label === 'Baseline' ? t('初始版本') : row.label;
const versionLabels = computed(() => new Map(versions.value.map(row => [row.version_id, labelOf(row)])));
const styleOf = row => ({adopted: adopted.value.has(row.version_id), selected: active.value?.version_id === row.version_id});
const measurements = row => (row.evaluation_ids || []).map(id => evaluations.value.get(id)).filter(Boolean);
const stageId = item => props.stages.find(stage => stage.id === item.call_id || stage.id === item.evaluation_id)?.id;
function evaluationLabel(item) {
  if (item.benchmark) return item.benchmark;
  if ('aime2024_accuracy' in (item.metrics || {})) return 'AIME 2024';
  if ('aime2025_accuracy' in (item.metrics || {})) return 'AIME 2025';
  return {search: t('搜索集'), heldout: t('封存集'), test: t('封存集'), dev: t('验证集')}[item.split] || item.split || t('评测');
}
function score(item) {
  const metrics = item?.metrics || {};
  for (const prefix of ['aime2024_', 'aime2025_', '']) {
    if (Number.isFinite(metrics[prefix + 'correct']) && Number.isFinite(metrics[prefix + 'count'])) {
      return `${metrics[prefix + 'correct']}/${metrics[prefix + 'count']}`;
    }
  }
  const key = [props.run.primary_metric, 'accuracy', ...Object.keys(metrics)].find(name => Number.isFinite(metrics[name]));
  return key ? formatMetric(metrics[key], metricDefinition(props.run, key)) : '';
}
function summary(row) {
  const items = measurements(row);
  return score(items.find(item => item.split === 'search') || items[0]);
}
</script>

<style scoped>
.lineage { margin: 0 0 20px; border-block: 1px solid var(--line, #dce2e9); }
.lineage-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 0; }
.lineage-head h2 { margin: 0; font-size: 1rem; }
.lineage-key { display: flex; align-items: center; gap: 7px; font-size: .82rem; }
.lineage-key span { width: 8px; height: 8px; border-radius: 50%; background: var(--orange, #b45b35); }
.lineage :deep(.trajectory-shell) { margin-bottom: 0; }
.version { display: flex; align-items: center; justify-content: space-between; gap: 6px; min-width: 132px; min-height: 48px;
  padding: 6px 9px; border: 1px solid var(--line, #cbd5df); border-radius: 6px; background: var(--surface, #fff); color: inherit; text-align: left; }
.version span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: .83rem; }
.version strong { font-size: .83rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
.version.adopted { border-color: var(--orange, #b45b35); }
.version.selected { outline: 2px solid var(--accent, #2158a0); outline-offset: -2px; }
.lineage-warning { margin: 0 0 8px; }
.lineage-groups { display: grid; gap: 12px; padding-block: 8px; }
.lineage-group { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
.lineage-group h3 { min-width: 90px; margin: 0; font-size: .85rem; }
.lineage-detail { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; padding: 12px 0; }
.selection-label { color: var(--orange, #b45b35); font-size: .82rem; font-weight: 600; }
.lineage-measurements, .lineage-sources { flex-basis: 100%; display: flex; align-items: center; flex-wrap: wrap; gap: 8px 16px; }
.lineage-measurement { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; font-size: .84rem; }
</style>
