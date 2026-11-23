<template>
  <details v-if="recent.length || run.observation_status === 'degraded'" class="observation-panel technical"
    :open="opened" @toggle="opened = $event.target.open">
    <summary>{{ t(titles[category]) }} <span v-if="run.observation_status === 'degraded'" class="bad">· {{ t('观测不完整') }}</span></summary>
    <div v-if="opened" class="observation-body">
      <div class="observation-tools">
        <ActionButton icon="list" :label="t(archive ? '最新事件' : '浏览事件记录')" :disabled="busy" @click="toggleArchive" />
        <template v-if="archive">
          <ActionButton icon="chevron-left" :label="t('上一页')" :disabled="busy || !past.length" @click="previous" />
          <ActionButton icon="chevron-right" :label="t('下一页')" :disabled="busy || nextCursor === null" @click="load(nextCursor)" />
        </template>
        <span v-if="error" class="bad" role="status">{{ t('事件记录加载失败') }}</span>
        <span v-if="run.observation_dropped" class="muted">{{ t('已丢弃 {p0} 条观测', {p0: run.observation_dropped}) }}</span>
      </div>
      <div class="observation-list" :aria-busy="busy">
        <button v-for="event in rows" :key="event.sequence" type="button" class="observation-row"
          :aria-pressed="active?.sequence === event.sequence" @click="selected = event.sequence">
          <span class="seq num">{{ event.sequence }}</span><span>{{ title(event) }}</span>
          <span class="muted">{{ context(event) }}</span>
        </button>
      </div>
      <section v-if="active" class="observation-detail">
        <header><h3>{{ title(active) }}</h3><time class="muted" :datetime="active.timestamp">{{ dateText(active.timestamp) }}</time></header>
        <dl class="facts">
          <template v-for="[key, value] in fields" :key="key"><dt>{{ key }}</dt><dd>{{ value }}</dd></template>
        </dl>
        <dl v-if="links.length" class="facts">
          <template v-for="[relation, ids] in links" :key="relation">
            <dt>{{ t(relations[relation] || relation) }}</dt><dd class="mono">{{ ids.join(', ') }}</dd>
          </template>
        </dl>
        <DecisionGraph v-if="showLineage" :run-id="`${run.run_id}:${active.kind}`" :rows="lineageRows"
          :selected="active.subject" @select="row => selected = row.sequence" />
        <SeriesChart v-if="chart" :title="chart.title" :sets="chart.sets" :definition="chart.definition"
          :axis-label="t('事件序号')" />
        <VersionContent v-if="version" :run-id="run.run_id" :version="version" :labels="versionLabels" />
        <ActionButton v-if="active.call_id && hasStage" :label="t('查看关联阶段')" @click="$emit('stage', active.call_id)" />
      </section>
    </div>
  </details>
</template>

<script setup>
import {computed, onBeforeUnmount, ref, watch} from 'vue';
import {dateText, t} from '../app/i18n.js';
import {formatMetric, metricDefinition, stagesOf} from '../app/insight.js';
import ActionButton from './ActionButton.vue';
import DecisionGraph from './DecisionGraph.vue';
import SeriesChart from './SeriesChart.vue';
import VersionContent from './VersionContent.vue';

const props = defineProps({run: {type: Object, required: true}, category: {type: String, required: true}, initiallyOpen: Boolean});
defineEmits(['stage']);
const titles = {decision: '决策与演化事件', quality: '评测事件', execution: '执行事件'};
const names = {'gepa.prompt': 'GEPA 解题提示词', 'gepa.proposal': 'GEPA 改写尝试',
  'gepa.population': 'GEPA 候选记录', 'gepa.screening': 'GEPA 候选筛选',
  'host.evaluated': '评测回执', 'host.operation': '操作回执',
  'host.measurement': '执行指标', 'host.version_evaluated': '版本评测引用', 'host.version_selected': '版本处置'};
const relations = {derived_from: '父版本', evidence: '参考', evaluates: '评测', content: '版本内容',
  selected: '已选定', applied: '已生效', kept: '保留原版本'};
const screeningStatuses = {no_usable_feedback: '跳过：没有可用反馈', perfect_feedback: '跳过：可用反馈全部答对',
  empty_reflection: '跳过：反思未返回文本', rejected: '小批得分未达到门槛',
  validating: '通过小批筛选，正在完整验证', admitted: '已进入候选池'};
const opened = ref(props.initiallyOpen), archive = ref(false), page = ref([]), selected = ref(null), busy = ref(false), error = ref(false);
const past = ref([]), cursor = ref(0), nextCursor = ref(null);
let request;
const recent = computed(() => (props.run.observation_events || []).filter(row => row.category === props.category));
const available = computed(() => archive.value ? page.value : recent.value);
const rows = computed(() => archive.value ? available.value : available.value.slice(-20).reverse());
const active = computed(() => rows.value.find(row => row.sequence === selected.value) || rows.value[0]);
const title = row => t(names[row.kind] || row.kind);
const context = row => row.kind === 'gepa.screening' ? [t('第 {p0} 轮', {p0: row.fields.round}),
  t(screeningStatuses[row.fields.status] || row.fields.status)].join(' · ') : [row.fields?.benchmark, row.fields?.split,
  relations[row.fields?.action] ? t(relations[row.fields.action]) : row.fields?.status,
  row.scope === 'run' || row.scope === row.call_id ? '' : row.scope].filter(Boolean).join(' · ');
const fields = computed(() => {
  const row = active.value;
  if (row?.kind === 'gepa.screening') {
    const f = row.fields;
    return [[t('筛选结果'), t(screeningStatuses[f.status] || f.status)],
      [t('改进方式'), t(f.strategy === 'merge' ? '合并' : '反思改写')],
      [t('小批所在集合'), t(f.split === 'validation' ? '验证集' : '搜索集')],
      [t('小批样本数'), f.batch_size],
      [t('可用反馈数'), f.usable_feedback === undefined ? undefined : `${f.usable_feedback} / ${f.batch_size}`],
      ...[['父代答对', f.parent_correct], ['第二个父代答对', f.second_parent_correct], ['子代答对', f.child_correct]]
        .map(([label, count]) => [t(label), count === undefined ? undefined : `${count} / ${f.batch_size}`]),
      [t('完整验证答对'), f.validation_correct === undefined ? undefined : `${f.validation_correct} / ${f.validation_size}`],
      [t('候选池索引'), f.pool_index]].filter(([, value]) => value !== undefined);
  }
  return Object.entries(row?.fields || {}).flatMap(([key, value]) => typeof value === 'object'
    ? Object.entries(value).map(([metric, n]) => [metric, formatMetric(n, metricDefinition(props.run, metric))]) : [[key, value]]);
});
const links = computed(() => Object.entries(active.value?.links || {}).filter(([, ids]) => ids.length));
const byId = computed(() => new Map((props.run.versions || []).map(row => [row.version_id, row])));
const versionLabels = computed(() => new Map([...byId.value].map(([id, row]) => [id, row.label])));
const version = computed(() => {
  const row = active.value;
  const id = row?.links?.content?.[0] || (row?.content_available ? row.subject : null);
  return id ? {...byId.value.get(id), version_id: id, content_available: row.content_available} : null;
});
const hasStage = computed(() => props.category === 'decision' && stagesOf(props.run).some(stage => stage.id === active.value?.call_id));
const lineageRows = computed(() => {
  if (active.value?.view !== 'lineage') return [];
  const entries = available.value.filter(row => row.kind === active.value.kind && row.subject);
  const subjects = new Map(entries.map(row => [row.subject, row]));
  return [...subjects.values()].map(row => ({decision_id: row.subject, sequence: row.sequence,
    label: `${title(row)} · ${row.subject.slice(0, 8)}`,
    depends_on_decision_ids: (row.links?.derived_from || []).filter(id => id !== row.subject && subjects.has(id))}));
});
const showLineage = computed(() => lineageRows.value.some(row => row.depends_on_decision_ids.length));
const chart = computed(() => {
  const row = active.value;
  if (row?.view !== 'series') return null;
  const {metrics = {}, benchmark, split, protocol, scorer, series, primary_metric: primary} = row.fields || {};
  const key = [primary, 'accuracy', 'score', ...Object.keys(metrics)].find(key => Number.isFinite(metrics[key]));
  if (!key) return null;
  const points = available.value.filter(event => event.kind === row.kind && event.fields?.benchmark === benchmark
    && event.fields?.split === split && event.fields?.protocol === protocol && event.fields?.scorer === scorer
    && event.fields?.series === series && (props.category === 'quality' || event.scope === row.scope))
    .filter(event => Number.isFinite(event.fields?.metrics?.[key]))
    .map(event => ({x: event.sequence, y: event.fields.metrics[key]}));
  return points.length < 2 ? null : {title: [benchmark, split, key].filter(Boolean).join(' · '),
    sets: [{name: key, points}], definition: metricDefinition(props.run, key)};
});
function reset() {
  request?.abort();
  busy.value = error.value = archive.value = false;
  page.value = []; past.value = []; cursor.value = 0; nextCursor.value = selected.value = null;
}
async function load(start, direction = 'next') {
  request?.abort();
  const controller = request = new AbortController();
  busy.value = true; error.value = false;
  try {
    const query = new URLSearchParams({cursor: String(start), category: props.category});
    const response = await fetch(`/api/runs/${encodeURIComponent(props.run.run_id)}/events?${query}`, {signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (controller.signal.aborted) return;
    if (data.schema_version !== 1 || !Array.isArray(data.events)) throw new Error('Invalid observation page');
    if (direction === 'previous') past.value.pop();
    else if (direction === 'next') past.value.push(cursor.value);
    archive.value = true; page.value = data.events; cursor.value = start;
    nextCursor.value = data.next_cursor; selected.value = null;
  } catch {
    if (!controller.signal.aborted) error.value = true;
  } finally {
    if (!controller.signal.aborted) busy.value = false;
  }
}
const toggleArchive = () => archive.value ? reset() : load(0, 'first');
const previous = () => load(past.value.at(-1), 'previous');
watch(() => [props.run.run_id, props.category], reset);
onBeforeUnmount(() => request?.abort());
</script>

<style scoped>
.observation-panel { margin: 0 0 16px; }
.observation-body { display: grid; gap: 12px; margin-top: 12px; }
.observation-tools, .observation-detail header { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
.observation-list { display: grid; max-height: 17rem; overflow-y: auto; border-block: 1px solid var(--line); }
.observation-row { display: grid; grid-template-columns: 4ch minmax(0, 1fr) minmax(0, 1fr); gap: 12px;
  color: inherit; background: transparent; border: 0; padding: 8px; text-align: left; overflow-wrap: anywhere; }
.observation-row[aria-pressed='true'] { background: var(--accent-soft, #edf2fa); }
.observation-detail { display: grid; min-width: 0; gap: 8px; }
.observation-detail h3 { font-size: .9rem; margin: 0; }
.observation-detail dd { overflow-wrap: anywhere; }
</style>
