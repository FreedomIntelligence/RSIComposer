<template>
  <header class="page-head"><h1>{{ t('实验') }}</h1>
    <ActionButton v-if="store.compareIds.length > 1" :label="t('对比已选 {p0} 个', {p0: store.compareIds.length})" @click="navigate('compare')" /></header>
  <p v-if="!store.runs.length" class="empty-state">{{ t('还没有实验。运行 rsisys run 后会出现在这里。') }}</p>
  <section v-for="group in groups" :key="group.id" class="section">
    <SectionTitle v-if="groups.length > 1 || group.id !== 'unassigned'" :title="group.name" :count="group.runs.length" />
    <div class="table-wrap">
      <table>
        <thead><tr><th class="check"></th><th>{{ t('实验') }}</th><th>{{ t('状态') }}</th><th class="num">{{ t('阶段') }}</th><th>{{ t('开始') }}</th><th class="num">{{ t('时长') }}</th></tr></thead>
        <tbody>
          <tr v-for="row in group.runs" :key="row.run_id" :class="{selected: row.run_id === store.runId}" @click="selectRun(row.run_id)">
            <td class="check" @click.stop><input type="checkbox" :checked="store.compareIds.includes(row.run_id)"
              :aria-label="t('对比 {p0}', {p0: runLabel(row)})" @change="toggleCompare(row.run_id, $event.target.checked)"></td>
            <td><span class="strong">{{ runLabel(row) }}</span><span v-if="row.simulated === true" class="tag">{{ t('模拟') }}</span>
              <small v-if="row.error" class="cell-sub bad">{{ row.error }}</small></td>
            <td><StatusBadge :status="row.status" /></td>
            <td class="num muted">{{ row.counts?.decisions ?? '—' }}</td>
            <td class="muted">{{ dateText(row.started_at) }}</td>
            <td class="num muted">{{ durationText(seconds(row)) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
<script setup>
import {computed} from 'vue';
import SectionTitle from '../components/SectionTitle.vue';
import ActionButton from '../components/ActionButton.vue';
import StatusBadge from '../components/StatusBadge.vue';
import {store, navigate, selectRun, toggleCompare} from '../app/store.js';
import {t, dateText, durationText} from '../app/i18n.js';
import {runLabel} from '../app/insight.js';
const seconds = row => {
  const start = Date.parse(row.started_at), end = Date.parse(row.updated_at);
  return Number.isFinite(start) && Number.isFinite(end) && end >= start ? (end - start) / 1000 : undefined;
};
// Membership comes only from the snapshot's explicit project; nothing is inferred from names.
const groups = computed(() => {
  const map = new Map();
  for (const row of store.runs) {
    const id = row.project?.project_id || 'unassigned';
    if (!map.has(id)) map.set(id, {id, name: id === 'unassigned' ? t('未分组') : row.project?.name || id, runs: []});
    map.get(id).runs.push(row);
  }
  const time = row => Date.parse(row.started_at) || Date.parse(row.updated_at) || 0;
  for (const group of map.values()) group.runs.sort((a, b) => time(b) - time(a));
  return [...map.values()].sort((a, b) => (a.id === 'unassigned') - (b.id === 'unassigned') || time(b.runs[0]) - time(a.runs[0]));
});
</script>
