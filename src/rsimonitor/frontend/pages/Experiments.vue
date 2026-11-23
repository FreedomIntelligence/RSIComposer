<template>
  <header class="page-head"><h1>{{ t('实验列表') }}</h1></header>
  <section v-if="store.runs.length" class="experiment-index">
    <div class="experiment-toolbar">
      <div class="segments" role="group" :aria-label="t('按状态筛选')">
        <button v-for="filter in filters" :key="filter.id" type="button" :aria-pressed="status === filter.id" @click="status = filter.id">
          {{ t(filter.label) }} <span class="num segment-count">{{ filter.count }}</span>
        </button>
      </div>
      <div class="experiment-search">
        <Icon name="search" />
        <input v-model.trim="query" type="search" :placeholder="t('搜索实验')" :aria-label="t('搜索实验')">
      </div>
    </div>
    <div v-if="query || status !== 'all' || store.compareIds.length" class="experiment-selection">
      <span class="muted num" aria-live="polite">{{ t('{p0} 条结果', {p0: matching.length}) }}</span>
      <ActionButton v-if="store.compareIds.length" icon="git-compare-arrows" :disabled="store.compareIds.length < 2"
        :label="t('对比已选 {p0} 个', {p0: store.compareIds.length})" @click="navigate('compare')" />
      <button v-if="store.compareIds.length" type="button" class="ghost" :title="t('清除勾选')" :aria-label="t('清除勾选')"
        @click="store.compareIds.slice().forEach(id => toggleCompare(id, false))"><Icon name="x" /></button>
    </div>
    <p v-if="!matching.length" class="empty-state">{{ t('没有匹配的实验') }}</p>
    <section v-for="group in groups" :key="group.id" class="experiment-group">
      <SectionTitle v-if="groups.length > 1 || group.id !== 'unassigned'" :title="group.name" :count="group.runs.length" />
      <div class="table-wrap">
      <table :aria-label="t('实验列表')">
        <thead><tr><th class="check"></th><th scope="col">{{ t('实验') }}</th><th scope="col">{{ t('状态') }}</th>
          <th scope="col" class="num run-stages">{{ t('决策数') }}</th><th scope="col" class="run-date">{{ t('开始') }}</th>
          <th scope="col" class="run-date">{{ t('最近更新') }}</th></tr></thead>
        <tbody>
          <tr v-for="row in group.runs" :key="row.run_id" :class="{selected: store.compareIds.includes(row.run_id)}" @click="selectRun(row.run_id)">
            <td class="check" @click.stop><input type="checkbox" :checked="store.compareIds.includes(row.run_id)"
              :aria-label="t('对比 {p0}', {p0: runLabel(row)})" @change="toggleCompare(row.run_id, $event.target.checked)"></td>
            <td><button class="run-open" type="button" @click.stop="selectRun(row.run_id)">{{ runLabel(row) }}</button><span v-if="row.simulated === true" class="tag">{{ t('模拟') }}</span>
              <small v-if="row.error" class="cell-sub bad">{{ row.error }}</small></td>
            <td><StatusBadge :status="row.status" /></td>
            <td class="num muted run-stages">{{ row.counts?.decisions ?? '—' }}</td>
            <td class="muted num run-date">{{ dateText(row.started_at) }}</td>
            <td class="muted num run-date">{{ dateText(row.updated_at) }}</td>
          </tr>
        </tbody>
      </table>
      </div>
    </section>
  </section>
  <p v-else class="empty-state">{{ t('还没有实验。运行 rsicomposer run 后会出现在这里。') }}</p>
</template>
<script setup>
import {computed, ref} from 'vue';
import Icon from '../components/Icon.vue';
import SectionTitle from '../components/SectionTitle.vue';
import ActionButton from '../components/ActionButton.vue';
import StatusBadge from '../components/StatusBadge.vue';
import {store, navigate, selectRun, toggleCompare} from '../app/store.js';
import {t, dateText} from '../app/i18n.js';
import {runLabel} from '../app/insight.js';
const query = ref('');
const status = ref('all');
const category = value => ['completed', 'finished', 'succeeded'].includes(value) ? 'complete'
  : ['starting', 'queued', 'pending', 'running', 'active', 'executing', 'waiting', 'progressed'].includes(value) ? 'active' : 'other';
const filters = computed(() => {
  const counts = {all: store.runs.length, active: 0, complete: 0, other: 0};
  for (const row of store.runs) counts[category(row.status)]++;
  return [{id: 'all', label: '全部'}, {id: 'active', label: '进行中'},
    {id: 'complete', label: '已完成'}, {id: 'other', label: '其他状态'}]
    .filter(({id}) => id === 'all' || id === status.value || counts[id]).map(filter => ({...filter, count: counts[filter.id]}));
});
const matching = computed(() => {
  const terms = query.value.toLowerCase().split(/\s+/).filter(Boolean);
  return store.runs.filter(row => {
    const text = [runLabel(row), row.run_id, row.project?.name].join(' ').toLowerCase();
    return (status.value === 'all' || status.value === category(row.status)) && terms.every(term => text.includes(term));
  });
});
// Membership comes only from the snapshot's explicit project; nothing is inferred from names.
const groups = computed(() => {
  const map = new Map();
  for (const row of matching.value) {
    const id = row.project?.project_id || 'unassigned';
    if (!map.has(id)) map.set(id, {id, name: id === 'unassigned' ? t('未分组') : row.project?.name || id, runs: []});
    map.get(id).runs.push(row);
  }
  const time = row => Date.parse(row.started_at) || Date.parse(row.updated_at) || 0;
  for (const group of map.values()) group.runs.sort((a, b) => time(b) - time(a));
  return [...map.values()].sort((a, b) => (a.id === 'unassigned') - (b.id === 'unassigned') || time(b.runs[0]) - time(a.runs[0]));
});
</script>
