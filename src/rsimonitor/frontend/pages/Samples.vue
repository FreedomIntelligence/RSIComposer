<template>
  <SectionTitle :title="t('逐样本结果')" :count="shown.length">
    <select v-if="evaluations.length > 1" v-model="store.evaluationId" :aria-label="t('评测')">
      <option value="">{{ t('全部评测') }}</option>
      <option v-for="row in evaluations" :key="row.evaluation_id" :value="row.evaluation_id">{{ row.label || stageLabel(run, row.call_id || row.decision_id) || row.evaluation_id }}</option>
    </select>
    <ActionButton v-if="shown.length" :label="t('导出')" icon="download" @click="download(`${run.run_id}-samples.json`, shown)" />
  </SectionTitle>
  <DataTable :headers="[t('样本'), t('阶段'), t('正确'), t('可解析'), t('预测'), t('标准答案')]"
    :rows="shown.map(row => [row.id, stageLabel(run, row.call_id || row.decision_id), flag(row.correct), flag(row.parsed), row.predicted, row.expected])"
    :empty="t('这次运行没有上报逐样本结果')" />
</template>
<script setup>
import {computed} from 'vue';
import SectionTitle from '../components/SectionTitle.vue';
import ActionButton from '../components/ActionButton.vue';
import DataTable from '../components/DataTable.vue';
import {store, download} from '../app/store.js';
import {t} from '../app/i18n.js';
import {rowsOf, stageLabel} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const evaluations = computed(() => rowsOf(props.run.evaluations).filter(row => row.evaluation_id));
const shown = computed(() => {
  const samples = rowsOf(props.run.samples), selected = evaluations.value.find(row => row.evaluation_id === store.evaluationId);
  return selected ? samples.filter(row => (row.call_id || row.decision_id) ===
    (selected.call_id || selected.decision_id || selected.evaluation_id)) : samples;
});
const flag = value => value === true ? '✓' : value === false ? '✕' : '—';
</script>
