<template>
  <DataTable :headers="headers" :rows="rows" :numeric="numeric" :empty="t('还没有上报评测')" />
</template>
<script setup>
import {computed} from 'vue';
import DataTable from './DataTable.vue';
import {t} from '../app/i18n.js';
import {evaluationRows, evaluationMetricKeys, metricDefinition, formatMetric, stageLabel} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const all = computed(() => evaluationRows(props.run));
// Columns follow what was reported; a column no row fills (split, model) is left out.
const keys = computed(() => evaluationMetricKeys([props.run]).slice(0, 4));
const hasSplit = computed(() => all.value.some(row => row.split));
const hasModel = computed(() => all.value.some(row => row.model || row.weight_identity?.model_id || row.candidate_id));
const headers = computed(() => ['#', t('评测'), ...(hasSplit.value ? [t('数据划分')] : []), ...(hasModel.value ? [t('模型')] : []),
  ...keys.value.map(key => metricDefinition(props.run, key).label || key)]);
const numeric = computed(() => [0, ...keys.value.map((_, index) => headers.value.length - keys.value.length + index)]);
const rows = computed(() => all.value.map((row, index) => [index + 1,
  row.label || row.protocol || stageLabel(props.run, row.call_id || row.decision_id) || t('评测'),
  ...(hasSplit.value ? [row.split] : []),
  ...(hasModel.value ? [row.model || row.weight_identity?.model_id || row.candidate_id] : []),
  ...keys.value.map(key => formatMetric(row.flat[key], metricDefinition(props.run, key)))]));
</script>
